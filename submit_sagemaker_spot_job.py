#!/usr/bin/env python3
"""
Local launcher to submit a SageMaker Spot training job that runs the existing notebook
by executing train_from_notebook.py inside a PyTorch Estimator.

Prerequisites:
- AWS credentials configured locally (e.g., via IAM role on Studio or ~/.aws/credentials)
- An S3 bucket you can write to (e.g. s3://your-bucket/path)
- IAM role ARN with SageMaker permissions (create training jobs, access S3)

This script packs the current working directory (including the notebook) as source_dir
and runs the entry point train_from_notebook.py on the training instance.
"""

import argparse
import os
import time
import shutil
import tempfile
from pathlib import Path

from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import sagemaker
import boto3


def parse_args():
    p = argparse.ArgumentParser(description="Submit SageMaker Spot training job from notebook")
    p.add_argument("--role-arn", required=True, help="IAM role ARN for SageMaker")
    p.add_argument("--bucket", required=True, help="S3 bucket (no s3:// prefix) e.g. my-bucket")
    p.add_argument("--prefix", default="imagenet-notebook-job", help="S3 prefix for job artifacts")
    p.add_argument("--instance-type", default="ml.p3.2xlarge", help="SageMaker instance type")
    p.add_argument("--instance-count", type=int, default=1, help="Number of instances (DDP if >1)")
    p.add_argument("--max-wait", type=int, default=7200, help="Max wait seconds for Spot capacity")
    p.add_argument("--max-run", type=int, default=7200, help="Max run seconds once started")
    p.add_argument("--job-name", default=None, help="Optional explicit job name")
    # Default to py311 as recent SageMaker PyTorch containers (e.g., 2.5.1) require Python 3.11
    p.add_argument("--py-version", default="py311", help="Python version tag for framework (e.g., py311)")
    p.add_argument("--framework-version", default="2.5.1", help="PyTorch version for SageMaker container (must be one of supported list)")
    p.add_argument("--no-spot", action="store_true", help="Disable spot instance usage")
    p.add_argument("--entry-point", default="train_from_notebook.py", help="Entry script inside source_dir (default: notebook executor)")
    p.add_argument("--no-wait", action="store_true", help="Submit job asynchronously and return immediately")
    p.add_argument("--input-mode", choices=["File", "FastFile", "Pipe"], default="File",
                   help="SageMaker channel input mode. FastFile streams from S3, reducing startup download time.")
    # DDP training options
    p.add_argument("--use-ddp", action="store_true", help="Force DDP training entry (train_ddp.py) even on single instance")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size for DDP script")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--nesterov", action="store_true")
    # Pipeline (train_ddp_pipeline.py) options
    p.add_argument("--scheduler", choices=["onecycle", "cosine"], default="onecycle",
                   help="LR scheduler for pipeline entry (onecycle or cosine)")
    p.add_argument("--final-epochs", type=int, default=None,
                   help="Epochs for final training stage in pipeline; if omitted, uses --epochs")
    p.add_argument("--onecycle-pct-start", type=float, default=0.1,
                   help="OneCycleLR pct_start (fraction of cycle spent increasing LR)")
    p.add_argument("--onecycle-max-lr", type=float, default=None,
                   help="Override OneCycleLR max_lr (defaults to LR discovered by LR finder)")
    p.add_argument("--onecycle-warmup-epochs", type=int, default=None,
                   help="Explicit warmup epochs for OneCycleLR (overrides pct_start when set)")
    # Augmentation / regularization passthrough for pipeline script
    p.add_argument("--aug-policy", choices=["none","autoaugment","randaugment"], default="none",
                   help="Augmentation policy for pipeline training script")
    p.add_argument("--randaugment-n", type=int, default=2, help="RandAugment number of ops")
    p.add_argument("--randaugment-m", type=int, default=9, help="RandAugment magnitude (0-10)")
    p.add_argument("--color-jitter", type=float, default=0.0, help="ColorJitter strength (0 disables)")
    p.add_argument("--random-erasing-p", type=float, default=0.0, help="RandomErasing probability (0 disables)")
    p.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing value")
    # DALI toggle
    p.add_argument("--use-dali", action="store_true", help="Enable NVIDIA DALI data pipeline if available in training container")
    p.add_argument("--auto-install-dali", action="store_true", help="Attempt to pip install NVIDIA DALI on the training container if missing (pipeline entry only)")
    p.add_argument("--prefetch-factor", type=int, default=None, help="Prefetch factor for DataLoader workers (pipeline entry only)")
    # Early stopping target
    p.add_argument("--target-val-top1", type=float, default=None, help="Stop training early when validation top1 (percent) reaches this value")
    # Compile toggle (pipeline)
    p.add_argument("--compile", action="store_true", help="Enable torch.compile in pipeline entry")
    p.add_argument("--compile-mode", choices=["default","reduce-overhead","max-autotune"], default="reduce-overhead",
                   help="torch.compile mode for pipeline entry")
    # LR finder/auto-scale controls
    p.add_argument("--lr-range-iters", type=int, default=100)
    p.add_argument("--lr-range-start", type=float, default=1e-5)
    p.add_argument("--lr-range-end", type=float, default=1.0)
    p.add_argument("--lr-finder-policy", choices=["min0.1","steepest","steepest-pre-min"], default="min0.1",
                   help="Policy for selecting LR after range test ('min0.1','steepest','steepest-pre-min').")
    p.add_argument("--lr-finder-at-final-batch", action="store_true",
                   help="Run LR finder at final (post-discovery) batch size instead of search batch fraction.")
    p.add_argument("--lr-auto-floor", type=float, default=1e-3)
    p.add_argument("--lr-auto-cap", type=float, default=2.5e-1)
    # Staged augmentation controls (pipeline)
    p.add_argument("--staged-aug", action="store_true", help="Enable 4-stage augmentation schedule (high OneCycle / high Cosine / medium OneCycle / medium Cosine)")
    p.add_argument("--stage1-frac", type=float, default=0.4, help="Fraction of total epochs for stage 1 (high aug, OneCycle)")
    p.add_argument("--stage2-frac", type=float, default=0.4, help="Fraction of total epochs for stage 2 (high aug, Cosine)")
    p.add_argument("--stage3-frac", type=float, default=0.1, help="Fraction of total epochs for stage 3 (medium aug, OneCycle)")
    p.add_argument("--stage4-frac", type=float, default=0.1, help="Fraction of total epochs for stage 4 (medium aug, Cosine)")
    # Per-stage overrides (optional)
    p.add_argument("--stage1-wd", type=float, default=None, help="Override weight decay for stage 1")
    p.add_argument("--stage2-wd", type=float, default=None, help="Override weight decay for stage 2")
    p.add_argument("--stage3-wd", type=float, default=None, help="Override weight decay for stage 3")
    p.add_argument("--stage4-wd", type=float, default=None, help="Override weight decay for stage 4")
    p.add_argument("--stage1-lr", type=float, default=None, help="Override base LR for stage 1")
    p.add_argument("--stage2-lr", type=float, default=None, help="Override base LR for stage 2")
    p.add_argument("--stage3-lr", type=float, default=None, help="Override base LR for stage 3")
    p.add_argument("--stage4-lr", type=float, default=None, help="Override base LR for stage 4")
    p.add_argument("--stage1-onecycle-max-lr", type=float, default=None, help="Override OneCycle max LR for stage 1 (if OneCycle)")
    p.add_argument("--stage3-onecycle-max-lr", type=float, default=None, help="Override OneCycle max LR for stage 3 (if OneCycle)")
    # Pipeline tuning/overrides for efficiency
    p.add_argument("--override-batch", type=int, default=None, help="Force per-GPU batch size for pipeline final training")
    p.add_argument("--override-lr", type=float, default=None, help="Force LR for final training (skip LR auto-scale)")
    p.add_argument("--override-wd", type=float, default=None, help="Force weight decay for final training")
    p.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps (0=auto in pipeline)")
    p.add_argument("--batch-safety-factor", type=float, default=None, help="Safety headroom multiplier applied to discovered max batch (default 0.9)")
    p.add_argument("--search-batch-fraction", type=float, default=None, help="Fraction of discovered max batch to use during LR/WD search phases (default 0.5)")
    # Skips for pipeline discovery phases
    p.add_argument("--skip-lr-finder", action="store_true", help="Skip LR finder in pipeline (requires override-lr)")
    p.add_argument("--skip-wd-finder", action="store_true", help="Skip WD finder in pipeline (requires override-wd)")
    p.add_argument("--skip-batch-finder", action="store_true", help="Skip batch-size finder in pipeline (requires override-batch)")
    p.add_argument("--one-stage-aug-split", type=float, default=None, help="When not using staged-aug, apply augmentation for first <frac> of epochs then none (e.g., 0.8)")
    # Checkpoint / resume (applies to both DDP and pipeline entries)
    p.add_argument("--checkpoint-s3-uri", type=str, default=None, help="S3 URI for SageMaker managed checkpoints (e.g. s3://bucket/path/checkpoints)")
    p.add_argument("--checkpoint-local-path", type=str, default="/opt/ml/checkpoints", help="Local path inside container for checkpoints")
    p.add_argument("--resume", action="store_true", help="Resume training (pipeline: load latest checkpoint & continue)")
    p.add_argument("--additional-epochs", type=int, default=0, help="Additional epochs beyond last completed when using --resume (pipeline)")
    # Pipeline worker optimizer cap
    p.add_argument("--max-workers", type=int, default=None, help="Max num_workers probe cap for pipeline worker optimization")
    return p.parse_args()


SUPPORTED_PYTORCH = [
    "1.13.1", "2.0.0", "2.0.1", "2.1.0", "2.1.2", "2.2.0", "2.3.0", "2.3.1", "2.4.1", "2.5.1"
]


def normalize_framework_version(raw: str) -> str:
    """Cope with shorthand like '2.2' -> '2.2.0' and validate against supported list."""
    if raw.count('.') == 1:  # add patch zero if missing
        candidate = raw + ".0"
        if candidate in SUPPORTED_PYTORCH:
            return candidate
    if raw in SUPPORTED_PYTORCH:
        return raw
    raise ValueError(
        f"framework_version '{raw}' is not supported. Choose one of: {SUPPORTED_PYTORCH}"
    )


def main():
    args = parse_args()
    try:
        args.framework_version = normalize_framework_version(args.framework_version)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return
    # Quick compatibility shim: certain framework versions only ship with specific Python versions
    # PyTorch 2.5.1 containers on SageMaker are only available for Python 3.11 (py311)
    if args.framework_version == "2.5.1" and args.py_version != "py311":
        print(f"[WARN] framework_version {args.framework_version} requires py311 on SageMaker. "
              f"Overriding requested {args.py_version} -> py311")
        args.py_version = "py311"
    sess = sagemaker.session.Session()
    print(f"Using AWS region: {sess.boto_region_name}")

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    job_name = args.job_name or f"imagenet-nb-{timestamp}"

    s3_output = f"s3://{args.bucket}/{args.prefix}/output"
    s3_code = f"s3://{args.bucket}/{args.prefix}/code"

    # Preflight: ensure bucket region matches current region (avoids subtle hangs)
    def _bucket_region(bucket: str) -> str:
        s3 = boto3.client("s3", region_name=sess.boto_region_name)
        resp = s3.get_bucket_location(Bucket=bucket)
        loc = resp.get("LocationConstraint")
        return loc or "us-east-1"

    try:
        bucket_region = _bucket_region(args.bucket)
        if bucket_region != sess.boto_region_name:
            print(f"[WARN] Bucket {args.bucket} is in region '{bucket_region}' but session is '{sess.boto_region_name}'.\n"
                  f"       For fastest uploads and to avoid job creation issues, use a bucket in '{sess.boto_region_name}' or set AWS region accordingly.")
    except Exception as e:
        print(f"[WARN] Could not verify bucket region: {e}")

    # Channels: Expect user to have uploaded ImageNet train/val tar or extracted structure
    # Provide placeholders. User should stage data separately (e.g., s3://bucket/imagenet/train/ ...)
    train_channel = f"s3://{args.bucket}/Datasets/imagenet1k/ILSVRC/Data/CLS-LOC/train/"
    val_channel = f"s3://{args.bucket}/Datasets/imagenet1k/ILSVRC/imagenet-sagemaker/val/"

    # Select entry point:
    # - If user explicitly provided an entry_point (not the default), always respect it
    # - Otherwise, auto-pick train_ddp.py when DDP or multi-instance is requested
    if args.entry_point != "train_from_notebook.py":
        entry_point = args.entry_point
    else:
        entry_point = "train_ddp.py" if (args.use_ddp or args.instance_count > 1) else args.entry_point

    # Enable torch.distributed launcher if using DDP script or multi-instance
    enable_dist = args.use_ddp or (args.instance_count > 1)

    # Hyperparameters passed to entry script
    # - For pipeline entry, pass pipeline-specific flags
    # - For DDP training entry, pass ddp-specific flags
    if entry_point == "train_ddp_pipeline.py":
        final_epochs = args.final_epochs if args.final_epochs is not None else args.epochs
        hps = {
            "scheduler": args.scheduler,
            "final-epochs": final_epochs,
            # Provide sensible defaults for pipeline discovery inputs
            # (pipeline has its own defaults; we only forward essentials)
        }
        if args.target_val_top1 is not None:
            hps["target-val-top1"] = args.target_val_top1
        if args.scheduler == "onecycle":
            hps["onecycle-pct-start"] = args.onecycle_pct_start
            if args.onecycle_max_lr is not None:
                hps["onecycle-max-lr"] = args.onecycle_max_lr
            if args.onecycle_warmup_epochs is not None:
                hps["onecycle-warmup-epochs"] = args.onecycle_warmup_epochs
        # Augmentation/regularization flags for pipeline
        if args.aug_policy is not None:
            hps["aug-policy"] = args.aug_policy
        hps["randaugment-n"] = args.randaugment_n
        hps["randaugment-m"] = args.randaugment_m
        hps["color-jitter"] = args.color_jitter
        hps["random-erasing-p"] = args.random_erasing_p
        hps["label-smoothing"] = args.label_smoothing
        hps["use-dali"] = args.use_dali
        if args.auto_install_dali:
            hps["auto-install-dali"] = True
        if args.prefetch_factor is not None:
            hps["prefetch-factor"] = args.prefetch_factor
        if args.max_workers is not None:
            hps["max-workers"] = args.max_workers
        if args.compile:
            hps["compile"] = True
            hps["compile-mode"] = args.compile_mode
        # LR finder/auto-scale controls
        hps["lr-range-iters"] = args.lr_range_iters
        hps["lr-range-start"] = args.lr_range_start
        hps["lr-range-end"] = args.lr_range_end
        hps["lr-finder-policy"] = args.lr_finder_policy
        hps["lr-auto-floor"] = args.lr_auto_floor
        hps["lr-auto-cap"] = args.lr_auto_cap
        # Optional skips for discovery phases
        if args.skip_lr_finder:
            hps["skip-lr-finder"] = True
        if args.skip_wd_finder:
            hps["skip-wd-finder"] = True
        if getattr(args, "skip_batch_finder", False):
            hps["skip-batch-finder"] = True
        # Optional overrides / tuning
        if args.override_batch is not None:
            hps["override-batch"] = args.override_batch
        if args.override_lr is not None:
            hps["override-lr"] = args.override_lr
        if args.override_wd is not None:
            hps["override-wd"] = args.override_wd
        if args.grad_accum is not None:
            hps["grad-accum"] = args.grad_accum
        if args.batch_safety_factor is not None:
            hps["batch-safety-factor"] = args.batch_safety_factor
        if args.search_batch_fraction is not None:
            hps["search-batch-fraction"] = args.search_batch_fraction
        # Resume/checkpoint options for pipeline
        if args.resume:
            hps["resume"] = True
        if args.additional_epochs:
            hps["additional-epochs"] = args.additional_epochs
        hps["checkpoint-dir"] = args.checkpoint_local_path
        # Staged augmentation flags
        if args.staged_aug:
            hps["staged-aug"] = True
            hps["stage1-frac"] = args.stage1_frac
            hps["stage2-frac"] = args.stage2_frac
            hps["stage3-frac"] = args.stage3_frac
            hps["stage4-frac"] = args.stage4_frac
            # Forward stage-specific overrides if provided
            if args.stage1_wd is not None: hps["stage1-wd"] = args.stage1_wd
            if args.stage2_wd is not None: hps["stage2-wd"] = args.stage2_wd
            if args.stage3_wd is not None: hps["stage3-wd"] = args.stage3_wd
            if args.stage4_wd is not None: hps["stage4-wd"] = args.stage4_wd
            if args.stage1_lr is not None: hps["stage1-lr"] = args.stage1_lr
            if args.stage2_lr is not None: hps["stage2-lr"] = args.stage2_lr
            if args.stage3_lr is not None: hps["stage3-lr"] = args.stage3_lr
            if args.stage4_lr is not None: hps["stage4-lr"] = args.stage4_lr
            if args.stage1_onecycle_max_lr is not None: hps["stage1-onecycle-max-lr"] = args.stage1_onecycle_max_lr
            if args.stage3_onecycle_max_lr is not None: hps["stage3-onecycle-max-lr"] = args.stage3_onecycle_max_lr
        # LR finder at final batch size toggle
        if getattr(args, "lr_finder_at_final_batch", False):
            hps["lr-finder-at-final-batch"] = True
        # Single-stage augmentation split (only meaningful when not staged-aug inside pipeline script)
        if args.one_stage_aug_split is not None:
            hps["one-stage-aug-split"] = args.one_stage_aug_split
    else:
        hps = {
            "epochs": args.epochs,
            "batch-size": args.batch_size,
            "lr": args.lr,
            "weight-decay": args.weight_decay,
            "workers": args.workers,
        }
        if args.target_val_top1 is not None:
            hps["target-val-top1"] = args.target_val_top1
        if args.nesterov:
            hps["nesterov"] = True

    # Build a minimal source_dir to avoid uploading large local datasets (e.g., imagenet1k)
    def build_source_dir() -> str:
        tmpdir = Path(tempfile.mkdtemp(prefix="sm_src_"))
        # Copy only the necessary files
        want = [
            "train_from_notebook.py",
            "train_ddp.py",
            "train_ddp_pipeline.py",
            "TSAI_mini_capstone_imagnet1k_resnet50.ipynb",
            "submit_sagemaker_spot_job.py",
            "requirements.txt",
        ]
        for fname in want:
            src = Path.cwd() / fname
            if src.exists():
                shutil.copy2(src, tmpdir / fname)
        # Create a tiny README for traceability
        (tmpdir / "README.txt").write_text("Minimal source package for SageMaker training job.\n")
        return str(tmpdir)

    minimal_source = build_source_dir()

    # Environment variables for stability and memory behavior
    env = {
        # Help reduce fragmentation-related OOM
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # Better NCCL error propagation when any rank fails (new var) and fallback old var
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        # Increase NCCL heartbeat timeout to reduce spurious watchdog terminations on heavy kernels
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "1200",
        # Enable detailed NCCL logging when needed for debugging
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "ALL",
    }

    estimator = PyTorch(
        entry_point=entry_point,
        source_dir=minimal_source,
        role=args.role_arn,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        framework_version=args.framework_version,
        py_version=args.py_version,
        output_path=s3_output,
        base_job_name=job_name,
        hyperparameters=hps,
        environment=env,
        use_spot_instances=not args.no_spot,
        max_wait=args.max_wait if not args.no_spot else None,
        max_run=args.max_run,
        enable_sagemaker_metrics=True,
        sagemaker_session=sess,
        distribution={"torch_distributed": {"enabled": enable_dist}},
        disable_profiler=True,
        checkpoint_s3_uri=args.checkpoint_s3_uri,
        checkpoint_local_path=args.checkpoint_local_path,
    )

    inputs = {
        "train": TrainingInput(s3_data=train_channel, input_mode=args.input_mode, distribution="FullyReplicated"),
        "val": TrainingInput(s3_data=val_channel, input_mode=args.input_mode, distribution="FullyReplicated"),
    }

    print(f"Submitting training job: {job_name}")
    print(f"Spot instances enabled: {not args.no_spot}")
    print(f"Output path: {s3_output}")
    print(f"Data channels: train={train_channel} val={val_channel}")
    print(f"Input mode: {args.input_mode}")
    print(f"Entry point: {entry_point} | torch.distributed enabled: {enable_dist}")
    if entry_point == "train_ddp_pipeline.py":
        line = f"Pipeline settings → scheduler={hps.get('scheduler')} final-epochs={hps.get('final-epochs')}"
        if hps.get('scheduler') == 'onecycle':
            line += f" onecycle-pct-start={hps.get('onecycle-pct-start')} onecycle-warmup-epochs={hps.get('onecycle-warmup-epochs','auto')} onecycle-max-lr={hps.get('onecycle-max-lr', 'auto')}"
        line += (f" aug-policy={hps.get('aug-policy','none')} randaugment(n={hps.get('randaugment-n')},m={hps.get('randaugment-m')})"
                 f" color-jitter={hps.get('color-jitter')} random-erasing-p={hps.get('random-erasing-p')}"
                 f" label-smoothing={hps.get('label-smoothing')} use-dali={hps.get('use-dali', False)} lr-finder-policy={hps.get('lr-finder-policy')}"
                 f" lr-range=[{hps.get('lr-range-start')},{hps.get('lr-range-end')}] lr-auto-floor={hps.get('lr-auto-floor')} lr-auto-cap={hps.get('lr-auto-cap')}")
        line += (f" compile={hps.get('compile', False)} compile-mode={hps.get('compile-mode','reduce-overhead')}"
                 f" max-workers={hps.get('max-workers','auto')} prefetch-factor={hps.get('prefetch-factor','default')}"
                 f" auto-install-dali={hps.get('auto-install-dali', False)}")
        if hps.get('staged-aug'):
            line += (f" staged-aug=True stage1-frac={hps.get('stage1-frac')} stage2-frac={hps.get('stage2-frac')}"
                     f" stage3-frac={hps.get('stage3-frac')} stage4-frac={hps.get('stage4-frac')}")
        if hps.get('lr-finder-at-final-batch'):
            line += " lr-finder-at-final-batch=True"
        if hps.get('target-val-top1') is not None:
            line += f" target-val-top1={hps.get('target-val-top1')}"
        print(line)
    print(f"Using PyTorch framework_version={args.framework_version} with {args.py_version}")

    import traceback
    try:
        print("Submitting to SageMaker... (this can take a few seconds while source is uploaded)")
        estimator.fit(inputs=inputs, job_name=job_name, wait=not args.no_wait)
    except Exception as e:
        print(f"[ERROR] Failed to submit or run training job: {e}")
        traceback.print_exc()
        raise
    finally:
        try:
            jt = getattr(estimator, "latest_training_job", None)
            if jt is not None:
                desc = jt.describe()
                print(f"TrainingJobArn: {desc.get('TrainingJobArn', 'N/A')}")
                print(f"TrainingJobStatus: {desc.get('TrainingJobStatus', 'N/A')}")
        except Exception:
            pass

    print("Training job submitted. Monitor in SageMaker console.")
    print(f"Executed notebook and artifacts will appear under: {s3_output}")


if __name__ == "__main__":
    main()
