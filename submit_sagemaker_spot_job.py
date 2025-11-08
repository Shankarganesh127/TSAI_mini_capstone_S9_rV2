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
    p.add_argument("--input-mode", choices=["File", "FastFile", "Pipe"], default="FastFile",
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
    else:
        hps = {
            "epochs": args.epochs,
            "batch-size": args.batch_size,
            "lr": args.lr,
            "weight-decay": args.weight_decay,
            "workers": args.workers,
        }
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
        ]
        for fname in want:
            src = Path.cwd() / fname
            if src.exists():
                shutil.copy2(src, tmpdir / fname)
        # Create a tiny README for traceability
        (tmpdir / "README.txt").write_text("Minimal source package for SageMaker training job.\n")
        return str(tmpdir)

    minimal_source = build_source_dir()

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
        use_spot_instances=not args.no_spot,
        max_wait=args.max_wait if not args.no_spot else None,
        max_run=args.max_run,
        enable_sagemaker_metrics=True,
        sagemaker_session=sess,
        distribution={"torch_distributed": {"enabled": enable_dist}},
        disable_profiler=True,
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
                 f" label-smoothing={hps.get('label-smoothing')}")
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
