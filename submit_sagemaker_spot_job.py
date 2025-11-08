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
from pathlib import Path

from sagemaker.pytorch import PyTorch
import sagemaker


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
    p.add_argument("--py-version", default="py310", help="Python version tag for framework")
    p.add_argument("--framework-version", default="2.2", help="PyTorch version for SageMaker container")
    p.add_argument("--no-spot", action="store_true", help="Disable spot instance usage")
    p.add_argument("--entry-point", default="train_from_notebook.py", help="Entry script inside source_dir (default: notebook executor)")
    # DDP training options
    p.add_argument("--use-ddp", action="store_true", help="Force DDP training entry (train_ddp.py) even on single instance")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size for DDP script")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--nesterov", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    sess = sagemaker.session.Session()

    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    job_name = args.job_name or f"imagenet-nb-{timestamp}"

    s3_output = f"s3://{args.bucket}/{args.prefix}/output"
    s3_code = f"s3://{args.bucket}/{args.prefix}/code"

    # Channels: Expect user to have uploaded ImageNet train/val tar or extracted structure
    # Provide placeholders. User should stage data separately (e.g., s3://bucket/imagenet/train/ ...)
    train_channel = f"s3://{args.bucket}/{args.prefix}/data/train/"
    val_channel = f"s3://{args.bucket}/{args.prefix}/data/val/"

    # Select entry point: use DDP script if requested or multi-instance
    entry_point = "train_ddp.py" if (args.use_ddp or args.instance_count > 1) else args.entry_point

    # Enable torch.distributed launcher if using DDP script or multi-instance
    enable_dist = args.use_ddp or (args.instance_count > 1)

    # Hyperparameters passed to entry script (DDP script consumes them; notebook executor ignores)
    hps = {
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "lr": args.lr,
        "weight-decay": args.weight_decay,
        "workers": args.workers,
    }
    if args.nesterov:
        hps["nesterov"] = True

    estimator = PyTorch(
        entry_point=entry_point,
        source_dir=str(Path.cwd()),
        role=args.role_arn,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        framework_version=args.framework_version,
        py_version=args.py_version,
        output_path=s3_output,
        base_job_name=job_name,
        hyperparameters=hps,
        use_spot_instances=not args.no-spot,
        max_wait=args.max_wait if not args.no-spot else None,
        max_run=args.max_run,
        enable_sagemaker_metrics=True,
        distribution={"torch_distributed": {"enabled": enable_dist}},
        disable_profiler=True,
    )

    inputs = {
        "train": train_channel,
        "val": val_channel,
    }

    print(f"Submitting training job: {job_name}")
    print(f"Spot instances enabled: {not args.no-spot}")
    print(f"Output path: {s3_output}")
    print(f"Data channels: train={train_channel} val={val_channel}")
    print(f"Entry point: {entry_point} | torch.distributed enabled: {enable_dist}")

    estimator.fit(inputs=inputs, job_name=job_name, wait=True)

    print("Training job submitted. Monitor in SageMaker console.")
    print(f"Executed notebook and artifacts will appear under: {s3_output}")


if __name__ == "__main__":
    main()
