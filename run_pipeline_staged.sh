#!/usr/bin/env bash
set -euo pipefail

# Quick launcher for staged augmentation pipeline on SageMaker Spot.
# Edit the variables below to match your environment, or override via env vars.

ROLE_ARN=${ROLE_ARN:-"arn:aws:iam::872109682518:role/service-role/AmazonSageMaker-ExecutionRole-20251009T010774"}
BUCKET=${BUCKET:-"tsai-era-v4-mini-capstone"}
PREFIX=${PREFIX:-"imagenet-pipeline-run"}

# Optional: S3 path where SageMaker will mirror checkpoints for managed recovery
CKPT_S3=${CKPT_S3:-"s3://$BUCKET/$PREFIX/checkpoints"}
CKPT_LOCAL=${CKPT_LOCAL:-"/opt/ml/checkpoints"}

INSTANCE_TYPE=${INSTANCE_TYPE:-"ml.p5.48xlarge"}

INSTANCE_COUNT=${INSTANCE_COUNT:-1}
MAX_RUN=${MAX_RUN:-86400}          # 24h
MAX_WAIT=${MAX_WAIT:-90000}

# Training config
FINAL_EPOCHS=${FINAL_EPOCHS:-150}
STAGE1_FRAC=${STAGE1_FRAC:-0.5}
STAGE2_FRAC=${STAGE2_FRAC:-0.3}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-10}

AUG_POLICY=${AUG_POLICY:-"randaugment"}
RA_N=${RA_N:-2}
RA_M=${RA_M:-8}
COLOR_JITTER=${COLOR_JITTER:-0.1}
RE_P=${RE_P:-0.05}
LABEL_SMOOTH=${LABEL_SMOOTH:-0.1}

USE_DALI=${USE_DALI:-1}            # 1=true, 0=false
AUTO_INSTALL_DALI=${AUTO_INSTALL_DALI:-1}
COMPILE=${COMPILE:-1}
COMPILE_MODE=${COMPILE_MODE:-"reduce-overhead"}

LR_RANGE_ITERS=${LR_RANGE_ITERS:-100}
LR_RANGE_START=${LR_RANGE_START:-1e-1}
LR_RANGE_END=${LR_RANGE_END:-1.5}
LR_FINDER_POLICY=${LR_FINDER_POLICY:-"steepest"}
LR_AUTO_FLOOR=${LR_AUTO_FLOOR:-0.1}
LR_AUTO_CAP=${LR_AUTO_CAP:-1.5}
MAX_WORKERS=${MAX_WORKERS:-12}
PREFETCH=${PREFETCH:-4}
ONECYCLE_MAX_LR=${ONECYCLE_MAX_LR:-}      # e.g. 1e-2 to override auto LR
OVERRIDE_BATCH=${OVERRIDE_BATCH:-}   # e.g. 1024 to skip safety shrink
OVERRIDE_LR=${OVERRIDE_LR:-}
OVERRIDE_WD=${OVERRIDE_WD:-}
GRAD_ACCUM=${GRAD_ACCUM:-1}          # integer or empty; empty lets pipeline auto decide
BATCH_SAFETY_FACTOR=${BATCH_SAFETY_FACTOR:-0.95}  # e.g. 1.0 to disable shrink
SEARCH_BATCH_FRACTION=${SEARCH_BATCH_FRACTION:-1} # override search fraction (default 0.5)

ENTRY_POINT=${ENTRY_POINT:-"train_ddp_pipeline.py"}
USE_DDP=${USE_DDP:-1}            # Enable torch.distributed launcher (multi-GPU on single instance)

# Mode: run (default) or resume
MODE=${1:-run}
ADDITIONAL_EPOCHS=${ADDITIONAL_EPOCHS:-0}

PY_ARGS=(
  --role-arn "$ROLE_ARN"
  --bucket "$BUCKET"
  --prefix "$PREFIX"
  --instance-type "$INSTANCE_TYPE"
  --instance-count "$INSTANCE_COUNT"
  --max-run "$MAX_RUN"
  --max-wait "$MAX_WAIT"
  --entry-point "$ENTRY_POINT"
  --scheduler onecycle
  --final-epochs "$FINAL_EPOCHS"
  --onecycle-warmup-epochs "$WARMUP_EPOCHS"
  --aug-policy "$AUG_POLICY"
  --randaugment-n "$RA_N"
  --randaugment-m "$RA_M"
  --color-jitter "$COLOR_JITTER"
  --random-erasing-p "$RE_P"
  --label-smoothing "$LABEL_SMOOTH"
  --lr-range-iters "$LR_RANGE_ITERS"
  --lr-range-start "$LR_RANGE_START"
  --lr-range-end "$LR_RANGE_END"
  --lr-finder-policy "$LR_FINDER_POLICY"
  --lr-auto-floor "$LR_AUTO_FLOOR"
  --lr-auto-cap "$LR_AUTO_CAP"
  --max-workers "$MAX_WORKERS"
  --prefetch-factor "$PREFETCH"
  --staged-aug
  --stage1-frac "$STAGE1_FRAC"
  --stage2-frac "$STAGE2_FRAC"
  --checkpoint-s3-uri "$CKPT_S3"
  --checkpoint-local-path "$CKPT_LOCAL"
)

if [[ -n "$OVERRIDE_BATCH" ]]; then PY_ARGS+=( --override-batch "$OVERRIDE_BATCH" ); fi
if [[ -n "$OVERRIDE_LR" ]]; then PY_ARGS+=( --override-lr "$OVERRIDE_LR" ); fi
if [[ -n "$OVERRIDE_WD" ]]; then PY_ARGS+=( --override-wd "$OVERRIDE_WD" ); fi
if [[ -n "$GRAD_ACCUM" ]]; then PY_ARGS+=( --grad-accum "$GRAD_ACCUM" ); fi
if [[ -n "$BATCH_SAFETY_FACTOR" ]]; then PY_ARGS+=( --batch-safety-factor "$BATCH_SAFETY_FACTOR" ); fi
if [[ -n "$SEARCH_BATCH_FRACTION" ]]; then PY_ARGS+=( --search-batch-fraction "$SEARCH_BATCH_FRACTION" ); fi
if [[ "$USE_DDP" == "1" ]]; then
  PY_ARGS+=( --use-ddp )
fi

if [[ "$USE_DALI" == "1" ]]; then
  PY_ARGS+=( --use-dali )
fi
if [[ "$AUTO_INSTALL_DALI" == "1" ]]; then
  PY_ARGS+=( --auto-install-dali )
fi
if [[ "$COMPILE" == "1" ]]; then
  PY_ARGS+=( --compile --compile-mode "$COMPILE_MODE" )
fi

if [[ "$MODE" == "resume" ]]; then
  PY_ARGS+=( --resume )
  if [[ "$ADDITIONAL_EPOCHS" != "0" ]]; then
    PY_ARGS+=( --additional-epochs "$ADDITIONAL_EPOCHS" )
  fi
fi

python submit_sagemaker_spot_job.py "${PY_ARGS[@]}"
