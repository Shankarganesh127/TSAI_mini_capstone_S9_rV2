#!/usr/bin/env bash
set -euo pipefail

# Enable expandable CUDA allocator segments to mitigate fragmentation-related OOMs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL stability and diagnostics
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1200
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

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
MAX_RUN=${MAX_RUN:-86400} #${MAX_RUN:-172800}          # 48h
MAX_WAIT=${MAX_WAIT:-90000} #${MAX_WAIT:-180000}

# Training config
FINAL_EPOCHS=${FINAL_EPOCHS:-200}
# Fractions to achieve epochs: 60 / 120 / 160 / 200 -> 0.4 / 0.3 / 0.2 / 0.1
STAGE1_FRAC=${STAGE1_FRAC:-0.4}
STAGE2_FRAC=${STAGE2_FRAC:-0.3}
STAGE3_FRAC=${STAGE3_FRAC:-0.2}
STAGE4_FRAC=${STAGE4_FRAC:-0.1}
STAGED_AUG=${STAGED_AUG:-1}  # 1 to enable 4-stage augmentation schedule
ONECYCLE_PCT_START=${ONECYCLE_PCT_START:-0.1}
# Leave empty to let pct_start control warmup length; set a number to override pct_start globally
WARMUP_EPOCHS=${WARMUP_EPOCHS:-}

AUG_POLICY=${AUG_POLICY:-"randaugment"}
RA_N=${RA_N:-2}
RA_M=${RA_M:-12}
COLOR_JITTER=${COLOR_JITTER:-0.1}
RE_P=${RE_P:-0.1}
LABEL_SMOOTH=${LABEL_SMOOTH:-0.1}

USE_DALI=${USE_DALI:-1}            # 1=true, 0=false
AUTO_INSTALL_DALI=${AUTO_INSTALL_DALI:-1}
COMPILE=${COMPILE:-1}
COMPILE_MODE=${COMPILE_MODE:-"reduce-overhead"}

LR_RANGE_ITERS=${LR_RANGE_ITERS:-200}
LR_RANGE_START=${LR_RANGE_START:-1e-2}
LR_RANGE_END=${LR_RANGE_END:-0.2}
LR_FINDER_POLICY=${LR_FINDER_POLICY:-"steepest-pre-min"}
LR_FINDER_AT_FINAL_BATCH=${LR_FINDER_AT_FINAL_BATCH:-1}  # 1=true run finder at final batch size
LR_AUTO_FLOOR=${LR_AUTO_FLOOR:-1e-2}
LR_AUTO_CAP=${LR_AUTO_CAP:-2e-1}
MAX_WORKERS=${MAX_WORKERS:-16}
PREFETCH=${PREFETCH:-4}
ONECYCLE_MAX_LR=${ONECYCLE_MAX_LR:-0.1}      # e.g. 1e-2 to override auto LR
TARGET_VAL_TOP1=${TARGET_VAL_TOP1:-80.0}       # e.g. 76.0 to early stop when reached
OVERRIDE_BATCH=${OVERRIDE_BATCH:-1024}   # e.g. 1024 to skip safety shrink
OVERRIDE_LR=${OVERRIDE_LR:-0.1}
OVERRIDE_WD=${OVERRIDE_WD:-5e-05}
GRAD_ACCUM=${GRAD_ACCUM:-1}          # integer or empty; empty lets pipeline auto decide
BATCH_SAFETY_FACTOR=${BATCH_SAFETY_FACTOR:-0.95}  # more headroom to avoid OOM
SEARCH_BATCH_FRACTION=${SEARCH_BATCH_FRACTION:-1.0} # use smaller batch during LR/WD search
SKIP_BATCH_FINDER=${SKIP_BATCH_FINDER:-1}   # 1 to skip batch finder (requires OVERRIDE_BATCH)
SKIP_LR_FINDER=${SKIP_LR_FINDER:-1}   # 1 to skip LR finder (requires OVERRIDE_LR)
SKIP_WD_FINDER=${SKIP_WD_FINDER:-1}   # 1 to skip WD finder (requires OVERRIDE_WD)
ONE_STAGE_AUG_SPLIT=${ONE_STAGE_AUG_SPLIT:-0.8}  # e.g. 0.8 to use heavy aug for first 80% then none for remaining (single-stage only)

# Per-stage optional overrides (blank to skip). These only apply if STAGED_AUG=1.
STAGE1_WD=${STAGE1_WD:-5e-5}
STAGE2_WD=${STAGE2_WD:-5e-5}
STAGE3_WD=${STAGE3_WD:-2e-5}
STAGE4_WD=${STAGE4_WD:-1e-5}
STAGE1_LR=${STAGE1_LR:-1e-2}          # OneCycle base (start) LR for stage 1
STAGE2_LR=${STAGE2_LR:-1e-3}          # Cosine start LR for stage 2
STAGE3_LR=${STAGE3_LR:-5e-4}          # OneCycle base LR for stage 3
STAGE4_LR=${STAGE4_LR:-5e-4}          # Cosine start LR for stage 4
STAGE1_ONECYCLE_MAX_LR=${STAGE1_ONECYCLE_MAX_LR:-1e-1}   # Peak LR stage 1
STAGE3_ONECYCLE_MAX_LR=${STAGE3_ONECYCLE_MAX_LR:-5e-3}   # Peak LR stage 3

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
  --onecycle-pct-start "$ONECYCLE_PCT_START"
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

if [[ -n "$ONECYCLE_MAX_LR" ]]; then
  PY_ARGS+=( --onecycle-max-lr "$ONECYCLE_MAX_LR" )
fi
# Only pass warmup epochs if explicitly set; otherwise pct_start will be used
if [[ -n "$WARMUP_EPOCHS" ]]; then
  PY_ARGS+=( --onecycle-warmup-epochs "$WARMUP_EPOCHS" )
fi
if [[ -n "$TARGET_VAL_TOP1" ]]; then
  PY_ARGS+=( --target-val-top1 "$TARGET_VAL_TOP1" )
fi
if [[ "$LR_FINDER_AT_FINAL_BATCH" == "1" ]]; then
  PY_ARGS+=( --lr-finder-at-final-batch )
fi
if [[ -n "$ONE_STAGE_AUG_SPLIT" ]]; then
  PY_ARGS+=( --one-stage-aug-split "$ONE_STAGE_AUG_SPLIT" )
fi
if [[ "$SKIP_BATCH_FINDER" == "1" ]]; then
  PY_ARGS+=( --skip-batch-finder )
fi
if [[ "$SKIP_LR_FINDER" == "1" ]]; then
  PY_ARGS+=( --skip-lr-finder )
fi
if [[ "$SKIP_WD_FINDER" == "1" ]]; then
  PY_ARGS+=( --skip-wd-finder )
fi

if [[ "$STAGED_AUG" == "1" ]]; then
  PY_ARGS+=( --staged-aug --stage1-frac "$STAGE1_FRAC" --stage2-frac "$STAGE2_FRAC" --stage3-frac "$STAGE3_FRAC" --stage4-frac "$STAGE4_FRAC" )
  # Forward overrides only if non-empty (default values count as provided)
  [[ -n "$STAGE1_WD" ]] && PY_ARGS+=( --stage1-wd "$STAGE1_WD" )
  [[ -n "$STAGE2_WD" ]] && PY_ARGS+=( --stage2-wd "$STAGE2_WD" )
  [[ -n "$STAGE3_WD" ]] && PY_ARGS+=( --stage3-wd "$STAGE3_WD" )
  [[ -n "$STAGE4_WD" ]] && PY_ARGS+=( --stage4-wd "$STAGE4_WD" )
  [[ -n "$STAGE1_LR" ]] && PY_ARGS+=( --stage1-lr "$STAGE1_LR" )
  [[ -n "$STAGE2_LR" ]] && PY_ARGS+=( --stage2-lr "$STAGE2_LR" )
  [[ -n "$STAGE3_LR" ]] && PY_ARGS+=( --stage3-lr "$STAGE3_LR" )
  [[ -n "$STAGE4_LR" ]] && PY_ARGS+=( --stage4-lr "$STAGE4_LR" )
  [[ -n "$STAGE1_ONECYCLE_MAX_LR" ]] && PY_ARGS+=( --stage1-onecycle-max-lr "$STAGE1_ONECYCLE_MAX_LR" )
  [[ -n "$STAGE3_ONECYCLE_MAX_LR" ]] && PY_ARGS+=( --stage3-onecycle-max-lr "$STAGE3_ONECYCLE_MAX_LR" )
fi

if [[ "$MODE" == "resume" ]]; then
  PY_ARGS+=( --resume )
  if [[ "$ADDITIONAL_EPOCHS" != "0" ]]; then
    PY_ARGS+=( --additional-epochs "$ADDITIONAL_EPOCHS" )
  fi
fi

python submit_sagemaker_spot_job.py "${PY_ARGS[@]}"
