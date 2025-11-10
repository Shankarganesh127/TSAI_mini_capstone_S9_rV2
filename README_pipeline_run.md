# Pipeline Training & Staged Augmentation

## Purpose
This README documents how to launch the pipeline training job on SageMaker Spot instances with checkpoint/resume and optional 3-stage augmentation.

## Base Submit Script
Use `submit_sagemaker_spot_job.py` to create a Spot training job.

## Required Environment
- AWS credentials / execution role with SageMaker + S3 access.
- S3 bucket containing ImageNet-style train/val directories or tars already extracted.
- This repository uploaded to Studio / local environment.

## Checkpointing & Resume
Estimator is configured to mirror checkpoints between `/opt/ml/model` and a local checkpoint dir (`/opt/ml/checkpoints`), which SageMaker will sync to S3 if `checkpoint_s3_uri` is set in an extended version (not shown in minimal submit script yet).
Pass `--resume` and optionally `--additional-epochs` in subsequent run to extend training beyond the last completed epoch.

## Staged Augmentation (Optional)
Enable `--staged-aug` for three phases:
1. Stage 1 (OneCycle, base augmentation) – fraction `--stage1-frac` (default 0.5)
2. Stage 2 (OneCycle, medium augmentation bump) – fraction `--stage2-frac` (default 0.3)
3. Stage 3 (Cosine, strong augmentation bump) – remaining epochs.

Augmentation bumps escalate `randaugment` magnitude, `color_jitter`, and `random_erasing_p` automatically when base policy is not `none`.

## Recommended Final Run Command (Example)
Adjust bucket, role ARN, and data channel prefixes for your environment. This example targets 150 total epochs, Spot with generous runtime.

```bash
python submit_sagemaker_spot_job.py \
  --role-arn arn:aws:iam::<ACCOUNT_ID>:role/<SAGEMAKER_EXEC_ROLE> \
  --bucket <YOUR_BUCKET> \
  --prefix imagenet-pipeline-run \
  --instance-type ml.p4d.24xlarge \
  --instance-count 1 \
  --max-run 86400 \
  --max-wait 90000 \
  --entry-point train_ddp_pipeline.py \
  --scheduler onecycle \
  --final-epochs 150 \
  --onecycle-warmup-epochs 10 \
  --aug-policy randaugment \
  --randaugment-n 2 \
  --randaugment-m 8 \
  --color-jitter 0.1 \
  --random-erasing-p 0.05 \
  --label-smoothing 0.1 \
  --use-dali \
  --auto-install-dali \
  --compile \
  --compile-mode reduce-overhead \
  --lr-range-iters 150 \
  --lr-range-start 1e-5 \
  --lr-range-end 1.0 \
  --lr-finder-policy min0.1 \
  --lr-auto-floor 1e-3 \
  --lr-auto-cap 2.0e-1 \
  --max-workers 16 \
  --prefetch-factor 4 \
  --staged-aug \
  --stage1-frac 0.5 \
  --stage2-frac 0.3
```

### Resume + Extend 30 More Epochs
If job interrupted after (e.g.) 120 epochs, resume to reach 180 total:
```bash
python submit_sagemaker_spot_job.py \
  --role-arn arn:aws:iam::<ACCOUNT_ID>:role/<SAGEMAKER_EXEC_ROLE> \
  --bucket <YOUR_BUCKET> \
  --prefix imagenet-pipeline-run-resume \
  --instance-type ml.p4d.24xlarge \
  --instance-count 1 \
  --max-run 100000 \
  --max-wait 110000 \
  --entry-point train_ddp_pipeline.py \
  --scheduler onecycle \
  --final-epochs 150 \
  --resume \
  --additional-epochs 30 \
  --staged-aug --stage1-frac 0.5 --stage2-frac 0.3
```
The script will detect the last checkpoint epoch and extend `final_epochs` = last_epoch + additional_epochs.

## Notes
- Adjust `--instance-count` >1 for multi-node DDP; ensure appropriate Spot capacity and increase `--max-wait`.
- For large models or multi-node, consider raising `--lr-range-iters` proportionally.
- If CUDA version mismatch occurs for DALI, swap wheel name in `requirements.txt` (e.g., use `nvidia-dali-cuda118`).
- Remove `--compile` if initial compilation overhead is undesirable.

## Troubleshooting
| Symptom | Resolution |
|---------|------------|
| DALI install fails | Check CUDA version in container, edit `requirements.txt` accordingly. |
| Resume ignored | Ensure `--resume` flag passed and checkpoint files exist in `/opt/ml/checkpoints` or `/opt/ml/model`. |
| OOM at start | Reduce `--randaugment-m`, or lower batch by providing `--override-batch` in pipeline (add flag if needed). |
| LR too high | Override with `--onecycle-max-lr` or set `--override-lr` (future enhancement to add override flag to submit script). |

## Future Enhancements
- Per-stage explicit augmentation flags.
- Automatic `max-run` calculation from epoch timing.
- Scheduler state restoration for exact LR continuity on resume.

---
Generated on: 2025-11-09
