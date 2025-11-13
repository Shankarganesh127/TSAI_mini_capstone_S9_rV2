#!/usr/bin/env python3
"""
Multi-step ImageNet training pipeline with DDP support:
 1. Optimize num_workers (rank0)
 2. Find max batch size (power-of-two) fitting GPU memory (rank0)
 3. LR range test (rank0 subset)
 4. Weight decay search (rank0 quick trial)
 5. Full training using discovered hyperparameters
 6. Unified logging + saving plots to logs/

Design Principles:
 - Expensive search steps only run on rank0; results broadcast to all ranks.
 - Memory-sensitive batch size finder uses a small number of batches.
 - LR Finder & WD search use limited iterations for speed; configurable.
 - All plots saved as PNG under /opt/ml/model/logs and also optionally to /opt/ml/model.
 - AMP (autocast + GradScaler) enabled if CUDA available.

Environment:
 - Expects ImageNet-like folder structure under /opt/ml/input/data/{train,val}
 - Use SageMaker PyTorch estimator with torch_distributed enabled.
"""
from __future__ import annotations
import argparse, os, time, json, math
# Prefer expandable CUDA segments to reduce fragmentation when close to VRAM limit
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('NCCL_ASYNC_ERROR_HANDLING', '1')
from pathlib import Path
from datetime import timedelta
from typing import List, Dict

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Global toggles (set in main from args)
CHANNELS_LAST = True
USE_DALI = False
DALI_AVAILABLE = False

# Augmentation/regularization config (set from args in main)
AUG_POLICY = 'none'            # one of: 'none', 'autoaugment', 'randaugment'
RAND_N = 2                     # RandAugment: number of ops
RAND_M = 9                     # RandAugment: magnitude (0-10)
COLOR_JITTER = 0.0             # strength for ColorJitter (0 disables)
RANDOM_ERASING_P = 0.0         # probability for RandomErasing (0 disables)
LABEL_SMOOTHING = 0.0          # training-time label smoothing
IO_RETRIES = 3                 # image load retry attempts
IO_RETRY_WAIT = 0.5            # seconds between retries
IO_SKIP_CORRUPT = True         # if True, replace unreadable images with blank image

def safe_pil_loader(path: str):
    """Robust PIL loader with retry + optional corrupt skip replacement."""
    for attempt in range(IO_RETRIES):
        try:
            with open(path, 'rb') as f:
                img = Image.open(f)
                return img.convert('RGB')
        except (OSError, ConnectionAbortedError):
            if attempt < IO_RETRIES - 1:
                time.sleep(IO_RETRY_WAIT)
            else:
                if IO_SKIP_CORRUPT:
                    return Image.new('RGB', (224, 224), (0, 0, 0))
                raise

# -----------------------------
# Unified Logging
# -----------------------------
import logging
import csv

def setup_logger(log_dir: Path, name: str = "pipeline") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    # Timestamped log file for each run
    ts = time.strftime('%Y%m%d-%H%M%S')
    log_path = log_dir / f"{name}_{ts}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = '%(asctime)s - %(levelname)s - %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(fh); logger.addHandler(ch)
    return logger

# -----------------------------
# Iteration CSV logger
# -----------------------------

class IterCSV:
    """Lightweight CSV appender for iteration-level logging per stage/phase.
    Writes header once if file doesn't exist. Safe to instantiate multiple times.
    """
    def __init__(self, path: Path, headers: list[str]):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Always include a timestamp column as the first column
        self.headers = (['timestamp'] + headers) if 'timestamp' not in headers else headers
        self._ensure_header()

    def _ensure_header(self):
        if not self.path.exists():
            with self.path.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self.headers)
                w.writeheader()

    def log(self, row: dict):
        # Prepend current timestamp in human-readable format
        now = time.strftime('%Y-%m-%d %H:%M:%S')
        row_with_ts = {'timestamp': now, **row}
        # Only keep known headers in row to avoid schema drift
        filtered = {k: row_with_ts.get(k) for k in self.headers}
        with self.path.open('a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=self.headers)
            w.writerow(filtered)

# -----------------------------
# DDP helpers
# -----------------------------

def is_dist() -> bool:
    return dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0

def is_primary() -> bool:
    return get_rank() == 0

def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1

def barrier():
    if is_dist():
        dist.barrier()

def _flag_path(name: str) -> Path:
    """Shared flag file path used to coordinate phases across ranks to avoid NCCL timeouts.
    We use /opt/ml/model/logs as a durable location inside SageMaker containers.
    """
    p = Path('/opt/ml/model/logs')
    p.mkdir(parents=True, exist_ok=True)
    return p / name

def _wait_for_flag(path: Path, timeout_s: int = 7200, poll_s: float = 1.0):
    """Wait until the given flag file exists or timeout elapses. Non-blocking for NCCL since
    we avoid entering collectives while waiting. Returns True if found, False if timed out.
    """
    t0 = time.time()
    while not path.exists():
        if (time.time() - t0) > timeout_s:
            return False
        time.sleep(poll_s)
    return True

class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34"""
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = F.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    """Bottleneck residual block for ResNet-50/101/152"""
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = F.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = F.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out

class ResNetImageNet(nn.Module):
    """ResNet for ImageNet-1K (224x224 input, 1000 classes)"""
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNetImageNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.fc(x)
        return x

# -----------------------------
# Data / Loaders
# -----------------------------

def build_transforms():
    # Build training transforms with optional strong augmentation and regularization
    train_list = [
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
    ]
    # Strong policy augmentations
    if AUG_POLICY == 'autoaugment':
        train_list.append(T.AutoAugment(T.AutoAugmentPolicy.IMAGENET))
    elif AUG_POLICY == 'randaugment':
        # Typical settings: n=2 ops, magnitude 9
        train_list.append(T.RandAugment(num_ops=RAND_N, magnitude=RAND_M))
    # Color jitter for additional regularization
    if COLOR_JITTER and COLOR_JITTER > 0:
        train_list.append(T.ColorJitter(brightness=COLOR_JITTER, contrast=COLOR_JITTER, saturation=COLOR_JITTER, hue=min(0.5, COLOR_JITTER/4)))
    # To tensor and normalize
    train_list.extend([
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    # Random erasing applied on tensors (after normalization)
    if RANDOM_ERASING_P and RANDOM_ERASING_P > 0:
        train_list.append(T.RandomErasing(p=RANDOM_ERASING_P, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'))
    train_tfms = T.Compose(train_list)
    val_tfms = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return train_tfms, val_tfms


def make_loaders(train_dir: str, val_dir: str, batch_size: int, workers: int, prefetch_factor: int = 2, force_no_dali: bool = False):
    # If DALI is requested and available, build DALI loaders instead
    if USE_DALI and DALI_AVAILABLE and not force_no_dali:
        return make_dali_loaders(train_dir, val_dir, batch_size, workers)
    train_tfms, val_tfms = build_transforms()
    train_ds = torchvision.datasets.ImageFolder(train_dir, transform=train_tfms, loader=safe_pil_loader)
    val_ds   = torchvision.datasets.ImageFolder(val_dir, transform=val_tfms, loader=safe_pil_loader)
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True) if is_dist() else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False, drop_last=False) if is_dist() else None
    pin = torch.cuda.is_available()
    common_kwargs = {}
    if workers > 0 and prefetch_factor is not None:
        common_kwargs['prefetch_factor'] = max(2, int(prefetch_factor))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=train_sampler is None,
                              sampler=train_sampler, num_workers=workers, pin_memory=pin,
                              persistent_workers=workers>0, **common_kwargs)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              sampler=val_sampler, num_workers=max(1, workers//2), pin_memory=pin,
                              persistent_workers=workers>0, **common_kwargs)
    return train_loader, val_loader, train_ds, val_ds

# -----------------------------
# DALI Loaders (optional)
# -----------------------------
def make_dali_loaders(train_dir: str, val_dir: str, batch_size: int, workers: int):
    import math as _math
    global USE_DALI
    try:
        from nvidia.dali.pipeline import pipeline_def
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types
        from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    except Exception as e:
        # Fallback silently to torchvision if import fails mid-run
        USE_DALI = False
        if is_primary():
            print(f"[DALI] Not available at runtime ({e}); falling back to torchvision loaders.")
        return make_loaders(train_dir, val_dir, batch_size, workers)

    # Count dataset sizes quickly using ImageFolder metadata (no transform)
    train_count = len(torchvision.datasets.ImageFolder(train_dir))
    val_count = len(torchvision.datasets.ImageFolder(val_dir))
    world = get_world_size(); shard = get_rank(); device_id = int(os.environ.get('LOCAL_RANK', 0))

    mean = [0.485*255, 0.456*255, 0.406*255]
    std = [0.229*255, 0.224*255, 0.225*255]

    @pipeline_def
    def train_pipe(data_root):
        jpegs, labels = fn.readers.file(file_root=data_root, random_shuffle=True, shard_id=shard, num_shards=world, name="Reader")
        images = fn.decoders.image_random_crop(jpegs, device="mixed", random_aspect_ratio=[0.8, 1.25], random_area=[0.08, 1.0], num_attempts=10)
        images = fn.resize(images, resize_x=224, resize_y=224)
        mirror = fn.random.coin_flip(probability=0.5)
        images = fn.crop_mirror_normalize(images.gpu(), dtype=types.FLOAT, output_layout="CHW", mean=mean, std=std, mirror=mirror)
        return images, labels

    @pipeline_def
    def val_pipe(data_root):
        jpegs, labels = fn.readers.file(file_root=data_root, random_shuffle=False, shard_id=shard, num_shards=world, name="ReaderVal")
        images = fn.decoders.image(jpegs, device="mixed")
        images = fn.resize(images, resize_shorter=256)
        images = fn.crop_mirror_normalize(images.gpu(), crop_h=224, crop_w=224, dtype=types.FLOAT, output_layout="CHW", mean=mean, std=std, mirror=0)
        return images, labels

    try:
        train_p = train_pipe(batch_size=batch_size, num_threads=max(2, workers), device_id=device_id, data_root=train_dir, seed=42)
        val_p = val_pipe(batch_size=batch_size, num_threads=max(2, workers//2), device_id=device_id, data_root=val_dir, seed=42)
        # Note: When reader_name is provided, DALI infers size from the reader. Do not pass 'size'.
        train_iter = DALIGenericIterator([train_p], output_map=["data","label"], reader_name="Reader", auto_reset=True, last_batch_policy=LastBatchPolicy.DROP)
        val_iter = DALIGenericIterator([val_p], output_map=["data","label"], reader_name="ReaderVal", auto_reset=True, last_batch_policy=LastBatchPolicy.PARTIAL)
    except RuntimeError as e:
        if 'CUDA allocation failed' in str(e):
            if is_primary():
                print(f"[DALI/OOM] CUDA OOM while building DALI pipeline (batch_size={batch_size}). Falling back to torchvision loaders.")
            USE_DALI = False
            return make_loaders(train_dir, val_dir, batch_size, workers)
        else:
            raise

    class DaliWrapper:
        def __init__(self, it, steps):
            self.it = it
            self._len = max(1, steps)
        def __len__(self):
            return self._len
        def __iter__(self):
            for batch in self.it:
                data = batch[0]["data"].contiguous()  # GPU tensor float32, NCHW
                label = batch[0]["label"].squeeze().long().contiguous()
                yield data, label
        def reset(self):
            try:
                self.it.reset()
            except Exception:
                pass

    train_steps = _math.floor((train_count//max(1,world))/batch_size)
    val_steps = _math.ceil((val_count//max(1,world))/batch_size)
    return DaliWrapper(train_iter, train_steps), DaliWrapper(val_iter, val_steps), None, None

# -----------------------------
# Worker Optimization (rank0)
# -----------------------------

def optimize_num_workers(train_dir: str, batch_size: int, max_workers: int, probe_batches: int, device: torch.device, logger, iter_log: IterCSV | None = None) -> int:
    train_tfms, _ = build_transforms()
    ds = torchvision.datasets.ImageFolder(train_dir, transform=train_tfms)
    stats = []
    for idx, nw in enumerate(range(1, max_workers+1), start=1):
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=torch.cuda.is_available())
        t0 = time.time(); n=0
        for x,_ in loader:
            x = x.to(device, non_blocking=True) if torch.cuda.is_available() else x
            n+=1
            if n>=probe_batches: break
        dt = max(1e-6, time.time()-t0)
        throughput = n/dt
        stats.append({'num_workers': nw, 'throughput': throughput})
        if iter_log is not None and is_primary():
            iter_log.log({'iteration': idx, 'num_workers': nw, 'throughput': round(throughput, 3), 'probe_batches': probe_batches, 'duration_sec': round(dt, 3)})
    best = max(stats, key=lambda s: s['throughput'])
    threshold = best['throughput'] * 0.95
    candidates = [s['num_workers'] for s in stats if s['throughput']>=threshold]
    suggestion = min(candidates)
    if is_primary():
        logger.info(f"[OPT_WORKERS] stats={stats} best={best} suggestion={suggestion}")
        plot_workers(stats, suggestion, Path('logs/workers.png'))
    return suggestion


def plot_workers(stats, suggestion, path: Path):
    xs = [s['num_workers'] for s in stats]; ys=[s['throughput'] for s in stats]
    plt.figure(figsize=(6,4)); plt.plot(xs, ys, marker='o'); plt.axvline(suggestion, color='red', linestyle='--');
    plt.title('Throughput vs num_workers'); plt.xlabel('num_workers'); plt.ylabel('batches/sec'); plt.grid(True, ls='--', alpha=0.5)
    path.parent.mkdir(exist_ok=True); plt.savefig(path); plt.close()

# -----------------------------
# Batch Size Finder (rank0)
# -----------------------------
class BatchSizeFinder:
    def __init__(self, model_fn, train_dir, device, workers, iter_log: IterCSV | None = None, logger: logging.Logger | None = None):
        self.model_fn = model_fn; self.train_dir = train_dir; self.device=device; self.workers=workers; self.iter_log = iter_log; self.logger = logger
        self.train_tfms, _ = build_transforms()
        self.ds = torchvision.datasets.ImageFolder(train_dir, transform=self.train_tfms)
    def _probe(self, bs, steps=5):
        loader = DataLoader(self.ds, batch_size=bs, shuffle=True, num_workers=self.workers, pin_memory=torch.cuda.is_available())
        model = self.model_fn().to(self.device)
        if CHANNELS_LAST:
            model = model.to(memory_format=torch.channels_last)
        # IMPORTANT: use train mode so BatchNorm and other layers allocate like real training
        model.train()
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        ok=True
        try:
            for i,(x,y) in enumerate(loader):
                if i>=steps: break
                x=x.to(self.device)
                if CHANNELS_LAST and x.is_floating_point():
                    x = x.to(memory_format=torch.channels_last)
                y=y.to(self.device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                    out=model(x); loss=nn.functional.cross_entropy(out,y)
                # Zero grads explicitly to avoid accumulation side-effects in probe
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = None
                loss.backward()  # allocate grads similar to training
        except RuntimeError as e:
            if 'out of memory' in str(e).lower(): ok=False
            else: raise
        mem = torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else 0.0
        del model, loader
        torch.cuda.empty_cache(); return {'bs':bs,'ok':ok,'mem_gb':mem}
    @staticmethod
    def _pow2_floor(n: int) -> int:
        return 1 if n <= 1 else 1 << (int(n).bit_length() - 1)

    def find(self, start=32, max_bs=2048):
        curve=[]; last_ok=0; first_oom=None; bs=start; attempt=0
        # Ensure we start at the next power-of-two >= start for consistent probing
        if bs & (bs - 1):
            # not a power of two, bump up to next power-of-two
            bs = 1 << (int(bs).bit_length())
        while bs<=max_bs:
            r=self._probe(bs); curve.append(r); attempt += 1
            if self.iter_log is not None and is_primary():
                self.iter_log.log({'attempt': attempt, 'batch_size': bs, 'ok': int(r['ok']), 'mem_gb': round(r['mem_gb'],3)})
            if self.logger is not None and is_primary():
                self.logger.info(f"[BSF] attempt={attempt} bs={bs} ok={int(r['ok'])} mem_gb={r['mem_gb']:.3f}")
            if r['ok']: last_ok=bs; bs*=2
            else: first_oom=bs; break
        if last_ok==0: return {'best':start,'curve':curve}
        if first_oom is None: return {'best':last_ok,'curve':curve}
        lo,hi=last_ok,first_oom
        while hi-lo>1:
            mid=(lo+hi)//2; r=self._probe(mid); curve.append(r); attempt += 1
            if self.iter_log is not None and is_primary():
                self.iter_log.log({'attempt': attempt, 'batch_size': mid, 'ok': int(r['ok']), 'mem_gb': round(r['mem_gb'],3)})
            if self.logger is not None and is_primary():
                self.logger.info(f"[BSF] bisect attempt={attempt} bs={mid} ok={int(r['ok'])} mem_gb={r['mem_gb']:.3f}")
            if r['ok']: lo=mid
            else: hi=mid
        return {'best':lo,'curve':curve}


def plot_batch_curve(curve, best, path: Path):
    bs=[c['bs'] for c in curve]; mem=[c['mem_gb'] for c in curve]; colors=['green' if c['ok'] else 'red' for c in curve]
    plt.figure(figsize=(7,4)); plt.bar(bs, mem, color=colors); plt.axvline(best, color='blue', ls='--'); plt.title('Max Memory vs Batch Size'); plt.xlabel('Batch Size'); plt.ylabel('Peak Mem GB');
    for b,m in zip(bs,mem): plt.text(b,m+0.1,f"{m:.1f}GB",ha='center',fontsize=8)
    path.parent.mkdir(exist_ok=True); plt.savefig(path); plt.close()

# -----------------------------
# LR Finder (rank0)
# -----------------------------
class LRFinderSimple:
    def __init__(self, model_fn, loader, device, iter_log: IterCSV | None = None, logger: logging.Logger | None = None):
        self.model_fn=model_fn; self.loader=loader; self.device=device; self.iter_log=iter_log; self.logger=logger
    def range_test(self, start_lr=1e-5, end_lr=1, num_iter=100):
        model=self.model_fn().to(self.device)
        if CHANNELS_LAST:
            model = model.to(memory_format=torch.channels_last)
        opt=SGD(model.parameters(), lr=start_lr, momentum=0.9, nesterov=True)
        criterion=nn.CrossEntropyLoss()
        lrs=[]; losses=[]
        lr_mult=(end_lr/start_lr)**(1/max(1,num_iter))
        lr=start_lr
        for i,(x,y) in enumerate(self.loader):
            if i>=num_iter: break
            x=x.to(self.device); y=y.to(self.device)
            if CHANNELS_LAST and x.is_floating_point():
                x = x.to(memory_format=torch.channels_last)
            for g in opt.param_groups: g['lr']=lr
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                out=model(x); loss=criterion(out,y)
            opt.zero_grad(set_to_none=True)
            scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            lrs.append(lr); losses.append(loss.item()); lr*=lr_mult
            if self.iter_log is not None and is_primary():
                self.iter_log.log({'iteration': i+1, 'lr': lrs[-1], 'loss': losses[-1]})
            if self.logger is not None and is_primary():
                self.logger.info(f"[LRF] iter={i+1} lr={lrs[-1]:.5e} loss={losses[-1]:.5f}")
        # Compute finite difference slopes in log space for 'steepest' policy
        import math
        log_lrs=[math.log10(x) for x in lrs]
        slopes=[]
        for i in range(2,len(losses)):
            # central diff on smoothed loss (simple 3-point moving average)
            prev=(losses[i-2]+losses[i-1]+losses[i])/3.0
            cur=(losses[i-1]+losses[i]+(losses[i]+1e-12))/3.0
            dl=cur-prev
            dloglr=log_lrs[i]-log_lrs[i-1] if log_lrs[i]!=log_lrs[i-1] else 1e-6
            slopes.append({'idx':i,'slope':dl/dloglr,'lr':lrs[i]})
        return {'lrs':lrs,'losses':losses,'slopes':slopes}


def plot_lr_curve(report, path: Path, policy: str = 'min0.1'):
    lrs=report['lrs']; losses=report['losses']; slopes=report.get('slopes',[])
    plt.figure(figsize=(7,4)); plt.plot(lrs, losses); plt.xscale('log'); plt.xlabel('LR (log)'); plt.ylabel('Loss'); plt.title(f'LR Range Test ({policy})'); plt.grid(True, ls='--', alpha=0.5)
    if policy=='min0.1':
        min_idx=int(torch.tensor(losses).argmin()); suggested=lrs[min_idx]*0.1
    elif policy=='steepest' and slopes:
        # Choose LR at most negative slope (largest loss drop per log LR)
        steep=min(slopes, key=lambda s: s['slope'])
        suggested=steep['lr']
    elif policy=='steepest-pre-min' and slopes:
        # Constrain to steepest slope that occurs at or before the minimum-loss index
        import math
        min_idx=int(torch.tensor(losses).argmin())
        # slopes[i]['idx'] aligns to losses index i; keep those with idx <= min_idx
        pre = [s for s in slopes if s['idx'] <= min_idx]
        if not pre:
            # Fallback to 'min0.1' if no pre-min slopes available
            suggested = lrs[min_idx]*0.1
        else:
            steep=min(pre, key=lambda s: s['slope'])
            suggested=steep['lr']
    else:
        min_idx=int(torch.tensor(losses).argmin()); suggested=lrs[min_idx]*0.1
    plt.axvline(suggested,color='red',ls='--'); plt.text(suggested,max(losses)*0.9,f"suggested={suggested:.2e}",color='red')
    # Optionally annotate steepest slope point
    if slopes and policy=='steepest':
        steep=min(slopes, key=lambda s: s['slope'])
        plt.axvline(steep['lr'], color='orange', ls=':')
        plt.text(steep['lr'], max(losses)*0.8, f"steep={steep['lr']:.2e}", color='orange')
    path.parent.mkdir(exist_ok=True); plt.savefig(path); plt.close()
    return suggested

# -----------------------------
# Weight Decay Finder (rank0)
# -----------------------------
class WDFinder:
    def __init__(self, model_fn, train_loader, val_loader, device, iter_log: IterCSV | None = None, logger: logging.Logger | None = None):
        self.model_fn=model_fn; self.train_loader=train_loader; self.val_loader=val_loader; self.device=device; self.iter_log=iter_log; self.logger=logger
    def test(self, wds: List[float], steps=50, lr=0.1):
        results=[]
        criterion=nn.CrossEntropyLoss()
        for wd in wds:
            model=self.model_fn().to(self.device)
            if CHANNELS_LAST:
                model = model.to(memory_format=torch.channels_last)
            opt=SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
            scaler=torch.amp.GradScaler(enabled=torch.cuda.is_available())
            model.train(); it=0
            for x,y in self.train_loader:
                if it>=steps: break
                x=x.to(self.device); y=y.to(self.device)
                if CHANNELS_LAST and x.is_floating_point():
                    x = x.to(memory_format=torch.channels_last)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                    out=model(x); loss=criterion(out,y)
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); it+=1
                if self.iter_log is not None and is_primary():
                    self.iter_log.log({'wd': wd, 'step': it, 'loss': float(loss.item())})
                if self.logger is not None and is_primary():
                    self.logger.info(f"[WD] wd={wd:.2e} step={it} loss={float(loss.item()):.5f}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # quick val
            model.eval(); correct=0; total=0
            with torch.no_grad():
                for vx,vy in self.val_loader:
                    vx=vx.to(self.device); vy=vy.to(self.device)
                    out=model(vx); pred=out.argmax(1); correct+=(pred==vy).sum().item(); total+=vy.size(0)
                    if total>=2048: break  # early limit for speed (kept small to avoid NCCL timeouts on idle ranks)
            acc=100.0*correct/total
            results.append({'wd':wd,'val_top1':acc})
            if self.logger is not None and is_primary():
                self.logger.info(f"[WD] wd={wd:.2e} val_top1={acc:.2f}% steps={it}")
            del model,opt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return results

    @staticmethod
    def adjust_weight_decay(selected_wd: float, eff_batch: int) -> float:
        """Heuristic WD adjustment based on effective batch size.
        For much larger batches, slightly increase WD; for tiny batches, decrease it.
        The output is softly clamped to a reasonable envelope.
        """
        base = float(selected_wd)
        if eff_batch >= 2048:
            base *= 1.8
        elif eff_batch >= 1024:
            base *= 1.5
        elif eff_batch >= 512:
            base *= 1.3
        elif eff_batch <= 64:
            base *= 0.7
        elif eff_batch <= 128:
            base *= 0.85
        # Soft clamp to reasonable envelope for ImageNet-scale training
        return min(max(base, 3e-6), 5e-4)


def plot_wd(results, path: Path):
    wds=[r['wd'] for r in results]; acc=[r['val_top1'] for r in results]
    plt.figure(figsize=(6,4)); plt.xscale('log'); plt.plot(wds, acc, 'o-'); plt.xlabel('Weight Decay'); plt.ylabel('Val Top1 %'); plt.title('WD Search'); plt.grid(True, ls='--', alpha=0.5)
    best=max(results,key=lambda r:r['val_top1']); plt.axvline(best['wd'],color='green',ls='--'); plt.text(best['wd'],max(acc)*0.9,f"best={best['wd']:.1e}",color='green')
    path.parent.mkdir(exist_ok=True); plt.savefig(path); plt.close(); return best['wd']

# -----------------------------
# Metric helpers
# -----------------------------

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    with torch.no_grad():
        maxk=max(topk); batch_size=target.size(0)
        _, pred=output.topk(maxk,1,True,True); pred=pred.t()
        correct=pred.eq(target.view(1,-1).expand_as(pred))
        res=[]
        for k in topk:
            correct_k=correct[:k].reshape(-1).float().sum(0,keepdim=True)
            res.append(correct_k.mul_(100.0/batch_size))
        return res

# -----------------------------
# Training Loop
# -----------------------------

def train(model, train_loader, val_loader, device, epochs, lr, wd, scheduler_type, logger, grad_accum: int = 1,
          label_smoothing: float = 0.0, start_epoch: int = 1, checkpoint_dir: Path | None = None,
          schedule_epochs: int | None = None, logs_dir: Path | None = None, stage_name: str | None = None,
          target_val_top1: float | None = None):
    """Train for a given number of epochs.
    start_epoch: epoch index to start from (1-based), used for resume.
    checkpoint_dir: optional directory to mirror checkpoints (managed checkpointing).
    """
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    steps_per_epoch = math.ceil(len(train_loader)/max(1, grad_accum))
    onecycle_pct_start = getattr(train, "onecycle_pct_start", 0.1)
    onecycle_max_lr = getattr(train, "onecycle_max_lr", None)
    # Number of epochs the scheduler should span in this call (default: remaining epochs)
    if schedule_epochs is None:
        schedule_epochs = max(1, epochs - start_epoch + 1)
    if scheduler_type == 'onecycle':
        max_lr = onecycle_max_lr if onecycle_max_lr is not None else lr
        scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=schedule_epochs, pct_start=onecycle_pct_start)
        if is_primary():
            logger.info(f"[SCHED] OneCycleLR max_lr={max_lr:.4e} pct_start={onecycle_pct_start:.2f} epochs={schedule_epochs}")
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=schedule_epochs)
        if is_primary():
            logger.info(f"[SCHED] CosineAnnealingLR T_max={schedule_epochs}")
    best_top1 = 0.0
    best_train_top1 = 0.0
    if is_primary():
        logger.info(f"[TRAIN] start_epoch={start_epoch} epochs={epochs} grad_accum={grad_accum} steps_per_epoch={steps_per_epoch}")
    epoch_times = []
    criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing)).to(device)
    # Iteration CSV logger for training stage (primary only)
    iter_log_enabled = getattr(train, 'iter_log_enabled', True)
    iter_log_interval = int(getattr(train, 'iter_log_interval', 50))
    iter_logger = None
    if is_primary() and iter_log_enabled:
        if logs_dir is None:
            logs_dir = Path('/opt/ml/model/logs')
        if stage_name is not None:
            iter_log_path = logs_dir / 'stages' / stage_name / 'iteration.csv'
        else:
            iter_log_path = logs_dir / 'iteration.csv'
        iter_logger = IterCSV(iter_log_path, headers=['epoch','step','lr','loss_scaled','loss_unscaled','interval_imgs_sec','cum_imgs_sec'])

    for epoch in range(start_epoch, epochs + 1):
        model.train(); epoch_loss=0.0; epoch_top1=0.0; total=0
        t_epoch_start=time.time()
        if is_primary():
            logger.info(f"[EPOCH-START] epoch={epoch} start={time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(t_epoch_start))}")
        if hasattr(train_loader, 'sampler') and isinstance(getattr(train_loader, 'sampler', None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        accum=0; last_log_t=time.time(); interval_imgs=0
        log_interval=getattr(train,'log_interval',50)
        opt_steps_done = 0  # count optimizer steps taken this epoch
        # derive a step-based logging interval to roughly match the microbatch interval
        step_log_interval = max(1, int(math.ceil(log_interval / max(1, grad_accum))))
        for i,(x,y) in enumerate(train_loader):
            x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
            if CHANNELS_LAST and x.is_floating_point():
                x = x.to(memory_format=torch.channels_last)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                out=model(x); loss=criterion(out,y)
            loss = loss / max(1, grad_accum)
            scaler.scale(loss).backward(); accum += 1
            do_step = (accum % max(1, grad_accum) == 0) or (i == len(train_loader)-1)
            if do_step:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                if scheduler_type=='onecycle': scheduler.step()
                opt_steps_done = min(steps_per_epoch, opt_steps_done + 1)
            with torch.no_grad(): top1,=accuracy(out,y,(1,))
            epoch_loss+=loss.item()*x.size(0); epoch_top1+=top1.item()*x.size(0)/100.0; total+=x.size(0)
            interval_imgs += x.size(0)
            # Log progress on optimizer-step boundaries using steps_per_epoch as denominator
            if is_primary() and do_step and (opt_steps_done % step_log_interval == 0 or opt_steps_done == steps_per_epoch):
                now=time.time(); dt=now-last_log_t; last_log_t=now
                cur_lr=optimizer.param_groups[0]['lr']
                pct = 100.0 * opt_steps_done / max(1, steps_per_epoch)
                imgs_sec = interval_imgs / max(1e-6, dt)
                elapsed_epoch = now - t_epoch_start
                cum_imgs_sec = total / max(1e-6, elapsed_epoch)
                logger.info(f"[PROGRESS] epoch={epoch} step={opt_steps_done}/{steps_per_epoch} ({pct:.1f}%) lr={cur_lr:.3e} loss={loss.item():.4f} interval_imgs/sec={imgs_sec:.1f} cum_imgs/sec={cum_imgs_sec:.1f}")
                interval_imgs = 0
            # Iteration CSV at finer cadence
            if iter_logger is not None and is_primary() and (do_step and ((opt_steps_done % max(1, step_log_interval) == 0) or (opt_steps_done == steps_per_epoch))):
                cur_lr = optimizer.param_groups[0]['lr']
                now = time.time(); elapsed_epoch = now - t_epoch_start
                cum_imgs_sec = total / max(1e-6, elapsed_epoch)
                loss_unscaled = loss.item() * max(1, grad_accum)
                iter_logger.log({'epoch': epoch, 'step': opt_steps_done, 'lr': cur_lr, 'loss_scaled': float(loss.item()), 'loss_unscaled': float(loss_unscaled), 'interval_imgs_sec': None, 'cum_imgs_sec': round(cum_imgs_sec,1)})
        if scheduler_type!='onecycle': scheduler.step()
        metrics=torch.tensor([epoch_loss, epoch_top1, float(total)], device=device)
        if is_dist(): dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        loss_epoch=metrics[0].item()/metrics[2].item(); top1_epoch=(metrics[1].item()/metrics[2].item())*100.0
        t_epoch_end=time.time(); dur=t_epoch_end-t_epoch_start
        if is_primary():
            # Update bests
            if top1_epoch > best_train_top1:
                best_train_top1 = top1_epoch
            logger.info(f"[TRAIN] epoch={epoch} loss={loss_epoch:.4f} top1={top1_epoch:.2f}% best_train_top1={best_train_top1:.2f}% best_val_top1={best_top1:.2f}%")
            logger.info(f"[EPOCH-END] epoch={epoch} end={time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(t_epoch_end))} duration_sec={dur:.1f}")
            # Global throughput across all ranks (metrics[2] already all-reduced)
            imgs_per_sec = metrics[2].item() / max(1e-6, dur)
            logger.info(f"[THROUGHPUT] epoch={epoch} imgs_per_sec_global={imgs_per_sec:.1f}")
        val_loss, val_top1 = validate(model, val_loader, device)
        if is_primary(): logger.info(f"[VAL] epoch={epoch} loss={val_loss:.4f} top1={val_top1:.2f}% best_train_top1={best_train_top1:.2f}% best_val_top1={max(best_top1, val_top1):.2f}%")
        # Early stopping broadcast logic
        stop_flag = torch.tensor([0], device=device)
        if target_val_top1 is not None and val_top1 >= target_val_top1:
            if is_primary(): logger.info(f"[EARLY-STOP] target {target_val_top1:.2f}% reached (val_top1={val_top1:.2f}%) at epoch={epoch}")
            stop_flag = torch.tensor([1], device=device)
        if is_dist():
            # Primary already sets flag; broadcast to others
            dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            break
        if is_primary():
            epoch_times.append({'epoch':epoch,'start_iso':time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(t_epoch_start)),
                                'end_iso':time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(t_epoch_end)),
                                'duration_sec':round(dur,1),'train_loss':round(loss_epoch,4),
                                'val_loss':round(val_loss,4),'val_top1':round(val_top1,2)})
            # Save checkpoints (last + best) each epoch
            save_checkpoint(model, optimizer, scaler, epoch, val_top1, Path('/opt/ml/model/checkpoint_last.pt'))
            if checkpoint_dir is not None:
                try:
                    save_checkpoint(model, optimizer, scaler, epoch, val_top1, checkpoint_dir / 'checkpoint_last.pt')
                except Exception:
                    pass
            if val_top1 > best_top1:
                best_top1 = val_top1
                save_checkpoint(model, optimizer, scaler, epoch, val_top1, Path('/opt/ml/model/checkpoint_best.pt'))
                if checkpoint_dir is not None:
                    try:
                        save_checkpoint(model, optimizer, scaler, epoch, val_top1, checkpoint_dir / 'checkpoint_best.pt')
                    except Exception:
                        pass
    if is_primary():
        logger.info(f"Training complete. Best val top1={best_top1:.2f}%")
        # write epoch metrics CSV and plots to run-specific (and stage-specific) directories
        import csv
        if logs_dir is None:
            logs_dir = Path('/opt/ml/model/logs')
        if stage_name is not None:
            out_dir = logs_dir / 'stages' / stage_name
        else:
            out_dir = logs_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / 'epoch_times.csv'
        with csv_path.open('w', newline='') as f:
            writer=csv.DictWriter(f, fieldnames=['epoch','start_iso','end_iso','duration_sec','train_loss','val_loss','val_top1'])
            writer.writeheader(); writer.writerows(epoch_times)
        try:
            ep=[e['epoch'] for e in epoch_times]; tl=[e['train_loss'] for e in epoch_times]; vl=[e['val_loss'] for e in epoch_times]; vt=[e['val_top1'] for e in epoch_times]
            plt.figure(figsize=(7,4)); plt.plot(ep, tl, label='train'); plt.plot(ep, vl, label='val'); plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.grid(True, ls='--', alpha=0.5); plt.title('Loss vs Epoch'); plt.savefig(out_dir / 'train_val_loss.png'); plt.close()
            plt.figure(figsize=(7,4)); plt.plot(ep, vt, label='val_top1'); plt.xlabel('epoch'); plt.ylabel('top1 %'); plt.grid(True, ls='--', alpha=0.5); plt.title('Val Top1 vs Epoch'); plt.savefig(out_dir / 'val_top1.png'); plt.close()
        except Exception:
            pass
    return {
        'best_val_top1': best_top1,
        'epochs_run': len(epoch_times),
        'early_stopped': (target_val_top1 is not None and best_top1 >= target_val_top1),
        'target_val_top1': target_val_top1
    }


def validate(model, loader, device):
    model.eval(); criterion=nn.CrossEntropyLoss().to(device)
    loss_sum=0.0; top1_sum=0.0; total=0
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
            if CHANNELS_LAST and x.is_floating_point():
                x = x.to(memory_format=torch.channels_last)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                out=model(x); loss=criterion(out,y)
            top1,=accuracy(out,y,(1,))
            loss_sum+=loss.item()*x.size(0); top1_sum+=top1.item()*x.size(0)/100.0; total+=x.size(0)
    metrics=torch.tensor([loss_sum, top1_sum, float(total)], device=device)
    if is_dist(): dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    return metrics[0].item()/metrics[2].item(), (metrics[1].item()/metrics[2].item())*100.0


def save_checkpoint(model, optimizer, scaler, epoch, val_top1, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    to_save = model.module if isinstance(model, DDP) else model
    torch.save({'epoch':epoch,'model_state':to_save.state_dict(),'optimizer_state':optimizer.state_dict(),'scaler_state':scaler.state_dict(),'val_top1':val_top1}, path)

# -----------------------------
# Model Builder
# -----------------------------

def build_model(num_classes=1000, sync_bn: bool = True):
    """Build the local ResNetImageNet model instead of torchvision variant.
    Uses Bottleneck blocks with [3,4,6,3] layers (ResNet-50 topology).
    SyncBN conversion is optional and used for DDP.
    Channels-last and torch.compile are applied elsewhere based on args and phase.
    """
    m = ResNetImageNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    if sync_bn and is_dist():
        m = torch.nn.SyncBatchNorm.convert_sync_batchnorm(m)
    return m

# -----------------------------
# Main
# -----------------------------

def init_dist():
    world_size=int(os.environ.get('WORLD_SIZE',1))
    if world_size>1 and not dist.is_initialized():
        # Extend process group timeout to accommodate long rank0-only phases (dataprobe/BS/LR/WD search)
        dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=7200))
        local_rank=int(os.environ.get('LOCAL_RANK',0)); torch.cuda.set_device(local_rank)


def main():
    ap=argparse.ArgumentParser()
    # helper to parse SageMaker-style boolean strings
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        return str(v).lower() in ("true","1","yes","y","t")
    ap.add_argument('--train-dir', default='/opt/ml/input/data/train')
    ap.add_argument('--val-dir', default='/opt/ml/input/data/val')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--scheduler', choices=['onecycle','cosine'], default='onecycle')
    ap.add_argument('--max-workers', type=int, default=8)
    ap.add_argument('--worker-probe-batches', type=int, default=32)
    ap.add_argument('--batch-start', type=int, default=32)
    ap.add_argument('--batch-max', type=int, default=1024)
    ap.add_argument('--batch-safety-factor', type=float, default=0.9,
                    help='Multiply discovered best batch by this factor for final training to reserve headroom.')
    ap.add_argument('--lr-range-iters', type=int, default=100)
    ap.add_argument('--lr-range-start', type=float, default=1e-5, help='Starting LR for LR range test')
    ap.add_argument('--lr-range-end', type=float, default=1.0, help='Ending LR for LR range test')
    ap.add_argument('--lr-finder-policy', choices=['min0.1','steepest','steepest-pre-min'], default='min0.1',
                    help="LR picker: 'min0.1' uses min-loss*0.1; 'steepest' uses largest negative slope; 'steepest-pre-min' constrains to steepest at or before min-loss")
    ap.add_argument('--wd-candidates', type=str, default='1e-5,3e-5,1e-4,3e-4,1e-3')
    ap.add_argument('--wd-steps', type=int, default=20)
    ap.add_argument('--final-epochs', type=int, default=10)
    ap.add_argument('--onecycle-pct-start', type=float, default=0.1)
    ap.add_argument('--onecycle-warmup-epochs', type=int, default=None, help='If set, overrides pct_start so that warmup spans this many epochs.')
    ap.add_argument('--onecycle-max-lr', type=float, default=None)
    ap.add_argument('--lr-auto-floor', type=float, default=1e-3, help='Lower clamp for batch-scaled LR')
    ap.add_argument('--lr-auto-cap', type=float, default=2.5e-1, help='Upper clamp for batch-scaled LR')
    # Augmentation & regularization flags
    ap.add_argument('--aug-policy', choices=['none','autoaugment','randaugment'], default='none', help='Strong augmentation policy to apply.')
    ap.add_argument('--randaugment-n', type=int, default=2, help='RandAugment number of ops (when aug-policy=randaugment).')
    ap.add_argument('--randaugment-m', type=int, default=9, help='RandAugment magnitude (0-10).')
    ap.add_argument('--color-jitter', type=float, default=0.0, help='ColorJitter strength (0 disables).')
    ap.add_argument('--random-erasing-p', type=float, default=0.0, help='Probability of RandomErasing (0 disables).')
    ap.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing value for CrossEntropyLoss.')
    # Per-stage overrides (optional; only used when --staged-aug enabled)
    ap.add_argument('--stage1-wd', type=float, default=None, help='Override weight decay for stage 1.')
    ap.add_argument('--stage2-wd', type=float, default=None, help='Override weight decay for stage 2.')
    ap.add_argument('--stage3-wd', type=float, default=None, help='Override weight decay for stage 3.')
    ap.add_argument('--stage4-wd', type=float, default=None, help='Override weight decay for stage 4.')
    ap.add_argument('--stage1-lr', type=float, default=None, help='Override base LR for stage 1.')
    ap.add_argument('--stage2-lr', type=float, default=None, help='Override base LR for stage 2.')
    ap.add_argument('--stage3-lr', type=float, default=None, help='Override base LR for stage 3.')
    ap.add_argument('--stage4-lr', type=float, default=None, help='Override base LR for stage 4.')
    ap.add_argument('--stage1-onecycle-max-lr', type=float, default=None, help='Override OneCycle max LR for stage 1 (if stage uses OneCycle).')
    ap.add_argument('--stage3-onecycle-max-lr', type=float, default=None, help='Override OneCycle max LR for stage 3 (if stage uses OneCycle).')
    # Memory/format/perf options
    ap.add_argument('--channels-last', dest='channels_last', action='store_true', help='Use channels_last memory format for tensors and model')
    ap.add_argument('--no-channels-last', dest='channels_last', action='store_false')
    ap.set_defaults(channels_last=True)
    ap.add_argument('--compile', type=str2bool, nargs='?', const=True, default=False, help='Enable torch.compile for final training model')
    ap.add_argument('--compile-mode', choices=['default','reduce-overhead','max-autotune'], default='reduce-overhead')
    ap.add_argument('--search-batch-fraction', type=float, default=0.5,
                    help='Fraction of discovered max batch size to use during LR/WD search phases (safety headroom to prevent OOM).')
    ap.add_argument('--lr-finder-at-final-batch', type=str2bool, nargs='?', const=True, default=False,
                    help='If set, run LR/WD search at the final batch size (overrides search-batch-fraction).')
    ap.add_argument('--override-batch', type=int, default=None)
    ap.add_argument('--override-lr', type=float, default=None)
    ap.add_argument('--override-lr-tolerance', type=float, default=0.05,
                    help='Relative tolerance for applying --override-lr. If the LR found by the finder differs from override by more than this fraction (e.g., 0.05 = 5%), the override is applied; otherwise the found LR is kept.')
    ap.add_argument('--override-wd', type=float, default=None)
    # Optional: skip search phases when override is provided (use str2bool for SageMaker hyperparams compatibility)
    ap.add_argument('--skip-batch-finder', type=str2bool, nargs='?', const=True, default=False,
                    help='Skip batch-size search and use --override-batch directly (if provided).')
    ap.add_argument('--skip-lr-finder', type=str2bool, nargs='?', const=True, default=False,
                    help='Skip LR range test and use --override-lr directly (if provided).')
    ap.add_argument('--skip-wd-finder', type=str2bool, nargs='?', const=True, default=False,
                    help='Skip weight-decay search and use --override-wd directly (if provided).')
    ap.add_argument('--grad-accum', type=int, default=0, help='Gradient accumulation steps; 0 = auto (preserve effective per-GPU batch from raw_best).')
    # DALI toggle
    ap.add_argument('--use-dali', type=str2bool, nargs='?', const=True, default=False,
                    help='Use NVIDIA DALI for data loading/augmentation if available in the environment.')
    ap.add_argument('--auto-install-dali', type=str2bool, nargs='?', const=True, default=False,
                    help='If set and DALI is requested but missing, attempt pip install nvidia-dali for current CUDA (rank0 only).')
    ap.add_argument('--prefetch-factor', type=int, default=2,
                    help='Prefetch factor for DataLoader workers (each worker preloads this many batches).')
    # IO robustness options
    ap.add_argument('--io-retries', type=int, default=3, help='Retry attempts for image loading errors.')
    ap.add_argument('--io-retry-wait', type=float, default=0.5, help='Seconds to wait between image load retries.')
    ap.add_argument('--io-skip-corrupt', action='store_true', help='Replace unreadable/corrupt images with a black placeholder instead of failing.')
    ap.add_argument('--no-io-skip-corrupt', dest='io_skip_corrupt', action='store_false')
    ap.set_defaults(io_skip_corrupt=True)
    # Resume / checkpoint arguments (matching DDP script semantics)
    ap.add_argument('--resume', type=str2bool, nargs='?', const=True, default=False, help='Resume from latest checkpoint in checkpoint-dir if available')
    ap.add_argument('--additional-epochs', type=int, default=0, help='Extend training beyond last completed epoch when resuming')
    ap.add_argument('--checkpoint-dir', type=str, default='/opt/ml/checkpoints', help='Directory for managed checkpoints (mirrors /opt/ml/model)')
    # Staged augmentation flags (optional multi-phase training)
    ap.add_argument('--staged-aug', type=str2bool, nargs='?', const=True, default=False, help='Enable multi-stage augmentation + scheduler sequence')
    ap.add_argument('--stage1-frac', type=float, default=1.0, help='Fraction of total epochs for stage 1 (high aug, OneCycle)')
    ap.add_argument('--stage2-frac', type=float, default=0.0, help='Fraction of total epochs for stage 2 (high aug, Cosine)')
    ap.add_argument('--stage3-frac', type=float, default=0.0, help='Fraction of total epochs for stage 3 (medium aug, OneCycle)')
    ap.add_argument('--stage4-frac', type=float, default=0.0, help='Fraction of total epochs for stage 4 (medium aug, Cosine)')
    ap.add_argument('--target-val-top1', type=float, default=None, help='Early stop if validation top1 (%) >= this value.')
    # Iteration logging controls
    ap.add_argument('--iter-log-interval', type=int, default=50, help='Steps between iteration CSV writes (train loop).')
    ap.add_argument('--no-iter-log', action='store_true', help='Disable iteration CSV logging entirely.')
    # Single-stage augmentation split: apply current aug for first frac, then no-augmentation for remainder
    ap.add_argument('--one-stage-aug-split', type=float, default=None,
                    help='When not using --staged-aug, apply current augmentation for the first <frac> of remaining epochs, then switch to no augmentation for the rest. Example: 0.8')

    # Parse CLI arguments (this was inadvertently omitted during refactor)
    args = ap.parse_args()

    init_dist()
    # Set global toggles from args
    global CHANNELS_LAST
    global AUG_POLICY, RAND_N, RAND_M, COLOR_JITTER, RANDOM_ERASING_P, LABEL_SMOOTHING
    global IO_RETRIES, IO_RETRY_WAIT, IO_SKIP_CORRUPT
    global USE_DALI, DALI_AVAILABLE
    CHANNELS_LAST = bool(args.channels_last)
    AUG_POLICY = args.aug_policy
    RAND_N = args.randaugment_n
    RAND_M = args.randaugment_m
    COLOR_JITTER = args.color_jitter
    RANDOM_ERASING_P = args.random_erasing_p
    LABEL_SMOOTHING = args.label_smoothing
    IO_RETRIES = max(1, args.io_retries)
    IO_RETRY_WAIT = max(0.0, args.io_retry_wait)
    IO_SKIP_CORRUPT = bool(args.io_skip_corrupt)
    USE_DALI = bool(args.use_dali)
    device=torch.device(f"cuda:{os.environ.get('LOCAL_RANK',0)}" if torch.cuda.is_available() else 'cpu')
    # Probe / optionally install DALI (rank0) then broadcast availability
    if USE_DALI:
        world = get_world_size()
        # Under DDP we avoid performing auto-install, but we still try to use an already-installed DALI.
        if bool(args.auto_install_dali) and world > 1:
            if is_primary():
                print("[DALI] Auto-install disabled under DDP (WORLD_SIZE>1); will try to use existing DALI if present.")
        try:
            import nvidia.dali as _dali  # type: ignore
            DALI_AVAILABLE = True
        except Exception as e:
            DALI_AVAILABLE = False
            if is_primary():
                print(f"[DALI] Requested but not available initially ({e}).")
            # Only attempt auto-install if user asked for it and we're not in DDP (to avoid long rank0 stalls)
            if bool(args.auto_install_dali) and world == 1 and is_primary():
                try:
                    import subprocess, sys
                    cuda_ver = (torch.version.cuda or '').strip()
                    # Map CUDA version to correct DALI wheel name
                    if cuda_ver.startswith('12'):
                        pkg = 'nvidia-dali-cuda120'
                    elif cuda_ver.startswith('11.8') or cuda_ver.startswith('11'):
                        pkg = 'nvidia-dali-cuda118'
                    else:
                        pkg = 'nvidia-dali-cuda120'
                    print(f"[DALI] Attempting auto-install of {pkg} via pip...")
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])
                    try:
                        import nvidia.dali as _dali  # re-probe
                        DALI_AVAILABLE = True
                        print("[DALI] Auto-install succeeded; enabling DALI.")
                    except Exception as e2:
                        print(f"[DALI] Auto-install attempted but still unavailable: {e2}")
                except Exception as e3:
                    if is_primary():
                        print(f"[DALI] Auto-install failed: {e3}")
            if not DALI_AVAILABLE and is_primary():
                print("[DALI] Falling back to torchvision loaders.")
    # Broadcast DALI availability so all ranks agree
    if is_dist():
        dali_tensor = torch.tensor([1 if DALI_AVAILABLE else 0], device=device)
        dist.broadcast(dali_tensor, src=0)
        DALI_AVAILABLE = bool(dali_tensor.item())
    # Log final DALI status once on primary
    if is_primary():
        logger_status = f"[DALI] requested={bool(args.use_dali)} available={DALI_AVAILABLE} enabled={bool(args.use_dali) and DALI_AVAILABLE} world_size={get_world_size()}"
        print(logger_status)

    logs_dir=Path('/opt/ml/model/logs'); logger=setup_logger(logs_dir,'pipeline')
    if is_primary(): logger.info(f"Args: {json.dumps(vars(args), indent=2)} world_size={get_world_size()}")
    # Configure training loop iteration logging toggles
    train.iter_log_enabled = (not args.no_iter_log)
    train.iter_log_interval = max(1, args.iter_log_interval)

    # 1. Optimize workers
    if is_primary():
        worker_iter_csv = IterCSV(logs_dir / 'workers_iter.csv', headers=['iteration','num_workers','throughput','probe_batches','duration_sec']) if train.iter_log_enabled else None
        workers = optimize_num_workers(args.train_dir, batch_size=args.batch_start, max_workers=args.max_workers, probe_batches=args.worker_probe_batches, device=device, logger=logger, iter_log=worker_iter_csv)
        # Signal readiness to other ranks to enter broadcast without incurring NCCL watchdog wait
        try:
            _flag_path('workers_ready.flag').touch()
        except Exception:
            pass
    else:
        workers = 4
        # Avoid entering NCCL broadcast until rank0 is ready
        _wait_for_flag(_flag_path('workers_ready.flag'))
    if is_dist():
        workers_tensor=torch.tensor([workers], device=device)
        dist.broadcast(workers_tensor, src=0)
        workers=int(workers_tensor.item())
    if is_primary(): logger.info(f"Selected num_workers={workers}")

    # 2. Batch size finder
    if is_primary():
        # Allow bypass when override-batch is provided
        if args.skip_batch_finder and args.override_batch is not None:
            best_batch = int(args.override_batch)
            logger.info(f"[BATCH FINDER] Skipped (skip-batch-finder=True); using override-batch={best_batch}")
            # Signal readiness to other ranks
            try:
                _flag_path('batch_ready.flag').touch()
            except Exception:
                pass
        else:
            if args.skip_batch_finder and args.override_batch is None:
                logger.warning("[BATCH FINDER] skip-batch-finder set but --override-batch missing; running finder.")
            # IMPORTANT: avoid SyncBatchNorm in single-rank search to prevent distributed collectives during finder
            # Using SyncBN here can trigger allreduce calls on rank0 while other ranks are idle, leading to NCCL timeouts.
            bs_iter_csv = IterCSV(logs_dir / 'batchsize_iter.csv', headers=['attempt','batch_size','ok','mem_gb']) if train.iter_log_enabled else None
            bs_finder=BatchSizeFinder(lambda: build_model(sync_bn=False), args.train_dir, device, workers, iter_log=bs_iter_csv, logger=logger)
            bs_report=bs_finder.find(start=args.batch_start, max_bs=args.batch_max)
            raw_best=bs_report['best']
            # Compute safety-adjusted batch WITHOUT power-of-two rounding per user request
            best_batch = int(max(1, raw_best * args.batch_safety_factor))
            if best_batch != raw_best:
                logger.info(f"Applying batch_safety_factor={args.batch_safety_factor} raw_best={raw_best} final_batch={best_batch} (no pow2 rounding)")
            plot_batch_curve(bs_report['curve'], raw_best, logs_dir/'batchsize.png')
            # Flag batch-size computation complete
            try:
                _flag_path('batch_ready.flag').touch()
            except Exception:
                pass
    else:
        best_batch=args.batch_start
        # Wait for rank0 to finish batch-size search before entering broadcast
        _wait_for_flag(_flag_path('batch_ready.flag'))
    if args.override_batch is not None:
        best_batch=args.override_batch
        # Enforce power-of-two when override is provided
        def _pow2_floor(n: int) -> int:
            return 1 if n <= 1 else 1 << (int(n).bit_length() - 1)
        best_batch_pow2 = _pow2_floor(best_batch)
        if best_batch_pow2 != best_batch and is_primary():
            logger.info(f"Override batch rounded down to power-of-two: {best_batch} -> {best_batch_pow2}")
        best_batch = best_batch_pow2
    if is_dist():
        bs_tensor=torch.tensor([best_batch], device=device); dist.broadcast(bs_tensor, src=0); best_batch=int(bs_tensor.item())
    if is_primary(): logger.info(f"Selected batch_size={best_batch}")
    if is_primary(): logger.info(f"IO settings: retries={IO_RETRIES} retry_wait={IO_RETRY_WAIT}s skip_corrupt={IO_SKIP_CORRUPT}")

    # ------------------------------------------------------------------
    # Early finalize grad accumulation and adjusted per-GPU batch size
    # so that ALL subsequent pipeline stages (LR finder, WD search, training)
    # use the same final batch formula: (raw_best / grad_accum) * safety_factor.
    # ------------------------------------------------------------------
    if 'raw_best' in locals():  # only rank0 has raw_best; adjust there and broadcast
        if args.grad_accum == 0:
            # Auto infer accumulation if safety factor reduced batch
            if best_batch < raw_best:
                auto_accum = max(1, raw_best // max(1, best_batch))
                grad_accum = auto_accum
                if is_primary():
                    logger.info(f"Auto gradient accumulation enabled early: raw_best={raw_best} reduced_batch={best_batch} grad_accum={grad_accum}")
            else:
                grad_accum = 1
        else:
            grad_accum = max(1, args.grad_accum)
        final_batch = int(max(1, (raw_best / max(1, grad_accum)) * args.batch_safety_factor))
        if final_batch != best_batch and is_primary():
            logger.info(f"[BATCH FINALIZE EARLY] raw_best={raw_best} grad_accum={grad_accum} safety_factor={args.batch_safety_factor} -> final_batch={final_batch} (was {best_batch})")
        best_batch = final_batch
        # Broadcast updated batch + grad_accum so all ranks are consistent
        if is_dist():
            bs_tensor = torch.tensor([best_batch], device=device); dist.broadcast(bs_tensor, src=0); best_batch = int(bs_tensor.item())
            ga_tensor = torch.tensor([grad_accum], device=device); dist.broadcast(ga_tensor, src=0); grad_accum = int(ga_tensor.item())
    else:
        # Non-primary ranks: receive grad_accum via later broadcast if needed; set placeholder
        grad_accum = max(1, args.grad_accum) if args.grad_accum != 0 else 1

    # For LR/WD search phases, user requested using the final adjusted batch size.
    search_batch = best_batch
    if is_primary():
        logger.info(f"[LR/WD SEARCH] Using FINAL adjusted batch_size={search_batch} for LR and WD search phases.")

    # Force torchvision loaders during search phases to reduce GPU memory pressure if DALI causes OOM
    train_loader_tmp, val_loader_tmp, _, _ = make_loaders(args.train_dir, args.val_dir, search_batch, workers, prefetch_factor=args.prefetch_factor, force_no_dali=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. LR Finder (supports skip via --skip-lr-finder + --override-lr)
    if is_primary():
        if bool(args.skip_lr_finder) and args.override_lr is not None:
            suggested_lr = float(args.override_lr)
            logger.info(f"[LR FINDER] Skipped (skip-lr-finder=True); using override-lr={suggested_lr:.4e}")
            try: _flag_path('lr_ready.flag').touch()
            except Exception: pass
        else:
            if bool(args.skip_lr_finder) and args.override_lr is None:
                logger.warning("[LR FINDER] skip-lr-finder set but --override-lr missing; running finder.")
            lr_iter_csv = IterCSV(logs_dir / 'lrfinder_iter.csv', headers=['iteration','lr','loss']) if train.iter_log_enabled else None
            lr_finder=LRFinderSimple(lambda: build_model(sync_bn=False), train_loader_tmp, device, iter_log=lr_iter_csv, logger=logger)
            lr_report=lr_finder.range_test(start_lr=args.lr_range_start, end_lr=args.lr_range_end, num_iter=args.lr_range_iters)
            suggested_lr=plot_lr_curve(lr_report, logs_dir/'lr_finder.png', policy=args.lr_finder_policy)
            try: _flag_path('lr_ready.flag').touch()
            except Exception: pass
    else:
        _wait_for_flag(_flag_path('lr_ready.flag'))
        suggested_lr = 0.01 if not bool(args.skip_lr_finder) or args.override_lr is None else float(args.override_lr)
    # Determine if LR override should be applied based on tolerance
    lr_override_applied = False
    if args.override_lr is not None:
        try:
            ov = float(args.override_lr)
            # Use override if found LR is not within +/- tolerance of override
            rel_diff = abs(suggested_lr - ov) / (abs(ov) if abs(ov) > 1e-12 else 1.0)
            if rel_diff > float(args.override_lr_tolerance):
                if is_primary():
                    logger.info(f"[LR OVERRIDE] Applying override_lf={ov:.4e} (found={suggested_lr:.4e}, rel_diff={rel_diff:.3f} > tol={args.override_lr_tolerance:.3f})")
                suggested_lr = ov
                lr_override_applied = True
            else:
                if is_primary():
                    logger.info(f"[LR OVERRIDE] Keeping found LR {suggested_lr:.4e} (within tol ±{args.override_lr_tolerance:.0%} of override {ov:.4e})")
        except Exception:
            # If parsing fails, fall back to original behavior (apply override directly)
            suggested_lr = args.override_lr
            lr_override_applied = True
    if is_dist():
        lr_tensor=torch.tensor([suggested_lr], device=device); dist.broadcast(lr_tensor, src=0); suggested_lr=float(lr_tensor.item())
    if is_primary(): logger.info(f"Selected learning_rate={suggested_lr:.4e}")
    # If LR finder was explicitly skipped with an override, treat it as an applied override to bypass auto-scaling
    if bool(args.skip_lr_finder) and (args.override_lr is not None):
        lr_override_applied = True

    # 4. Weight Decay Finder (supports skip via --skip-wd-finder + --override-wd)
    if is_primary():
        if bool(args.skip_wd_finder) and args.override_wd is not None:
            best_wd = float(args.override_wd)
            logger.info(f"[WD FINDER] Skipped (skip-wd-finder=True); using override-wd={best_wd:.2e}")
            try: _flag_path('wd_ready.flag').touch()
            except Exception: pass
        else:
            if bool(args.skip_wd_finder) and args.override_wd is None:
                logger.warning("[WD FINDER] skip-wd-finder set but --override-wd missing; running finder.")
            wd_values = [float(w) for w in args.wd_candidates.split(',')]
            wd_iter_csv = IterCSV(logs_dir / 'wdsearch_iter.csv', headers=['wd','step','loss']) if train.iter_log_enabled else None
            wd_finder = WDFinder(lambda: build_model(sync_bn=False), train_loader_tmp, val_loader_tmp, device, iter_log=wd_iter_csv, logger=logger)
            try:
                wd_results = wd_finder.test(wd_values, steps=args.wd_steps, lr=suggested_lr)
                best_wd = plot_wd(wd_results, logs_dir / 'weight_decay.png')
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e).lower():
                    if is_primary():
                        logger.warning(f"[WD SEARCH] CUDA OOM during WD search with batch={search_batch}; retrying with force_no_dali and reduced batch")
                    USE_DALI = False
                    retry_bs = max(1, search_batch//2)
                    train_loader_tmp2, val_loader_tmp2, _, _ = make_loaders(args.train_dir, args.val_dir, retry_bs, workers, prefetch_factor=args.prefetch_factor, force_no_dali=True)
                    wd_finder2 = WDFinder(lambda: build_model(sync_bn=False), train_loader_tmp2, val_loader_tmp2, device, iter_log=wd_iter_csv, logger=logger)
                    wd_results = wd_finder2.test(wd_values, steps=args.wd_steps, lr=suggested_lr)
                    best_wd = plot_wd(wd_results, logs_dir / 'weight_decay.png')
                else:
                    if is_primary():
                        logger.error(f"[WD SEARCH] Exception during WD search: {e}; falling back to default 1e-4")
                    best_wd = 1e-4
            try: _flag_path('wd_ready.flag').touch()
            except Exception: pass
    else:
        _wait_for_flag(_flag_path('wd_ready.flag'))
        best_wd = (float(args.override_wd) if (bool(args.skip_wd_finder) and args.override_wd is not None) else 1e-4)
    if args.override_wd is not None: best_wd=args.override_wd
    else:
        # Adjust WD using heuristic based on effective global batch (per-GPU batch * world)
        eff_batch_for_wd = max(1, best_batch) * max(1, get_world_size())
        try:
            best_wd_adjusted = WDFinder.adjust_weight_decay(best_wd, eff_batch_for_wd)
            if is_primary(): logger.info(f"[WD AUTO] raw={best_wd:.2e} eff_batch={eff_batch_for_wd} adjusted={best_wd_adjusted:.2e}")
            best_wd = best_wd_adjusted
        except Exception:
            pass
    if is_dist():
        barrier()
        wd_tensor=torch.tensor([best_wd], device=device)
        dist.broadcast(wd_tensor, src=0)
        best_wd=float(wd_tensor.item())
    if is_primary(): logger.info(f"Selected weight_decay={best_wd:.2e}")

    barrier()

    # 5. Full training (with optional resume)
    # Rebuild loaders for full training (ensure no leftover reduced batch)
    # Use DALI (if available) only for full training (unless user disabled via --use-dali False)
    train_loader, val_loader, _, _ = make_loaders(args.train_dir, args.val_dir, best_batch, workers, prefetch_factor=args.prefetch_factor, force_no_dali=not (USE_DALI and DALI_AVAILABLE))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    barrier()
    # Build final model
    model=build_model(sync_bn=True).to(device)
    # Enable cudnn benchmark for final training stability/performance (after batch size fixed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    # Apply channels_last & compile only once for final model
    barrier()
    if CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    barrier()
    if bool(args.compile) and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode=args.compile_mode)
            if is_primary():
                logger.info(f"[OPT] torch.compile applied (mode={args.compile_mode})")
        except Exception as e:
            if is_primary():
                logger.info(f"[OPT] torch.compile skipped: {e}")
    barrier()
    # Resume logic: load latest checkpoint BEFORE constructing scheduler and starting epochs
    last_epoch_completed = 0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    os.environ['PIPELINE_CHECKPOINT_DIR'] = str(checkpoint_dir)

    def latest_ckpt_path():
        # Prefer checkpoint-dir, fallback to /opt/ml/model
        cand1 = checkpoint_dir / 'checkpoint_last.pt'
        cand2 = Path('/opt/ml/model') / 'checkpoint_last.pt'
        if cand1.exists(): return cand1
        if cand2.exists(): return cand2
        return None

    barrier()
    if args.resume:
        ck = latest_ckpt_path()
        if ck and ck.exists():
            if is_primary():
                print(f"[RESUME] Loading checkpoint {ck}")
            state = torch.load(ck, map_location=device)
            model.load_state_dict(state.get('model_state', {}))
            last_epoch_completed = int(state.get('epoch', 0))
            # Optimizer/scaler states will be recreated later after LR/WD discovery (we search before training), so we deliberately skip them here.
        else:
            if is_primary():
                print("[RESUME] No checkpoint found; starting from scratch.")

    # Adjust total final epochs if resuming with additional-epochs
    final_epochs = args.final_epochs
    if args.resume and last_epoch_completed > 0 and args.additional_epochs > 0:
        final_epochs = last_epoch_completed + args.additional_epochs
        if is_primary():
            print(f"[RESUME] Extending training: last_epoch={last_epoch_completed} additional={args.additional_epochs} -> final_epochs={final_epochs}")

    barrier()
    # Wrap with DDP after (potential) weight loading
    if is_dist():
        model=DDP(model, device_ids=[int(os.environ.get('LOCAL_RANK',0))], output_device=int(os.environ.get('LOCAL_RANK',0)), find_unused_parameters=False)

    # Inject OneCycle parameters into train function (simple way without refactor)
    # Override pct_start if warmup epochs explicitly provided
    if args.scheduler == 'onecycle' and args.onecycle_warmup_epochs is not None:
        computed_pct = args.onecycle_warmup_epochs / max(1, final_epochs)
        # Clamp to (0, 0.9] to avoid extremely long warmup impacting decay phase
        computed_pct = max(0.0, min(0.9, computed_pct))
        if is_primary():
            logger.info(f"[SCHED] onecycle warmup_epochs={args.onecycle_warmup_epochs} final_epochs={args.final_epochs} -> pct_start={computed_pct:.3f}")
        args.onecycle_pct_start = computed_pct
    train.onecycle_pct_start = args.onecycle_pct_start
    train.onecycle_max_lr = args.onecycle_max_lr
    try:
        # grad_accum and final best_batch already finalized earlier; proceed to auto-scale LR from search effective batch
        # to final effective global batch (considering world size and grad accumulation)
        world = get_world_size()
        eff_train_global = max(1, best_batch) * max(1, grad_accum) * max(1, world)
        eff_search_global = max(1, search_batch)  # search ran on rank0 only
        if not lr_override_applied:
            scale = eff_train_global / eff_search_global
            auto_lr = suggested_lr * scale
            # Dynamic floor/cap heuristics: wider range for very small batch, tighter for huge batches
            base_floor = args.lr_auto_floor
            base_cap = args.lr_auto_cap
            # Adjust cap down slightly if batch is extremely large to prevent instability
            if eff_train_global >= 1024:
                base_cap = min(base_cap, 0.15)
            elif eff_train_global <= 128:
                base_cap = max(base_cap, 0.3)  # allow exploration for small batch
            # If scaled LR is far below floor relative to scale, nudge upward (avoid underfitting)
            lr_floor = base_floor
            lr_cap = base_cap
            # Soft clamp with bias toward median of (floor,cap) when far outside
            median_lr = (lr_floor + lr_cap) / 2.0
            auto_lr_clamped = min(max(auto_lr, lr_floor), lr_cap)
            if auto_lr < lr_floor * 0.5:
                auto_lr_clamped = max(median_lr, lr_floor)
            elif auto_lr > lr_cap * 1.5:
                auto_lr_clamped = min(median_lr, lr_cap)
            if is_primary():
                logger.info(f"[LR AUTO] found={suggested_lr:.2e} scale={scale:.2f} auto_raw={auto_lr:.2e} final={auto_lr_clamped:.2e} (floor={lr_floor:.0e}, cap={lr_cap:.2f}) eff_train_global={eff_train_global}")
            suggested_lr = auto_lr_clamped
        # Broadcast final LR to all ranks
        if is_dist():
            lr_tensor=torch.tensor([suggested_lr], device=device); dist.broadcast(lr_tensor, src=0); suggested_lr=float(lr_tensor.item())
        # Conditional: single-stage vs staged augmentation
        if not bool(args.staged_aug):
            remaining = max(1, final_epochs - last_epoch_completed)
            split = args.one_stage_aug_split
            if split is None or split <= 0.0 or split >= 1.0:
                # No split requested: run single continuous stage with current augmentation
                if is_primary():
                    logger.info(f"[TRAIN-ONE] epochs={final_epochs} start_epoch={last_epoch_completed+1} batch={best_batch} lr={suggested_lr:.4e} wd={best_wd:.2e} scheduler={args.scheduler}")
                train_loader, val_loader, _, _ = make_loaders(args.train_dir, args.val_dir, best_batch, workers, prefetch_factor=args.prefetch_factor)
                single_summary = train(model, train_loader, val_loader, device, final_epochs, suggested_lr, best_wd, args.scheduler, logger,
                      grad_accum=grad_accum, label_smoothing=LABEL_SMOOTHING, start_epoch=last_epoch_completed + 1,
                      checkpoint_dir=checkpoint_dir, schedule_epochs=remaining, target_val_top1=args.target_val_top1)
                last_epoch_completed = last_epoch_completed + max(0, int(single_summary.get('epochs_run', remaining)))
            else:
                # Two-phase within single stage: heavy aug for first fraction, then no augmentation
                split = max(0.0, min(1.0, float(split)))
                e_heavy = max(0, int(round(remaining * split)))
                e_none = max(0, remaining - e_heavy)
                # Phase 1: heavy augmentation (use current settings as-is)
                if e_heavy > 0:
                    if is_primary():
                        logger.info(f"[ONE-STAGE] Phase1 heavy augmentation epochs={e_heavy} (split={split:.2f})")
                    train_loader, val_loader, _, _ = make_loaders(args.train_dir, args.val_dir, best_batch, workers, prefetch_factor=args.prefetch_factor)
                    summary1 = train(model, train_loader, val_loader, device, last_epoch_completed + e_heavy, suggested_lr, best_wd, args.scheduler, logger,
                          grad_accum=grad_accum, label_smoothing=LABEL_SMOOTHING, start_epoch=last_epoch_completed + 1,
                          checkpoint_dir=checkpoint_dir, schedule_epochs=e_heavy, target_val_top1=args.target_val_top1)
                    last_epoch_completed = last_epoch_completed + max(0, int(summary1.get('epochs_run', e_heavy)))
                # Phase 2: no augmentation (disable aug knobs and rebuild loaders)
                if e_none > 0 and last_epoch_completed < final_epochs:
                    # Disable augmentation globally for transforms
                    AUG_POLICY = 'none'
                    RAND_N = 0; RAND_M = 0
                    COLOR_JITTER = 0.0
                    RANDOM_ERASING_P = 0.0
                    if is_primary():
                        logger.info(f"[ONE-STAGE] Phase2 no-augmentation epochs={e_none}")
                    train_loader, val_loader, _, _ = make_loaders(args.train_dir, args.val_dir, best_batch, workers, prefetch_factor=args.prefetch_factor)
                    summary2 = train(model, train_loader, val_loader, device, final_epochs, suggested_lr, best_wd, args.scheduler, logger,
                          grad_accum=grad_accum, label_smoothing=LABEL_SMOOTHING, start_epoch=last_epoch_completed + 1,
                          checkpoint_dir=checkpoint_dir, schedule_epochs=e_none, target_val_top1=args.target_val_top1)
                    last_epoch_completed = last_epoch_completed + max(0, int(summary2.get('epochs_run', e_none)))
            if is_primary():
                summary = {
                    'final_epochs': final_epochs,
                    'staged': False,
                    'last_epoch_completed': last_epoch_completed,
                    'resumed': bool(args.resume and last_epoch_completed > 0),
                    'batch_size': best_batch,
                    'learning_rate_final': suggested_lr,
                    'weight_decay_final': best_wd,
                    'early_stopped': single_summary.get('early_stopped', False),
                    'target_val_top1': args.target_val_top1,
                    'best_val_top1': single_summary.get('best_val_top1'),
                }
                with open('/opt/ml/model/logs/summary.json','w') as f:
                    json.dump(summary, f, indent=2)
        else:
            if is_primary():
                logger.info(f"[STAGES] Preparing 4-stage schedule total_epochs={final_epochs} f1={args.stage1_frac} f2={args.stage2_frac} f3={args.stage3_frac} f4={args.stage4_frac}")
            # Stage fractions: clamp to [0,1], then allocate epochs with rounding and fix remainder on last stage
            f1 = max(0.0, min(1.0, args.stage1_frac))
            f2 = max(0.0, min(1.0, args.stage2_frac))
            f3 = max(0.0, min(1.0, args.stage3_frac))
            fsum = f1 + f2 + f3 + max(0.0, min(1.0, args.stage4_frac))
            if fsum <= 0:
                f1, f2, f3 = 0.4, 0.4, 0.1  # fall back to defaults
            total = final_epochs
            e1 = max(0, int(round(total * f1)))
            e2 = max(0, int(round(total * f2)))
            e3 = max(0, int(round(total * f3)))
            e4 = max(0, total - e1 - e2 - e3)
            # Ensure at least 1 epoch if total > 0
            if total > 0 and (e1 + e2 + e3 + e4) == 0:
                e1 = 1
            # Absolute epoch indices for each stage end (inclusive)
            current_start = last_epoch_completed + 1
            abs1_end = current_start + e1 - 1
            abs2_end = abs1_end + e2
            abs3_end = abs2_end + e3
            abs4_end = last_epoch_completed + total  # must equal
            if is_primary():
                logger.info(f"[STAGES] Stage1 (high, OneCycle): {current_start}-{abs1_end}")
                logger.info(f"[STAGES] Stage2 (high, Cosine): {abs1_end+1}-{abs2_end}")
                logger.info(f"[STAGES] Stage3 (medium, OneCycle): {abs2_end+1}-{abs3_end}")
                logger.info(f"[STAGES] Stage4 (light, Cosine): {abs3_end+1}-{abs4_end}")

            def set_aug(level: str):
                """Configure global augmentation knobs for a given level ('high', 'medium', or 'light')."""
                global AUG_POLICY, RAND_N, RAND_M, COLOR_JITTER, RANDOM_ERASING_P
                if level == 'high':
                    # High intensity: RandAugment n=2, m>=12, moderate CJ, stronger RE
                    if AUG_POLICY != 'autoaugment':
                        AUG_POLICY = 'randaugment'
                    RAND_N = 2
                    RAND_M = max(RAND_M, 12)
                    COLOR_JITTER = max(COLOR_JITTER, 0.1)
                    RANDOM_ERASING_P = max(RANDOM_ERASING_P, 0.2)
                elif level == 'medium':
                    # Medium intensity: RandAugment n=1, m>=7, weaker CJ, moderate RE
                    if AUG_POLICY != 'autoaugment':
                        AUG_POLICY = 'randaugment'
                    RAND_N = 1
                    RAND_M = max(RAND_M, 7)
                    COLOR_JITTER = max(COLOR_JITTER, 0.05)
                    RANDOM_ERASING_P = max(RANDOM_ERASING_P, 0.10)
                elif level == 'light':
                    # Light: essentially disable strong aug; keep base transforms only
                    AUG_POLICY = 'none'
                    RAND_N = 0; RAND_M = 0
                    COLOR_JITTER = 0.0
                    RANDOM_ERASING_P = 0.0
                if is_primary():
                    logger.info(f"[AUG] level={level} policy={AUG_POLICY} rand(n={RAND_N},m={RAND_M}) cj={COLOR_JITTER} re_p={RANDOM_ERASING_P}")

            def rebuild_loaders():
                return make_loaders(args.train_dir, args.val_dir, best_batch, workers, prefetch_factor=args.prefetch_factor)

            stages = [
                { 'idx': 1, 'name': 'stage1_high_onecycle',   'end': abs1_end, 'level': 'high',  'sched': 'onecycle' },
                { 'idx': 2, 'name': 'stage2_high_cosine',     'end': abs2_end, 'level': 'high',  'sched': 'cosine'   },
                { 'idx': 3, 'name': 'stage3_medium_onecycle', 'end': abs3_end, 'level': 'medium','sched': 'onecycle' },
                { 'idx': 4, 'name': 'stage4_light_cosine',    'end': abs4_end, 'level': 'light', 'sched': 'cosine'   },
            ]

            # Iterate stages and train remaining epochs in each
            stage_summary = {'early_stopped': False, 'best_val_top1': None}
            for idx, st in enumerate(stages, start=1):
                s_end = st['end']
                # Skip zero-length stages or already completed epochs
                if s_end < last_epoch_completed + 1:
                    continue
                if s_end < current_start:
                    continue
                # If stage duration is zero, skip
                sched_len = max(0, s_end - max(current_start, 1) + 1)
                if sched_len <= 0:
                    continue
                # Apply augmentation level and rebuild loaders
                set_aug(st['level'])
                # Stage-specific WD override
                stage_wd = best_wd
                if st['idx'] == 1 and args.stage1_wd is not None:
                    stage_wd = float(args.stage1_wd)
                elif st['idx'] == 2 and args.stage2_wd is not None:
                    stage_wd = float(args.stage2_wd)
                elif st['idx'] == 3 and args.stage3_wd is not None:
                    stage_wd = float(args.stage3_wd)
                elif st['idx'] == 4 and args.stage4_wd is not None:
                    stage_wd = float(args.stage4_wd)
                # Stage-specific LR override
                stage_lr = suggested_lr
                if st['idx'] == 1 and args.stage1_lr is not None:
                    stage_lr = float(args.stage1_lr)
                elif st['idx'] == 2 and args.stage2_lr is not None:
                    stage_lr = float(args.stage2_lr)
                elif st['idx'] == 3 and args.stage3_lr is not None:
                    stage_lr = float(args.stage3_lr)
                elif st['idx'] == 4 and args.stage4_lr is not None:
                    stage_lr = float(args.stage4_lr)
                # Stage-specific OneCycle max_lr override
                if st['sched'] == 'onecycle':
                    if st['idx'] == 1 and args.stage1_onecycle_max_lr is not None:
                        train.onecycle_max_lr = float(args.stage1_onecycle_max_lr)
                    elif st['idx'] == 3 and args.stage3_onecycle_max_lr is not None:
                        train.onecycle_max_lr = float(args.stage3_onecycle_max_lr)
                    else:
                        train.onecycle_max_lr = args.onecycle_max_lr
                train_loader, val_loader, _, _ = rebuild_loaders()
                s_start = max(current_start, 1)
                # Stage-specific logger and naming
                stage_dir = logs_dir / 'stages' / st['name']
                stage_logger = setup_logger(stage_dir, st['name']) if is_primary() else logger
                if is_primary():
                    extra = ''
                    if st['sched'] == 'onecycle':
                        oc_max = getattr(train, 'onecycle_max_lr', None)
                        if oc_max is not None:
                            extra = f" max_lr={oc_max:.4e} pct_start={getattr(train,'onecycle_pct_start',0.1):.2f}"
                    stage_logger.info(f"[STAGE-START] {st['name']} start={s_start} end={s_end} sched={st['sched']} batch={best_batch} lr={stage_lr:.4e} wd={stage_wd:.2e}{extra}")
                stage_summary = train(model, train_loader, val_loader, device, s_end, stage_lr, stage_wd, st['sched'], stage_logger,
                      grad_accum=grad_accum, label_smoothing=LABEL_SMOOTHING, start_epoch=s_start,
                      checkpoint_dir=checkpoint_dir, schedule_epochs=sched_len, logs_dir=logs_dir, stage_name=st['name'],
                      target_val_top1=args.target_val_top1)
                # Compute actual last epoch completed for this stage (accounts for early stop)
                last_epoch_completed = max(last_epoch_completed, s_start + max(0, int(stage_summary.get('epochs_run', sched_len))) - 1)
                current_start = last_epoch_completed + 1
                if stage_summary.get('early_stopped'):
                    if is_primary(): stage_logger.info(f"[EARLY-STOP] Halting further stages after {st['name']} (target reached).")
                    break
            if is_primary():
                summary = {
                    'final_epochs': final_epochs,
                    'staged': True,
                    'stage1_epochs': e1,
                    'stage2_epochs': e2,
                    'stage3_epochs': e3,
                    'stage4_epochs': e4,
                    'last_epoch_completed': last_epoch_completed,
                    'resumed': bool(args.resume and last_epoch_completed > 0),
                    'batch_size': best_batch,
                    'learning_rate_final': suggested_lr,
                    'weight_decay_final': best_wd,
                    'target_val_top1': args.target_val_top1,
                    'early_stopped': stage_summary.get('early_stopped', False),
                    'best_val_top1': stage_summary.get('best_val_top1'),
                }
                with open('/opt/ml/model/logs/summary.json','w') as f:
                    json.dump(summary, f, indent=2)
    except Exception as e:
        if is_primary():
            print(f"[ERROR] Exception during staged training: {e}")
        raise

    if is_dist():
        # Clean shutdown to avoid NCCL lingering warnings; ensure no further collectives.
        dist.destroy_process_group()

if __name__=='__main__':
    main()
