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
from typing import List, Dict

import torch
import torch.distributed as dist
from torch import nn
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

def setup_logger(log_dir: Path, name: str = "pipeline") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(fh); logger.addHandler(ch)
    return logger

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


def make_loaders(train_dir: str, val_dir: str, batch_size: int, workers: int, prefetch_factor: int = 2):
    # If DALI is requested and available, build DALI loaders instead
    if USE_DALI and DALI_AVAILABLE:
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
    try:
        from nvidia.dali.pipeline import pipeline_def
        import nvidia.dali.fn as fn
        import nvidia.dali.types as types
        from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
    except Exception as e:
        # Fallback silently to torchvision if import fails mid-run
        global USE_DALI
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

    train_p = train_pipe(batch_size=batch_size, num_threads=max(1, workers), device_id=device_id, data_root=train_dir, seed=42)
    val_p = val_pipe(batch_size=batch_size, num_threads=max(1, workers//2), device_id=device_id, data_root=val_dir, seed=42)
    train_iter = DALIGenericIterator([train_p], output_map=["data","label"], reader_name="Reader", auto_reset=True, last_batch_policy=LastBatchPolicy.DROP, size=train_count//max(1,world))
    val_iter = DALIGenericIterator([val_p], output_map=["data","label"], reader_name="ReaderVal", auto_reset=True, last_batch_policy=LastBatchPolicy.PARTIAL, size=val_count//max(1,world))

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

def optimize_num_workers(train_dir: str, batch_size: int, max_workers: int, probe_batches: int, device: torch.device, logger) -> int:
    train_tfms, _ = build_transforms()
    ds = torchvision.datasets.ImageFolder(train_dir, transform=train_tfms)
    stats = []
    for nw in range(1, max_workers+1):
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=torch.cuda.is_available())
        t0 = time.time(); n=0
        for x,_ in loader:
            x = x.to(device, non_blocking=True) if torch.cuda.is_available() else x
            n+=1
            if n>=probe_batches: break
        dt = max(1e-6, time.time()-t0)
        throughput = n/dt
        stats.append({'num_workers': nw, 'throughput': throughput})
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
    def __init__(self, model_fn, train_dir, device, workers):
        self.model_fn = model_fn; self.train_dir = train_dir; self.device=device; self.workers=workers
        self.train_tfms, _ = build_transforms()
        self.ds = torchvision.datasets.ImageFolder(train_dir, transform=self.train_tfms)
    def _probe(self, bs, steps=5):
        loader = DataLoader(self.ds, batch_size=bs, shuffle=True, num_workers=self.workers, pin_memory=torch.cuda.is_available())
        model = self.model_fn().to(self.device)
        if CHANNELS_LAST:
            model = model.to(memory_format=torch.channels_last)
        model.eval()
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
                loss.backward()  # allocate grads
        except RuntimeError as e:
            if 'out of memory' in str(e).lower(): ok=False
            else: raise
        mem = torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else 0.0
        del model, loader
        torch.cuda.empty_cache(); return {'bs':bs,'ok':ok,'mem_gb':mem}
    def find(self, start=32, max_bs=2048):
        curve=[]; last_ok=0; first_oom=None; bs=start
        while bs<=max_bs:
            r=self._probe(bs); curve.append(r)
            if r['ok']: last_ok=bs; bs*=2
            else: first_oom=bs; break
        if last_ok==0: return {'best':start,'curve':curve}
        if first_oom is None: return {'best':last_ok,'curve':curve}
        lo,hi=last_ok,first_oom
        while hi-lo>1:
            mid=(lo+hi)//2; r=self._probe(mid); curve.append(r)
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
    def __init__(self, model_fn, loader, device):
        self.model_fn=model_fn; self.loader=loader; self.device=device
    def range_test(self, start_lr=1e-5, end_lr=1, num_iter=100):
        model=self.model_fn().to(self.device)
        if CHANNELS_LAST:
            model = model.to(memory_format=torch.channels_last)
        opt=SGD(model.parameters(), lr=start_lr, momentum=0.9)
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
    def __init__(self, model_fn, train_loader, val_loader, device):
        self.model_fn=model_fn; self.train_loader=train_loader; self.val_loader=val_loader; self.device=device
    def test(self, wds: List[float], steps=50, lr=0.1):
        results=[]
        criterion=nn.CrossEntropyLoss()
        for wd in wds:
            model=self.model_fn().to(self.device)
            if CHANNELS_LAST:
                model = model.to(memory_format=torch.channels_last)
            opt=SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
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
            del model,opt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return results


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

def train(model, train_loader, val_loader, device, epochs, lr, wd, scheduler_type, logger, grad_accum: int = 1, label_smoothing: float = 0.0):
    scaler=torch.amp.GradScaler(enabled=torch.cuda.is_available())
    optimizer=SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    steps_per_epoch=math.ceil(len(train_loader)/max(1, grad_accum))
    # Scheduler selection (OneCycle supports pct_start and custom max_lr)
    onecycle_pct_start = getattr(train, "onecycle_pct_start", 0.1)  # default if not patched
    onecycle_max_lr = getattr(train, "onecycle_max_lr", None)
    if scheduler_type=='onecycle':
        max_lr = onecycle_max_lr if onecycle_max_lr is not None else lr
        scheduler=OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs, pct_start=onecycle_pct_start)
        if is_primary():
            logger.info(f"[SCHED] OneCycleLR max_lr={max_lr:.4e} pct_start={onecycle_pct_start:.2f}")
    else:
        scheduler=CosineAnnealingLR(optimizer, T_max=epochs)
        if is_primary():
            logger.info(f"[SCHED] CosineAnnealingLR T_max={epochs}")
    best_top1=0.0
    if is_primary():
        logger.info(f"[TRAIN] grad_accum={grad_accum} steps_per_epoch={steps_per_epoch}")
    epoch_times=[]
    # Training loss (with optional label smoothing)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing)).to(device)
    for epoch in range(1, epochs+1):
        model.train(); epoch_loss=0.0; epoch_top1=0.0; total=0
        t_epoch_start=time.time()
        if is_primary():
            logger.info(f"[EPOCH-START] epoch={epoch} start={time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(t_epoch_start))}")
        if hasattr(train_loader, 'sampler') and isinstance(getattr(train_loader, 'sampler', None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        accum=0; last_log_t=time.time(); interval_imgs=0
        log_interval=getattr(train,'log_interval',50)
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
            with torch.no_grad(): top1,=accuracy(out,y,(1,))
            epoch_loss+=loss.item()*x.size(0); epoch_top1+=top1.item()*x.size(0)/100.0; total+=x.size(0)
            interval_imgs += x.size(0)
            if is_primary() and ((i+1) % max(1,log_interval)==0):
                now=time.time(); dt=now-last_log_t; last_log_t=now
                cur_lr=optimizer.param_groups[0]['lr']; pct=100.0*(i+1)/len(train_loader)
                imgs_sec = interval_imgs / max(1e-6, dt)
                elapsed_epoch = now - t_epoch_start
                cum_imgs_sec = total / max(1e-6, elapsed_epoch)
                logger.info(f"[PROGRESS] epoch={epoch} step={i+1}/{len(train_loader)} ({pct:.1f}%) lr={cur_lr:.3e} loss={loss.item():.4f} interval_imgs/sec={imgs_sec:.1f} cum_imgs/sec={cum_imgs_sec:.1f}")
                interval_imgs = 0
        if scheduler_type!='onecycle': scheduler.step()
        metrics=torch.tensor([epoch_loss, epoch_top1, float(total)], device=device)
        if is_dist(): dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        loss_epoch=metrics[0].item()/metrics[2].item(); top1_epoch=(metrics[1].item()/metrics[2].item())*100.0
        t_epoch_end=time.time(); dur=t_epoch_end-t_epoch_start
        if is_primary():
            logger.info(f"[TRAIN] epoch={epoch} loss={loss_epoch:.4f} top1={top1_epoch:.2f}%")
            logger.info(f"[EPOCH-END] epoch={epoch} end={time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(t_epoch_end))} duration_sec={dur:.1f}")
            # Global throughput across all ranks (metrics[2] already all-reduced)
            imgs_per_sec = metrics[2].item() / max(1e-6, dur)
            logger.info(f"[THROUGHPUT] epoch={epoch} imgs_per_sec_global={imgs_per_sec:.1f}")
        val_loss, val_top1 = validate(model, val_loader, device)
        if is_primary(): logger.info(f"[VAL] epoch={epoch} loss={val_loss:.4f} top1={val_top1:.2f}%")
        if is_primary():
            epoch_times.append({'epoch':epoch,'start_iso':time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(t_epoch_start)),
                                'end_iso':time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(t_epoch_end)),
                                'duration_sec':round(dur,1),'train_loss':round(loss_epoch,4),
                                'val_loss':round(val_loss,4),'val_top1':round(val_top1,2)})
            if val_top1>best_top1:
                best_top1=val_top1
                save_checkpoint(model, optimizer, scaler, epoch, val_top1, Path('/opt/ml/model/checkpoint_best.pt'))
            save_checkpoint(model, optimizer, scaler, epoch, val_top1, Path('/opt/ml/model/checkpoint_last.pt'))
    if is_primary():
        logger.info(f"Training complete. Best val top1={best_top1:.2f}%")
        # write epoch metrics CSV and plots
        import csv
        csv_path=Path('/opt/ml/model/logs/epoch_times.csv'); csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open('w', newline='') as f:
            writer=csv.DictWriter(f, fieldnames=['epoch','start_iso','end_iso','duration_sec','train_loss','val_loss','val_top1'])
            writer.writeheader(); writer.writerows(epoch_times)
        try:
            ep=[e['epoch'] for e in epoch_times]; tl=[e['train_loss'] for e in epoch_times]; vl=[e['val_loss'] for e in epoch_times]; vt=[e['val_top1'] for e in epoch_times]
            plt.figure(figsize=(7,4)); plt.plot(ep, tl, label='train'); plt.plot(ep, vl, label='val'); plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.grid(True, ls='--', alpha=0.5); plt.title('Loss vs Epoch'); plt.savefig('/opt/ml/model/logs/train_val_loss.png'); plt.close()
            plt.figure(figsize=(7,4)); plt.plot(ep, vt, label='val_top1'); plt.xlabel('epoch'); plt.ylabel('top1 %'); plt.grid(True, ls='--', alpha=0.5); plt.title('Val Top1 vs Epoch'); plt.savefig('/opt/ml/model/logs/val_top1.png'); plt.close()
        except Exception:
            pass


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
    """Builds the model. SyncBN conversion is optional and used for DDP.
    Channels-last and torch.compile are applied elsewhere based on args and phase.
    """
    m = torchvision.models.resnet50(weights=None, num_classes=num_classes)
    if sync_bn and is_dist():
        m = torch.nn.SyncBatchNorm.convert_sync_batchnorm(m)
    return m

# -----------------------------
# Main
# -----------------------------

def init_dist():
    world_size=int(os.environ.get('WORLD_SIZE',1))
    if world_size>1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://')
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
    ap.add_argument('--epochs', type=int, default=5)
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
    ap.add_argument('--lr-finder-policy', choices=['min0.1','steepest'], default='min0.1',
                    help="LR picker: 'min0.1' uses min-loss*0.1; 'steepest' uses largest negative slope region")
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
    # Memory/format/perf options
    ap.add_argument('--channels-last', dest='channels_last', action='store_true', help='Use channels_last memory format for tensors and model')
    ap.add_argument('--no-channels-last', dest='channels_last', action='store_false')
    ap.set_defaults(channels_last=True)
    ap.add_argument('--compile', type=str2bool, nargs='?', const=True, default=False, help='Enable torch.compile for final training model')
    ap.add_argument('--compile-mode', choices=['default','reduce-overhead','max-autotune'], default='reduce-overhead')
    ap.add_argument('--search-batch-fraction', type=float, default=0.5,
                    help='Fraction of discovered max batch size to use during LR/WD search phases (safety headroom to prevent OOM).')
    ap.add_argument('--override-batch', type=int, default=None)
    ap.add_argument('--override-lr', type=float, default=None)
    ap.add_argument('--override-wd', type=float, default=None)
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
    args=ap.parse_args()

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
        try:
            import nvidia.dali as _dali  # type: ignore
            DALI_AVAILABLE = True
        except Exception as e:
            DALI_AVAILABLE = False
            if is_primary():
                print(f"[DALI] Requested but not available initially ({e}).")
            if bool(args.auto_install_dali) and is_primary():
                try:
                    import subprocess, sys
                    cuda_ver = (torch.version.cuda or '').strip()
                    # Map CUDA version to correct DALI wheel name
                    # CUDA 12.x -> nvidia-dali-cuda120, CUDA 11.8 -> nvidia-dali-cuda118 (most common images)
                    if cuda_ver.startswith('12'):
                        pkg = 'nvidia-dali-cuda120'
                    elif cuda_ver.startswith('11.8') or cuda_ver.startswith('11'):
                        pkg = 'nvidia-dali-cuda118'
                    else:
                        # Fallback: try CUDA 12.0 build which works across 12.x toolkits in most SM images
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

    logs_dir=Path('/opt/ml/model/logs'); logger=setup_logger(logs_dir,'pipeline')
    if is_primary(): logger.info(f"Args: {json.dumps(vars(args), indent=2)} world_size={get_world_size()}")

    # 1. Optimize workers
    if is_primary():
        workers = optimize_num_workers(args.train_dir, batch_size=args.batch_start, max_workers=args.max_workers, probe_batches=args.worker_probe_batches, device=device, logger=logger)
    else:
        workers = 4
    if is_dist():
        workers_tensor=torch.tensor([workers], device=device)
        dist.broadcast(workers_tensor, src=0)
        workers=int(workers_tensor.item())
    if is_primary(): logger.info(f"Selected num_workers={workers}")

    # 2. Batch size finder
    if is_primary():
        bs_finder=BatchSizeFinder(lambda: build_model(sync_bn=False), args.train_dir, device, workers)
        bs_report=bs_finder.find(start=args.batch_start, max_bs=args.batch_max)
        raw_best=bs_report['best']
        best_batch=int(max(1, raw_best * args.batch_safety_factor))
        if best_batch < raw_best:
            logger.info(f"Applying batch_safety_factor={args.batch_safety_factor} raw_best={raw_best} final_batch={best_batch}")
        plot_batch_curve(bs_report['curve'], raw_best, logs_dir/'batchsize.png')
    else:
        best_batch=args.batch_start
    if args.override_batch is not None: best_batch=args.override_batch
    if is_dist():
        bs_tensor=torch.tensor([best_batch], device=device); dist.broadcast(bs_tensor, src=0); best_batch=int(bs_tensor.item())
    if is_primary(): logger.info(f"Selected batch_size={best_batch}")
    if is_primary(): logger.info(f"IO settings: retries={IO_RETRIES} retry_wait={IO_RETRY_WAIT}s skip_corrupt={IO_SKIP_CORRUPT}")

    # Build a temporary loader for LR & WD tests using reduced batch for headroom
    search_batch = max(1, int(best_batch * args.search_batch_fraction))
    if search_batch < best_batch and is_primary():
        logger.info(f"Using reduced batch_size={search_batch} (fraction={args.search_batch_fraction}) for LR/WD search phases to provide memory headroom (final batch={best_batch}).")
    train_loader_tmp, val_loader_tmp, _, _ = make_loaders(args.train_dir, args.val_dir, search_batch, workers, prefetch_factor=args.prefetch_factor)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 3. LR Finder
    if is_primary():
        lr_finder=LRFinderSimple(lambda: build_model(sync_bn=False), train_loader_tmp, device)
        lr_report=lr_finder.range_test(start_lr=args.lr_range_start, end_lr=args.lr_range_end, num_iter=args.lr_range_iters)
        suggested_lr=plot_lr_curve(lr_report, logs_dir/'lr_finder.png', policy=args.lr_finder_policy)
    else:
        suggested_lr=0.01
    if args.override_lr is not None: suggested_lr=args.override_lr
    if is_dist():
        lr_tensor=torch.tensor([suggested_lr], device=device); dist.broadcast(lr_tensor, src=0); suggested_lr=float(lr_tensor.item())
    if is_primary(): logger.info(f"Selected learning_rate={suggested_lr:.4e}")

    # 4. Weight Decay Finder
    if is_primary():
        wd_values=[float(w) for w in args.wd_candidates.split(',')]
        wd_finder=WDFinder(lambda: build_model(sync_bn=False), train_loader_tmp, val_loader_tmp, device)
        wd_results=wd_finder.test(wd_values, steps=args.wd_steps, lr=suggested_lr)
        best_wd=plot_wd(wd_results, logs_dir/'weight_decay.png')
        # Signal to other ranks that WD search is complete to avoid long NCCL blocking waits
        try:
            (logs_dir / 'wd_ready.flag').touch()
        except Exception:
            pass
    else:
        best_wd=1e-4
    if args.override_wd is not None: best_wd=args.override_wd
    if is_dist():
        # For non-primary ranks, avoid entering a long blocking broadcast until rank0 has finished WD search
        if not is_primary():
            flag_path = logs_dir / 'wd_ready.flag'
            waited = 0
            while not flag_path.exists():
                time.sleep(2)
                waited += 2
                if waited % 60 == 0 and is_primary():
                    pass
        wd_tensor=torch.tensor([best_wd], device=device); dist.broadcast(wd_tensor, src=0); best_wd=float(wd_tensor.item())
    if is_primary(): logger.info(f"Selected weight_decay={best_wd:.2e}")

    barrier()

    # 5. Full training
    # Rebuild loaders for full training (ensure no leftover reduced batch)
    train_loader, val_loader, _, _ = make_loaders(args.train_dir, args.val_dir, best_batch, workers, prefetch_factor=args.prefetch_factor)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model=build_model(sync_bn=True).to(device)
    # Enable cudnn benchmark for final training stability/performance (after batch size fixed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    # Apply channels_last & compile only once for final model
    if CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    if bool(args.compile) and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode=args.compile_mode)
            if is_primary():
                logger.info(f"[OPT] torch.compile applied (mode={args.compile_mode})")
        except Exception as e:
            if is_primary():
                logger.info(f"[OPT] torch.compile skipped: {e}")
    if is_dist():
        model=DDP(model, device_ids=[int(os.environ.get('LOCAL_RANK',0))], output_device=int(os.environ.get('LOCAL_RANK',0)), find_unused_parameters=False)
    # Inject OneCycle parameters into train function (simple way without refactor)
    # Override pct_start if warmup epochs explicitly provided
    if args.scheduler == 'onecycle' and args.onecycle_warmup_epochs is not None:
        computed_pct = args.onecycle_warmup_epochs / max(1, args.final_epochs)
        # Clamp to (0, 0.9] to avoid extremely long warmup impacting decay phase
        computed_pct = max(0.0, min(0.9, computed_pct))
        if is_primary():
            logger.info(f"[SCHED] onecycle warmup_epochs={args.onecycle_warmup_epochs} final_epochs={args.final_epochs} -> pct_start={computed_pct:.3f}")
        args.onecycle_pct_start = computed_pct
    train.onecycle_pct_start = args.onecycle_pct_start
    train.onecycle_max_lr = args.onecycle_max_lr
    try:
        # Auto gradient accumulation: if grad_accum==0 and safety factor shrank batch, try to keep raw effective batch
        if args.grad_accum == 0:
            if 'raw_best' in locals() and best_batch < raw_best:
                # approximate accumulation factor
                auto_accum = max(1, raw_best // max(1, best_batch))
                grad_accum = auto_accum
                if is_primary():
                    logger.info(f"Auto gradient accumulation enabled: raw_best={raw_best} reduced_batch={best_batch} grad_accum={grad_accum}")
            else:
                grad_accum = 1
        else:
            grad_accum = max(1, args.grad_accum)
        # Auto-scale LR from search effective batch to final effective global batch (considering world size and grad accumulation)
        world = get_world_size()
        eff_train_global = max(1, best_batch) * max(1, grad_accum) * max(1, world)
        eff_search_global = max(1, search_batch)  # search ran on rank0 only
        if args.override_lr is None:
            scale = eff_train_global / eff_search_global
            auto_lr = suggested_lr * scale
            lr_floor = args.lr_auto_floor
            lr_cap = args.lr_auto_cap
            auto_lr_clamped = min(max(auto_lr, lr_floor), lr_cap)
            if is_primary():
                logger.info(f"[LR AUTO] found={suggested_lr:.2e} scale={scale:.2f} auto={auto_lr:.2e} clamped={auto_lr_clamped:.2e} (floor={lr_floor:.0e}, cap={lr_cap:.2f})")
            suggested_lr = auto_lr_clamped
        # Broadcast final LR to all ranks
        if is_dist():
            lr_tensor=torch.tensor([suggested_lr], device=device); dist.broadcast(lr_tensor, src=0); suggested_lr=float(lr_tensor.item())
        if is_primary():
            logger.info(f"Starting full training: epochs={args.final_epochs} batch_size={best_batch} lr={suggested_lr:.4e} wd={best_wd:.2e} scheduler={args.scheduler}")
        train(model, train_loader, val_loader, device, args.final_epochs, suggested_lr, best_wd, args.scheduler, logger, grad_accum=grad_accum, label_smoothing=LABEL_SMOOTHING)
    except Exception as e:
        if is_primary():
            logger.error(f"[ERROR] Exception during training phase: {e}")
        raise

    # Synchronize all ranks before writing summary / destroying process group.
    if is_dist():
        barrier()

    if is_primary():
        logger.info("Pipeline completed successfully.")
        # Write summary JSON
        summary={
            'num_workers': workers,
            'batch_size': best_batch,
            'lr': suggested_lr,
            'weight_decay': best_wd,
            'scheduler': args.scheduler,
            'epochs': args.final_epochs
        }
        with open('/opt/ml/model/logs/summary.json','w') as f: json.dump(summary,f,indent=2)

    if is_dist():
        # Clean shutdown to avoid NCCL lingering warnings; ensure no further collectives.
        dist.destroy_process_group()

if __name__=='__main__':
    main()
