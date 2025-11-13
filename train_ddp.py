#!/usr/bin/env python3
"""
DDP training script for ImageNet-1K compatible folder structure using SageMaker channels.
- Uses DistributedDataParallel (DDP) across all available GPUs and instances
- Loads data from SageMaker channels: /opt/ml/input/data/{train,val}
- Saves checkpoints and logs under /opt/ml/model (uploaded to S3 by SageMaker)
- Mixed precision (AMP) enabled by default

This script is standalone and does not depend on the notebook.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as T


def is_primary() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0


def setup_ddp(backend: str = "nccl"):
    """Initialize torch.distributed from SageMaker environment or torchrun."""
    if dist.is_initialized():
        return
    # SageMaker sets these env vars
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        # Single process (still works on 1 GPU without DDP)
        return


def get_device() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SM_LOCAL_RANK", 0)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def build_dataloaders(train_dir: str, val_dir: str, batch_size: int, workers: int) -> tuple[DataLoader, DataLoader]:
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_tfms = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        
        normalize,
    ])
    val_tfms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])

    train_ds = torchvision.datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = torchvision.datasets.ImageFolder(val_dir, transform=val_tfms)

    # Distributed samplers
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=max(1, workers // 2),
        pin_memory=pin,
        persistent_workers=workers > 0,
    )
    return train_loader, val_loader


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
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
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
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
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super().__init__()
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
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
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

def build_model(num_classes=1000, sync_bn=True) -> nn.Module:
    model = ResNetImageNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    if dist.is_initialized() and sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_ckpt(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, log_interval=50):
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    running_loss = 0.0
    running_top1 = 0.0
    total = 0

    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    for i, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            top1, = accuracy(out, y, topk=(1,))
        running_loss += loss.item() * x.size(0)
        running_top1 += top1.item() * x.size(0) / 100.0
        total += x.size(0)

        if is_primary() and (i + 1) % log_interval == 0:
            print(f"Epoch {epoch} | Step {i+1}/{len(loader)} | Loss {loss.item():.4f} | Top1 {top1.item():.2f}%", flush=True)

    # Reduce metrics across ranks
    metrics = torch.tensor([running_loss, running_top1, float(total)], device=device)
    if dist.is_initialized():
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    loss_epoch = metrics[0].item() / metrics[2].item()
    top1_epoch = (metrics[1].item() / metrics[2].item()) * 100.0
    return loss_epoch, top1_epoch


def validate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    running_loss = 0.0
    running_top1 = 0.0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                out = model(x)
                loss = criterion(out, y)
            top1, = accuracy(out, y, topk=(1,))
            running_loss += loss.item() * x.size(0)
            running_top1 += top1.item() * x.size(0) / 100.0
            total += x.size(0)

    metrics = torch.tensor([running_loss, running_top1, float(total)], device=device)
    if dist.is_initialized():
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    val_loss = metrics[0].item() / metrics[2].item()
    val_top1 = (metrics[1].item() / metrics[2].item()) * 100.0
    return val_loss, val_top1


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--train-dir", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--val-dir", type=str, default="/opt/ml/input/data/val")
    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--target-val-top1", type=float, default=None, help="Early stop if validation top1 >= this percentage")
    parser.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--nesterov", action="store_true")

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    setup_ddp()
    device = get_device()

    if is_primary():
        print("Config:", json.dumps(vars(args), indent=2))
        print(f"World size: {dist.get_world_size() if dist.is_initialized() else 1}")

    train_loader, val_loader = build_dataloaders(
        args.train_dir, args.val_dir, args.batch_size, args.workers
    )

    model = build_model(num_classes=1000).to(device)

    if dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=args.nesterov, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    best_top1 = 0.0
    best_train_top1 = 0.0
    model_dir = Path("/opt/ml/model")
    model_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_top1 = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        val_loss, val_top1 = validate(model, val_loader, device)
        scheduler.step()

        if is_primary():
            # Track running bests
            if train_top1 > best_train_top1:
                best_train_top1 = train_top1
            if val_top1 > best_top1:
                best_top1 = val_top1
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f} train_top1={train_top1:.2f}% (best_train_top1={best_train_top1:.2f}%) "
                f"val_loss={val_loss:.4f} val_top1={val_top1:.2f}% (best_val_top1={best_top1:.2f}%)",
                flush=True,
            )
            # Save last
            to_save = model.module if isinstance(model, DDP) else model
            save_ckpt({
                "epoch": epoch,
                "model_state": to_save.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "val_top1": val_top1,
            }, model_dir / "checkpoint_last.pt")
            # Save best
            if val_top1 >= best_top1:
                best_top1 = val_top1
                save_ckpt({
                    "epoch": epoch,
                    "model_state": to_save.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "val_top1": val_top1,
                }, model_dir / "checkpoint_best.pt")
            # Early stopping check
            stop_flag = 0
            if args.target_val_top1 is not None and val_top1 >= args.target_val_top1:
                print(f"[EARLY-STOP] Target {args.target_val_top1:.2f}% reached (val_top1={val_top1:.2f}%) at epoch {epoch}", flush=True)
                stop_flag = 1
            # Broadcast decision
            flag_tensor = torch.tensor([stop_flag], device=device)
            if dist.is_initialized():
                dist.broadcast(flag_tensor, src=0)
            if stop_flag == 1:
                break
        else:
            # Non-primary waits for potential stop signal
            flag_tensor = torch.tensor([0], device=device)
            if dist.is_initialized():
                dist.broadcast(flag_tensor, src=0)
            if flag_tensor.item() == 1:
                break

    if dist.is_initialized():
        dist.destroy_process_group()

    if is_primary():
        print("Training complete. Checkpoints saved to /opt/ml/model")


if __name__ == "__main__":
    main()
