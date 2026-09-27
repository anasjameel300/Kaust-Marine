"""
Master Automated Baseline Training Suite for Fish-Vista (10 Families)
KAUST Marine Vision - Research Grade Deep Learning Platform

Executes end-to-end sequential training across our full suite of dedicated
traditional deep learning architectures on the high-resolution Fish-Vista dataset:
  1. YOLO11s-cls          (Real-time CSPDarknet backbone)
  2. MobileNetV4-Small    (Universal edge-optimized inverted residual CNN)
  3. ConvNeXt-Tiny        (Modernized large-kernel 7x7 depthwise CNN)
  4. Swin-Tiny            (Hierarchical Shifted-Window Vision Transformer)
  5. ResNet50d            (Classic academic benchmark deep residual network)

Features:
- Aspect-ratio preserving Letterbox Padding (114, 114, 114) for all PyTorch models.
- GPU VRAM memory flushing between runs to prevent CUDA OOM on 4 GB RTX 3050.
- Summary leaderboard comparing Top-1 Acc, Top-5 Acc, and runtime at the end.

Usage:
  # Train all 5 baseline models sequentially with one single command:
  python scripts/train_all_fish_vista_baselines.py

  # Or train a specific subset:
  python scripts/train_all_fish_vista_baselines.py --models yolo mobilenetv4 convnext swin resnet50
"""

import os
import sys
import time
import argparse
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

# Paths
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = WORKSPACE_DIR / "datasets" / "fish_vista_10_families"
RUNS_DIR = WORKSPACE_DIR / "runs" / "classify_fish_vista"

# Aspect-ratio preserving Letterbox Pad
class LetterboxPad:
    def __init__(self, target_size: int = 224, fill: int = 114):
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == 0 or h == 0:
            return Image.new("RGB", (self.target_size, self.target_size), (self.fill, self.fill, self.fill))
        scale = min(self.target_size / w, self.target_size / h)
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = img.resize((nw, nh), resample=Image.BILINEAR)

        pad_left = (self.target_size - nw) // 2
        pad_right = self.target_size - nw - pad_left
        pad_top = (self.target_size - nh) // 2
        pad_bottom = self.target_size - nh - pad_top

        return ImageOps.expand(resized, (pad_left, pad_top, pad_right, pad_bottom), fill=(self.fill, self.fill, self.fill))

# Model Configurations for PyTorch / timm
TIMM_MODELS = {
    "mobilenetv4": {
        "timm_name": "mobilenetv4_conv_small",
        "save_dir": RUNS_DIR / "mobilenetv4_small",
        "desc": "MobileNetV4-Small (Edge Inverted Residual CNN)",
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 50
    },
    "convnext": {
        "timm_name": "convnext_tiny",
        "save_dir": RUNS_DIR / "convnext_tiny",
        "desc": "ConvNeXt-Tiny (Modern 7x7 Depthwise ConvNet)",
        "batch_size": 32,
        "lr": 5e-4,
        "weight_decay": 1e-2,
        "epochs": 50
    },
    "swin": {
        "timm_name": "swin_tiny_patch4_window7_224",
        "save_dir": RUNS_DIR / "swin_tiny",
        "desc": "Swin-Tiny (Shifted-Window Vision Transformer)",
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 5e-2,
        "epochs": 50
    },
    "resnet50": {
        "timm_name": "resnet50d",
        "save_dir": RUNS_DIR / "resnet50d",
        "desc": "ResNet50d (Classic Deep Residual Network Baseline)",
        "batch_size": 32,
        "lr": 5e-4,
        "weight_decay": 1e-3,
        "epochs": 50
    }
}

def get_dataloaders(batch_size: int):
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Missing train or val directory under {DATASET_DIR}")

    train_transforms = transforms.Compose([
        LetterboxPad(target_size=224, fill=114),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.RandomRotation(degrees=8),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        LetterboxPad(target_size=224, fill=114),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transforms)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, train_dataset.classes

def train_yolo() -> dict:
    """Trains YOLO11s-cls using Ultralytics engine."""
    from ultralytics import YOLO
    print("\n" + "#" * 80)
    print(" [1/5] TRAINING: YOLO11s-cls (CSPDarknet Backbone)")
    print("#" * 80)

    start_time = time.time()
    save_dir = RUNS_DIR / "yolo11s_cls"
    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO("yolo11s-cls.pt")
    results = model.train(
        data=str(DATASET_DIR),
        epochs=50,
        imgsz=224,
        batch=32,
        device=device,
        workers=4,
        project=str(RUNS_DIR),
        name="yolo11s_cls",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        dropout=0.15,
        amp=True,
        cache=False,
        patience=5,
        verbose=False
    )

    elapsed_min = (time.time() - start_time) / 60.0
    top1 = getattr(results, "top1", 0.0)
    if isinstance(top1, float) and top1 <= 1.0:
        top1 *= 100.0

    top5 = getattr(results, "top5", 0.0)
    if isinstance(top5, float) and top5 <= 1.0:
        top5 *= 100.0

    # Flush GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_key": "yolo11s",
        "description": "YOLO11s-cls (CSPDarknet)",
        "top1_acc": top1,
        "top5_acc": top5,
        "runtime_min": elapsed_min,
        "best_path": str(save_dir / "weights" / "best.pt")
    }

def train_timm_model(model_key: str, patience: int = 7, min_epochs: int = 15) -> dict:
    """Trains a PyTorch model via timm with AMP and Cosine Annealing."""
    config = TIMM_MODELS[model_key]
    save_dir = config["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 80)
    print(f" TRAINING: {config['desc']}")
    print(f" Batch: {config['batch_size']} | LR: {config['lr']} | Max Epochs: {config['epochs']}")
    print(f" Save Dir: {save_dir}")
    print("#" * 80)

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, class_names = get_dataloaders(config["batch_size"])
    num_classes = len(class_names)

    model = timm.create_model(config["timm_name"], pretrained=True, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    best_val_top5 = 0.0
    patience_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{config['epochs']:02d} [Train]", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100.0 * train_correct / train_total:.1f}%"})

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct_top1 = 0
        val_correct_top5 = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, pred_top1 = outputs.max(1)
                val_total += labels.size(0)
                val_correct_top1 += pred_top1.eq(labels).sum().item()

                maxk = min(5, num_classes)
                _, pred_topk = outputs.topk(maxk, 1, True, True)
                val_correct_top5 += pred_topk.eq(labels.view(-1, 1).expand_as(pred_topk)).sum().item()

        epoch_train_loss = train_loss / max(1, train_total)
        epoch_train_acc = 100.0 * train_correct / max(1, train_total)
        epoch_val_loss = val_loss / max(1, val_total)
        epoch_val_acc = 100.0 * val_correct_top1 / max(1, val_total)
        epoch_val_top5 = 100.0 * val_correct_top5 / max(1, val_total)

        print(f"Epoch {epoch:02d}/{config['epochs']:02d} | Train: {epoch_train_acc:.1f}% (loss {epoch_train_loss:.3f}) | Val Top-1: {epoch_val_acc:.2f}% | Top-5: {epoch_val_top5:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_val_top5 = epoch_val_top5
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_name": config["timm_name"],
                "state_dict": model.state_dict(),
                "val_acc_top1": epoch_val_acc,
                "val_acc_top5": epoch_val_top5,
                "class_names": class_names,
                "preprocessing": "letterbox_pad_224"
            }
            torch.save(checkpoint, save_dir / "best.pth")
            print(f"  >>> Best Checkpoint Saved! (New Best Val Acc: {best_val_acc:.2f}%)")
        else:
            if epoch >= min_epochs:
                patience_counter += 1
                print(f"  [Patience: {patience_counter}/{patience}] No validation improvement.")
                if patience_counter >= patience:
                    print(f"\n[Early Stopping] No validation improvement for {patience} consecutive epochs. Stopping run.")
                    break

        # Always save latest state checkpoint
        last_checkpoint = {
            "epoch": epoch,
            "model_name": config["timm_name"],
            "state_dict": model.state_dict(),
            "val_acc_top1": epoch_val_acc,
            "val_acc_top5": epoch_val_top5,
            "class_names": class_names,
        }
        torch.save(last_checkpoint, save_dir / "last.pth")

    elapsed_min = (time.time() - start_time) / 60.0

    # Memory cleanup
    del model, optimizer, scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_key": model_key,
        "description": config["desc"],
        "top1_acc": best_val_acc,
        "top5_acc": best_val_top5,
        "runtime_min": elapsed_min,
        "best_path": str(save_dir / "best.pth")
    }

def main():
    parser = argparse.ArgumentParser(description="Automated Baseline Training Suite on Fish-Vista (10 Families)")
    parser.add_argument("--models", nargs="+", default=["yolo", "mobilenetv4", "convnext", "swin", "resnet50"],
                        choices=["yolo", "mobilenetv4", "convnext", "swin", "resnet50"],
                        help="List of models to train sequentially (default: all 5 models)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    args = parser.parse_args()

    print("=" * 85)
    print(" KAUST MARINE: AUTOMATED MULTI-MODEL BASELINE TRAINING SUITE")
    print(f" Selected Architectures to Train: {', '.join(args.models)}")
    print(f" Dataset: {DATASET_DIR}")
    print("=" * 85)

    summary_results = []
    suite_start = time.time()

    for idx, model_name in enumerate(args.models, 1):
        print(f"\n>>> Executing Run [{idx}/{len(args.models)}]: {model_name.upper()}...")
        try:
            if model_name == "yolo":
                res = train_yolo()
            else:
                res = train_timm_model(model_name, patience=args.patience)
            summary_results.append(res)
        except Exception as e:
            print(f"\n[ERROR] Failed to train {model_name}: {e}")
            import traceback
            traceback.print_exc()

    total_time = (time.time() - suite_start) / 60.0

    # Print Leaderboard Table
    print("\n\n" + "=" * 95)
    print(" " * 28 + "FISH-VISTA 10-FAMILY BASELINE BENCHMARK LEADERBOARD")
    print("=" * 95)
    print(f" {'Architecture':<32} | {'Top-1 Val Acc':<15} | {'Top-5 Val Acc':<15} | {'Runtime':<12} | {'Status'}")
    print("-" * 95)
    for r in summary_results:
        top1_str = f"{r['top1_acc']:.2f}%" if r['top1_acc'] > 0 else "N/A"
        top5_str = f"{r['top5_acc']:.2f}%" if r['top5_acc'] > 0 else "N/A"
        print(f" {r['description']:<32} | {top1_str:<15} | {top5_str:<15} | {r['runtime_min']:.1f} min{'':<4} | Completed")
    print("-" * 95)
    print(f" Total Suite Training Time: {total_time:.1f} minutes")
    print("=" * 95)

if __name__ == "__main__":
    main()
