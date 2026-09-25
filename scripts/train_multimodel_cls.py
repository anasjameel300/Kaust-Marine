"""
Multi-Model Taxonomic Fish Family Classification Suite
KAUST Marine Vision - Research Grade Deep Learning Platform

Trains non-YOLO diverse deep learning architectures on the balanced 10-Family Red Sea dataset:
1. ConvNeXt-Tiny (Modern Pure ConvNet with 7x7 Depthwise Kernels)
2. MobileNetV4-Conv-Small (Ultra-Efficient Inverted Residual Architecture)
3. Swin-Tiny (Hierarchical Vision Transformer with Shifted Window Attention)

Supports individual training via `--model <name>` or sequential full-suite training via `--model all`.
Optimized for NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) using PyTorch AMP.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from pathlib import Path

# Paths
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = WORKSPACE_DIR / "datasets" / "taxonomic_cls" / "families"
RUNS_DIR = WORKSPACE_DIR / "runs" / "classify"

# Architecture Configurations
MODEL_CONFIGS = {
    "convnext": {
        "timm_name": "convnext_tiny",
        "save_dir": RUNS_DIR / "convnext_tiny",
        "description": "ConvNeXt-Tiny (Pure ConvNet with 7x7 Large Receptive Field)",
        "batch_size": 32,
        "lr": 5e-4,
        "weight_decay": 1e-2,
        "epochs": 50
    },
    "mobilenetv4": {
        "timm_name": "mobilenetv4_conv_small",
        "save_dir": RUNS_DIR / "mobilenetv4_small",
        "description": "MobileNetV4-Small (Universal Efficient Inverted Residual CNN)",
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 50
    },
    "swin": {
        "timm_name": "swin_tiny_patch4_window7_224",
        "save_dir": RUNS_DIR / "swin_tiny",
        "description": "Swin-Tiny Transformer (Shifted Window Self-Attention)",
        "batch_size": 32,
        "lr": 3e-4,
        "weight_decay": 5e-2,
        "epochs": 50
    }
}

def get_dataloaders(batch_size: int):
    """Constructs train and validation dataloaders with domain-specific augmentation."""
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Train/val directories missing under {DATASET_DIR}")

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transforms)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, train_dataset.classes

def train_single_model(model_key: str, patience: int = 5, min_epochs: int = 25):
    from tqdm import tqdm
    config = MODEL_CONFIGS[model_key]
    save_dir = config["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75, flush=True)
    print(f"Training Architecture: {config['description']}", flush=True)
    print(f"Max Epochs: {config['epochs']} | Min Epochs before Early Stop: {min_epochs} | Patience: {patience}", flush=True)
    print(f"Batch Size: {config['batch_size']} | Learning Rate: {config['lr']}", flush=True)
    print(f"Artifact Save Directory: {save_dir}", flush=True)
    print("=" * 75, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

    # Build dataloaders
    train_loader, val_loader, class_names = get_dataloaders(config["batch_size"])
    num_classes = len(class_names)
    print(f"[Classes] {num_classes} Red Sea Families: {class_names}", flush=True)

    # Initialize pretrained backbone from timm
    print(f"[Model] Initializing pretrained {config['timm_name']} backbone...", flush=True)
    model = timm.create_model(config["timm_name"], pretrained=True, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    best_top1_acc = 0.0
    epochs_no_improve = 0
    history = []

    start_time = time.time()
    total_epochs = config["epochs"]

    print(f"\n[Training] Starting {total_epochs} Max Epochs with Early Stopping (Patience: {patience}, Min Epochs: {min_epochs})...\n", flush=True)

    for epoch in range(1, total_epochs + 1):
        # Training Phase with live tqdm bar
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Train]", unit="batch", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct_train += preds.eq(labels).sum().item()
            total_train += labels.size(0)

            current_acc = (correct_train / total_train) * 100
            current_loss = running_loss / total_train
            pbar.set_postfix({"Loss": f"{current_loss:.4f}", "Acc": f"{current_acc:.1f}%", "LR": f"{optimizer.param_groups[0]['lr']:.2e}"})

        scheduler.step()
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # Validation Phase with live tqdm bar
        model.eval()
        val_loss = 0.0
        correct_top1 = 0
        correct_top5 = 0
        total_val = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Val]  ", unit="batch", leave=False)
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, top1_preds = outputs.max(1)
                correct_top1 += top1_preds.eq(labels).sum().item()

                maxk = min(5, num_classes)
                _, top5_preds = outputs.topk(maxk, 1, True, True)
                correct_top5 += top5_preds.eq(labels.view(-1, 1).expand_as(top5_preds)).sum().item()
                total_val += labels.size(0)

                val_pbar.set_postfix({"Top1": f"{(correct_top1/total_val)*100:.1f}%", "Top5": f"{(correct_top5/total_val)*100:.1f}%"})

        val_loss = val_loss / total_val
        val_top1 = correct_top1 / total_val
        val_top5 = correct_top5 / total_val

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_top1_acc": round(val_top1, 4),
            "val_top5_acc": round(val_top5, 4),
            "lr": round(optimizer.param_groups[0]["lr"], 6)
        }
        history.append(epoch_stats)

        is_best = val_top1 > best_top1_acc
        if is_best:
            best_top1_acc = val_top1
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_name": config["timm_name"],
                "class_names": class_names,
                "state_dict": model.state_dict(),
                "val_top1_acc": val_top1,
                "val_top5_acc": val_top5
            }, str(save_dir / "best.pth"))
        else:
            epochs_no_improve += 1

        marker = " 🏆 (New Best)" if is_best else ""
        print(f"Epoch [{epoch:02d}/{total_epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.1f}% || "
              f"Val Loss: {val_loss:.4f} | Val Top-1: {val_top1*100:5.1f}% | Top-5: {val_top5*100:5.1f}%{marker}", flush=True)

        # Early Stopping Logic: only activates after min_epochs
        if epoch >= min_epochs and epochs_no_improve >= patience:
            print(f"\n⏹ Early Stopping Triggered at Epoch {epoch}!", flush=True)
            print(f"  • No validation improvement for {patience} consecutive epochs after epoch {min_epochs}.", flush=True)
            print(f"  • Best checkpoint preserved with Top-1 Accuracy: {best_top1_acc * 100:.2f}%\n", flush=True)
            break

    total_time = round(time.time() - start_time, 1)

    # Save summary metadata
    summary = {
        "model_key": model_key,
        "timm_name": config["timm_name"],
        "description": config["description"],
        "best_val_top1_acc": round(best_top1_acc, 4),
        "total_training_seconds": total_time,
        "stopped_epoch": epoch,
        "classes": class_names,
        "history": history
    }

    with open(save_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 75, flush=True)
    print(f"Finished {config['timm_name']} in {total_time}s (Stopped at Epoch {epoch})", flush=True)
    print(f"Best Top-1 Validation Accuracy: {best_top1_acc * 100:.2f}%", flush=True)
    print(f"Checkpoint saved: {save_dir / 'best.pth'}", flush=True)
    print("-" * 75 + "\n", flush=True)

    return summary

def main():
    parser = argparse.ArgumentParser(description="Multi-Model Taxonomic Fish Classifier Sequential Training")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "convnext", "mobilenetv4", "swin"],
                        help="Model architecture to train (default: all - trains all 3 sequentially)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Number of epochs without val improvement before early stopping (default: 5)")
    parser.add_argument("--min-epochs", type=int, default=20,
                        help="Minimum epochs to train before early stopping can trigger (default: 20)")
    args = parser.parse_args()

    models_to_train = ["convnext", "mobilenetv4", "swin"] if args.model == "all" else [args.model]

    print("=" * 75, flush=True)
    print("KAUST Marine Vision - Multi-Model Pipeline Sequence", flush=True)
    print(f"Planned Models to Train: {', '.join(models_to_train).upper()}", flush=True)
    print(f"Early Stopping Policy: Patience = {args.patience} epochs (Active after Epoch {args.min_epochs})", flush=True)
    print("=" * 75, flush=True)

    results = {}
    for idx, m in enumerate(models_to_train, 1):
        print(f"\n>>> Starting Pipeline Stage [{idx}/{len(models_to_train)}]: {m.upper()} <<<", flush=True)
        results[m] = train_single_model(m, patience=args.patience, min_epochs=args.min_epochs)

    print("\n" + "=" * 75, flush=True)
    print("🏆 ALL MULTI-MODEL ARCHITECTURES TRAINED & VALIDATED SUCCESSFULLY!", flush=True)
    print("=" * 75, flush=True)
    print(f"{'Model Architecture':<24} | {'Best Top-1 Acc':>14} | {'Stopped Epoch':>13} | {'Training Time':>13}")
    print("-" * 75, flush=True)
    for m, res in results.items():
        print(f"{res['timm_name']:<24} | {res['best_val_top1_acc']*100:>13.2f}% | {res['stopped_epoch']:>13} | {res['total_training_seconds']:>12.1f}s", flush=True)
    print("=" * 75 + "\n", flush=True)

if __name__ == "__main__":
    main()

