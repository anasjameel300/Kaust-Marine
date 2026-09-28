"""
BioCLIP + LoRA Fine-Tuning & Zero-Shot Benchmark
KAUST Marine Vision - Research Grade Deep Learning Platform

Evaluates BioCLIP (imageomics/bioclip, ViT-B/16 trained on TreeOfLife-10M):
1. Zero-Shot Evaluation: Evaluates out-of-the-box text-image taxonomic classification.
2. Parameter-Efficient Fine-Tuning (LoRA): Injects low-rank adapters into the visual
   transformer encoder while keeping the backbone frozen.

Requirements:
  pip install open-clip-torch

Usage:
  python scripts/train_bioclip_lora.py --epochs 50 --patience 5 --batch_size 16
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Paths
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = WORKSPACE_DIR / "datasets" / "fish_vista_10_families"
SAVE_DIR = WORKSPACE_DIR / "runs" / "classify_fish_vista" / "bioclip_lora"

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

class LoRALinear(nn.Module):
    """Low-Rank Adaptation layer for linear projections."""
    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

        self.rank = rank
        self.scaling = alpha / rank

        in_features = original_linear.in_features
        out_features = original_linear.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.original_linear(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return orig_out + lora_out

def apply_lora_to_clip_visual(visual_encoder: nn.Module, rank: int = 8, alpha: float = 16.0):
    """Traverses visual transformer attention blocks and wraps in_proj or qkv projections with LoRA."""
    injected_count = 0
    for name, module in visual_encoder.named_modules():
        # Check standard PyTorch MultiheadAttention in_proj_weight or custom Linear projections
        if hasattr(module, "in_proj_weight") and isinstance(module.in_proj_weight, nn.Parameter):
            # Standard nn.MultiheadAttention uses in_proj_weight parameter
            pass
        elif hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
            module.qkv = LoRALinear(module.qkv, rank=rank, alpha=alpha)
            injected_count += 1
        elif hasattr(module, "out_proj") and isinstance(module.out_proj, nn.Linear):
            module.out_proj = LoRALinear(module.out_proj, rank=rank, alpha=alpha)
            injected_count += 1
    return injected_count

class BioCLIPWithLoRA(nn.Module):
    def __init__(self, open_clip_model, num_classes: int = 10, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.visual = open_clip_model.visual

        # Freeze all vision encoder parameters
        for param in self.visual.parameters():
            param.requires_grad = False

        # Apply LoRA to attention modules
        injected = apply_lora_to_clip_visual(self.visual, rank=rank, alpha=alpha)
        print(f"[LoRA] Injected LoRA into {injected} vision transformer layers.")

        # Determine feature output dimension
        embed_dim = 512
        if hasattr(self.visual, "output_dim"):
            embed_dim = self.visual.output_dim
        elif hasattr(open_clip_model, "text_projection") and open_clip_model.text_projection is not None:
            embed_dim = open_clip_model.text_projection.shape[1]

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.visual(x)
        # Ensure 2D (B, D)
        if feats.ndim > 2:
            feats = feats[:, 0]
        logits = self.classifier(feats)
        return logits

def evaluate_zero_shot(clip_model, tokenizer, val_loader, class_names, device):
    """Evaluates BioCLIP in pure zero-shot mode using biological taxonomic prompts."""
    print("\n" + "=" * 80)
    print(" [BENCHMARK 1] EVALUATING BIOCLIP ZERO-SHOT ACCURACY")
    print("=" * 80)
    clip_model.eval()

    # Create descriptive taxonomic prompts
    prompts = [f"a scientific photo of {cls_name}, a family of marine reef fish." for cls_name in class_names]
    text_tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="BioCLIP Zero-Shot"):
            images = images.to(device)
            labels = labels.to(device)

            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Cosine similarity logits
            similarity = (100.0 * image_features @ text_features.T)
            _, pred_top1 = similarity.max(1)
            total += labels.size(0)
            correct_top1 += pred_top1.eq(labels).sum().item()

            maxk = min(5, len(class_names))
            _, pred_topk = similarity.topk(maxk, 1, True, True)
            correct_top5 += pred_topk.eq(labels.view(-1, 1).expand_as(pred_topk)).sum().item()

    acc_top1 = 100.0 * correct_top1 / max(1, total)
    acc_top5 = 100.0 * correct_top5 / max(1, total)
    print(f"\n>>> BioCLIP Zero-Shot Top-1 Accuracy: {acc_top1:.2f}% | Top-5: {acc_top5:.2f}%")
    print("=" * 80 + "\n")
    return acc_top1, acc_top5

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
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    val_transforms = transforms.Compose([
        LetterboxPad(target_size=224, fill=114),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transforms)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, train_dataset.classes

def main():
    parser = argparse.ArgumentParser(description="BioCLIP Zero-Shot & LoRA Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs (default: 50)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha")
    args = parser.parse_args()

    try:
        import open_clip
    except ImportError:
        print("[ERROR] 'open-clip-torch' package is required for BioCLIP.")
        print("Please install via: pip install open-clip-torch")
        sys.exit(1)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(" KAUST MARINE: BIOCLIP (ViT-B/16) ZERO-SHOT & LoRA BENCHMARK")
    print(f" Backbone: imageomics/bioclip (TreeOfLife-10M)")
    print(f" Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    train_loader, val_loader, class_names = get_dataloaders(args.batch_size)
    num_classes = len(class_names)

    # 1. Load official BioCLIP model
    print("[BioCLIP] Downloading / Loading 'hf-hub:imageomics/bioclip'...")
    clip_model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
    clip_model.to(device)

    # 2. Evaluate Zero-Shot Baseline
    evaluate_zero_shot(clip_model, tokenizer, val_loader, class_names, device)

    # 3. Build LoRA Classifier
    print("=" * 80)
    print(" [BENCHMARK 2] TRAINING BIOCLIP + LoRA ADAPTER")
    print("=" * 80)
    lora_model = BioCLIPWithLoRA(clip_model, num_classes=num_classes, rank=args.rank, alpha=args.alpha)
    lora_model.to(device)

    trainable_tensors = [p for p in lora_model.parameters() if p.requires_grad]
    trainable_cnt = sum(p.numel() for p in trainable_tensors)
    total_cnt = sum(p.numel() for p in lora_model.parameters())
    print(f"[Parameters] Trainable: {trainable_cnt:,} / {total_cnt:,} ({100.0 * trainable_cnt / total_cnt:.2f}%)")

    optimizer = torch.optim.AdamW(trainable_tensors, lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    best_val_top5 = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        lora_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]", leave=False)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = lora_model(images)
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
        lora_model.eval()
        val_loss = 0.0
        val_correct_top1 = 0
        val_correct_top5 = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    outputs = lora_model(images)
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

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train: {epoch_train_acc:.1f}% | Val Top-1: {epoch_val_acc:.2f}% | Top-5: {epoch_val_top5:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_val_top5 = epoch_val_top5
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_name": "bioclip_vitb16_lora",
                "rank": args.rank,
                "alpha": args.alpha,
                "state_dict": lora_model.state_dict(),
                "val_acc_top1": epoch_val_acc,
                "val_acc_top5": epoch_val_top5,
                "class_names": class_names,
            }
            torch.save(checkpoint, SAVE_DIR / "best.pth")
            print(f"  >>> Best Checkpoint Saved! (New Best Val Acc: {best_val_acc:.2f}%)")
        else:
            if epoch >= 10:
                patience_counter += 1
                print(f"  [Patience: {patience_counter}/{args.patience}] No validation improvement.")
                if patience_counter >= args.patience:
                    print(f"\n[Early Stopping] No validation improvement for {args.patience} consecutive epochs.")
                    break

    print("\n" + "=" * 80)
    print(" BioCLIP + LoRA Fine-Tuning Complete")
    print(f" Best Top-1 Validation Accuracy: {best_val_acc:.2f}%")
    print(f" Saved Checkpoint: {SAVE_DIR / 'best.pth'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
