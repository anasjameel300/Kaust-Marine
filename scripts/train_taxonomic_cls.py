"""
Stage 2: Taxonomic Marine Fish Family Classifier Training
KAUST Marine Vision - Research Grade Deep Learning Platform

Trains YOLO11s-cls on the balanced 10-Family Red Sea dataset:
- Families: Lutjanidae (Snappers), Scaridae (Parrotfish), Balistidae (Triggerfish),
            Siganidae (Rabbitfish), Acanthuridae (Surgeonfish), Chaetodontidae (Butterflyfish),
            Holocentridae (Squirrelfish), Labridae (Wrasses), Nemipteridae (Breams), Pomacentridae (Damselfish)
- Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
- Optimization: PyTorch AMP (Automatic Mixed Precision) + Cosine LR Decay
"""

import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

# Workspace Paths
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = WORKSPACE_DIR / "datasets" / "taxonomic_cls" / "families"
PROJECT_DIR = WORKSPACE_DIR / "runs" / "classify"
RUN_NAME = "kaust_taxonomic_cls_v1"

def train_classifier():
    print("=" * 70)
    print("KAUST Marine Vision: Training Stage 2 Taxonomic Marine Classifier")
    print("=" * 70)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found at {DATASET_DIR}. Run balance_and_prepare_taxonomic_dataset.py first.")

    # Device configuration
    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[Hardware] Training Device: {device_name} (device={device})")

    # Load pretrained YOLO11s-cls backbone
    model = YOLO("yolo11s-cls.pt")
    print(f"[Architecture] Initialized YOLO11s-cls backbone")

    # Hyperparameters optimized for RTX 3050 (4 GB VRAM)
    results = model.train(
        data=str(DATASET_DIR),
        epochs=30,
        imgsz=224,
        batch=32,
        device=device,
        workers=4,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        dropout=0.15,
        amp=True,
        cache=True,
        verbose=True
    )

    print("\n" + "=" * 70)
    print(f"Training Complete! Artifacts saved to: {PROJECT_DIR / RUN_NAME}")
    print(f"Best Checkpoint: {PROJECT_DIR / RUN_NAME / 'weights' / 'best.pt'}")
    print("=" * 70)

if __name__ == "__main__":
    train_classifier()
