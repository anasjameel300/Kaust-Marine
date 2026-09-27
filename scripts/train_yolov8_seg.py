"""
Stage 1: YOLOv8-seg COCO-Pretrained Instance Segmentation Training
KAUST Marine Vision - Research Grade Deep Learning Platform

Trains YOLOv8s-seg (Small, 11.2M params) initialized with official COCO segmentation weights
on the clean 3-class marine dataset:
- Classes: 0: fish, 1: reefs, 2: human_divers
- Epochs: 50
- Early Stopping Patience: 5 epochs
- Checkpoint System: Automatically saves weights/best.pt and periodic checkpoints
- Hardware: Optimized for NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) with PyTorch AMP
"""

import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

# Paths
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
DATASET_YAML = WORKSPACE_DIR / "datasets" / "underwater_yolo_seg" / "dataset.yaml"
PROJECT_DIR = WORKSPACE_DIR / "runs" / "segment" / "models"
RUN_NAME = "kaust_yolov8s_seg_v1"

class CleanTerminalCallback:
    """
    Clean, spacious progress callback that logs epoch validation metrics
    and explicitly notifies when a new best checkpoint is saved.
    """
    def __init__(self):
        self.best_fitness = 0.0

    def on_fit_epoch_end(self, trainer):
        try:
            epoch = trainer.epoch + 1
            total_epochs = trainer.epochs
            
            # GPU VRAM tracking
            if torch.cuda.is_available():
                gpu_mem = f"{torch.cuda.memory_reserved(0) / (1024**3):.2f} GB"
            else:
                gpu_mem = "CPU"

            metrics = getattr(trainer, "metrics", {}) or {}
            box_map50 = metrics.get("metrics/mAP50(B)", 0.0)
            mask_map50 = metrics.get("metrics/mAP50(M)", 0.0)
            mask_map50_95 = metrics.get("metrics/mAP50-95(M)", 0.0)
            fitness = getattr(trainer, "fitness", 0.0) or 0.0

            is_new_best = False
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                is_new_best = True

            print("\n" + "=" * 72)
            print(f"  🏁 EPOCH {epoch:02d} / {total_epochs:02d}  |  GPU VRAM: {gpu_mem}  |  Fitness: {fitness:.4f}")
            print("-" * 72)
            print(f"  [VALIDATION PERFORMANCE]")
            print(f"    • Box mAP50       : {box_map50 * 100:.1f}%")
            print(f"    • Mask mAP50      : {mask_map50 * 100:.1f}%")
            print(f"    • Mask mAP50-95   : {mask_map50_95 * 100:.1f}%")
            print()
            if is_new_best:
                print(f"  ⭐ [CHECKPOINT SAVED] New best model recorded! -> weights/best.pt")
            else:
                print(f"  🔒 [HISTORICAL BEST] Preserved peak fitness: {self.best_fitness:.4f}")
            print("=" * 72 + "\n")
        except Exception:
            pass

def main():
    print("=" * 72)
    print("  KAUST Marine Vision: Training YOLOv8-seg (COCO Pretrained)")
    print("=" * 72)

    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"Dataset YAML not found at: {DATASET_YAML}")

    # Device configuration
    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[Hardware] Device: {device_name} (device={device})")

    # Load official COCO-pretrained YOLOv8s-seg weights
    print("[Architecture] Initializing YOLOv8s-seg with official COCO pretrained weights...")
    model = YOLO("yolov8s-seg.pt")

    # Attach our clean terminal callback
    clean_cb = CleanTerminalCallback()
    model.add_callback("on_fit_epoch_end", clean_cb.on_fit_epoch_end)

    print(f"[Hyperparameters] Epochs=50, Patience=5, Batch=8, ImgSize=640, AMP=True")
    print(f"[Output Directory] {PROJECT_DIR / RUN_NAME}")
    print("=" * 72 + "\n")

    # Run training
    results = model.train(
        data=str(DATASET_YAML),
        epochs=50,
        patience=5,             # Stop early if no improvement for 5 consecutive epochs
        imgsz=640,
        batch=8,                # Safe for 4 GB VRAM on RTX 3050 Laptop
        workers=2,
        device=device,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        save=True,              # Ensure checkpoints are saved
        save_period=5,          # Save backup checkpoint every 5 epochs
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,
        overlap_mask=True,
        mask_ratio=4,
        amp=True,               # Automatic Mixed Precision
        verbose=True
    )

    print("\n" + "=" * 72)
    print("  TRAINING SESSION COMPLETE")
    print(f"  Best Weights Checkpoint : {PROJECT_DIR / RUN_NAME / 'weights' / 'best.pt'}")
    print(f"  Last Weights Checkpoint : {PROJECT_DIR / RUN_NAME / 'weights' / 'last.pt'}")
    print("=" * 72 + "\n")

if __name__ == "__main__":
    main()
