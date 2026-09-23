import os
import torch
from pathlib import Path
from ultralytics import YOLO

class CleanTerminalCallback:
    """
    Custom callback to display clean, spacious, research-grade training progress.
    Avoids cramped text and highlights peak model checkpoint saves.
    """
    def __init__(self):
        self.best_fitness = 0.0

    def on_fit_epoch_end(self, trainer):
        try:
            epoch = trainer.epoch + 1
            total_epochs = trainer.epochs
            
            # GPU memory usage
            gpu_mem = f"{torch.cuda.memory_reserved(0) / (1024**3):.2f} GB" if torch.cuda.is_available() else "CPU"

            # Validation metrics
            metrics = getattr(trainer, "metrics", {}) or {}
            box_map50 = metrics.get("metrics/mAP50(B)", 0.0)
            mask_map50 = metrics.get("metrics/mAP50(M)", 0.0)
            mask_map50_95 = metrics.get("metrics/mAP50-95(M)", 0.0)
            fitness = getattr(trainer, "fitness", 0.0) or 0.0

            # Check if new best checkpoint was saved
            is_new_best = False
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                is_new_best = True

            print("\n" + "=" * 70)
            print(f"  🏁 EPOCH {epoch} / {total_epochs}  |  GPU VRAM: {gpu_mem}  |  Fitness: {fitness:.4f}")
            print("-" * 70)
            print(f"  [VALIDATION METRICS - OVERALL]")
            print(f"    • Box mAP50       : {box_map50 * 100:.1f}%")
            print(f"    • Mask mAP50      : {mask_map50 * 100:.1f}%")
            print(f"    • Mask mAP50-95   : {mask_map50_95 * 100:.1f}%")
            print()
            if is_new_best:
                print(f"  ⭐ [CHECKPOINT] NEW BEST MODEL SAVED! -> weights/best.pt")
            else:
                print(f"  🔒 [CHECKPOINT] Historical Best Preserved (Peak Fitness: {self.best_fitness:.4f})")
            print("=" * 70 + "\n")
        except Exception:
            pass

def train():
    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("\n" + "=" * 70)
    print(f"  KAUST Marine Fish Instance Segmentation Training")
    print(f"  Model Architecture: YOLO11s-seg (Small, 9.4M params)")
    print(f"  Hardware: {device_name} (device={device})")
    print("=" * 70 + "\n")

    dataset_yaml = Path(__file__).resolve().parent.parent / "datasets" / "underwater_yolo_seg" / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"dataset.yaml not found at {dataset_yaml}")

    print("[1/2] Loading YOLO11s-seg (Small) backbone...")
    model = YOLO("yolo11s-seg.pt")

    # Attach our clean terminal formatter callback
    clean_cb = CleanTerminalCallback()
    model.add_callback("on_fit_epoch_end", clean_cb.on_fit_epoch_end)

    print("[2/2] Launching training on clean 3-class marine dataset...")
    results = model.train(
        data=str(dataset_yaml),
        epochs=40,
        imgsz=640,
        batch=8,
        workers=2,
        device=device,
        project="models",
        name="kaust_fish_seg_v2",
        exist_ok=True,
        save=True,
        save_period=5,
        patience=12,
        verbose=True
    )

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("  Best weights saved to: models/kaust_fish_seg_v2/weights/best.pt")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    train()
