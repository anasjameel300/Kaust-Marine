import os
import torch
from pathlib import Path
from ultralytics import YOLO

class CleanTerminalCallback:
    """
    Custom callback to display clean, spacious, research-grade training progress.
    Prevents terminal clutter, tracks peak mAP, and clearly indicates best checkpoint saves.
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

            print("\n" + "=" * 72)
            print(f"  🏁 [STAGE 2 FINE-TUNE] EPOCH {epoch} / {total_epochs}  |  GPU VRAM: {gpu_mem}  |  Fitness: {fitness:.4f}")
            print("-" * 72)
            print(f"  [VALIDATION METRICS - OVERALL]")
            print(f"    • Box mAP50       : {box_map50 * 100:.1f}%")
            print(f"    • Mask mAP50      : {mask_map50 * 100:.1f}%")
            print(f"    • Mask mAP50-95   : {mask_map50_95 * 100:.1f}%")
            print()
            if is_new_best:
                print(f"  ⭐ [CHECKPOINT] NEW BEST FINE-TUNED MODEL SAVED! -> weights/best.pt")
            else:
                print(f"  🔒 [CHECKPOINT] Historical Best Preserved (Peak Fitness: {self.best_fitness:.4f})")
            print("=" * 72 + "\n")
        except Exception:
            pass

def finetune():
    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    # Base weights from Stage 1 Peak (Epoch 40 best.pt)
    base_dir = Path(__file__).resolve().parent.parent
    weights_path = base_dir / "runs" / "segment" / "models" / "kaust_fish_seg_v2" / "weights" / "best.pt"

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found at: {weights_path}\n"
            "Please ensure Stage 1 training has completed before running fine-tuning."
        )

    dataset_yaml = base_dir / "datasets" / "underwater_yolo_seg" / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found at: {dataset_yaml}")

    print("\n" + "=" * 72)
    print("  KAUST Marine Vision - Stage 2 Warm Fine-Tuning")
    print(f"  Base Checkpoint : {weights_path.name} (from Stage 1 v2)")
    print(f"  Architecture    : YOLO11s-seg (9.4M params)")
    print(f"  Compute Device  : {device_name} (device={device})")
    print("  Strategy        :")
    print("    - Warm initialization (lr0=0.001, lrf=0.0001, cosine decay)")
    print("    - Early backbone freeze (freeze=6 layers) to avoid overfitting")
    print("    - Dynamic underwater color augmentation (HSV jitter & flip)")
    print("    - Mosaic augmentation disabled for final 5 epochs (close_mosaic=5)")
    print("=" * 72 + "\n")

    # Load model with stage 1 weights
    model = YOLO(str(weights_path))

    # Register clean terminal callback
    callback = CleanTerminalCallback()
    model.add_callback("on_fit_epoch_end", callback.on_fit_epoch_end)

    # Launch Fine-Tuning
    results = model.train(
        data=str(dataset_yaml),
        epochs=25,
        batch=8,
        imgsz=640,
        device=device,
        project="runs/segment/models",
        name="kaust_fish_seg_finetune_v1",
        exist_ok=True,
        workers=2,
        
        # Warm learning rate schedule
        lr0=0.001,           # 10x lower than initial training
        lrf=0.0001,          # Final learning rate
        cos_lr=True,         # Cosine annealing
        warmup_epochs=2,     # Smooth ramp-up
        
        # Freezing early backbone to preserve edge/texture filters & stop overfitting
        freeze=6,
        
        # Underwater domain augmentations
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        fliplr=0.5,
        mosaic=0.5,
        close_mosaic=5,      # Turn off mosaic for final 5 epochs for clean boundaries
        
        # Checkpointing & Optimization
        patience=10,
        save=True,
        save_period=-1,      # Only save best and last
        plots=True,
        verbose=True
    )

    print("\n" + "=" * 72)
    print("  🎉 Stage 2 Warm Fine-Tuning Completed Successfully!")
    print(f"  Best Weights Saved To: runs/segment/models/kaust_fish_seg_finetune_v1/weights/best.pt")
    print("=" * 72 + "\n")

    # Final detailed class evaluation
    print("[Evaluation] Running full per-class validation on best weights...")
    val_model = YOLO("runs/segment/models/kaust_fish_seg_finetune_v1/weights/best.pt")
    val_results = val_model.val(data=str(dataset_yaml), device=device)

    print("\n" + "=" * 72)
    print("  📊 FINAL PER-CLASS VALIDATION SCORES")
    print("-" * 72)
    names = val_results.names
    for cls_id, cls_name in names.items():
        if cls_id < len(val_results.seg.map50):
            m50 = val_results.seg.map50[cls_id] * 100
            m50_95 = val_results.seg.map[cls_id] * 100
            p = val_results.seg.p[cls_id] * 100
            r = val_results.seg.r[cls_id] * 100
            print(f"  Class {cls_id} ({cls_name:<14}): Precision={p:5.1f}% | Recall={r:5.1f}% | Mask mAP50={m50:5.1f}% | mAP50-95={m50_95:5.1f}%")
    print("=" * 72 + "\n")

if __name__ == "__main__":
    finetune()
