"""
Dual-Benchmark Comparative Evaluation Suite
KAUST Marine Vision - Research Grade Deep Learning Platform

Evaluates all trained classification architectures across two distinct regimes:
1. Controlled In-Domain Benchmark: Fish-Vista 10-Family validation split (museum vouchers).
2. Real-World Out-of-Distribution (OOD) Benchmark: Wild Red Sea & coastal in-situ photos.

Models Evaluated:
- Dedicated Task CNNs / Transformers: YOLO11s-cls, ResNet50d, MobileNetV4, ConvNeXt, Swin-Tiny
- Vision Foundation Models (LoRA): DINOv2-Small + LoRA, BioCLIP (ViT-B/16) + LoRA

Calculates:
- Top-1 and Top-5 Accuracy for both domains
- Domain Robustness Degradation: Delta Acc = Acc_controlled - Acc_real_world
- GPU Inference Latency (ms per image)
- Outputs a publication-ready comparative leaderboard table.

Usage:
  python scripts/evaluate_dual_benchmark.py
"""

import os
import sys
import time
import json
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
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

IN_DOMAIN_DIR = WORKSPACE_DIR / "datasets" / "fish_vista_10_families" / "val"
OOD_DIR = WORKSPACE_DIR / "datasets" / "evaluation_ood_redsea"
RUNS_DIR = WORKSPACE_DIR / "runs" / "classify_fish_vista"
REPORT_DIR = WORKSPACE_DIR / "reports"

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

def get_eval_loader(data_dir: Path, mean=None, std=None, batch_size=32):
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    eval_transforms = transforms.Compose([
        LetterboxPad(target_size=224, fill=114),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = datasets.ImageFolder(str(data_dir), transform=eval_transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return loader, dataset.classes

def evaluate_pytorch_model(model: nn.Module, loader: DataLoader, device: torch.device):
    """Evaluates top-1 and top-5 accuracy and average per-image latency on GPU."""
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    latencies = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                outputs = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / images.size(0))

            _, pred_top1 = outputs.max(1)
            total += labels.size(0)
            correct_top1 += pred_top1.eq(labels).sum().item()

            maxk = min(5, outputs.size(1))
            _, pred_topk = outputs.topk(maxk, 1, True, True)
            correct_top5 += pred_topk.eq(labels.view(-1, 1).expand_as(pred_topk)).sum().item()

    acc_top1 = 100.0 * correct_top1 / max(1, total)
    acc_top5 = 100.0 * correct_top5 / max(1, total)
    avg_latency_ms = (sum(latencies) / max(1, len(latencies))) * 1000.0
    return acc_top1, acc_top5, avg_latency_ms

def evaluate_yolo(weights_path: Path, data_dir: Path, device: torch.device):
    """Evaluates YOLO11s-cls weights using native batch inference on raw images."""
    from ultralytics import YOLO
    model = YOLO(str(weights_path))

    image_paths = []
    labels = []
    for fam_dir in sorted(data_dir.iterdir()):
        if fam_dir.is_dir():
            for img_p in fam_dir.glob("*.jpg"):
                image_paths.append(str(img_p))
                labels.append(fam_dir.name.lower())

    correct_top1 = 0
    correct_top5 = 0
    total = len(image_paths)
    latencies = []

    batch_size = 32
    yolo_names = {k: v.lower() for k, v in model.names.items()}
    dev_str = 0 if device.type == "cuda" else "cpu"

    for i in range(0, total, batch_size):
        batch_p = image_paths[i:i + batch_size]
        batch_l = labels[i:i + batch_size]

        t0 = time.perf_counter()
        preds = model.predict(batch_p, batch=len(batch_p), device=dev_str, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) / len(batch_p))

        for res, gt in zip(preds, batch_l):
            probs = res.probs
            top1_name = yolo_names[probs.top1]
            top5_names = [yolo_names[idx] for idx in probs.top5]
            if top1_name == gt:
                correct_top1 += 1
            if gt in top5_names:
                correct_top5 += 1

    acc_top1 = 100.0 * correct_top1 / max(1, total)
    acc_top5 = 100.0 * correct_top5 / max(1, total)
    avg_latency_ms = (sum(latencies) / max(1, len(latencies))) * 1000.0
    return acc_top1, acc_top5, avg_latency_ms

def main():
    print("=" * 95)
    print(" " * 20 + "KAUST MARINE: DUAL-BENCHMARK COMPARATIVE EVALUATION")
    print("=" * 95)
    print(f" In-Domain Benchmark:  {IN_DOMAIN_DIR} (Controlled Museum Vouchers)")
    print(f" Real-World Benchmark: {OOD_DIR} (Wild In-Situ Red Sea Reefs)")
    print("=" * 95)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Evaluation Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Standard loaders
    in_loader, in_classes = get_eval_loader(IN_DOMAIN_DIR, batch_size=32)
    ood_loader, ood_classes = get_eval_loader(OOD_DIR, batch_size=32)

    # BioCLIP specific loaders (CLIP normalization)
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    in_loader_clip, _ = get_eval_loader(IN_DOMAIN_DIR, mean=clip_mean, std=clip_std, batch_size=16)
    ood_loader_clip, _ = get_eval_loader(OOD_DIR, mean=clip_mean, std=clip_std, batch_size=16)

    results = []

    # 1. Evaluate YOLO11s-cls
    yolo_pt = RUNS_DIR / "yolo11s_cls" / "weights" / "best.pt"
    if yolo_pt.exists():
        print("\n>>> [1/7] Evaluating YOLO11s-cls (CSPDarknet Backbone)...")
        in_t1, in_t5, lat = evaluate_yolo(yolo_pt, IN_DOMAIN_DIR, device)
        ood_t1, ood_t5, _ = evaluate_yolo(yolo_pt, OOD_DIR, device)
        results.append({
            "model": "YOLO11s-cls",
            "type": "Edge Real-Time CNN",
            "in_top1": in_t1,
            "in_top5": in_t5,
            "ood_top1": ood_t1,
            "ood_top5": ood_t5,
            "delta": in_t1 - ood_t1,
            "latency_ms": lat
        })
        print(f"    In-Domain: {in_t1:.2f}% | Real-World OOD: {ood_t1:.2f}% | Robustness Drop: -{in_t1 - ood_t1:.2f}% | Latency: {lat:.1f}ms")

    # 2. Evaluate ResNet50d
    resnet_pth = RUNS_DIR / "resnet50d" / "best.pth"
    if resnet_pth.exists():
        print("\n>>> [2/7] Evaluating ResNet50d (Classic Deep Residual CNN)...")
        ckpt = torch.load(resnet_pth, map_location=device, weights_only=False)
        model = timm.create_model("resnet50d", pretrained=False, num_classes=10)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)

        in_t1, in_t5, lat = evaluate_pytorch_model(model, in_loader, device)
        ood_t1, ood_t5, _ = evaluate_pytorch_model(model, ood_loader, device)
        results.append({
            "model": "ResNet50d",
            "type": "Classic Residual CNN",
            "in_top1": in_t1,
            "in_top5": in_t5,
            "ood_top1": ood_t1,
            "ood_top5": ood_t5,
            "delta": in_t1 - ood_t1,
            "latency_ms": lat
        })
        print(f"    In-Domain: {in_t1:.2f}% | Real-World OOD: {ood_t1:.2f}% | Robustness Drop: -{in_t1 - ood_t1:.2f}% | Latency: {lat:.1f}ms")
        del model
        torch.cuda.empty_cache()

    # 3. Evaluate MobileNetV4-Small
    mb_pth = RUNS_DIR / "mobilenetv4_small" / "best.pth"
    if mb_pth.exists():
        print("\n>>> [3/7] Evaluating MobileNetV4-Small (Inverted Residual CNN)...")
        ckpt = torch.load(mb_pth, map_location=device, weights_only=False)
        model = timm.create_model("mobilenetv4_conv_small", pretrained=False, num_classes=10)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)

        in_t1, in_t5, lat = evaluate_pytorch_model(model, in_loader, device)
        ood_t1, ood_t5, _ = evaluate_pytorch_model(model, ood_loader, device)
        results.append({
            "model": "MobileNetV4-Small",
            "type": "Edge Optimized CNN",
            "in_top1": in_t1,
            "in_top5": in_t5,
            "ood_top1": ood_t1,
            "ood_top5": ood_t5,
            "delta": in_t1 - ood_t1,
            "latency_ms": lat
        })
        print(f"    In-Domain: {in_t1:.2f}% | Real-World OOD: {ood_t1:.2f}% | Robustness Drop: -{in_t1 - ood_t1:.2f}% | Latency: {lat:.1f}ms")
        del model
        torch.cuda.empty_cache()

    # 4. Evaluate ConvNeXt-Tiny
    conv_pth = RUNS_DIR / "convnext_tiny" / "best.pth"
    if conv_pth.exists():
        print("\n>>> [4/7] Evaluating ConvNeXt-Tiny (Modern 7x7 ConvNet)...")
        ckpt = torch.load(conv_pth, map_location=device, weights_only=False)
        model = timm.create_model("convnext_tiny", pretrained=False, num_classes=10)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)

        in_t1, in_t5, lat = evaluate_pytorch_model(model, in_loader, device)
        ood_t1, ood_t5, _ = evaluate_pytorch_model(model, ood_loader, device)
        results.append({
            "model": "ConvNeXt-Tiny",
            "type": "Large-Kernel ConvNet",
            "in_top1": in_t1,
            "in_top5": in_t5,
            "ood_top1": ood_t1,
            "ood_top5": ood_t5,
            "delta": in_t1 - ood_t1,
            "latency_ms": lat
        })
        print(f"    In-Domain: {in_t1:.2f}% | Real-World OOD: {ood_t1:.2f}% | Robustness Drop: -{in_t1 - ood_t1:.2f}% | Latency: {lat:.1f}ms")
        del model
        torch.cuda.empty_cache()

    # 5. Evaluate Swin-Tiny
    swin_pth = RUNS_DIR / "swin_tiny" / "best.pth"
    if swin_pth.exists():
        print("\n>>> [5/7] Evaluating Swin-Tiny (Shifted-Window Vision Transformer)...")
        ckpt = torch.load(swin_pth, map_location=device, weights_only=False)
        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=10)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)

        in_t1, in_t5, lat = evaluate_pytorch_model(model, in_loader, device)
        ood_t1, ood_t5, _ = evaluate_pytorch_model(model, ood_loader, device)
        results.append({
            "model": "Swin-Tiny",
            "type": "Hierarchical ViT",
            "in_top1": in_t1,
            "in_top5": in_t5,
            "ood_top1": ood_t1,
            "ood_top5": ood_t5,
            "delta": in_t1 - ood_t1,
            "latency_ms": lat
        })
        print(f"    In-Domain: {in_t1:.2f}% | Real-World OOD: {ood_t1:.2f}% | Robustness Drop: -{in_t1 - ood_t1:.2f}% | Latency: {lat:.1f}ms")
        del model
        torch.cuda.empty_cache()

    # 6. Evaluate DINOv2-Small + LoRA
    dino_pth = RUNS_DIR / "dinov2_lora" / "best.pth"
    if dino_pth.exists():
        print("\n>>> [6/7] Evaluating DINOv2-Small + LoRA (Self-Supervised VFM)...")
        from scripts.train_dinov2_lora import DINOv2WithLoRAClassifier
        ckpt = torch.load(dino_pth, map_location=device, weights_only=False)
        rank = ckpt.get("rank", 8)
        alpha = ckpt.get("alpha", 16.0)
        model = DINOv2WithLoRAClassifier(num_classes=10, rank=rank, alpha=alpha)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)

        in_t1, in_t5, lat = evaluate_pytorch_model(model, in_loader, device)
        ood_t1, ood_t5, _ = evaluate_pytorch_model(model, ood_loader, device)
        results.append({
            "model": "DINOv2-Small + LoRA",
            "type": "Vision Foundation Model (PEFT)",
            "in_top1": in_t1,
            "in_top5": in_t5,
            "ood_top1": ood_t1,
            "ood_top5": ood_t5,
            "delta": in_t1 - ood_t1,
            "latency_ms": lat
        })
        print(f"    In-Domain: {in_t1:.2f}% | Real-World OOD: {ood_t1:.2f}% | Robustness Drop: -{in_t1 - ood_t1:.2f}% | Latency: {lat:.1f}ms")
        del model
        torch.cuda.empty_cache()

    # 7. Evaluate BioCLIP (ViT-B/16) + LoRA
    bioclip_pth = RUNS_DIR / "bioclip_lora" / "best.pth"
    if bioclip_pth.exists():
        print("\n>>> [7/7] Evaluating BioCLIP (ViT-B/16) + LoRA (Biological VFM)...")
        try:
            import open_clip
            from scripts.train_bioclip_lora import BioCLIPWithLoRA
            ckpt = torch.load(bioclip_pth, map_location=device, weights_only=False)
            clip_base, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
            model = BioCLIPWithLoRA(clip_base, num_classes=10, rank=ckpt.get("rank", 8), alpha=ckpt.get("alpha", 16.0))
            model.load_state_dict(ckpt["state_dict"])
            model.to(device)

            in_t1, in_t5, lat = evaluate_pytorch_model(model, in_loader_clip, device)
            ood_t1, ood_t5, _ = evaluate_pytorch_model(model, ood_loader_clip, device)
            results.append({
                "model": "BioCLIP (ViT-B) + LoRA",
                "type": "Biological VFM (PEFT)",
                "in_top1": in_t1,
                "in_top5": in_t5,
                "ood_top1": ood_t1,
                "ood_top5": ood_t5,
                "delta": in_t1 - ood_t1,
                "latency_ms": lat
            })
            print(f"    In-Domain: {in_t1:.2f}% | Real-World OOD: {ood_t1:.2f}% | Robustness Drop: -{in_t1 - ood_t1:.2f}% | Latency: {lat:.1f}ms")
            del model, clip_base
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    [Notice] BioCLIP eval skipped or encountered error: {e}")

    # Output Leaderboard Table
    print("\n\n" + "=" * 105)
    print(" " * 28 + "DUAL-BENCHMARK SCIENTIFIC COMPARISON LEADERBOARD")
    print("=" * 105)
    print(f" {'Architecture':<24} | {'Paradigm':<22} | {'Controlled Acc':<15} | {'Real-World OOD':<15} | {'Robustness Drop':<16} | {'GPU Latency'}")
    print("-" * 105)
    for r in results:
        print(f" {r['model']:<24} | {r['type']:<22} | {r['in_top1']:6.2f}% (T5:{r['in_top5']:5.1f}%) | {r['ood_top1']:6.2f}% (T5:{r['ood_top5']:5.1f}%) | -{r['delta']:5.2f}%{'':<9} | {r['latency_ms']:5.1f} ms")
    print("=" * 105)

    # Save JSON and Markdown Report
    json_path = REPORT_DIR / "dual_benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_path = REPORT_DIR / "dual_benchmark_leaderboard.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Dual-Benchmark Scientific Leaderboard: Controlled vs. Real-World Red Sea\n\n")
        f.write("| Architecture | Paradigm | Controlled (Fish-Vista) | Real-World (Wild Red Sea) | Robustness Drop (Δ) | GPU Latency |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
        for r in results:
            f.write(f"| **{r['model']}** | {r['type']} | **{r['in_top1']:.2f}%** | **{r['ood_top1']:.2f}%** | -{r['delta']:.2f}% | {r['latency_ms']:.1f} ms |\n")
        f.write(f"\n*Generated automatically from {len(results)} evaluated models.*\n")

    print(f"\nReports saved successfully to:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown Table: {md_path}\n")

if __name__ == "__main__":
    main()
