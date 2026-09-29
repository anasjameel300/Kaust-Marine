"""
Multi-Model Taxonomic Fish Classification Engine
KAUST Marine Vision - Research Grade Deep Learning Platform

Integrates 4 distinct architectures into an ensemble voting system:
1. YOLO11s-cls (CSPDarknet - Local fast texture extractor)
2. ConvNeXt-Tiny (Modern Pure ConvNet with 7x7 Depthwise Kernels)
3. MobileNetV4-Small (Universal Efficient Inverted Residual CNN)
4. Swin-Tiny (Hierarchical Vision Transformer with Shifted Window Attention)

Provides:
- Softmax probability fusion (Soft-Voting)
- Epistemic uncertainty & agreement metrics (Unanimous, Majority, Disagreement)
- Metadata & FishBase allometric lookup for downstream biomass estimation
"""

import os
import cv2
import json
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import timm
from torchvision import transforms
from ultralytics import YOLO
from app.dual_crop_engine import dual_crop_extractor, to_base64_jpeg

# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
RUNS_DIR = PROJECT_ROOT / "runs" / "classify_fish_vista"
if not RUNS_DIR.exists():
    RUNS_DIR = PROJECT_ROOT / "runs" / "classify"
METADATA_PATH = PROJECT_ROOT / "datasets" / "taxonomic_cls" / "taxonomy_metadata.json"

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Taxonomic Engine] Initializing inference on {device}...")

# Load taxonomy metadata
if METADATA_PATH.exists():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        TAXONOMY_METADATA = json.load(f)
else:
    TAXONOMY_METADATA = {}

# Canonical 10 Red Sea Marine Families (Alphabetical order as loaded by PyTorch ImageFolder)
CLASS_NAMES = [
    "Acanthuridae",
    "Balistidae",
    "Chaetodontidae",
    "Holocentridae",
    "Labridae",
    "Lutjanidae",
    "Nemipteridae",
    "Pomacentridae",
    "Scaridae",
    "Siganidae"
]

# Family metadata & FishBase allometric parameters (W = a * L^b)
FAMILY_PROFILES = {
    "Lutjanidae": {
        "common_name": "Snapper",
        "scientific_representative": "Lutjanus fulvus",
        "trophic_guild": "Commercial Predator / Catch",
        "fb_a": 0.0140,
        "fb_b": 3.02,
        "is_commercial": True
    },
    "Scaridae": {
        "common_name": "Parrotfish",
        "scientific_representative": "Scarus spp",
        "trophic_guild": "Bioeroding Herbivore / Energetics",
        "fb_a": 0.0182,
        "fb_b": 3.01,
        "is_commercial": True
    },
    "Balistidae": {
        "common_name": "Triggerfish",
        "scientific_representative": "Balistapus undulatus",
        "trophic_guild": "Hard-shelled Invertivore",
        "fb_a": 0.0275,
        "fb_b": 2.95,
        "is_commercial": True
    },
    "Siganidae": {
        "common_name": "Rabbitfish / Spinefoot",
        "scientific_representative": "Siganus fuscescens",
        "trophic_guild": "Commercial Herbivore / Food Fishery",
        "fb_a": 0.0135,
        "fb_b": 2.98,
        "is_commercial": True
    },
    "Acanthuridae": {
        "common_name": "Surgeonfish & Tangs",
        "scientific_representative": "Acanthurus nigrofuscus",
        "trophic_guild": "Algal Turf Grazer",
        "fb_a": 0.0210,
        "fb_b": 3.00,
        "is_commercial": False
    },
    "Chaetodontidae": {
        "common_name": "Butterflyfish",
        "scientific_representative": "Chaetodon lunulatus",
        "trophic_guild": "Obligate Corallivore (Indicator)",
        "fb_a": 0.0234,
        "fb_b": 3.05,
        "is_commercial": False
    },
    "Holocentridae": {
        "common_name": "Squirrelfish & Soldierfish",
        "scientific_representative": "Myripristis kuntee",
        "trophic_guild": "Nocturnal Reef Carnivore",
        "fb_a": 0.0195,
        "fb_b": 3.03,
        "is_commercial": False
    },
    "Labridae": {
        "common_name": "Wrasse",
        "scientific_representative": "Hemigymnus fasciatus",
        "trophic_guild": "Invertebrate Carnivore",
        "fb_a": 0.0152,
        "fb_b": 3.04,
        "is_commercial": False
    },
    "Nemipteridae": {
        "common_name": "Monocle Bream",
        "scientific_representative": "Scolopsis bilineata",
        "trophic_guild": "Benthic Sand Forager",
        "fb_a": 0.0165,
        "fb_b": 2.99,
        "is_commercial": True
    },
    "Pomacentridae": {
        "common_name": "Damselfish / Clownfish",
        "scientific_representative": "Dascyllus reticulatus",
        "trophic_guild": "Planktivore & Anemone Mutualist",
        "fb_a": 0.0225,
        "fb_b": 3.08,
        "is_commercial": False
    }
}

def letterbox_pad(img_bgr: np.ndarray, target_size: int = 224, pad_color: Tuple[int, int, int] = (114, 114, 114)) -> np.ndarray:
    """
    Resizes image maintaining original aspect ratio and pads remaining borders
    with neutral gray (114, 114, 114) to reach exact target_size x target_size.
    Eliminates artificial stretching or squishing of slender vs deep-bodied fish.
    """
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_size, target_size, 3), pad_color[0], dtype=np.uint8)
    scale = min(target_size / h, target_size / w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_top = (target_size - nh) // 2
    pad_bottom = target_size - nh - pad_top
    pad_left = (target_size - nw) // 2
    pad_right = target_size - nw - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=pad_color
    )
    return padded

# Image transforms for PyTorch models (ImageNet normalization)
transform_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transforms for CLIP / BioCLIP
transform_clip = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

def apply_adaptive_roi_gating(p_glob: np.ndarray, p_head: Optional[np.ndarray] = None, p_caudal: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Adaptive Anchor-Preserving Gating (Option A):
    - The global full-body distribution (p_glob) acts as the primary anchor.
    - Sub-region anatomical crops (head, caudal) only reinforce the prediction
      if they agree with the primary anchor.
    - If a sub-region is noisy or confused (e.g. an isolated tail fin), it is gated out
      so it NEVER penalizes or dilutes the confident global organism prediction.
    """
    if p_head is None or p_caudal is None:
        return p_glob.copy()

    p_fused = p_glob.copy()
    top_idx = int(np.argmax(p_glob))

    # 1. Cephalic Reinforcement (Anterior snout/jaw keypoints)
    head_top_idx = int(np.argmax(p_head))
    if head_top_idx == top_idx:
        head_conf = float(p_head[head_top_idx])
        p_fused[top_idx] = min(0.99, p_fused[top_idx] + 0.08 * head_conf)

    # 2. Caudal Reinforcement (Posterior peduncle/spine keypoints)
    caudal_top_idx = int(np.argmax(p_caudal))
    if caudal_top_idx == top_idx:
        caudal_conf = float(p_caudal[caudal_top_idx])
        p_fused[top_idx] = min(0.99, p_fused[top_idx] + 0.06 * caudal_conf)

    # Re-normalize probability distribution
    sum_val = float(np.sum(p_fused))
    if sum_val > 0:
        p_fused = p_fused / sum_val
    return p_fused

class MultiModelTaxonomicEngine:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.load_models()

    def load_models(self):
        # 1. BioCLIP + LoRA (In-Situ Wild Champion)
        bioclip_path = RUNS_DIR / "bioclip_lora" / "best.pth"
        if bioclip_path.exists():
            try:
                import sys
                if str(PROJECT_ROOT) not in sys.path:
                    sys.path.insert(0, str(PROJECT_ROOT))
                import open_clip
                from scripts.train_bioclip_lora import BioCLIPWithLoRA
                print(f"[Taxonomic Engine] Loading BioCLIP + LoRA ({bioclip_path.name})...")
                ck = torch.load(str(bioclip_path), map_location=device, weights_only=False)
                clip_base, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
                m = BioCLIPWithLoRA(clip_base, num_classes=len(CLASS_NAMES), rank=ck.get("rank", 8), alpha=ck.get("alpha", 16.0))
                m.load_state_dict(ck["state_dict"])
                m.to(device).eval()
                self.models["bioclip"] = m
            except Exception as e:
                print(f"[Taxonomic Engine] Warning: Failed to load BioCLIP LoRA: {e}")

        # 2. ResNet50d (Landed Catch Champion)
        resnet_path = RUNS_DIR / "resnet50d" / "best.pth"
        if resnet_path.exists():
            try:
                print(f"[Taxonomic Engine] Loading ResNet50d ({resnet_path.name})...")
                ck = torch.load(str(resnet_path), map_location=device, weights_only=False)
                m = timm.create_model("resnet50d", pretrained=False, num_classes=len(CLASS_NAMES))
                m.load_state_dict(ck["state_dict"])
                m.to(device).eval()
                self.models["resnet50"] = m
            except Exception as e:
                print(f"[Taxonomic Engine] Warning: Failed to load ResNet50: {e}")

        # 3. DINOv2-Small + LoRA (VFM Champion)
        dino_path = RUNS_DIR / "dinov2_lora" / "best.pth"
        if dino_path.exists():
            try:
                import sys
                if str(PROJECT_ROOT) not in sys.path:
                    sys.path.insert(0, str(PROJECT_ROOT))
                from scripts.train_dinov2_lora import DINOv2WithLoRAClassifier
                print(f"[Taxonomic Engine] Loading DINOv2-Small + LoRA ({dino_path.name})...")
                ck = torch.load(str(dino_path), map_location=device, weights_only=False)
                m = DINOv2WithLoRAClassifier(num_classes=len(CLASS_NAMES), rank=ck.get("rank", 8), alpha=ck.get("alpha", 16.0))
                m.load_state_dict(ck["state_dict"])
                m.to(device).eval()
                self.models["dinov2"] = m
            except Exception as e:
                print(f"[Taxonomic Engine] Warning: Failed to load DINOv2 LoRA: {e}")

        # 4. YOLO11s-cls
        yolo_path = RUNS_DIR / "yolo11s_cls" / "weights" / "best.pt"
        if not yolo_path.exists():
            yolo_path = PROJECT_ROOT / "runs" / "classify" / "kaust_taxonomic_cls_v1" / "weights" / "best.pt"
        if yolo_path.exists():
            print(f"[Taxonomic Engine] Loading YOLO11s-cls ({yolo_path.name})...")
            self.models["yolo"] = YOLO(str(yolo_path))

        # 5. Swin-Tiny Transformer
        swin_path = RUNS_DIR / "swin_tiny" / "best.pth"
        if swin_path.exists():
            try:
                print(f"[Taxonomic Engine] Loading Swin-Tiny Transformer ({swin_path.name})...")
                ck = torch.load(str(swin_path), map_location=device, weights_only=False)
                m = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=len(CLASS_NAMES))
                m.load_state_dict(ck["state_dict"])
                m.to(device).eval()
                self.models["swin"] = m
            except Exception as e:
                print(f"[Taxonomic Engine] Warning: Failed to load Swin: {e}")

        # 6. MobileNetV4-Small
        mob_path = RUNS_DIR / "mobilenetv4_small" / "best.pth"
        if mob_path.exists():
            try:
                ck = torch.load(str(mob_path), map_location=device, weights_only=False)
                m = timm.create_model("mobilenetv4_conv_small", pretrained=False, num_classes=len(CLASS_NAMES))
                m.load_state_dict(ck["state_dict"])
                m.to(device).eval()
                self.models["mobilenetv4"] = m
            except Exception:
                pass

        # 7. ConvNeXt-Tiny
        conv_path = RUNS_DIR / "convnext_tiny" / "best.pth"
        if conv_path.exists():
            try:
                ck = torch.load(str(conv_path), map_location=device, weights_only=False)
                m = timm.create_model("convnext_tiny", pretrained=False, num_classes=len(CLASS_NAMES))
                m.load_state_dict(ck["state_dict"])
                m.to(device).eval()
                self.models["convnext"] = m
            except Exception:
                pass

        print(f"[Taxonomic Engine] Active Classifier Backbones: {list(self.models.keys())}")

    def classify_crop(self, crop_bgr: np.ndarray, mode: str = "bioclip",
                      full_img_bgr: Optional[np.ndarray] = None, box: Optional[list] = None,
                      enable_dual_crop: bool = True) -> Dict[str, Any]:
        """
        Classifies a single cropped fish instance using the chosen mode:
        mode in ['bioclip', 'resnet50', 'dinov2', 'ensemble', 'swin', 'yolo', 'mobilenetv4', 'convnext']
        Supports Dual-Crop Multi-Region Feature Extraction (Fish-Vista CVPR 2025):
        Global letterbox (55%) + Cephalic ROI (25%) + Caudal Peduncle ROI (20%).
        """
        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 5:
            return self._format_result("Pomacentridae", 0.50, {}, "Fallback")

        # 1. Multi-Region Anatomical Crop Extraction
        if full_img_bgr is not None and box is not None:
            regions = dual_crop_extractor.extract_regions(full_img_bgr, box)
        else:
            regions = dual_crop_extractor.extract_from_crop(crop_bgr)

        global_bgr = regions["global"]
        head_bgr = regions["head"]
        caudal_bgr = regions["caudal"]

        global_rgb = cv2.cvtColor(global_bgr, cv2.COLOR_BGR2RGB)
        tensor_img = transform_pipeline(global_rgb).unsqueeze(0).to(device)
        tensor_clip = transform_clip(global_rgb).unsqueeze(0).to(device)

        if enable_dual_crop:
            head_rgb = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2RGB)
            caudal_rgb = cv2.cvtColor(caudal_bgr, cv2.COLOR_BGR2RGB)
            tensor_head_img = transform_pipeline(head_rgb).unsqueeze(0).to(device)
            tensor_head_clip = transform_clip(head_rgb).unsqueeze(0).to(device)
            tensor_caudal_img = transform_pipeline(caudal_rgb).unsqueeze(0).to(device)
            tensor_caudal_clip = transform_clip(caudal_rgb).unsqueeze(0).to(device)

        model_probs: Dict[str, np.ndarray] = {}
        model_predictions: Dict[str, str] = {}
        model_confidences: Dict[str, float] = {}
        head_trait_preds: Dict[str, str] = {}
        caudal_trait_preds: Dict[str, str] = {}

        # 1. BioCLIP + LoRA (In-Situ Wild Champion)
        if "bioclip" in self.models and (mode in ["ensemble", "bioclip"]):
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    if enable_dual_crop:
                        batch_clip = torch.cat([tensor_clip, tensor_head_clip, tensor_caudal_clip], dim=0)
                        logits = self.models["bioclip"](batch_clip)
                        probs = F.softmax(logits, dim=1).cpu().numpy()
                        p_glob, p_head, p_caudal = probs[0], probs[1], probs[2]
                        p_fused = apply_adaptive_roi_gating(p_glob, p_head, p_caudal)
                        head_trait_preds["BioCLIP"] = CLASS_NAMES[int(np.argmax(p_head))]
                        caudal_trait_preds["BioCLIP"] = CLASS_NAMES[int(np.argmax(p_caudal))]
                    else:
                        logits = self.models["bioclip"](tensor_clip)
                        p_fused = F.softmax(logits, dim=1).cpu().numpy()[0]

                    model_probs["BioCLIP-LoRA"] = p_fused
                    top_idx = int(np.argmax(p_fused))
                    model_predictions["BioCLIP-LoRA"] = CLASS_NAMES[top_idx]
                    model_confidences["BioCLIP-LoRA"] = round(float(p_fused[top_idx]), 3)

        # 2. ResNet50d (Landed Catch Champion)
        if "resnet50" in self.models and (mode in ["ensemble", "resnet50"]):
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    if enable_dual_crop:
                        batch_img = torch.cat([tensor_img, tensor_head_img, tensor_caudal_img], dim=0)
                        logits = self.models["resnet50"](batch_img)
                        probs = F.softmax(logits, dim=1).cpu().numpy()
                        p_glob, p_head, p_caudal = probs[0], probs[1], probs[2]
                        p_fused = apply_adaptive_roi_gating(p_glob, p_head, p_caudal)
                        head_trait_preds["ResNet50d"] = CLASS_NAMES[int(np.argmax(p_head))]
                        caudal_trait_preds["ResNet50d"] = CLASS_NAMES[int(np.argmax(p_caudal))]
                    else:
                        logits = self.models["resnet50"](tensor_img)
                        p_fused = F.softmax(logits, dim=1).cpu().numpy()[0]

                    model_probs["ResNet50d"] = p_fused
                    top_idx = int(np.argmax(p_fused))
                    model_predictions["ResNet50d"] = CLASS_NAMES[top_idx]
                    model_confidences["ResNet50d"] = round(float(p_fused[top_idx]), 3)

        # 3. DINOv2-Small + LoRA
        if "dinov2" in self.models and (mode in ["ensemble", "dinov2"]):
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    if enable_dual_crop:
                        batch_img = torch.cat([tensor_img, tensor_head_img, tensor_caudal_img], dim=0)
                        logits = self.models["dinov2"](batch_img)
                        probs = F.softmax(logits, dim=1).cpu().numpy()
                        p_glob, p_head, p_caudal = probs[0], probs[1], probs[2]
                        p_fused = apply_adaptive_roi_gating(p_glob, p_head, p_caudal)
                        head_trait_preds["DINOv2"] = CLASS_NAMES[int(np.argmax(p_head))]
                        caudal_trait_preds["DINOv2"] = CLASS_NAMES[int(np.argmax(p_caudal))]
                    else:
                        logits = self.models["dinov2"](tensor_img)
                        p_fused = F.softmax(logits, dim=1).cpu().numpy()[0]

                    model_probs["DINOv2-LoRA"] = p_fused
                    top_idx = int(np.argmax(p_fused))
                    model_predictions["DINOv2-LoRA"] = CLASS_NAMES[top_idx]
                    model_confidences["DINOv2-LoRA"] = round(float(p_fused[top_idx]), 3)

        # 4. Swin Transformer
        if "swin" in self.models and (mode in ["ensemble", "swin"]):
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    if enable_dual_crop:
                        batch_img = torch.cat([tensor_img, tensor_head_img, tensor_caudal_img], dim=0)
                        logits = self.models["swin"](batch_img)
                        probs = F.softmax(logits, dim=1).cpu().numpy()
                        p_glob, p_head, p_caudal = probs[0], probs[1], probs[2]
                        p_fused = apply_adaptive_roi_gating(p_glob, p_head, p_caudal)
                    else:
                        logits = self.models["swin"](tensor_img)
                        p_fused = F.softmax(logits, dim=1).cpu().numpy()[0]

                    model_probs["Swin-Tiny"] = p_fused
                    top_idx = int(np.argmax(p_fused))
                    model_predictions["Swin-Tiny"] = CLASS_NAMES[top_idx]
                    model_confidences["Swin-Tiny"] = round(float(p_fused[top_idx]), 3)

        # 5. YOLO11s-cls
        if "yolo" in self.models and (mode in ["ensemble", "yolo"]):
            if enable_dual_crop:
                res_glob = self.models["yolo"].predict(global_rgb, verbose=False, device=0 if torch.cuda.is_available() else "cpu")[0]
                p_glob = np.zeros(len(CLASS_NAMES))
                if res_glob.probs is not None:
                    pd = res_glob.probs.data.cpu().numpy()
                    yn = res_glob.names
                    yd = {yn[i]: float(pd[i]) for i in range(len(pd))}
                    p_glob = np.array([yd.get(c, 0.0) for c in CLASS_NAMES])
                    if p_glob.sum() > 0: p_glob = p_glob / p_glob.sum()

                res_head = self.models["yolo"].predict(head_rgb, verbose=False, device=0 if torch.cuda.is_available() else "cpu")[0]
                p_head = np.zeros(len(CLASS_NAMES))
                if res_head.probs is not None:
                    pd = res_head.probs.data.cpu().numpy()
                    yn = res_head.names
                    yd = {yn[i]: float(pd[i]) for i in range(len(pd))}
                    p_head = np.array([yd.get(c, 0.0) for c in CLASS_NAMES])
                    if p_head.sum() > 0: p_head = p_head / p_head.sum()

                res_caudal = self.models["yolo"].predict(caudal_rgb, verbose=False, device=0 if torch.cuda.is_available() else "cpu")[0]
                p_caudal = np.zeros(len(CLASS_NAMES))
                if res_caudal.probs is not None:
                    pd = res_caudal.probs.data.cpu().numpy()
                    yn = res_caudal.names
                    yd = {yn[i]: float(pd[i]) for i in range(len(pd))}
                    p_caudal = np.array([yd.get(c, 0.0) for c in CLASS_NAMES])
                    if p_caudal.sum() > 0: p_caudal = p_caudal / p_caudal.sum()

                p_fused = apply_adaptive_roi_gating(p_glob, p_head, p_caudal)
            else:
                res_glob = self.models["yolo"].predict(global_rgb, verbose=False, device=0 if torch.cuda.is_available() else "cpu")[0]
                p_fused = np.zeros(len(CLASS_NAMES))
                if res_glob.probs is not None:
                    pd = res_glob.probs.data.cpu().numpy()
                    yn = res_glob.names
                    yd = {yn[i]: float(pd[i]) for i in range(len(pd))}
                    p_fused = np.array([yd.get(c, 0.0) for c in CLASS_NAMES])
                    if p_fused.sum() > 0: p_fused = p_fused / p_fused.sum()

            model_probs["YOLO11s-cls"] = p_fused
            top_idx = int(np.argmax(p_fused))
            model_predictions["YOLO11s-cls"] = CLASS_NAMES[top_idx]
            model_confidences["YOLO11s-cls"] = round(float(p_fused[top_idx]), 3)

        # 6. MobileNetV4
        if "mobilenetv4" in self.models and (mode == "mobilenetv4"):
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logits = self.models["mobilenetv4"](tensor_img)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    model_probs["MobileNetV4"] = probs
                    top_idx = int(np.argmax(probs))
                    model_predictions["MobileNetV4"] = CLASS_NAMES[top_idx]
                    model_confidences["MobileNetV4"] = round(float(probs[top_idx]), 3)

        if not model_probs:
            return self._format_result("Pomacentridae", 0.50, {}, "Fallback")

        # Decision Strategy: Scientific Ensemble Soft-Voting or Single Model
        if mode == "ensemble":
            MODEL_WEIGHTS = {
                "BioCLIP-LoRA": 2.5,
                "DINOv2-LoRA": 1.5,
                "ResNet50d": 1.0,
                "Swin-Tiny": 1.0,
                "YOLO11s-cls": 1.0,
                "MobileNetV4": 0.5,
                "ConvNeXt": 0.5
            }
            total_weight = sum(MODEL_WEIGHTS.get(k, 1.0) for k in model_probs.keys())
            weighted_sum = sum(model_probs[k] * MODEL_WEIGHTS.get(k, 1.0) for k in model_probs.keys())
            mean_probs = weighted_sum / max(1e-5, total_weight)
            final_idx = int(np.argmax(mean_probs))
            predicted_family = CLASS_NAMES[final_idx]
            final_conf = round(float(mean_probs[final_idx]), 3)

            # Consensus & Agreement Scoring
            votes = list(model_predictions.values())
            vote_count = votes.count(predicted_family)
            total_voters = len(votes)

            if vote_count == total_voters:
                agreement_status = f"Unanimous ({vote_count}/{total_voters} Agree)"
                consensus_level = "high"
            elif vote_count >= (total_voters / 2):
                agreement_status = f"Majority ({vote_count}/{total_voters} Agree)"
                consensus_level = "moderate"
            else:
                agreement_status = f"Uncertain / Split Vote ({vote_count}/{total_voters})"
                consensus_level = "low"
        else:
            model_key_name = list(model_probs.keys())[0]
            probs = model_probs[model_key_name]
            final_idx = int(np.argmax(probs))
            predicted_family = CLASS_NAMES[final_idx]
            final_conf = round(float(probs[final_idx]), 3)
            agreement_status = f"Single Model ({model_key_name})"
            consensus_level = "high"

        # Multi-Region ROI payload
        roi_info = None
        if enable_dual_crop:
            head_fam = list(head_trait_preds.values())[0] if head_trait_preds else predicted_family
            caudal_fam = list(caudal_trait_preds.values())[0] if caudal_trait_preds else predicted_family
            head_agrees = bool(head_fam == predicted_family)
            caudal_agrees = bool(caudal_fam == predicted_family)

            if head_agrees and caudal_agrees:
                consensus_badge = "Anatomical Consensus"
                consensus_color = "#34d399"
            elif head_agrees:
                consensus_badge = "Cephalic Validated"
                consensus_color = "#38bdf8"
            elif caudal_agrees:
                consensus_badge = "Caudal Validated"
                consensus_color = "#38bdf8"
            else:
                consensus_badge = "Macro-Anchored"
                consensus_color = "#94a3b8"

            roi_info = {
                "head_thumb": to_base64_jpeg(head_bgr, quality=80, max_dim=120),
                "caudal_thumb": to_base64_jpeg(caudal_bgr, quality=80, max_dim=120),
                "cephalic_family": head_fam,
                "caudal_family": caudal_fam,
                "consensus_badge": consensus_badge,
                "consensus_color": consensus_color,
                "anatomical_match": bool(head_agrees or caudal_agrees),
                "has_dual_crop": True
            }

        return self._format_result(
            predicted_family=predicted_family,
            confidence=final_conf,
            model_votes=model_predictions,
            agreement_status=agreement_status,
            consensus_level=consensus_level,
            individual_confs=model_confidences,
            rois=roi_info
        )

    def _format_result(self, predicted_family: str, confidence: float, model_votes: Dict[str, str],
                       agreement_status: str, consensus_level: str = "high", individual_confs: Dict[str, float] = None,
                       rois: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        profile = FAMILY_PROFILES.get(predicted_family, {
            "common_name": "Reef Fish",
            "scientific_representative": "Teleostei",
            "trophic_guild": "Marine Organism",
            "fb_a": 0.0200,
            "fb_b": 3.00,
            "is_commercial": False
        })

        # Out-Of-Distribution (OOD) & Uncertainty Filter:
        # If confidence is below 0.65 or consensus is low (split vote), flag as unverified
        is_verified = bool(confidence >= 0.65 and consensus_level in ["high", "moderate"])

        return {
            "family": predicted_family,
            "common_name": profile["common_name"],
            "scientific_name": profile["scientific_representative"],
            "trophic_guild": profile["trophic_guild"],
            "is_commercial": profile["is_commercial"],
            "confidence": confidence,
            "is_verified": is_verified,
            "agreement_status": agreement_status,
            "consensus_level": consensus_level,
            "model_votes": model_votes,
            "individual_confidences": individual_confs or {},
            "fb_a": profile["fb_a"],
            "fb_b": profile["fb_b"],
            "rois": rois
        }

# Global singleton engine
taxonomic_engine = MultiModelTaxonomicEngine()
