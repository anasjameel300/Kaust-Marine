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
from typing import Dict, Any, List, Tuple
import timm
from torchvision import transforms
from ultralytics import YOLO

# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
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

# Image transforms for PyTorch models
transform_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MultiModelTaxonomicEngine:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.load_models()

    def load_models(self):
        # 1. YOLO11s-cls
        yolo_path = RUNS_DIR / "kaust_taxonomic_cls_v1" / "weights" / "best.pt"
        if yolo_path.exists():
            print(f"[Taxonomic Engine] Loading YOLO11s-cls ({yolo_path.name})...")
            self.models["yolo"] = YOLO(str(yolo_path))
        else:
            print("[Taxonomic Engine] Warning: YOLO11s-cls weights not found.")

        # 2. ConvNeXt-Tiny
        conv_path = RUNS_DIR / "convnext_tiny" / "best.pth"
        if conv_path.exists():
            print(f"[Taxonomic Engine] Loading ConvNeXt-Tiny ({conv_path.name})...")
            ck = torch.load(str(conv_path), map_location=device)
            m = timm.create_model("convnext_tiny", pretrained=False, num_classes=len(CLASS_NAMES))
            m.load_state_dict(ck["state_dict"])
            m.to(device).eval()
            self.models["convnext"] = m
        else:
            print("[Taxonomic Engine] Warning: ConvNeXt weights not found.")

        # 3. MobileNetV4-Small
        mob_path = RUNS_DIR / "mobilenetv4_small" / "best.pth"
        if mob_path.exists():
            print(f"[Taxonomic Engine] Loading MobileNetV4-Small ({mob_path.name})...")
            ck = torch.load(str(mob_path), map_location=device)
            m = timm.create_model("mobilenetv4_conv_small", pretrained=False, num_classes=len(CLASS_NAMES))
            m.load_state_dict(ck["state_dict"])
            m.to(device).eval()
            self.models["mobilenetv4"] = m
        else:
            print("[Taxonomic Engine] Warning: MobileNetV4 weights not found.")

        # 4. Swin-Tiny Transformer
        swin_path = RUNS_DIR / "swin_tiny" / "best.pth"
        if swin_path.exists():
            print(f"[Taxonomic Engine] Loading Swin-Tiny Transformer ({swin_path.name})...")
            ck = torch.load(str(swin_path), map_location=device)
            m = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=len(CLASS_NAMES))
            m.load_state_dict(ck["state_dict"])
            m.to(device).eval()
            self.models["swin"] = m
        else:
            print("[Taxonomic Engine] Warning: Swin Transformer weights not found.")

        print(f"[Taxonomic Engine] Active Classifier Backbones: {list(self.models.keys())}")

    def classify_crop(self, crop_bgr: np.ndarray, mode: str = "ensemble") -> Dict[str, Any]:
        """
        Classifies a single cropped fish instance using the chosen mode:
        mode in ['ensemble', 'swin', 'convnext', 'mobilenetv4', 'yolo']
        """
        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 5:
            # Fallback for degenerate crop
            return self._format_result("Pomacentridae", 0.50, {}, "Fallback")

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor_img = transform_pipeline(crop_rgb).unsqueeze(0).to(device)

        model_probs: Dict[str, np.ndarray] = {}
        model_predictions: Dict[str, str] = {}
        model_confidences: Dict[str, float] = {}

        # 1. ConvNeXt
        if "convnext" in self.models and (mode in ["ensemble", "convnext"]):
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logits = self.models["convnext"](tensor_img)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    model_probs["ConvNeXt-Tiny"] = probs
                    top_idx = int(np.argmax(probs))
                    model_predictions["ConvNeXt-Tiny"] = CLASS_NAMES[top_idx]
                    model_confidences["ConvNeXt-Tiny"] = round(float(probs[top_idx]), 3)

        # 2. MobileNetV4
        if "mobilenetv4" in self.models and (mode in ["ensemble", "mobilenetv4"]):
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logits = self.models["mobilenetv4"](tensor_img)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    model_probs["MobileNetV4"] = probs
                    top_idx = int(np.argmax(probs))
                    model_predictions["MobileNetV4"] = CLASS_NAMES[top_idx]
                    model_confidences["MobileNetV4"] = round(float(probs[top_idx]), 3)

        # 3. Swin Transformer
        if "swin" in self.models and (mode in ["ensemble", "swin"]):
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    logits = self.models["swin"](tensor_img)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    model_probs["Swin-Tiny"] = probs
                    top_idx = int(np.argmax(probs))
                    model_predictions["Swin-Tiny"] = CLASS_NAMES[top_idx]
                    model_confidences["Swin-Tiny"] = round(float(probs[top_idx]), 3)

        # 4. YOLO11s-cls
        if "yolo" in self.models and (mode in ["ensemble", "yolo"]):
            # Ultralytics accepts BGR / RGB array directly
            results = self.models["yolo"].predict(crop_rgb, verbose=False, device=0 if torch.cuda.is_available() else "cpu")[0]
            if results.probs is not None:
                probs_data = results.probs.data.cpu().numpy()
                # Re-map according to model names
                yolo_names = results.names
                yolo_dict = {yolo_names[i]: float(probs_data[i]) for i in range(len(probs_data))}
                probs = np.array([yolo_dict.get(c, 0.0) for c in CLASS_NAMES])
                # Normalize sum
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                model_probs["YOLO11s-cls"] = probs
                top_idx = int(np.argmax(probs))
                model_predictions["YOLO11s-cls"] = CLASS_NAMES[top_idx]
                model_confidences["YOLO11s-cls"] = round(float(probs[top_idx]), 3)

        if not model_probs:
            return self._format_result("Pomacentridae", 0.50, {}, "Fallback")

        # Decision Strategy: Ensemble Soft-Voting or Single Model
        if mode == "ensemble":
            # Average probability distributions
            prob_matrix = np.array(list(model_probs.values()))
            mean_probs = np.mean(prob_matrix, axis=0)
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
            # Single model output
            model_key_name = list(model_probs.keys())[0]
            probs = model_probs[model_key_name]
            final_idx = int(np.argmax(probs))
            predicted_family = CLASS_NAMES[final_idx]
            final_conf = round(float(probs[final_idx]), 3)
            agreement_status = f"Single Model ({model_key_name})"
            consensus_level = "high"

        return self._format_result(
            predicted_family=predicted_family,
            confidence=final_conf,
            model_votes=model_predictions,
            agreement_status=agreement_status,
            consensus_level=consensus_level,
            individual_confs=model_confidences
        )

    def _format_result(self, predicted_family: str, confidence: float, model_votes: Dict[str, str],
                       agreement_status: str, consensus_level: str = "high", individual_confs: Dict[str, float] = None) -> Dict[str, Any]:
        profile = FAMILY_PROFILES.get(predicted_family, {
            "common_name": "Reef Fish",
            "scientific_representative": "Teleostei",
            "trophic_guild": "Marine Organism",
            "fb_a": 0.0200,
            "fb_b": 3.00,
            "is_commercial": False
        })

        return {
            "family": predicted_family,
            "common_name": profile["common_name"],
            "scientific_name": profile["scientific_representative"],
            "trophic_guild": profile["trophic_guild"],
            "is_commercial": profile["is_commercial"],
            "confidence": confidence,
            "agreement_status": agreement_status,
            "consensus_level": consensus_level,
            "model_votes": model_votes,
            "individual_confidences": individual_confs or {},
            "fb_a": profile["fb_a"],
            "fb_b": profile["fb_b"]
        }

# Global singleton engine
taxonomic_engine = MultiModelTaxonomicEngine()
