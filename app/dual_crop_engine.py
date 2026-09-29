"""
Dual-Crop Multi-Region Feature Extraction Engine
KAUST Marine Vision - Research Grade Deep Learning Platform

Implements the multi-region trait extraction methodology formulated in the
Fish-Vista research paper (Mehrab et al., CVPR 2025).

Architecture:
- Stream A: Full-Body Letterbox Crop (Macro-morphology: L/D aspect ratio, general color).
- Stream B: Diagnostic Anatomical Micro-Crops:
    1. Cephalic ROI (Anterior 35%): Snout angle, eye diameter ratio, jaw structure.
    2. Caudal Peduncle ROI (Posterior 25%): Caudal spine (scalpel marker), tail geometry.
- Fusion: Combines global and local trait embeddings to resolve fine-grained taxonomic confusion.
"""

import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

def letterbox_pad(img_bgr: np.ndarray, target_size: int = 224, fill: int = 114) -> np.ndarray:
    """Proportionally scales and pads image to target_size x target_size with neutral gray."""
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_size, target_size, 3), fill, dtype=np.uint8)
    scale = min(target_size / h, target_size / w)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_top = (target_size - nh) // 2
    pad_bottom = target_size - nh - pad_top
    pad_left = (target_size - nw) // 2
    pad_right = target_size - nw - pad_left

    return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(fill, fill, fill))

class DualCropExtractor:
    """
    Extracts oriented multi-region anatomical crops from a segmented fish instance.
    Uses oriented bounding box geometry to isolate head and tail regions automatically.
    """
    def __init__(self, target_size: int = 224):
        self.target_size = target_size

    def extract_from_crop(self, raw_crop: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extracts global, cephalic (head), and caudal (tail) regions directly from a cropped fish instance.
        """
        if raw_crop is None or raw_crop.size == 0 or raw_crop.shape[0] < 5 or raw_crop.shape[1] < 5:
            empty = np.full((self.target_size, self.target_size, 3), 114, dtype=np.uint8)
            return {"global": empty, "head": empty, "caudal": empty}

        global_crop = letterbox_pad(raw_crop, target_size=self.target_size)
        ch, cw = raw_crop.shape[:2]

        if cw >= ch:
            # Horizontal aspect fish
            head_w = max(10, int(cw * 0.38))
            tail_w = max(10, int(cw * 0.30))
            roi_left = raw_crop[:, :head_w]
            roi_right = raw_crop[:, -tail_w:]
            head_crop = letterbox_pad(roi_left, target_size=self.target_size)
            caudal_crop = letterbox_pad(roi_right, target_size=self.target_size)
        else:
            # Vertical aspect fish
            head_h = max(10, int(ch * 0.38))
            tail_h = max(10, int(ch * 0.30))
            roi_top = raw_crop[:head_h, :]
            roi_bottom = raw_crop[-tail_h:, :]
            head_crop = letterbox_pad(roi_top, target_size=self.target_size)
            caudal_crop = letterbox_pad(roi_bottom, target_size=self.target_size)

        return {
            "global": global_crop,
            "head": head_crop,
            "caudal": caudal_crop
        }

    def extract_regions(self, full_img_bgr: np.ndarray, box: list, polygon: Optional[list] = None) -> Dict[str, np.ndarray]:
        """
        Extracts:
        - global_crop: Full fish with aspect-ratio preserving padding
        - head_crop: Cephalic region (anterior 35%)
        - caudal_crop: Caudal peduncle & tail region (posterior 25%)
        """
        h_img, w_img = full_img_bgr.shape[:2]
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(w_img, int(box[2]))
        y2 = min(h_img, int(box[3]))

        raw_crop = full_img_bgr[y1:y2, x1:x2]
        return self.extract_from_crop(raw_crop)

def to_base64_jpeg(img_bgr: np.ndarray, quality: int = 80, max_dim: int = 120) -> str:
    """Encodes a BGR numpy image into a compact base64 JPEG data URL for frontend rendering."""
    if img_bgr is None or img_bgr.size == 0:
        return ""
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    success, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return ""
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

class DualCropFusionClassifier(nn.Module):
    """
    Gated Multi-Region Trait Fusion Head.
    Combines:
    - Global feature vector z_global (body shape, overall color)
    - Cephalic feature vector z_head (snout, eye, mouth)
    - Caudal feature vector z_caudal (scalpel spine, fin morphology)
    """
    def __init__(self, embed_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.embed_dim = embed_dim
        # Fusion projection: [global || head || caudal] -> 3 * embed_dim -> embed_dim -> num_classes
        self.fusion_gate = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, z_global: torch.Tensor, z_head: torch.Tensor, z_caudal: torch.Tensor) -> torch.Tensor:
        # Concatenate multi-region trait vectors
        fused = torch.cat([z_global, z_head, z_caudal], dim=-1)
        logits = self.fusion_gate(fused)
        return logits

# Global extractor instance
dual_crop_extractor = DualCropExtractor(target_size=224)
