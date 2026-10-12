# Automated Marine Fish Monitoring, Taxonomy & Biomass Assessment Platform
### Research-Grade Computer Vision Platform for KAUST VSRP Internship

**Target Laboratories at KAUST (Division of Biological and Environmental Science and Engineering - BESE):**
* **Dr. Jessica Zamborain-Mason** (*Integrated Reef Fisheries Laboratory*) – *Using AI to Automate Red Sea Fisheries Monitoring*
* **Dr. Renato Morais** (*Reef–Ocean Connectivity Knowledge Hub / ROCKNHub*) – *Upscaling Marine Fisheries Stock Assessment in the Red Sea*

---

## 1. Executive Summary & Scientific Overview

This repository houses a unified, production-grade deep learning platform for **automated marine organism instance segmentation, fine-grained taxonomic classification, anatomical morphometrics, and allometric stock assessment**.

Designed specifically for the research workflows of **Dr. Jessica Zamborain-Mason** (artisanal catch monitoring & length sizing) and **Dr. Renato Morais** (in-situ BRUVs video surveys & reef biomass production), the system provides an end-to-end 3-stage ecological pipeline:
1. **Stage 1 (Sub-pixel Instance Segmentation):** Clean polygon contours isolating individual organisms from complex coral reef substrate and high-turbidity water columns with dedicated diver/gear rejection (`fish`, `reefs`, `human_divers`).
2. **Stage 2 (Fine-Grained Taxonomy & Vision Foundation Models):** 10-family marine classification powered by Parameter-Efficient Fine-Tuning (PEFT/LoRA) on Vision Foundation Models (**BioCLIP**, **DINOv2**) and high-resolution convolutional/transformer backbones (**ResNet50d**, **Swin-Tiny**, **YOLO11s-cls**).
3. **Stage 3 (Dual-Crop Trait Extraction & Allometric Biomass):** Multi-region anatomical trait extraction (Cephalic anterior 35% & Caudal posterior 25%) backed by an **Adaptive Anchor-Preserving Gating** mechanism and dynamic FishBase allometric equations ($W = a \cdot L^b$) computing individual biomass (g), total biomass (kg), commercial yield (%), and Shannon Diversity ($H'$).

---

## 2. Research Architecture & Pipeline

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                   RAW UNDERWATER OR LANDED CATCH INPUT                 │
 │                (In-Situ BRUVs Survey OR Artisanal Catch Tray)          │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │           STAGE 1: YOLO11s-seg INSTANCE SEGMENTATION ENGINE            │
 │   • Backbone: YOLO11s-seg (63.3% mAP50, 60.2% Recall)                  │
 │   • 3-Class Schema: fish, reefs, human_divers (Rejects divers/benthic)  │
 │   • Scale-Sliced Tiling (SAHI) for distant/dense organism schools      │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │          STAGE 2: FINE-GRAINED TAXONOMIC CLASSIFICATION (PEFT)         │
 │   • BioCLIP (ViT-B) + LoRA: In-Situ Wild Champion (89.6% Top-5)        │
 │   • ResNet50d: Landed Catch Champion (97.8% In-Domain Top-1)           │
 │   • DINOv2-Small + LoRA (98.1%) & Swin-Tiny (97.5%)                    │
 │   • Dual-Crop Multi-Region ROIs: Cephalic (35%) & Caudal (25%)         │
 │   • Adaptive Anchor-Preserving Gating: Shields against fragment noise  │
 └───────────────────────────────────┬────────────────────────────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │              STAGE 3: ALLOMETRIC BIOMASS & COMMUNITY ECOLOGY           │
 │   • Oriented Morphometrics: Snout-to-tail Fork Length & Body Depth     │
 │   • Physical Calibration: In-Situ BRUVs (22 px/cm) / Tray (35 px/cm)  │
 │   • FishBase Allometry: W = a * L^b (Family-specific parameters)       │
 │   • Community Metrics: Total Biomass (kg), Commercial %, Shannon H'    │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Empirical Landmark Discovery: The "Underwater Illusion"

We conducted a rigorous head-to-head dual-benchmark evaluation comparing **Controlled Museum Vouchers (Fish-Vista Holdout)** against **Real-World Wild In-Situ Photography (250 research-grade Red Sea photos from iNaturalist across 10 families)**:

| Architecture | Paradigm | Controlled (Fish-Vista) | Real-World (Wild Red Sea) | Robustness Drop ($\Delta$) | GPU Latency | Primary PI Alignment |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **BioCLIP (ViT-B) + LoRA** 🏆 | **Biological VFM (PEFT)** | **96.44% (T5: 100%)** | **51.20% (T5: 89.6%)** | **-45.24% (Champion)** | 7.9 ms | **Dr. Renato Morais (In-Situ BRUVs)** |
| **DINOv2-Small + LoRA** | Vision Foundation (PEFT) | **98.08% (T5: 99.7%)** | 23.20% (T5: 73.2%) | -74.88% | 4.7 ms | General VFM Benchmark |
| **Swin-Tiny** | Shifted Window ViT | 97.53% (T5: 99.7%) | 23.20% (T5: 61.6%) | -74.33% | 6.2 ms | Hierarchical Feature Benchmark |
| **ResNet50d** 🏆 | Deep Residual CNN | **97.81% (T5: 99.7%)** | 17.20% (T5: 66.0%) | -80.61% | 6.7 ms | **Dr. Jessica Zamborain-Mason (Catch Boards)** |
| **YOLO11s-cls** | CSPDarknet Real-Time | 95.26% (T5: 99.2%) | 14.00% (T5: 56.0%) | -81.26% | 160.1 ms | Embedded Low-Power Hardware |
| **ConvNeXt-Tiny** | 7x7 Depthwise ConvNet | 80.27% (T5: 96.2%) | 12.00% (T5: 57.2%) | -68.27% | 5.0 ms | Pure Convolutional Baseline |
| **MobileNetV4-Small** | Edge Inverted Residual | 81.10% (T5: 98.9%) | 7.60% (T5: 56.0%) | -73.50% | **2.9 ms** | Ultra-Fast Edge Mobile |

### Critical Scientific Insights:
1. **The "Underwater Illusion":** On clean museum studio vouchers, traditional CNNs appear solved ($95\% - 98\%$). However, when plunged into the wild Red Sea, standard CNNs suffer a catastrophic **$\sim 80\%$ collapse** due to water column red-light attenuation, turbidity, and coral camouflage.
2. **BioCLIP Dominance in the Wild:** BioCLIP + LoRA retained **$51.20\%$ Top-1 and $89.60\%$ Top-5 accuracy in wild conditions**—surpassing ResNet50 by $+34.0\%$ absolute (3x higher). Its TreeOfLife-10M biological pre-training confers superior invariance to aquatic lighting and benthic clutter.
3. **BioCLIP Zero-Shot Failure Verified:** Out-of-the-box BioCLIP scored only $27.40\%$ zero-shot on Fish-Vista families. Parameter-Efficient LoRA fine-tuning surged it by **$+69.04\%$ to $96.44\%$**, proving the necessity of fine-tuning.

---

## 4. Dual-Crop Multi-Region Feature Extraction & Adaptive Gating

Following the *Fish-Vista* formulation (Mehrab et al., CVPR 2025), the system isolates diagnostic anatomical sub-regions:
* **Stream A (Global Letterbox):** Aspect-ratio preserved whole-organism view capturing length-to-depth ratio ($L/D$) and overall counter-shading coloration.
* **Stream B1 (Cephalic ROI - Anterior 35%):** Diagnostic snout angle, eye diameter ratio, and jaw dentition (e.g. parrotfish fused beak plates).
* **Stream B2 (Caudal Peduncle ROI - Posterior 25%):** Diagnostic caudal spine (e.g. Acanthuridae scalpel marker), peduncle depth, and fin fork geometry.

### Adaptive Anchor-Preserving Gating & Research Limitations:
* **The Scientific Boundary:** Pretrained vision foundation backbones (BioCLIP, DINOv2, ResNet) were trained on full-body organism vouchers. Evaluating severed anatomical fragments (such as an isolated caudal peduncle without body context) in a zero-shot manner introduces semantic ambiguity (e.g., an isolated surgeonfish tail resembling a damselfish).
* **Current Mitigation:** The system preserves the whole-organism prediction as the invariant anchor. Agreeing sub-regions boost confidence (`Cephalic Validated`, `Anatomical Consensus`), while ambiguous or noisy part predictions are safely gated out.
* **Ongoing Limitation & Future Direction:** Dedicated supervised fine-tuning of region-specific anatomical classification heads (rather than evaluating general whole-body encoders on severed crops) remains an essential next research milestone for full anatomical trait resolution.

---

## 5. Allometric Biomass & Community Ecology Engine

Using oriented bounding rectangle geometry, each detected fish is measured in calibrated metric units ($cm$):

$$\text{Weight (g)} = a \cdot \text{Length}^{b}$$

Where $a$ and $b$ are allometric scaling parameters dynamically retrieved from FishBase for the classified family:
* **Total Biomass:** Aggregate biomass in kilograms: $W_{\text{total}} = \sum \frac{W_i}{1000}$
* **Commercial Yield (%):** Percentage of total biomass belonging to commercial food fish families (e.g. *Lutjanidae*, *Siganidae*, *Scaridae*, *Balistidae*).
* **Shannon Diversity Index ($H'$):** Ecological community diversity metric based on family relative abundance:
  $$H' = -\sum_{i=1}^{S} p_i \ln(p_i)$$

---

## 6. Automated BRUVs Video Survey & MaxN Ecological Tracking

To address the video survey workflows of **Dr. Renato Morais**, the platform features an automated underwater video survey engine:
* **The Double-Counting Challenge:** In continuous video surveys, the same individual fish enters and leaves the camera view repeatedly. Summing frame detections creates severe ecological overestimation.
* **MaxN Ecological Standard:** MaxN is defined as the maximum number of individuals of a given family/species visible simultaneously in any single video frame. Because all counted fish are visible on screen at the exact same moment, double-counting is mathematically impossible.
* **ByteTrack Multi-Object Tracking:** Combines YOLO11s-seg with ByteTrack Kalman filtering to track persistent fish identities across frames, extracting the highest-confidence crop per track for Stage 2 taxonomic classification.
* **Temporal Sub-Sampling & Fast Inference:** Ingests video at 2–4 FPS, reducing GPU latency while retaining full tracking continuity.
* **Interactive Research Portal:**
  * Native web-standard H.264 video playback with live bounding boxes, segmentation masks, and tracking IDs.
  * Temporal fish density sparkline displaying community density fluctuations across time.
  * Clickable peak keyframe snapshot seek that jumps and pauses on the exact peak community frame.
  * Interactive fish track cards displaying track lifespan, in-peak presence, and metric sizing.

---

## 7. Project Structure

```text
Kaust Marine/
├── README.md                      # Complete research documentation & user guide
├── journey_document.md            # Comprehensive project journal & audit trail
├── IMPLEMENTATION_PLAN.md         # Phase-by-phase implementation plan
├── app/                           # Full-stack testing web application
│   ├── server.py                  # FastAPI inference backend & model server
│   ├── taxonomic_engine.py        # Multi-model classification & adaptive gating
│   ├── dual_crop_engine.py        # Anatomical ROI extraction & fusion
│   ├── biomass_engine.py          # Metric sizing, FishBase allometry & Shannon H'
│   ├── video_engine.py            # ByteTrack tracking & MaxN video processing
│   ├── samples/                   # Benchmark validation imagery & sample Red Sea video
│   └── static/                    # Frontend assets
│       ├── index.html             # Responsive obsidian-slate dashboard
│       ├── css/style.css          # Glassmorphism dark-mode styles
│       └── js/app.js              # Canvas visualizer & client controller
├── datasets/                      # Datasets (managed locally via .gitignore)
│   ├── underwater_yolo_seg/       # 3-class normalized YOLO segmentation dataset
│   ├── fish_vista_10_families/    # 10-family Fish-Vista museum vouchers (1,808 crops)
│   └── evaluation_ood_redsea/     # 250 wild in-situ Red Sea evaluation photos
├── runs/                          # Model weights & training logs
│   ├── segment/models/kaust_fish_seg_v2/ # 🏆 YOLO11s-seg Champion (best.pt)
│   └── classify_fish_vista/       # 🏆 7 Fine-Tuned Taxonomic Model Weights
│       ├── bioclip_lora/best.pth  # BioCLIP + LoRA (In-Situ Wild Champion - 89.6%)
│       ├── resnet50d/best.pth     # ResNet50d (Landed Catch Champion - 97.8%)
│       ├── dinov2_lora/best.pth   # DINOv2-Small + LoRA (98.1%)
│       ├── yolo11s_cls/weights/best.pt # YOLO11s-cls (95.3%)
│       └── swin_tiny/best.pth     # Swin-Tiny Transformer (97.5%)
└── scripts/                       # Training, evaluation & download scripts
    ├── train_bioclip_lora.py      # BioCLIP LoRA fine-tuning & zero-shot script
    ├── train_dinov2_lora.py       # DINOv2 LoRA fine-tuning script
    ├── evaluate_dual_benchmark.py # Dual-benchmark evaluation runner
    ├── download_fish_vista_10_families.py # Fish-Vista selective downloader
    ├── download_redsea_ood_eval.py# Wild Red Sea iNaturalist downloader
    ├── download_sample_video.py   # Ingest sample Red Sea underwater video
    └── process_bruvs_video.py     # Standalone CLI BRUVs video processor
```

---

## 8. Quickstart & How to Run

### 1. Prerequisites
Ensure you have Python 3.11 with CUDA support (recommended via Miniconda):

```bash
# Clone the repository
git clone https://github.com/anasjameel300/Kaust-Marine.git
cd "Kaust-Marine"

# Activate the conda environment
conda activate kaust_marine

# Install dependencies
pip install ultralytics fastapi uvicorn python-multipart opencv-python numpy timm torchvision open_clip_torch
```

### 2. Launch the Research Portal
Run the FastAPI web application:

```powershell
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at: **`http://127.0.0.1:8000`**

### 3. Running Dual-Benchmark Evaluations
To reproduce the dual-benchmark leaderboard across all 7 models:

```powershell
python scripts/evaluate_dual_benchmark.py
```
Results will be output to [`reports/dual_benchmark_leaderboard.md`](reports/dual_benchmark_leaderboard.md).
