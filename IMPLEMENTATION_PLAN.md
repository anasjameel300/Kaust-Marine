# Automated Marine Fish Monitoring & Stock Assessment System
### Research-Grade Implementation Plan for KAUST VSRP Internship

**Author:** Candidate for KAUST VSRP (Red Sea Fisheries & AI)  
**Target Laboratories:** 
- **Dr. Jessica Zamborain-Mason** (*Integrated Reef Fisheries Laboratory*) – *Using AI to Automate Red Sea Fisheries Monitoring*
- **Dr. Renato Morais** (*Reef–Ocean Connectivity Knowledge Hub / ROCKNHub*) – *Upscaling Marine Fisheries Stock Assessment in the Red Sea*

---

## 1. Executive Summary & Methodological Rationale

### 1.1 Academic & Institutional Context
Both KAUST laboratories operate under the **Division of Biological and Environmental Sciences and Engineering (BESE)** and the **Red Sea Research Platform**. While their research entry points address distinct domains, they share a unified mathematical core:

* **Dr. Jessica Zamborain-Mason (Artisanal Catch & Stock Sustainability):**
  - Focuses on aquatic food systems, artisanal fisheries sustainability, and socio-ecological resilience.
  - Core Goal: Developing computer vision models to **identify species and estimate size from landed catch images** (e.g., market stalls, boat decks, measuring boards).
  - Key Requirement: Fine-grained species classification + accurate metric physical sizing under variable real-world contexts.

* **Dr. Renato Morais (Macroecology, Reef Energetics & Biomass):**
  - Focuses on coral reef fish biomass production, energetics, and seascape connectivity (author of the `rfishprod` framework).
  - Core Goal: Using **underwater video systems (e.g., BRUVs / stereo-video)** combined with remote sensing and spatial ecology to project **fish biomass and productivity** across whole reef systems.
  - Key Requirement: Automated in-situ fish detection + length estimation converted to biological wet-weight biomass ($W = a \cdot L^b$).

---

### 1.2 The Unified Single-System Architecture
Rather than fragmenting into two disparate projects, we build a **single unified computer vision and ecological assessment platform**. 

The fundamental pipeline is identical across both environments:
$$\text{Raw Image / Video Frame} \xrightarrow{\textbf{Stage 1: Localizer}} \text{Bounding Box } [x_{\min}, y_{\min}, x_{\max}, y_{\max}] \xrightarrow{\textbf{Stage 2: Classifier}} \text{Family / Species ID} \xrightarrow{\textbf{Stage 3: Calibrator}} L \text{ (cm)} \to W = a L^b$$

```
                         ┌──────────────────────────────────────────────┐
                         │   RAW FIELD IMAGE / VIDEO FRAME INPUT        │
                         │   (In-Situ BRUVs Video OR Landed Catch Photo)│
                         └──────────────────────┬───────────────────────┘
                                                │
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │    STAGE 1: CLASS-AGNOSTIC FISH DETECTOR     │
                         │   (FishDet-M YOLO Backbone Localization)     │
                         │   Outputs: Clean Bounding Boxes [x, y, w, h] │
                         └──────────────────────┬───────────────────────┘
                                                │ Extracted Bounding Box Crops
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │   STAGE 2: TAXONOMIC SPECIES CLASSIFIER      │
                         │   (Fine-Grained Marine Taxonomic Backbone)   │
                         │   Outputs: Family / Genus / Species + Conf   │
                         └──────────────────────┬───────────────────────┘
                                                │ Species ID + Box Dimensions
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │    STAGE 3: SCIENTIFIC CALIBRATION MODULE    │
                         │   Derives: pixels_per_cm -> Metric Length L  │
                         │   • Reference Scale / Measuring Board / Rig  │
                         └──────────────────────┬───────────────────────┘
                                                │
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │    STAGE 4: ECOLOGICAL & STOCK ANALYTICS     │
                         │   • Biomass: W = a * L^b (FishBase params)   │
                         │   • Stock Yield / Size-Frequency Curve       │
                         │   • Shannon-Wiener Biodiversity Index (H')   │
                         └──────────────────────────────────────────────┘
```

---

### 1.3 Why the Decoupled Two-Stage Architecture is Academically Essential

In terrestrial object detection, classes are roughly balanced. In marine ecology, datasets suffer from **extreme long-tail class imbalance**:
* A few dominant species (e.g., damselfish or small snappers) account for over 80% of all video frames.
* Ecologically critical apex predators and high-value commercial species (e.g., groupers, emperors, large trevallies) appear only a few dozen times.
* Research benchmarks (such as Boulais et al., CVPR 2021) demonstrated that training single-stage multi-class detectors directly on imbalanced marine datasets causes overall mAP to collapse to **29% – 39%**, with rare commercial species dropping near zero.

**The Decoupled Solution:**
1. **Stage 1 (Localization):** Eliminates species imbalance completely by treating every target as class `"fish"`. It learns the morphological boundaries of marine organisms against complex backgrounds (water columns, turbidity, coral heads, backscatter, boat decks) with high recall ($>0.85$).
2. **Stage 2 (Taxonomic Identification):** Operates on localized crops, allowing class-weighted loss functions (Focal Loss) and specialized data augmentation (Mixup, CutMix) to classify rare commercial families accurately without degrading spatial localization.
3. **Cross-Domain Generalization:** The spatial detector trained on diverse aquatic domains generalizes across both underwater BRUVs (Dr. Morais) and landed catch surfaces (Dr. Zamborain-Mason).

---

## 2. Technical Breakdown by Stage

### Stage 1: Marine Organism Instance Segmentation (Sub-Pixel Contours & Diver Rejection)
* **Status:** **Completed & Verified (Production Baseline v2 Champion)**
* **Architectural Decision (Instance Segmentation vs. Bounding Boxes):**
  - Simple bounding boxes capture excessive background coral/water and cannot measure true curved fork length, caudal peduncle width, or true cross-sectional area needed for biomass.
  - We upgraded directly to **Native Instance Segmentation** (`YOLO11s-seg`), generating sub-pixel polygon masks that precisely isolate the organism's silhouette.
* **Benchmark Dataset:**
  - **Dataset Name:** **UIIS (Underwater Image Instance Segmentation) Benchmark**
  - **Dataset Source:** [Roboflow Universe - Underwater Instance Segmentation v3i](https://universe.roboflow.com/roboflow-100/underwater-instance-segmentation)
  - **Scale:** 2,480 training images, 275 validation images formatted in COCO Segmentation.
* **Custom Dataset Conversion & Class Pruning (`scripts/convert_coco_to_yolo.py`):**
  - Converted complex COCO segmentation polygons to normalized YOLO polygon coordinates (6-decimal precision).
  - Formatted into a clean, scientifically grounded 3-class structure:
    - **Class 0 (`fish`):** 9,507 train / 917 val masks (target organism).
    - **Class 1 (`reefs`):** 5,494 train / 604 val masks (acts as a negative attractor against coral camouflage confusion).
    - **Class 2 (`human_divers`):** 172 train / 26 val masks (explicitly classifies and rejects divers/gear).
  - Filtered out irrelevant classes (`robots`, `wrecks`, `sea_floor`, `aquatic_plants`) into natural background.
* **Trained Models & Verified Benchmarks:**
  - **Stage 1 Baseline Champion (`kaust_fish_seg_v2`):** Trained for 40 full epochs on an NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) using Automatic Mixed Precision (AMP).
    - **Fish Mask mAP50:** **63.3%**
    - **Fish Mask Recall:** **60.2%** (high ecological sensitivity; catches hidden/distant fish)
    - **Fish Mask Precision:** **61.2%**
    - **Diver Rejection Precision:** **69.8%**
    - **Pixel Silhouette:** Generous, complete snout-to-tail contours without clipping thin fins.
  - **Stage 2 Warm Fine-Tuned Checkpoint (`kaust_fish_seg_finetune_v1`):** 25 epochs warm fine-tuning with cosine annealing (`lr0=0.001`), 6-layer backbone freeze, and benthic augmentations.
    - **Fish Mask Precision:** **73.4%** (+12.2% surge)
    - **Fish Mask Recall:** 51.6% (conservative boundary fit, best for ultra-strict commercial identification).
  - **COCO-Pretrained YOLOv8s-seg Experiment (`kaust_yolov8s_seg_v1`):**
    - **Hypothesis:** Investigate if native YOLOv8 architecture (used in the original Roboflow UIIS benchmark) outperforms YOLO11 on small aquatic instances.
    - **Training:** Trained via `scripts/train_yolov8_seg.py` for 47 epochs (`patience=5`, early stopping).
    - **Results:** **68.8% Precision**, **58.2% mAP50**. Solid boundary fit, but did not match YOLO11s Baseline's 63.3% mAP50 or 60.2% recall on camouflaged reef targets.
  - **Methodology Pivot: Discarding Multi-Model Segmentation Ensemble:**
    - Fusing YOLOv8s and YOLO11s via IoU consensus was implemented and evaluated in the portal. Real-world validation showed it produced outputs identical to YOLOv8 alone without improving recall, while doubling GPU compute latency.
    - **Resolution:** Dropped the segmentation ensemble. Locked **Stage 1 Baseline (`kaust_fish_seg_v2`)** as the permanent production default, retaining Fine-Tuned and COCO as selectable options.
  - **Post-Processing & Boundary Cleaning:**
    - Replaced destructive morphological erosion (which broke fin tips and created long bridging lines between fish and corals) with `cv2.approxPolyDP` ($\epsilon = 0.75$) and area filtering ($> 20\text{ px}^2$).

---

### Stage 2: Fine-Grained Taxonomic Classification (Family & Species ID)
* **Status:** **Completed & Integrated (Decoupled On-Demand Architecture)**
* **Objective:** Take the cropped bounding-box regions produced by Stage 1 and classify each organism into its respective taxonomic family and species.
* **Balanced Dataset Engineering (`datasets/taxonomic_cls`):**
  - Evaluated the Fish4Knowledge ground-truth dataset and discovered severe raw class imbalance (>12,000 for top species vs. ~200 for rare species).
  - Consolidated into **10 Red Sea ecological families**: *Acanthuridae*, *Balistidae*, *Chaetodontidae*, *Holocentridae*, *Labridae*, *Lutjanidae*, *Nemipteridae*, *Pomacentridae*, *Scaridae*, *Siganidae*.
  - Balanced each family to 180 train and 45 validation crops (**2,250 total balanced crops**).
* **Trained Models & Verified Benchmarks:**
  1. **YOLO11s-cls:** 25 epochs, **99.78% Top-1 Val Accuracy**.
  2. **Swin-Tiny Transformer:** 26 epochs, **98.89% Top-1 Val Accuracy**.
  3. **MobileNetV4-Small:** 25 epochs, **98.00% Top-1 Val Accuracy**.
  4. **ConvNeXt-Tiny:** 26 epochs, **97.33% Top-1 Val Accuracy**.
* **Ensemble Inference:** Soft-voting consensus combining softmax probability vectors from all 4 backbones.
* **Architecture Pivot (On-Demand Execution):**
  - Running all 4 deep classification backbones on every image load caused a 4-second latency bottleneck.
  - **Resolution:** Decoupled Stage 2. Stage 1 executes instantly (~50ms) on image load or preset click. Stage 2 executes strictly on-demand when the researcher taps **"Run Identification (Species ID)"**.

---

### Stage 3: Metric Sizing & Allometric Biomass Conversion Engine
* **Status:** **Completed & Hardened**
* **The Monocular Scale Ambiguity Problem:**
  In a 2D camera image, an object of length $L$ at distance $Z$ subtends $p$ pixels:
  $$p = f \cdot \frac{L}{Z}$$
  Without distance $Z$ or a known scale reference, true metric length cannot be determined from monocular pixels alone.

* **The Scientific Sizing Solution:**
  - **Catch Monitoring Mode (Dr. Zamborain-Mason):** Fishermen place catch on a measuring board (`35.0 px/cm`) or next to a known reference scale. The calibration engine calculates `pixels_per_cm` $\to$ computes metric fork length $L$ (cm).
  - **BRUVs Survey Mode (Dr. Renato Morais):** Known rig geometry (standard bait arm length = 1.2 m, `22.0 px/cm`) provides the metric scale baseline.
  - **Custom Calibration Slider:** Allows live manual adjustment from 10.0 to 60.0 px/cm.

* **Biomass Conversion ($W = a \cdot L^b$):**
  Metric length $L$ is converted to wet-weight biomass using peer-reviewed FishBase allometric parameters:
  $$W = a \cdot L^b$$
  *(When taxonomy is unclassified, robust ecological defaults $a = 0.02, b = 3.0$ are applied; once classified, family-specific parameters are dynamically injected)*.

* **Community-Level Ecological Analytics:**
  - **Abundance:** Individual count per class and target organism.
  - **Total Biomass:** Aggregate biomass in kilograms.
  - **Commercial Catch Yield:** Percentage and mass of commercial target species vs. non-target catch.
  - **Shannon-Wiener Biodiversity Index ($H'$):**
    $$H' = -\sum_{i=1}^{S} p_i \ln(p_i)$$
* **Stability Hardening:**
  - Resolved an unhandled `NoneType` AttributeError in `biomass_engine.py` when taxonomy is idle, ensuring pure segmentation runs stably at 50ms without crashing.

---

### Stage 4: Full-Stack Web Application & Research Dashboard
* **Status:** **Completed, Operational & Scientifically De-Slopped (Live at http://127.0.0.1:8000)**
* **Technology Stack:**
  - **Backend:** FastAPI (Python) asynchronous GPU-accelerated inference service (`app/server.py`).
  - **Frontend:** Modern Scientific Dark Dashboard (Vanilla CSS + HTML5 + JavaScript) with responsive canvas rendering (`app/static/`).
  - **CV / Math Engine:** Ultralytics YOLO11-seg inference + OpenCV polygon approximation + Oriented bounding rectangle morphometrics.
* **Operational Features Delivered:**
  1. **Automatic Segmentation on Load & Preset Click:** Image selection or upload immediately triggers Stage 1 segmentation without requiring manual button presses.
  2. **Refined Vector Overlay:** Sub-pixel polygon masks with clean 1.6px boundaries and 18% translucent fills, keeping underlying fish scales and coral textures visible.
  3. **3-Model Backbone Switcher:** Clean selector between **Stage 1 Baseline** (Default), **Fine-Tuned**, and **COCO YOLOv8s-seg**.
  4. **Verified Sample Imagery:** 6 unique, verified coral reef fish presets (Blue Tang, Rainbow Parrotfish, Rock Beauty, Clownfish family, Emerald Parrotfish, Deep Blue Triggerfish) with automatic mtime cache busting (`?v={mtime}`).
  5. **De-Slopped Design System:** Removed cartoon emojis and saturated neon borders; implemented a professional obsidian-slate design system with minimalist metric tiles and structured instance cards.
  6. **Annotated Image Export:** One-click download of full-resolution segmented photos with contours.

---

## 3. Phase-by-Phase Roadmap & Current Status

### Phase 1: Stage 1 Setup & Model Development (COMPLETED)
- [x] Ingested the **UIIS Benchmark** (2,480 train images, 275 validation images).
- [x] Converted annotations to 3-class schema (`fish`, `reefs`, `human_divers`).
- [x] Trained **YOLO11s-seg Baseline Champion** (`kaust_fish_seg_v2`, 40 epochs): **63.3% mAP50**, **60.2% Recall**.
- [x] Trained **Stage 2 Warm Fine-Tuned** model (`kaust_fish_seg_finetune_v1`, 25 epochs): **73.4% Precision**.
- [x] Trained **COCO YOLOv8s-seg** model (`kaust_yolov8s_seg_v1`, 47 epochs): **68.8% Precision**, **58.2% mAP50**.
- [x] Tested and discarded multi-model segmentation ensemble due to redundancy and latency; locked Baseline v2 as default.
- [x] Removed morphological erosion bridge-line artifacts using `approxPolyDP(0.75)`.

### Phase 2: Stage 2 Taxonomic Multi-Model Classifier (COMPLETED)
- [x] Solved Fish4Knowledge class imbalance by constructing balanced 10-family dataset (2,250 crops).
- [x] Trained and verified 4 deep classification backbones: YOLO11s-cls (99.78%), Swin-Tiny (98.89%), MobileNetV4 (98.00%), ConvNeXt-Tiny (97.33%).
- [x] Integrated soft-voting ensemble consensus in `app/taxonomic_engine.py`.
- [x] Decoupled Stage 2 into an on-demand operation to keep image upload latency under 50ms.

### Phase 3: Stage 3 Metric Sizing & Allometric Biomass Engine (COMPLETED)
- [x] Implemented physical calibration system with BRUVs rig (22 px/cm), measuring board (35 px/cm), and custom slider.
- [x] Built allometric biomass engine ($W = a \cdot L^b$) with dynamic FishBase parameters.
- [x] Implemented community metrics (Total Biomass kg, Commercial Yield %, Shannon Diversity $H'$).
- [x] Hardened null-reference handling for unclassified instances in `biomass_engine.py`.

### Phase 4: Stage 4 Web Testing Portal & UX De-Slopping (COMPLETED)
- [x] Built FastAPI server (`app/server.py`) and responsive canvas visualizer (`app/static/`).
- [x] Fixed automatic segmentation trigger upon image load and preset thumbnail selection.
- [x] Replaced non-fish turtle and duplicate presets with 6 verified reef fish samples, adding cache-busting URLs.
- [x] De-slopped the interface: eliminated cartoon emojis, refined canvas strokes to 1.6px with 18% fill, and implemented a clean scientific obsidian-slate design system.

### Phase 5: Scientific Benchmark: Vision Foundation Models (LoRA) vs. Task-Specific Backbones on Fish-Vista (ACTIVE)
- [x] Verified *Fish-Vista* paper findings (Mehrab et al., CVPR 2025): zero-shot CLIP/BioCLIP fails on micro-traits; aspect squishing destroys body depth ratios.
- [x] Integrated letterbox padding (`114, 114, 114`) in `taxonomic_engine.py` to preserve biological morphometrics.
- [x] Added epistemic uncertainty rejection filter (< 65% confidence flagged as unverified specimen).
- [x] Authored selective Fish-Vista ingestion script (`scripts/download_fish_vista_10_families.py`) and downloaded 10 Red Sea families (`1,443` train, `365` val crops).
- [x] Trained and verified dedicated traditional baselines on Fish-Vista:
  - **ResNet50d:** **97.81%** Top-1 Val Acc, **99.73%** Top-5 Val Acc.
  - **YOLO11s-cls:** **95.34%** Top-1 Val Acc, **99.45%** Top-5 Val Acc.
  - **Swin-Tiny:** **97.53%** Top-1 Val Acc, **99.73%** Top-5 Val Acc.
  - **MobileNetV4-Small:** **81.10%** Top-1 Val Acc, **98.90%** Top-5 Val Acc.
  - **ConvNeXt-Tiny:** **80.27%** Top-1 Val Acc, **96.16%** Top-5 Val Acc.
- [x] **Step 1 (Completed):** Parameter-Efficient Fine-Tuning (PEFT/LoRA) on Vision Foundation Models:
  - **DINOv2-Small (`dinov2_vits14`) + LoRA**: **98.08%** Top-1 Val Acc (0.4M trainable parameters).
  - **BioCLIP (`imageomics/bioclip` ViT-B/16) + LoRA**: **96.44%** Top-1 Val Acc, **100.00%** Top-5 Val Acc (surging from 27.40% zero-shot).
- [x] **Step 2 (Completed):** Dual-Benchmark Comparative Evaluation:
  - Downloaded 250 verified wild in-situ Red Sea photos (`datasets/evaluation_ood_redsea`) across 10 families.
  - Executed automated evaluation across all 7 models ([reports/dual_benchmark_leaderboard.md](file:///c:/Users/anasjameel/Desktop/Kaust%20Marine/reports/dual_benchmark_leaderboard.md)).
  - Proved the "Underwater Illusion": CNNs suffer an ~80% collapse in the wild (14-17%), while BioCLIP + LoRA dominates at **51.20% Top-1 and 89.60% Top-5**.
- [x] **Step 3 (Completed):** Dual-Crop Multi-Region Feature Extraction (Fish-Vista CVPR 2025):
  - Created `app/dual_crop_engine.py` isolating Stream A (Global letterbox) + Stream B1 (Cephalic anterior 35% ROI) + Stream B2 (Caudal posterior 25% ROI).
  - Wired into `app/taxonomic_engine.py` with multi-region trait fusion ($0.55 \cdot P_{\text{global}} + 0.25 \cdot P_{\text{head}} + 0.20 \cdot P_{\text{caudal}}$) to resolve camouflage and lighting ambiguities.
- [x] **Step 4 (Completed):** Full-Stack Dashboard Integration & Model Suite:
  - Integrated all 7 models into `app/taxonomic_engine.py`, `app/server.py`, and `app/static/`: BioCLIP + LoRA (In-Situ Wild Champion), ResNet50d (Landed Catch Champion), DINOv2-LoRA, YOLO11s-cls, Swin-Tiny, and Scientific Ensemble.
  - Added interactive Dual-Crop ROI visualizer in the specimen cards displaying high-res base64 Cephalic & Caudal thumbnails with anatomical consensus badges.
- [x] **Step 5 (Completed):** Adaptive Anchor-Preserving Gating & Scientific Transparency Modal (`👁`):
  - Solved semantic OOD dilution via Option A adaptive gating: the global body remains the invariant anchor ($P_{\text{global}}$); agreeing sub-regions boost confidence while ambiguous isolated parts are safely filtered out.
  - Integrated interactive research notes modal (`👁`) in both the sidebar control panel and specimen cards detailing the current status, biological motivation, and future multi-task training path.

### Phase 6: Documentation & PI Application Dossier (ACTIVE NEXT STEP)
- [ ] Prepare tailored research dossier and publication manuscript highlighting:
  - In-situ BRUVs biomass and diversity assessment for **Dr. Renato Morais** (*ROCKNHub*).
  - Landed catch measuring board sizing and commercial yield tracking for **Dr. Jessica Zamborain-Mason** (*Integrated Reef Fisheries Lab*).
  - Scientific paper manuscript: *"Benchmarking Parameter-Efficient Vision Foundation Models versus Specialized Architectures for Fine-Grained Marine Fish Taxonomy and In-Situ Biomass Estimation"*.

