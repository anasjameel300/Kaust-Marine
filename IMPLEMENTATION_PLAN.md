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


---

### Stage 2: Fine-Grained Taxonomic Classification (Family & Species ID)
* **Status:** **Candidate Review Phase (To be finalized after Stage 1 verification)**
* **Objective:** Take the cropped bounding-box regions produced by Stage 1 and classify each organism into its respective taxonomic family and species.
* **Target Ecological & Commercial Taxa (Red Sea / Indo-Pacific Shared Guilds):**
  - **Snappers (*Lutjanidae*):** *Lutjanus* spp. (Red Snapper, Two-spot Snapper / *Bahara*, Lane Snapper)
  - **Groupers (*Serranidae*):** *Epinephelus* spp., *Plectropomus* spp. (Greasy Grouper / *Hamour*, Coral Trout / *Najil*)
  - **Emperors (*Lethrinidae*):** *Lethrinus* spp. (Spangled Emperor / *Shaoor*)
  - **Jacks & Trevallies (*Carangidae*):** *Caranx* spp., *Seriola* spp. (Giant Trevally / *Bayad*, Amberjack)
  - **Triggerfish (*Balistidae*):** *Balistes* spp., *Balistoides* spp. (Titan Triggerfish)
  - **Barracudas (*Sphyraenidae*):** *Sphyraena barracuda*
* **Candidate Datasets Under Consideration for Stage 2:**
  1. **Candidate A: OzFish Labeled Crops (AIMS / UWA / Pawsey)**
     - ~80,000 labeled crops covering 500+ species and 70 families of Indo-Pacific reef fish.
     - Direct family/genus metadata links. Strong academic relevance to Dr. Morais's background.
  2. **Candidate B: Fish4Knowledge Ground-Truth Species Dataset (Univ. of Edinburgh)**
     - 27,370 verified fish crops categorized into 23 distinct coral reef species clusters.
     - Clean, verified annotations from fixed underwater reef observatories.
  3. **Candidate C: Large-Scale Marine Fish Dataset (Kaggle / Commercial Fisheries)**
     - High-resolution crops of commercial marine species (Sea Bass, Bream, Mackerel).
     - Well-suited for validating the market-stall / landed catch classification path.
* **Decision Policy:** We will inspect class counts and species overlap for these candidates and select the optimal training set once Stage 1 localization is visually verified.

---

### Stage 3: Metric Sizing & Allometric Biomass Conversion Engine
* **The Monocular Scale Ambiguity Problem:**
  In a 2D camera image, an object of length $L$ at distance $Z$ subtends $p$ pixels:
  $$p = f \cdot \frac{L}{Z}$$
  Without distance $Z$ or a known scale reference, true metric length cannot be determined from monocular pixels alone.

* **The Scientific Sizing Solution:**
  - **Catch Monitoring Mode (Dr. Zamborain-Mason):** Fishermen place catch on a measuring board, beside an ArUco fiducial marker, or next to a known reference scale. The calibration engine calculates `pixels_per_cm` $\to$ computes metric fork length $L$ (cm).
  - **BRUVs Survey Mode (Dr. Renato Morais):** Known rig geometry (standard bait arm length = 1.2 m, bait canister diameter = 8.0 cm) provides the metric scale baseline.

* **Biomass Conversion ($W = a \cdot L^b$):**
  Metric length $L$ is converted to wet-weight biomass using peer-reviewed FishBase allometric parameters:
  $$W = a \cdot L^b$$
  *(e.g., for Lutjanus snappers, $a \approx 0.014$, $b \approx 3.02$; for Epinephelus groupers, $a \approx 0.013$, $b \approx 3.04$)*.

* **Community-Level Ecological Analytics:**
  - **Abundance:** Individual count per species/family.
  - **Total Biomass:** Aggregate biomass in grams/kilograms.
  - **Shannon-Wiener Biodiversity Index ($H'$):**
    $$H' = -\sum_{i=1}^{S} p_i \ln(p_i)$$

---

### Stage 4: Full-Stack Web Application & Research Dashboard
* **Status:** **Completed & Operational (Live at http://127.0.0.1:8000)**
* **Technology Stack:**
  - **Backend:** FastAPI (Python) asynchronous GPU-accelerated inference service (`app/server.py`).
  - **Frontend:** Glassmorphism UI (Vanilla CSS + HTML5 + JavaScript) with responsive canvas rendering (`app/static/`).
  - **CV / Math Engine:** Ultralytics YOLO11-seg inference + OpenCV polygon approximation + Oriented bounding rectangle morphometrics.
* **Operational Features Delivered:**
  1. **Dual-Mode Input:** Drag-and-drop custom field imagery (underwater surveys or catch trays) with dynamic file decoding, plus 6 preloaded benchmark validation presets.
  2. **Sub-Pixel Polygon Visualizer:** High-precision canvas overlays with neon contour glow, translucent mask fill, and floating metadata badges.
  3. **Live Model Architecture Switcher:** Real-time dropdown to toggle between the **Stage 1 Baseline Champion** (high recall & complete silhouette) and the **Stage 2 Fine-Tuned** checkpoint (ultra-high precision).
  4. **Multi-Class Filters & Sensitivity:** Independent toggles for `fish`, `reefs`, and `human_divers`, paired with a live confidence threshold slider (0.10 to 0.80).
  5. **Automated Morphometrics:** Calculates oriented snout-to-tail fork length (px), body depth (px), and 2D contour surface area (px²) in real time.
  6. **Annotated Image Export:** One-click download of full-resolution segmented photos with contours.

---

## 3. Phase-by-Phase Roadmap & Current Status

### Phase 1: Stage 1 Setup & Model Development (COMPLETED)
- [x] Ingested the **UIIS (Underwater Image Instance Segmentation) Benchmark** (2,480 train images, 275 validation images).
- [x] Built the custom conversion engine (`scripts/convert_coco_to_yolo.py`) to restructure raw COCO annotations into a clean 3-class schema (`fish`, `reefs`, `human_divers`) and filter background noise.
- [x] Configured the local deep learning environment: Miniconda Python 3.11 with PyTorch 2.6.0+cu124 and CUDA 12.4 on an NVIDIA GeForce RTX 3050 Laptop GPU.
- [x] Trained native **YOLO11s-seg** for 40 full epochs (`kaust_fish_seg_v2`), achieving **63.3% Mask mAP50**, **60.2% Recall**, and **69.8% Diver Precision**.
- [x] Executed Stage 2 warm fine-tuning experiment (`kaust_fish_seg_finetune_v1`), achieving **73.4% Fish Precision**.
- [x] Validated hands-on that the Stage 1 Baseline is our superior production champion for ecological abundance counts and complete anatomical length sizing.

### Phase 2: Stage 4 Web Testing Portal Development (COMPLETED)
- [x] Built the FastAPI backend (`app/server.py`) with GPU memory management and dynamic file upload handling.
- [x] Developed the responsive HTML5/Canvas testing dashboard (`app/static/`) with neon contours and morphometric cards.
- [x] Integrated dual-checkpoint model caching and runtime switching between Baseline v2 and Fine-Tuned v1.
- [x] Validated multi-class filtering, default "Fish Only" display, and orientation handling across devices.

### Phase 3: Stage 2 Taxonomic Classifier Development (Active Next Milestone)
- [ ] Evaluate candidate species datasets (OzFish, Fish4Knowledge, Kaggle Marine) for Red Sea commercial family overlap (Snappers, Groupers, Emperors, Trevallies).
- [ ] Train/fine-tune the taxonomic classifier using class-weighted Focal Loss on segmented crops.
- [ ] Chain Stage 1 instance segmentations into Stage 2 taxonomic family/species prediction.

### Phase 4: Sizing, Biomass & Calibration Engine Integration (Upcoming)
- [ ] Implement 2-point reference calibration to establish metric `pixels_per_cm` (measuring board / bait canister).
- [ ] Connect oriented snout-to-tail fork length $L$ (cm) to the FishBase allometric biomass database ($W = a \cdot L^b$).
- [ ] Validate metric length and biomass calculations against known ground-truth benchmarks.

### Phase 5: Documentation & PI Application Dossier (Upcoming)
- [ ] Capture visual walkthroughs and demonstration recordings of detections, calibration, and biomass estimation.
- [ ] Prepare tailored research project dossiers highlighting:
  - In-situ BRUVs biomass capabilities for **Dr. Renato Morais**.
  - Landed catch measuring board sizing for **Dr. Jessica Zamborain-Mason**.

