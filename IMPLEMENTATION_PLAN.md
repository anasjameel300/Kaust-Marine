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

### Stage 1: Class-Agnostic Fish Detection (Full Scenes with Natural Background)
* **Status:** **Finalized & Selected**
* **Selected Benchmark Dataset:** **FishDet-M** *(Khalifa University Center for Autonomous Robotic Systems / Hugging Face `GE9X/FishDet-M`)*
* **Dataset Characteristics:**
  - 105,556 raw full-scene images with 296,885 annotated fish bounding boxes.
  - Consolidates 13 diverse aquatic visual domains (coral reefs, open ocean, brackish waters, turbid coastal habitats).
  - Formatted in clean COCO bounding boxes.
* **Architecture:** YOLOv8 / YOLO11 class-agnostic detector.
* **Primary Objective:** Detect all fish in the frame with high recall and clean spatial bounding boxes, regardless of water turbidity, backscatter, or lighting conditions.
* **Milestone 1 Verification Gate:**
  1. Download benchmark subset / weights from FishDet-M.
  2. Run inference on benchmark field frames (including turbid underwater BRUV frames and high-density schooling scenes).
  3. **Visual & Metric Verification:** Inspect output bounding boxes to verify that fish are cleanly, tightly, and accurately bounded ("circled in") with minimal false positives on background coral/rocks.

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

* **Technology Stack:**
  - **Backend:** FastAPI (Python) asynchronous inference service.
  - **Frontend:** Modern Vanilla CSS + HTML5 + JavaScript (Sleek dark theme, glassmorphism, responsive canvas overlays).
  - **CV / Math Engine:** Ultralytics YOLO inference + OpenCV calibration + NumPy allometric computation.

* **Interactive Features:**
  1. **Dual-Mode Image / Video Upload:** Drag-and-drop custom images (underwater survey frames or landed catch photos) or select pre-loaded benchmark frames.
  2. **Interactive Canvas Visualizer:** Displays detected fish with clean bounding boxes, confidence badges, and taxonomic labels.
  3. **Interactive 2-Point Calibration Tool:** User clicks two reference points on a known object (measuring mat / bait canister) to establish `pixels_per_cm`.
  4. **Live Ecological Dashboard:** Real-time biomass calculation, species composition pie chart, and Shannon diversity index.
  5. **Survey Export Engine:** One-click export to CSV / JSON survey reports matching NOAA / FAO fisheries monitoring standards.

---

## 3. Phase-by-Phase Roadmap

### Phase 1: Stage 1 Setup & Verification (Current Priority)
- [ ] Trace and resolve data access bottlenecks (documented NOAA PARR access constraints).
- [ ] Adopt decoupled two-stage architecture and finalize **FishDet-M** for Stage 1.
- [ ] Ingest FishDet-M benchmark samples and configure the YOLO localization pipeline.
- [ ] **Quality Gate Milestone:** Run inference on test frames and verify that fish are accurately localized and bounded ("circled in") across varying water conditions.

### Phase 2: Stage 2 Taxonomic Classifier Development
- [ ] Evaluate candidate species datasets (OzFish, Fish4Knowledge, Kaggle Marine).
- [ ] Finalize the species dataset with the strongest Indo-Pacific / Red Sea commercial family overlap.
- [ ] Train/fine-tune the taxonomic classifier using class-weighted Focal Loss.
- [ ] Integrate Stage 1 crops with Stage 2 classification into an end-to-end inference chain.

### Phase 3: Sizing, Biomass & Calibration Engine Integration
- [ ] Connect detected box dimensions to the existing `engine/calibration.py` module.
- [ ] Connect classified species to the existing `engine/allometrics.py` FishBase database.
- [ ] Validate metric length and biomass calculations against known ground-truth benchmarks.

### Phase 4: Full-Stack Web Application Development
- [ ] Build FastAPI asynchronous endpoints (`/detect`, `/calibrate`, `/biomass`, `/export`).
- [ ] Build the interactive HTML5/Canvas frontend with the 2-point calibration tool.
- [ ] Implement the live ecological dashboard and survey export functionality.

### Phase 5: Documentation & PI Application Dossier
- [ ] Capture visual walkthrough screenshots and recordings of detections, calibration, and biomass estimation.
- [ ] Prepare tailored project dossiers and email summaries highlighting:
  - In-situ BRUVs capabilities for **Dr. Renato Morais**.
  - Landed catch measuring board capabilities for **Dr. Jessica Zamborain-Mason**.
