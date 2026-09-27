### Research-Grade Computer Vision Platform (KAUST Fisheries Monitoring)

---

## 1. Project Overview & Multi-Stage Architecture

The platform operates as a 3-stage modular pipeline:
1. **Stage 1: Instance Segmentation & Morphometrics** (YOLO11s-seg) — Isolates fish contours, rejects reefs and divers, and extracts oriented snout-to-tail length ($L_{\text{px}}$) and body depth ($D_{\text{px}}$).
2. **Stage 2: Taxonomic Family Classification** — Crops individual fish instances and predicts biological family via multi-model soft-voting.
3. **Stage 3: Sizing & Biomass Engine** — Applies optical calibration ($L_{\text{cm}} = L_{\text{px}} / \text{scale}$) and FishBase allometrics ($W = a \cdot L^b$) to compute individual weight and community stock metrics.

---

## 2. Stage 1: Instance Segmentation Execution

### Dataset
* **Source:** UIIS Benchmark + Marine COCO dataset.
* **Class Re-engineering:** Pruned to 3 relevant classes:
  * `fish` (9,507 train / 917 val)
  * `reefs` (5,494 train / 604 val) — negative substrate attractor
  * `human_divers` (172 train / 26 val) — diver rejection
* **Noise Pruning:** Distractor classes (`robots`, `wrecks`, `sea_floor`, `aquatic_plants`) mapped to background.

### Checkpoints Trained & Results
* Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), CUDA 12.4, AMP enabled.
* **Stage 1 Baseline (`kaust_fish_seg_v2`):**
  * Epochs: 40 | Backbone: YOLO11s-seg
  * Mask mAP50: **63.3%** | Mask Recall: **60.2%** | Mask Precision: **61.2%**
  * Role: High recall model for distant and camouflaged fish.
* **Stage 2 Fine-Tuned (`kaust_fish_seg_finetune_v1`):**
  * Epochs: 25 | Backbone: YOLO11s-seg (warm start, 6-layer freeze)
  * Mask mAP50: **63.3%** | Mask Precision: **73.4%** | Mask Recall: 51.6%
  * Role: High precision model for low false-positive operational needs.

### Server Implementation
* **Multi-Teacher Consensus:** Fuses candidate masks from Baseline and Fine-Tuned checkpoints via IoU ($> 0.45$) and containment filtering ($> 0.70$).
* **Scale-Sliced Tiling:** Slices input into macro full-frame + micro 640px sliding patches (20% overlap) to capture small background fish.

---

## 3. Stage 2: Taxonomic Multi-Model Classification

### Problem: Fish4Knowledge Raw Class Imbalance
The raw Fish4Knowledge ground-truth dataset (27,370 crops across 23 species) suffered from severe long-tail distribution:
* Top species (`fish_01`, *Dascyllus reticulatus*): > 12,000 images.
* Rare species (`fish_23`, *Siganus fuscescens*): ~200 images.

### Data Balancing Solution
1. Consolidated species into **10 Red Sea ecological families**:
   * *Acanthuridae*, *Balistidae*, *Chaetodontidae*, *Holocentridae*, *Labridae*, *Lutjanidae*, *Nemipteridae*, *Pomacentridae*, *Scaridae*, *Siganidae*.
2. Downsampled and balanced each family to an exact distribution:
   * **Train:** 180 images per family (1,800 total).
   * **Val:** 45 images per family (450 total).
   * **Total Dataset:** Exactly 2,250 balanced image crops.

### Models Trained & In-Distribution Benchmark Metrics
All 4 models were trained locally with PyTorch 2.6.0 + CUDA:

| Model | Architecture | Epochs | Top-1 Val Acc | Top-5 Val Acc | Checkpoint Location |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **YOLO11s-cls** | CSPDarknet | 25 | **99.78%** | 100.0% | `runs/classify/kaust_taxonomic_cls_v1/weights/best.pt` |
| **Swin-Tiny** | Shifted Window ViT | 26 | **98.89%** | 100.0% | `runs/classify/swin_tiny/best.pth` |
| **MobileNetV4-Small** | Inverted Residual CNN | 25 | **98.00%** | 100.0% | `runs/classify/mobilenetv4_small/best.pth` |
| **ConvNeXt-Tiny** | Modern 7x7 ConvNet | 26 | **97.33%** | 100.0% | `runs/classify/convnext_tiny/best.pth` |

---

## 4. Real-World Field Validation: Observed Failures & Error Analysis

When tested on real out-of-distribution underwater photography, the 98–99% benchmark accuracy broke down. Key failure cases reported:

### Failure Case 1: *Acanthurus nigrofuscus* (Brown Surgeonfish)
* **Ground Truth:** *Acanthuridae*
* **Model Prediction:** *Holocentridae* (Squirrelfish)
* **Confidence Score:** 38%
* **Error Analysis:** Missed the smooth oval body profile, small eyes, and diagnostic dark spots at the posterior dorsal/anal fin base. Assigned to *Holocentridae* despite that family having large nocturnal eyes, spiny gill armor, and vertical ribbing.

### Failure Case 2: *Zebrasoma scopas* (Twotone Tang)
* **Ground Truth:** *Acanthuridae*
* **Model Prediction:** *Chaetodontidae* (Butterflyfish)
* **Confidence Score:** 38%
* **Error Analysis:** Over-indexed on the elongated, pointed snout. Missed the tall sail-like dorsal fin and overlooked the defensive spine (scalpel) on the caudal peduncle (the definitive family marker).

### Failure Case 3: *Coris gaimard* (Red-Lined Wrasse, Juvenile & Adult)
* **Ground Truth:** *Labridae*
* **Model Prediction:** *Pomacentridae* (Damselfish/Clownfish)
* **Error Analysis:** Juvenile wrasse mimics clownfish patterns (bright body, white saddles). The training set for *Labridae* only contained heavy-bodied *Hemigymnus* wrasses; zero slender *Coris* or juvenile stages existed. Model mapped visual mimicry cues directly to *Pomacentridae*.

### Root Causes of Failure
1. **Dataset Domain Shift:** Fish4Knowledge crops are blurry, low-resolution ($80\times 80$ to $120\times 120$) captures from 2011 fixed-position cameras in Taiwan. Models learned camera noise, background water haze, and fixed sensor artifacts rather than biological anatomy.
2. **Aspect Ratio Distortion:** Resizing crops directly to square $224 \times 224$ distorted deep-bodied profiles (e.g. *Zebrasoma*), removing diagnostic fin-to-body ratios.
3. **Lack of Diagnostic Trait Localization:** Standard classification backbones make decisions from global color patches rather than localizing diagnostic traits (caudal spines, eye diameter ratios, fin ray morphology).
4. **No Uncertainty Filter:** At 38% confidence, models were guessing, yet the system still generated downstream biomass numbers.

---

## 5. Next Steps & Upgrade Roadmap

1. **Aspect-Ratio Preserving Preprocessing:** Replace direct $224 \times 224$ stretching with letterbox padding to maintain body depth/length proportions.
2. **Epistemic Uncertainty Filter:** Implement an Out-of-Distribution (OOD) rejection threshold. If prediction confidence is $< 70\%$ or entropy is high, flag the specimen as unverified rather than calculating incorrect biomass.
3. **Trait-Grounded Feature Representation:** Transition from training standard CNNs from scratch on low-resolution crops to foundation model feature extraction (e.g. DINOv2 / BioCLIP linear probing or LoRA adaptation) with diagnostic trait supervision.
4. **Red Sea Species Data Expansion:** Augment the 10 target families with clear, multi-angle in-situ photos covering juvenile stages and missing regional genera (*Coris*, *Acanthurus sohal*, *Naso*).

---

## 6. Stage 1 Iterations, Troubleshooting & Model Divergence

To reach our final production baseline, we tested multiple segmentation methodologies, post-processing strategies, and architectural checkpoints:

### 6.1 Post-Processing & Stray Line Artifact Removal
* **The Problem:** In our early browser portal, users noticed random residual bridge lines and "ghost" contours connecting distant fish to coral substrates.
* **What We Tried:** Initially, morphological operations (`cv2.morphologyEx` erosion and dilation) were applied to clean up boundary noise. However, the erosion broke thin caudal fin tips into disconnected pixels, and subsequent contour bridging drew long interpolation lines across the image.
* **The Solution & Outcome:** We eliminated destructive morphological erosion. Instead, we selected the largest continuous polygon (`max(contours, key=cv2.contourArea)`), applied gentle polygon simplification (`cv2.approxPolyDP` with $\epsilon = 0.75$), and discarded substrate noise specks smaller than $20\text{ px}^2$. This completely eliminated stray line artifacts while preserving delicate pectoral and caudal fin margins.

### 6.2 The YOLOv8s-seg (COCO Pretrained) Experiment
* **The Hypothesis:** Because the Roboflow benchmark from which the UIIS dataset was derived originally utilized a YOLOv8 backbone, we investigated whether training a native YOLOv8s-seg model from COCO pretrained weights would yield smoother contours than YOLO11s.
* **Implementation:** We authored `scripts/train_yolov8_seg.py`, training `yolov8s-seg.pt` on the 3-class dataset for 50 planned epochs with a checkpointing system and early stopping patience of 5 epochs.
* **Outcome:** The run converged at epoch 47, reaching **68.8% Precision** and **58.2% mAP50**. While solid, it did not surpass our YOLO11s Baseline champion (**63.3% mAP50**, **60.2% Recall**) on difficult, shadowed, or distant reef targets.

### 6.3 The Multi-Model Segmentation Ensemble Attempt
* **The Hypothesis:** We attempted to fuse candidates from both YOLOv8s-seg and YOLO11s Baseline via IoU ($>0.38$) and containment ($>0.65$) consensus merging to get the "best of both worlds".
* **What Went Wrong:** When evaluated on real field test images, the multi-model ensemble produced outputs virtually identical to the COCO YOLOv8 model alone. It provided no tangible recall improvement for small organisms, while doubling inference latency and introducing unnecessary pipeline complexity.
* **The Final Resolution:** We formally removed the multi-model segmentation ensemble from the architecture. We locked **Stage 1 Baseline (`kaust_fish_seg_v2`)** as the permanent default champion (highest recall and most complete anatomical coverage), retaining **Stage 2 Fine-Tuned (`finetuned`)** and **COCO Pretrained (`yolov8`)** as selectable comparative options in a clean 3-choice dropdown.

---

## 7. Pipeline Decoupling, Stability Engineering & UI Modernization

### 7.1 Decoupling Stage 1 Segmentation and Stage 2 Taxonomy
* **The Problem:** Initially, when an image was uploaded, the server automatically ran both Stage 1 segmentation and all 4 Stage 2 taxonomic classifiers across every detected crop. This introduced a 4-second latency spike per image, making the testing portal feel sluggish and laggy.
* **The Solution:** We decoupled Stage 2 execution. As soon as an image is loaded, only fast Stage 1 instance segmentation and geometric morphometrics execute (~50ms). Stage 2 taxonomic classification was converted to an **on-demand** action that only executes when the researcher explicitly taps **"Run Identification (Species ID)"**.

### 7.2 The HTTP 500 Null Reference Crash
* **The Bug:** After making taxonomy on-demand, users reported that clicking any preset or uploading an image failed completely with HTTP 500 Internal Server Errors, and the portal only worked when tapping "Reclassify Taxonomy".
* **Root Cause:** In `app/server.py`, when `run_classification=False`, `d["taxonomy"]` was set to `None`. In `app/biomass_engine.py`, the community metrics calculation used `d.get("taxonomy", {}).get("is_commercial", False)`. Because `d["taxonomy"]` was explicitly `None` (rather than absent), Python evaluated `None.get()`, crashing the server with `AttributeError: 'NoneType' object has no attribute 'get'`.
* **The Fix:** We hardened `biomass_engine.py` with `(d.get("taxonomy") or {}).get(...)` and `(d.get("biomass") or {}).get(...)`. Now pure segmentation and automatic biomass estimation execute without errors in under 50ms.

### 7.3 Automatic Segmentation & Browser Caching Fix
* **The Problem:** The user noted that automatic segmentation was not occurring upon selecting preset thumbnails, and Sample 2 displayed a turtle while Samples 1 and 3 were identical duplicates.
* **Root Causes:**
  1. In `app.js`, `loadImageAndSegment` compared the relative path (`/samples/sample_1.jpg`) against the absolute image property (`http://127.0.0.1:8000/...`). The comparison failed, setting `onload` on an already-cached image URL where the browser never fired `onload`.
  2. The browser aggressively cached old thumbnail images on disk.
* **The Fix:**
  1. Rebuilt `loadImageAndSegment` with fresh `new Image()` instances and an immediate execution guard (`if (img.complete && img.naturalWidth > 0) onReady()`).
  2. Replaced the turtle and duplicate images with 6 verified, distinct reef fish species in `app/samples/`.
  3. Added timestamp-based query parameters (`?v={int(mtime)}`) to `/api/samples` and script tags to guarantee immediate browser cache invalidation.

### 7.4 Dashboard De-Slopping & Scientific UI Redesign
* **The Problem:** The UI had accumulated tacky cartoon emojis (`🐟`, `⚖️`, `🎣`, `🌿`, `⚡`, `🏆`, `🌟`), competing saturated rainbow border outlines, and radioactive 18px neon glow filters that obscured fish morphology.
* **The Solution:** Completely overhauled `style.css` and `index.html` into a minimalist, professional marine research laboratory dashboard:
  - Eliminated all emojis across buttons, badges, and metric tiles.
  - Established a cohesive obsidian-slate palette (`#070b14` background, subtle `#1e293b` borders, refined `#10b981` / `#38bdf8` accents).
  - Reduced canvas polygon strokes to crisp 1.6px lines with an 18% translucent mask fill, keeping natural fish scales and reef textures clearly visible.
  - Replaced crowded sidebar badges with sleek, high-contrast metric grids and status pills.
