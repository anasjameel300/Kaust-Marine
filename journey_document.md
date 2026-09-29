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

---

## 8. Scientific Expansion: Fish-Vista Benchmark & Vision Foundation Models (LoRA Adaptation)

### 8.1 Empirical Verification via *Fish-Vista* (Mehrab et al., CVPR 2025)
* **Scientific Context:** Zero-shot foundation models (CLIP, BioCLIP) fail on fine-grained fish taxonomy because global self-attention and contrastive pooling wash out micro-anatomical traits (e.g. 2mm scalpel spine on *Acanthuridae*, spiny dorsal ray counts and eye-to-head ratios in *Holocentridae*).
* **Immediate Mitigations Integrated:**
  1. **Letterbox Padding (`taxonomic_engine.py`):** Replaced aspect-ratio squishing with proportional scaling and neutral gray `(114, 114, 114)` padding to $224 \times 224$, preserving true biological body depth-to-length ratios.
  2. **Epistemic Uncertainty Gate:** Instances with confidence $< 65\%$ or consensus disagreement are marked as `is_verified = False`, displayed as `"Unverified Specimen (Needs Taxonomic Review) — Best Guess: <Family> (<Confidence>%)"`.

### 8.2 The 10-Family Selective Fish-Vista Ingestion Strategy
* **Why Replace Fish4Knowledge:** Fish4Knowledge consists of blurry, low-resolution ($80\times 80$ to $120\times 120$) 2011 Taiwanese camera crops, causing artificial camera-noise learning.
* **The Focused 10-Family Core:** Rather than pulling all 69,269 images across 4,316 species, we ingest only the 10 Red Sea ecological & commercial families (*Acanthuridae*, *Balistidae*, *Chaetodontidae*, *Holocentridae*, *Labridae*, *Lutjanidae*, *Nemipteridae*, *Pomacentridae*, *Scaridae*, *Siganidae*).
* **Execution Tool:** Authored `scripts/download_fish_vista_10_families.py` to stream and parse metadata CSVs from Hugging Face (`imageomics/fish-vista`), reporting exact species diversity and downloading quota-balanced splits (`train` / `val`) with live progress tracking.

### 8.3 Scientific Benchmark: Parameter-Efficient VFMs (LoRA) vs. Task-Specific Backbones
To establish a publishable, research-grade comparative study, we formulate an experimental benchmark:
1. **Vision Foundation Models via PEFT/LoRA:**
   - **DINOv2-Small (`dinov2_vits14`):** Self-supervised ViT adapted via Low-Rank Adaptation ($r=8, \alpha=16$) on attention projection layers. Fits within ~3.1 GB VRAM on NVIDIA RTX 3050 (4 GB).
   - **BioCLIP (`imageomics/bioclip` ViT-B/16):** Domain-pretrained on TreeOfLife-10M, adapted via LoRA on visual projection heads.
2. **Dedicated Task Architectures (Trained from Scratch / ImageNet Warm):**
   - YOLO11s-cls (CSPDarknet)
   - Swin-Tiny (Shifted Window ViT)
   - MobileNetV4-Small (Inverted Residual CNN)
   - ConvNeXt-Tiny (Modernized 7x7 Depthwise ConvNet)
3. **Dual-Evaluation Protocol:**
   - **Benchmark 1 (Controlled / High-Resolution):** Fish-Vista 10-Family holdout validation set.
   - **Benchmark 2 (Real-World Out-of-Distribution):** In-situ Red Sea BRUVs underwater video and artisanal landed catch photos.
   - **Ecological Impact Metric:** Quantifying downstream allometric biomass distortion ($\Delta \text{Biomass} = |W_{\text{predicted}} - W_{\text{true}}|$) caused by taxonomic misclassifications.

### 8.4 Dual-Crop Multi-Region Feature Extraction Concept
* **Stream A (Global Letterbox Crop):** Full-body silhouette preserving aspect ratio ($L/D$) and gross pigmentation.
* **Stream B (Diagnostic ROIs):** Automatic anatomical extraction along the oriented minimum bounding axis:
  - *Cephalic Region (front 35%):* Captures snout angle, jaw structure, and eye diameter ratio.
  - *Caudal Peduncle Region (rear 25%):* Captures caudal spine morphology and tail geometry.
* **Fusion Head:** Cross-attention / gated concatenation combining global and micro-trait embeddings prior to classification.

---

### 8.5 Fish-Vista Dedicated Baselines: Verified Empirical Results
The full suite of dedicated traditional architectures was trained on the newly ingested Fish-Vista 10-family dataset (`1,443` train, `365` val crops across 10 Red Sea families) using letterbox padding (`114, 114, 114`) and early stopping (`patience=5`):

| Architecture | Paradigm | Epochs | Top-1 Val Acc | Top-5 Val Acc | Checkpoint Location |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **DINOv2-Small + LoRA** | Self-Supervised VFM (PEFT) | 15 (early stopped) | **98.08%** | **99.73%** | `runs/classify_fish_vista/dinov2_lora/best.pth` |
| **ResNet50d** | Deep Residual CNN | 19 (early stopped) | **97.81%** | **99.73%** | `runs/classify_fish_vista/resnet50d/best.pth` |
| **Swin-Tiny** | Shifted Window ViT | 16 (early stopped) | **97.53%** | **99.73%** | `runs/classify_fish_vista/swin_tiny/best.pth` |
| **BioCLIP (ViT-B) + LoRA** | Biological VFM (PEFT) | 12 (early stopped) | **96.44%** | **100.00%** | `runs/classify_fish_vista/bioclip_lora/best.pth` |
| **YOLO11s-cls** | CSPDarknet Backbone | 22 (early stopped) | **95.34%** | **99.45%** | `runs/classify_fish_vista/yolo11s_cls/weights/best.pt` |
| **MobileNetV4-Small** | Edge Inverted Residual CNN | 5 (early stopped) | **81.10%** | **98.90%** | `runs/classify_fish_vista/mobilenetv4_small/best.pth` |
| **ConvNeXt-Tiny** | Modern 7x7 Depthwise CNN | 31 (early stopped) | **80.27%** | **96.16%** | `runs/classify_fish_vista/convnext_tiny/best.pth` |

---

### 8.6 The Dual-Benchmark Landmark Discovery: Controlled vs. Wild Red Sea OOD
We assembled a dedicated Out-of-Distribution (OOD) test benchmark of **250 research-grade, in-situ underwater photos** from the Red Sea & coastal reefs (Saudi Arabia, Egypt, Jordan, Sudan) via iNaturalist (`datasets/evaluation_ood_redsea`) and evaluated all 7 architectures head-to-head against the controlled Fish-Vista holdout:

| Architecture | Paradigm | Controlled (Fish-Vista) | Real-World (Wild Red Sea) | Robustness Drop ($\Delta$) | GPU Latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BioCLIP (ViT-B) + LoRA** | Biological VFM (PEFT) | 96.44% (T5: 100.0%) | **51.20% (T5: 89.6%)** | **-45.24% (Champion)** | 7.9 ms |
| **DINOv2-Small + LoRA** | Vision Foundation Model (PEFT) | **98.08%** (T5: 99.7%) | 23.20% (T5: 73.2%) | -74.88% | **4.7 ms** |
| **Swin-Tiny** | Hierarchical ViT | 97.53% (T5: 99.7%) | 23.20% (T5: 61.6%) | -74.33% | 6.2 ms |
| **ResNet50d** | Classic Residual CNN | 97.81% (T5: 99.7%) | 17.20% (T5: 66.0%) | -80.61% | 6.7 ms |
| **YOLO11s-cls** | Edge Real-Time CNN | 95.26% (T5: 99.2%) | 14.00% (T5: 56.0%) | -81.26% | 160.1 ms |
| **ConvNeXt-Tiny** | Large-Kernel ConvNet | 80.27% (T5: 96.2%) | 12.00% (T5: 57.2%) | -68.27% | 5.0 ms |
| **MobileNetV4-Small** | Edge Optimized CNN | 81.10% (T5: 98.9%) | 7.60% (T5: 56.0%) | -73.50% | 2.9 ms |

#### Key Empirical Insights:
1. **The "Underwater Illusion":** On clean museum studio vouchers, traditional CNNs appear nearly solved (95%–98%). However, standard CNNs suffer a catastrophic ~80% collapse when plunged into the wild Red Sea due to water column red-light attenuation, caustics, and coral background camouflage.
2. **BioCLIP Dominance in the Wild:** BioCLIP + LoRA retained **51.20% Top-1 and 89.60% Top-5 accuracy** in wild conditions—surpassing ResNet50 by +34.0% absolute (3x higher) and YOLO by 3.6x. Its TreeOfLife-10M natural pre-training confers superior invariance to aquatic lighting and substrate clutter.
3. **Application Alignment:**
   - **Dr. Jessica Zamborain-Mason (Landed Catch):** ResNet50d (97.8%) and YOLO11s (95.3%) remain the champion edge architectures for flat measuring boards and market stalls.
   - **Dr. Renato Morais (In-Situ BRUVs & Reef Video):** BioCLIP + LoRA (89.6% Top-5) is the only viable candidate architecture for wild underwater video deployments.

**Completed Next Steps (Section 8.7):**

### 8.7 Site Integration & Dual-Crop Multi-Region Feature Extraction Operational
1. **Live Portal Classifier Suite:**
   - Integrated all 7 trained models into `app/taxonomic_engine.py` and the frontend UI (`app/static/index.html`):
     - **BioCLIP + LoRA (In-Situ Wild Champion — 89.6% T5 / 51.2% T1)** [Default]
     - **ResNet50d (Landed Catch Champion — 97.8% Top-1)**
     - **DINOv2-Small + LoRA (Foundation Model — 98.1% Top-1)**
     - **Scientific Ensemble (Weighted VFM + CNN Consensus)**
     - **Swin-Tiny Transformer (Window Attention — 97.5%)**
     - **YOLO11s-cls (Fast Edge Inference — 95.3%)**
     - **MobileNetV4-Small & ConvNeXt-Tiny (Edge Baselines)**
2. **Dual-Crop Multi-Region Trait Extraction (Fish-Vista CVPR 2025):**
   - Implemented `app/dual_crop_engine.py` and integrated it directly into `/api/segment`:
     - **Stream A (Global Letterbox):** Aspect-ratio preserved whole-organism view capturing length-to-depth ratio and overall counter-shading.
     - **Stream B1 (Cephalic ROI - Anterior 35%):** Diagnostic snout angle, eye diameter ratio, and mouth orientation.
     - **Stream B2 (Caudal Peduncle ROI - Posterior 25%):** Diagnostic caudal spine (scalpel marker), peduncle depth, and fin fork geometry.
     - **Multi-Region Anatomical Fusion:** Fuses regional logits ($0.55 \cdot P_{\text{global}} + 0.25 \cdot P_{\text{head}} + 0.20 \cdot P_{\text{caudal}}$) to resolve camouflage ambiguities.
3. **Interactive Visual Dashboard:**
   - Each segmented fish card in the UI displays real-time high-resolution base64 thumbnails of both the Cephalic (Anterior) and Caudal (Posterior) crops alongside regional family predictions and an anatomical consensus verification badge.
