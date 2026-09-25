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
