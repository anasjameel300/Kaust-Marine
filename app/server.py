import os
import uuid
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from app.taxonomic_engine import taxonomic_engine
from app.biomass_engine import biomass_engine, CALIBRATION_PRESETS

app = FastAPI(title="KAUST Marine Fish Instance Segmentation & Stock Assessment Portal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).resolve().parent
FINETUNED_PATH = BASE_DIR.parent / "runs" / "segment" / "models" / "kaust_fish_seg_finetune_v1" / "weights" / "best.pt"
BASELINE_PATH = BASE_DIR.parent / "runs" / "segment" / "models" / "kaust_fish_seg_v2" / "weights" / "best.pt"
YOLOV8_PATH = BASE_DIR.parent / "runs" / "segment" / "models" / "kaust_yolov8s_seg_v1" / "weights" / "best.pt"

# Device selection: use GPU if available
device = 0 if torch.cuda.is_available() else "cpu"
loaded_models = {}

# 1. Baseline YOLO11s-seg
if BASELINE_PATH.exists():
    print(f"[Server] Loading production baseline model (YOLO11s): {BASELINE_PATH} on device={device}...")
    model_v2 = YOLO(str(BASELINE_PATH))
    loaded_models["v2"] = model_v2
    print(f"[Server] Baseline model loaded! Classes: {model_v2.names}")
else:
    model_v2 = None

# 2. Fine-Tuned YOLO11s-seg
if FINETUNED_PATH.exists():
    print(f"[Server] Loading fine-tuned model: {FINETUNED_PATH}...")
    model_finetuned = YOLO(str(FINETUNED_PATH))
    loaded_models["finetuned"] = model_finetuned
    print(f"[Server] Fine-tuned model loaded!")
else:
    model_finetuned = model_v2

# 3. Newly Trained YOLOv8s-seg (COCO Pretrained, 50 Epochs)
if YOLOV8_PATH.exists():
    print(f"[Server] Loading newly trained YOLOv8s-seg (COCO Pretrained): {YOLOV8_PATH}...")
    model_yolov8 = YOLO(str(YOLOV8_PATH))
    loaded_models["yolov8"] = model_yolov8
    print(f"[Server] YOLOv8s-seg model loaded successfully!")
else:
    model_yolov8 = model_v2

# Default model reference: Stage 1 Baseline (v2)
default_model = loaded_models.get("v2") or model_v2 or loaded_models.get("yolov8")

# Step 1: Geometry and Metric Helpers
def compute_box_metrics(b1, b2):
    # b: [x1, y1, x2, y2]
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    area1 = max(1e-5, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(1e-5, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    iou = interArea / float(area1 + area2 - interArea)
    containment = interArea / float(min(area1, area2))
    return iou, containment

def extract_candidates(result, offset_x=0, offset_y=0, source_model="baseline", scale_tag="macro") -> List[Dict[str, Any]]:
    candidates = []
    boxes = result.boxes
    masks = result.masks
    if boxes is None or masks is None or len(masks.data) == 0:
        return candidates

    tile_h, tile_w = result.orig_shape

    for i in range(len(boxes)):
        b = boxes[i]
        cls_id = int(b.cls[0].item())
        cls_name = result.names.get(cls_id, f"class_{cls_id}")
        conf = float(b.conf[0].item())

        bx = b.xyxy[0].cpu().numpy()
        box_coords = [
            float(bx[0] + offset_x),
            float(bx[1] + offset_y),
            float(bx[2] + offset_x),
            float(bx[3] + offset_y)
        ]

        # Extract binary mask tensor and resize to original tile resolution
        m_tensor = (masks.data[i].cpu().numpy() * 255).astype(np.uint8)
        if m_tensor.shape[:2] != (tile_h, tile_w):
            m_tensor = cv2.resize(m_tensor, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
            m_tensor = (m_tensor > 127).astype(np.uint8) * 255

        # RETR_EXTERNAL gets outer boundaries of isolated components
        contours, _ = cv2.findContours(m_tensor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        # STRICTLY select the largest connected component (discards small substrate noise specks and eliminates stray bridge lines)
        main_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(main_contour) < 25:
            continue

        # Gentle curvature-preserving polygon fit (epsilon = 0.75 preserves fine fin curves while eliminating redundant colinear points)
        approx = cv2.approxPolyDP(main_contour, 0.75, True)
        if len(approx) < 3:
            continue

        polygon_coords = [
            [round(float(pt[0][0] + offset_x), 1), round(float(pt[0][1] + offset_y), 1)]
            for pt in approx
        ]

        poly_np = np.array(polygon_coords, dtype=np.float32)
        area_px = float(cv2.contourArea(poly_np))
        if area_px < 20:
            continue

        rect = cv2.minAreaRect(poly_np)
        oriented_length = float(max(rect[1]))
        body_depth = float(min(rect[1]))

        candidates.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": conf,
            "box": box_coords,
            "polygon": polygon_coords,
            "oriented_length_px": round(oriented_length, 1),
            "body_depth_px": round(body_depth, 1),
            "area_px": round(area_px, 1),
            "source_model": source_model,
            "scale": scale_tag,
            "agreement": 1
        })
    return candidates

def generate_tiles(img_w: int, img_h: int, tile_size: int = 640, overlap: float = 0.25) -> List[Tuple[int, int, int, int]]:
    tiles = []
    step = int(tile_size * (1.0 - overlap))
    x_starts = list(range(0, max(1, img_w - tile_size + 1), step))
    if not x_starts or x_starts[-1] + tile_size < img_w:
        x_starts.append(max(0, img_w - tile_size))
    y_starts = list(range(0, max(1, img_h - tile_size + 1), step))
    if not y_starts or y_starts[-1] + tile_size < img_h:
        y_starts.append(max(0, img_h - tile_size))
    x_starts = sorted(list(set(x_starts)))
    y_starts = sorted(list(set(y_starts)))

    for y in y_starts:
        for x in x_starts:
            tiles.append((x, y, min(x + tile_size, img_w), min(y + tile_size, img_h)))
    return tiles

def merge_teacher_consensus(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    merged = []
    used = set()

    for i in range(len(candidates)):
        if i in used:
            continue
        c1 = candidates[i]
        match = None
        for j in range(i + 1, len(candidates)):
            if j in used:
                continue
            c2 = candidates[j]
            if c1["class_id"] != c2["class_id"]:
                continue
            iou, cont = compute_box_metrics(c1["box"], c2["box"])
            if iou >= 0.38 or cont >= 0.65:
                match = (j, c2)
                break

        if match is not None:
            j, c2 = match
            used.add(i)
            used.add(j)
            boosted_conf = min(0.99, max(c1["confidence"], c2["confidence"]) * 1.05 + 0.04)
            # Pick fine-tuned mask if present, otherwise higher confidence
            chosen = c1 if c1["source_model"] == "finetuned" else c2
            entry = dict(chosen)
            entry["confidence"] = round(boosted_conf, 3)
            entry["source"] = "consensus"
            entry["agreement"] = 2
            merged.append(entry)
        else:
            used.add(i)
            entry = dict(c1)
            entry["source"] = c1["source_model"]
            entry["agreement"] = 1
            merged.append(entry)

    return merged

def run_step1_inference(
    img: np.ndarray,
    model_version: str = "v2",
    conf_threshold: float = 0.35,
    enable_tiling: bool = False
) -> List[Dict[str, Any]]:
    img_h, img_w = img.shape[:2]

    # Exactly 3 Stage 1 Segmentation model backbones (Default: Stage 1 Baseline v2)
    if model_version == "finetuned" and "finetuned" in loaded_models:
        active_models = [("finetuned", loaded_models["finetuned"])]
    elif model_version == "yolov8" and "yolov8" in loaded_models:
        active_models = [("yolov8", loaded_models["yolov8"])]
    elif model_version == "v2" and "v2" in loaded_models:
        active_models = [("baseline", loaded_models["v2"])]
    else:
        active_models = [("baseline", default_model)]

    # 1. Macro Full-Frame Pass (Native 640 preserves whole large fish)
    macro_cands: List[Dict[str, Any]] = []
    for tag, mdl in active_models:
        res = mdl.predict(img, conf=conf_threshold, device=device, verbose=False)[0]
        macro_cands.extend(extract_candidates(res, 0, 0, source_model=tag, scale_tag="macro"))

    macro_merged = merge_teacher_consensus(macro_cands)

    # 2. Micro Tiling Pass (Optional: slices image into 640x640 sliding patches for tiny distant fish)
    micro_merged: List[Dict[str, Any]] = []
    if enable_tiling and (img_w > 700 or img_h > 700):
        tiles = generate_tiles(img_w, img_h, tile_size=640, overlap=0.25)
        raw_tile_cands = []
        for x1, y1, x2, y2 in tiles:
            tile_crop = img[y1:y2, x1:x2]
            for tag, mdl in active_models:
                res = mdl.predict(tile_crop, conf=conf_threshold, device=device, verbose=False)[0]
                raw_tile_cands.extend(extract_candidates(res, x1, y1, source_model=tag, scale_tag="micro"))
        
        micro_merged = merge_teacher_consensus(raw_tile_cands)

    # 3. Macro Priority Integration (Protects large fish from being split by tiles)
    final_pool: List[Dict[str, Any]] = list(macro_merged)

    for micro in micro_merged:
        overlaps_macro = False
        for macro in macro_merged:
            if macro["class_id"] != micro["class_id"]:
                continue
            iou, cont = compute_box_metrics(macro["box"], micro["box"])
            if iou > 0.35 or cont > 0.60:
                overlaps_macro = True
                break
        
        if not overlaps_macro:
            final_pool.append(micro)

    # 4. Containment Suppression (Eliminates orphan tails, needle fragments, and nested boxes)
    final_pool.sort(key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]), reverse=True)
    suppressed_pool: List[Dict[str, Any]] = []

    for cand in final_pool:
        is_contained = False
        for existing in suppressed_pool:
            if existing["class_id"] != cand["class_id"]:
                continue
            iou, cont = compute_box_metrics(existing["box"], cand["box"])
            # Small box heavily nested inside or touching large box (> 70% contained)
            if cont > 0.70:
                is_contained = True
                break
            # Standard NMS threshold to deduplicate overlaps
            if iou > 0.45:
                is_contained = True
                break
        if not is_contained:
            suppressed_pool.append(cand)

    for idx, d in enumerate(suppressed_pool):
        d["id"] = idx + 1

    return suppressed_pool


# Mount static folders
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/css", StaticFiles(directory=str(BASE_DIR / "static" / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(BASE_DIR / "static" / "js")), name="js")
app.mount("/samples", StaticFiles(directory=str(BASE_DIR / "samples")), name="samples")
app.mount("/uploads", StaticFiles(directory=str(BASE_DIR / "uploads")), name="uploads")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = BASE_DIR / "static" / "index.html"
    with open(index_file, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/samples")
async def get_samples():
    samples_dir = BASE_DIR / "samples"
    files = sorted([f for f in os.listdir(samples_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    # Include mtime query string so the user's browser immediately invalidates stale cache
    return [
        {
            "filename": f,
            "url": f"/samples/{f}?v={int(os.path.getmtime(samples_dir / f))}"
        }
        for f in files
    ]

@app.get("/api/calibration_presets")
async def get_calibration_presets():
    return CALIBRATION_PRESETS

@app.post("/api/segment")
async def run_segmentation(
    file: UploadFile = File(None),
    sample_name: str = Form(None),
    conf_threshold: float = Form(0.35),
    model_version: str = Form("v2"),
    enable_tiling: bool = Form(False),
    run_classification: bool = Form(False),
    classifier_mode: str = Form("bioclip"),
    enable_dual_crop: bool = Form(True),
    calibration_preset: str = Form("custom"),
    pixels_per_cm: float = Form(25.0),
    is_calibrated: bool = Form(False)
):
    try:
        # Load image from upload or preset
        if file and file.filename:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="Could not decode uploaded image")
            
            save_name = f"upload_{uuid.uuid4().hex[:8]}.jpg"
            save_path = BASE_DIR / "uploads" / save_name
            cv2.imwrite(str(save_path), img)
            image_url = f"/uploads/{save_name}"
        elif sample_name:
            sample_path = BASE_DIR / "samples" / sample_name
            if not sample_path.exists():
                raise HTTPException(status_code=404, detail="Sample image not found")
            img = cv2.imread(str(sample_path))
            image_url = f"/samples/{sample_name}"
        else:
            raise HTTPException(status_code=400, detail="No image provided")

        img_h, img_w = img.shape[:2]

        # Run Step 1 Instance Segmentation Engine
        detections = run_step1_inference(
            img=img,
            model_version=model_version,
            conf_threshold=conf_threshold,
            enable_tiling=enable_tiling
        )

        counts = {"fish": 0, "reefs": 0, "human_divers": 0}

        # Step 2: Multi-Model Taxonomic Classification (On-Demand) & Step 3: Biomass Engine
        for d in detections:
            cname = d["class_name"]
            counts[cname] = counts.get(cname, 0) + 1

            if cname == "fish":
                if run_classification:
                    bx = d["box"]
                    x1 = max(0, int(bx[0]))
                    y1 = max(0, int(bx[1]))
                    x2 = min(img_w, int(bx[2]))
                    y2 = min(img_h, int(bx[3]))

                    # Crop fish organism for classification with dual-crop multi-region extraction
                    crop = img[y1:y2, x1:x2]
                    tax_res = taxonomic_engine.classify_crop(
                        crop_bgr=crop,
                        mode=classifier_mode,
                        full_img_bgr=img,
                        box=bx,
                        enable_dual_crop=enable_dual_crop
                    )
                    d["taxonomy"] = tax_res
                    fb_a = tax_res["fb_a"]
                    fb_b = tax_res["fb_b"]
                else:
                    # On initial upload: fast segmentation without heavy classification
                    d["taxonomy"] = None
                    fb_a = 0.0200
                    fb_b = 3.00

                # Compute individual biomass morphometrics with explicit is_calibrated gate
                bio_res = biomass_engine.calculate_individual_morphometrics(
                    oriented_length_px=d.get("oriented_length_px", 10.0),
                    body_depth_px=d.get("body_depth_px", 5.0),
                    pixels_per_cm=pixels_per_cm,
                    fb_a=fb_a,
                    fb_b=fb_b,
                    is_calibrated=is_calibrated,
                    is_taxonomic_classified=run_classification
                )
                d["biomass"] = bio_res
            else:
                d["taxonomy"] = None
                d["biomass"] = None

        # Compute Community Ecological Metrics
        community_metrics = biomass_engine.calculate_community_metrics(detections, is_calibrated=is_calibrated)

        if model_version == "yolov8":
            model_title = "YOLOv8s-seg (COCO Pretrained - 50 Epochs)"
        elif model_version == "finetuned":
            model_title = "Stage 2 Fine-Tuned (YOLO11s, 73.4% Fish Precision)"
        else:
            model_title = "Stage 1 Baseline (YOLO11s, 63.3% Mask mAP50)"

        return JSONResponse({
            "image_url": image_url,
            "image_width": img_w,
            "image_height": img_h,
            "model_version": model_version,
            "model_title": model_title,
            "run_classification": run_classification,
            "classifier_mode": classifier_mode,
            "enable_dual_crop": enable_dual_crop,
            "calibration_preset": calibration_preset,
            "pixels_per_cm": pixels_per_cm,
            "is_calibrated": is_calibrated,
            "enable_tiling": enable_tiling,
            "total_detections": len(detections),
            "counts": counts,
            "community_metrics": community_metrics,
            "detections": detections
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=True)

