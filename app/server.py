import os
import uuid
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

app = FastAPI(title="KAUST Marine Fish Instance Segmentation Test Portal")

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

# Default to the newly fine-tuned model (73.4% precision)
WEIGHTS_PATH = FINETUNED_PATH if FINETUNED_PATH.exists() else BASELINE_PATH

if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"Model weights not found at {WEIGHTS_PATH}")

# Device selection: use GPU if available
device = 0 if torch.cuda.is_available() else "cpu"
print(f"[Server] Loading fine-tuned model: {WEIGHTS_PATH} on device={device}...")
model = YOLO(str(WEIGHTS_PATH))
print(f"[Server] Model loaded successfully! Classes: {model.names}")

# Cache loaded models for quick switching
loaded_models = {"finetuned": model}
if BASELINE_PATH.exists():
    loaded_models["v2"] = YOLO(str(BASELINE_PATH))


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
    return [{"filename": f, "url": f"/samples/{f}"} for f in files]

@app.post("/api/segment")
async def run_segmentation(
    file: UploadFile = File(None),
    sample_name: str = Form(None),
    conf_threshold: float = Form(0.35),
    model_version: str = Form("finetuned")
):
    try:
        # Select active model
        active_model = loaded_models.get(model_version, model)

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

        # Run inference using the active YOLO11s-seg model
        results = active_model.predict(img, conf=conf_threshold, device=device, verbose=False)[0]


        detections: List[Dict[str, Any]] = []
        counts = {"fish": 0, "reefs": 0, "human_divers": 0}

        boxes = results.boxes
        masks = results.masks

        if boxes is not None and masks is not None and len(masks.xy) > 0:
            for i in range(len(boxes)):
                b = boxes[i]
                cls_id = int(b.cls[0].item())
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                conf = float(b.conf[0].item())
                counts[cls_name] = counts.get(cls_name, 0) + 1

                # Raw polygon vertices
                poly = masks.xy[i]
                if len(poly) < 3:
                    continue

                poly_np = np.array(poly, dtype=np.float32)

                # Smooth / simplify polygon
                perimeter = cv2.arcLength(poly_np.reshape((-1, 1, 2)), True)
                epsilon = max(0.8, 0.006 * perimeter)
                approx = cv2.approxPolyDP(poly_np.reshape((-1, 1, 2)), epsilon, True)
                polygon_coords = [[round(float(pt[0][0]), 1), round(float(pt[0][1]), 1)] for pt in approx]

                # Oriented dimensions (snout-to-tail fork length & body depth)
                rect = cv2.minAreaRect(poly_np)
                oriented_length = float(max(rect[1]))
                body_depth = float(min(rect[1]))
                area_px = float(cv2.contourArea(poly_np))

                detections.append({
                    "id": i + 1,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 3),
                    "oriented_length_px": round(oriented_length, 1),
                    "body_depth_px": round(body_depth, 1),
                    "area_px": round(area_px, 1),
                    "polygon": polygon_coords
                })

        return JSONResponse({
            "image_url": image_url,
            "image_width": img_w,
            "image_height": img_h,
            "model_version": model_version,
            "model_title": "Stage 2 Fine-Tuned (73.4% Fish Precision)" if model_version == "finetuned" else "Stage 1 Baseline (63.3% mAP50)",
            "total_detections": len(detections),
            "counts": counts,
            "detections": detections
        })


    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=True)
