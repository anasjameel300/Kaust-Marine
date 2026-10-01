"""
BRUVs Video Survey & MaxN Ecological Processing Engine
KAUST Marine Vision - Automated Video Monitoring & Tracking

Integrates:
1. YOLO11s-seg instance segmentation + ByteTrack multi-object tracking (MOT)
2. Temporal sub-sampling (2-4 FPS) for fast GPU processing
3. MaxN calculation (maximum simultaneous individuals in any single frame)
4. Track-level taxonomic classification & FishBase allometric biomass
5. Annotated video rendering & peak-frame snapshot generation
"""

import os
import uuid
import math
import shutil
import subprocess
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from ultralytics import YOLO

from app.taxonomic_engine import taxonomic_engine
from app.biomass_engine import biomass_engine

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
BASELINE_WEIGHTS = PROJECT_ROOT / "runs" / "segment" / "models" / "kaust_fish_seg_v2" / "weights" / "best.pt"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

device = 0 if torch.cuda.is_available() else "cpu"

class VideoProcessingEngine:
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or BASELINE_WEIGHTS
        self.model = None
        self._load_model()

    def _load_model(self):
        if self.model_path.exists():
            print(f"[VideoEngine] Loading YOLO11s-seg tracking model from {self.model_path} on device={device}...")
            self.model = YOLO(str(self.model_path))
            print("[VideoEngine] Tracking model loaded successfully!")
        else:
            print(f"[VideoEngine] Warning: Model weights not found at {self.model_path}")
            self.model = None

    def process_video(
        self,
        video_path: str,
        sample_fps: float = 3.0,
        max_duration_sec: float = 45.0,
        conf_threshold: float = 0.35,
        is_calibrated: bool = False,
        pixels_per_cm: float = 22.0,
        run_classification: bool = True,
        classifier_mode: str = "bioclip"
    ) -> Dict[str, Any]:
        """
        Processes a video file with ByteTrack tracking and MaxN computation.
        """
        v_path = Path(video_path)
        if not v_path.exists():
            if (BASE_DIR / "samples" / video_path).exists():
                v_path = BASE_DIR / "samples" / video_path
            elif (BASE_DIR / "uploads" / video_path).exists():
                v_path = BASE_DIR / "uploads" / video_path
            elif (PROJECT_ROOT / video_path).exists():
                v_path = PROJECT_ROOT / video_path
            else:
                raise FileNotFoundError(f"Video file not found: {video_path}")
        video_path = str(v_path)

        if self.model is None:
            self._load_model()
            if self.model is None:
                raise RuntimeError("YOLO11s-seg model weights could not be loaded")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_duration = total_frames / max(1.0, orig_fps)

        # Frame step for temporal sub-sampling
        target_sample_fps = max(1.0, float(sample_fps))
        frame_step = max(1, int(round(orig_fps / target_sample_fps)))
        effective_fps = orig_fps / frame_step

        # Max frames to process
        max_frames_to_process = int(round(min(total_duration, float(max_duration_sec)) * orig_fps))

        # Output video writer setup
        unique_id = uuid.uuid4().hex[:8]
        out_video_name = f"annotated_bruvs_{unique_id}.mp4"
        out_video_path = UPLOADS_DIR / out_video_name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(
            str(out_video_path),
            fourcc,
            effective_fps,
            (orig_width, orig_height)
        )

        # Tracking registries
        tracks_registry: Dict[int, Dict[str, Any]] = {}
        timeline: List[Dict[str, Any]] = []

        max_n_total = 0
        peak_timestamp = "00:00"
        peak_second = 0.0
        peak_frame_img: Optional[np.ndarray] = None
        peak_frame_active_track_ids: List[int] = []

        frame_idx = 0
        processed_count = 0

        # Palette: Clean scientific emerald for fish
        poly_color = (16, 185, 129)     # BGR: (green)
        poly_fill = (16, 185, 129)

        while cap.isOpened() and frame_idx < max_frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            current_sec = frame_idx / orig_fps
            m = int(current_sec // 60)
            s = current_sec % 60
            time_str = f"{m:02d}:{s:04.1f}"

            # Run YOLO11s-seg tracking with ByteTrack
            results = self.model.track(
                frame,
                tracker="bytetrack.yaml",
                persist=True,
                conf=conf_threshold,
                device=device,
                verbose=False
            )[0]

            annotated_frame = frame.copy()
            overlay = frame.copy()

            boxes = results.boxes
            masks = results.masks

            simultaneous_fish_in_frame = 0
            current_frame_track_ids = []

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    b = boxes[i]
                    cls_id = int(b.cls[0].item())
                    cls_name = results.names.get(cls_id, f"class_{cls_id}")
                    conf = float(b.conf[0].item())

                    # Filter only fish organisms for MaxN
                    if cls_name != "fish":
                        continue

                    # Extract track ID (ByteTrack assigns persistent IDs)
                    track_id = int(b.id[0].item()) if (b.id is not None and len(b.id) > 0) else (i + 1)
                    current_frame_track_ids.append(track_id)
                    simultaneous_fish_in_frame += 1

                    bx = b.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = max(0, int(bx[0])), max(0, int(bx[1])), min(orig_width, int(bx[2])), min(orig_height, int(bx[3]))

                    # Extract contour polygon if mask is present
                    poly_pts = None
                    oriented_len = float(max(1.0, x2 - x1))
                    oriented_dep = float(max(1.0, y2 - y1))

                    if masks is not None and i < len(masks.data):
                        m_pts = masks.xy[i]
                        if len(m_pts) >= 3:
                            poly_pts = np.array(m_pts, dtype=np.int32)
                            cv2.fillPoly(overlay, [poly_pts], poly_fill)
                            cv2.polylines(annotated_frame, [poly_pts], True, poly_color, 2)

                            # Calculate oriented minimum bounding rectangle
                            rect = cv2.minAreaRect(poly_pts)
                            w_r, h_r = rect[1]
                            oriented_len = max(w_r, h_r)
                            oriented_dep = min(w_r, h_r)

                    if poly_pts is None:
                        # Fallback to bounding box if polygon mask is degenerate
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), poly_color, 2)

                    # Update Track Registry
                    crop = frame[y1:y2, x1:x2].copy() if (y2 > y1 and x2 > x1) else None

                    if track_id not in tracks_registry:
                        tracks_registry[track_id] = {
                            "track_id": track_id,
                            "first_seen_sec": round(current_sec, 2),
                            "last_seen_sec": round(current_sec, 2),
                            "frame_appearances": 1,
                            "max_conf": conf,
                            "best_crop": crop,
                            "lengths_px": [oriented_len],
                            "depths_px": [oriented_dep],
                            "crop_box": [x1, y1, x2, y2]
                        }
                    else:
                        t = tracks_registry[track_id]
                        t["last_seen_sec"] = round(current_sec, 2)
                        t["frame_appearances"] += 1
                        t["lengths_px"].append(oriented_len)
                        t["depths_px"].append(oriented_dep)
                        if conf > t["max_conf"] and crop is not None and crop.size > 0:
                            t["max_conf"] = conf
                            t["best_crop"] = crop
                            t["crop_box"] = [x1, y1, x2, y2]

                    # Draw neat tracking badge
                    label_text = f"Fish #{track_id} ({int(conf * 100)}%)"
                    badge_x = x1
                    badge_y = max(16, y1 - 6)
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(annotated_frame, (badge_x, badge_y - th - 3), (badge_x + tw + 4, badge_y + 3), (15, 23, 42), -1)
                    cv2.rectangle(annotated_frame, (badge_x, badge_y - th - 3), (badge_x + tw + 4, badge_y + 3), poly_color, 1)
                    cv2.putText(annotated_frame, label_text, (badge_x + 2, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (248, 250, 252), 1, cv2.LINE_AA)

            # Blend transparent polygon masks (18% opacity)
            cv2.addWeighted(overlay, 0.18, annotated_frame, 0.82, 0, annotated_frame)

            # Draw Timestamp & Simultaneous Count Overlay
            hud_text = f"Time: {time_str} | In-Frame: {simultaneous_fish_in_frame} | MaxN: {max_n_total}"
            cv2.rectangle(annotated_frame, (8, 8), (320, 32), (15, 23, 42), -1)
            cv2.rectangle(annotated_frame, (8, 8), (320, 32), (56, 189, 248), 1)
            cv2.putText(annotated_frame, hud_text, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 189, 248), 1, cv2.LINE_AA)

            out_writer.write(annotated_frame)

            # Record timeline entry
            timeline.append({
                "second": round(current_sec, 1),
                "timestamp": time_str,
                "count": simultaneous_fish_in_frame
            })

            # Check for Peak Frame (MaxN)
            if simultaneous_fish_in_frame > max_n_total:
                max_n_total = simultaneous_fish_in_frame
                peak_timestamp = time_str
                peak_second = round(current_sec, 1)
                peak_frame_img = annotated_frame.copy()
                peak_frame_active_track_ids = list(current_frame_track_ids)

            frame_idx += 1
            processed_count += 1

        cap.release()
        out_writer.release()

        # Step 1.5: Transcode with FFmpeg to H.264 (avc1 / yuv420p) for universal browser playback
        final_video_name = out_video_name
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            fallback_candidates = [
                Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
            ]
            for fc in fallback_candidates:
                if fc.exists():
                    found = list(fc.glob("**/bin/ffmpeg.exe"))
                    if found:
                        ffmpeg_bin = str(found[0])
                        break

        if ffmpeg_bin:
            web_video_name = f"web_{out_video_name}"
            web_video_path = UPLOADS_DIR / web_video_name
            try:
                cmd = [
                    ffmpeg_bin, "-y",
                    "-i", str(out_video_path),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "veryfast",
                    "-movflags", "+faststart",
                    str(web_video_path)
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                if web_video_path.exists() and web_video_path.stat().st_size > 0:
                    final_video_name = web_video_name
            except Exception as e:
                print(f"[VideoEngine] FFmpeg H.264 transcode fallback: {e}")

        # If peak_frame_img wasn't set (e.g. 0 fish), take last frame
        if peak_frame_img is None:
            peak_frame_img = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)

        # Save peak frame keyframe snapshot
        peak_frame_name = f"peak_frame_{unique_id}.jpg"
        peak_frame_path = UPLOADS_DIR / peak_frame_name
        cv2.imwrite(str(peak_frame_path), peak_frame_img)

        # Step 2: Track-Level Taxonomic Classification & Biomass Computation
        unique_tracks_list: List[Dict[str, Any]] = []
        family_breakdown: Dict[str, int] = {}

        for tid, tinfo in tracks_registry.items():
            crop = tinfo.get("best_crop")
            med_len_px = float(np.median(tinfo["lengths_px"])) if tinfo["lengths_px"] else 50.0
            med_dep_px = float(np.median(tinfo["depths_px"])) if tinfo["depths_px"] else 20.0

            tax_res = None
            pred_family = "Reef Fish (Unclassified)"
            common_name = ""
            scientific_name = ""
            fb_a = 0.0200
            fb_b = 3.00
            is_comm = False

            if run_classification and crop is not None and crop.size > 0:
                try:
                    tax_res = taxonomic_engine.classify_crop(
                        crop_bgr=crop,
                        mode=classifier_mode,
                        enable_dual_crop=False
                    )
                    pred_family = tax_res.get("family", "Reef Fish")
                    common_name = tax_res.get("common_name", "")
                    scientific_name = tax_res.get("scientific_name", "")
                    fb_a = tax_res.get("fb_a", 0.0200)
                    fb_b = tax_res.get("fb_b", 3.00)
                    is_comm = tax_res.get("is_commercial", False)
                except Exception as e:
                    print(f"[VideoEngine] Track #{tid} taxonomic classification failed: {e}")

            # Compute individual track biomass
            bio_res = biomass_engine.calculate_individual_morphometrics(
                oriented_length_px=med_len_px,
                body_depth_px=med_dep_px,
                pixels_per_cm=pixels_per_cm,
                fb_a=fb_a,
                fb_b=fb_b,
                is_calibrated=is_calibrated,
                is_taxonomic_classified=bool(tax_res is not None)
            )

            # Record family breakdown for MaxN active tracks
            if tid in peak_frame_active_track_ids:
                family_breakdown[pred_family] = family_breakdown.get(pred_family, 0) + 1

            unique_tracks_list.append({
                "track_id": tid,
                "first_seen": f"{int(tinfo['first_seen_sec']//60):02d}:{tinfo['first_seen_sec']%60:04.1f}",
                "last_seen": f"{int(tinfo['last_seen_sec']//60):02d}:{tinfo['last_seen_sec']%60:04.1f}",
                "duration_sec": round(tinfo["last_seen_sec"] - tinfo["first_seen_sec"], 1),
                "frame_appearances": tinfo["frame_appearances"],
                "family": pred_family,
                "common_name": common_name,
                "scientific_name": scientific_name,
                "is_commercial": is_comm,
                "confidence": round(float(tinfo["max_conf"]), 2),
                "biomass": bio_res,
                "in_peak_frame": bool(tid in peak_frame_active_track_ids)
            })

        # If family breakdown is empty but max_n > 0, provide unclassified count
        if not family_breakdown and max_n_total > 0:
            family_breakdown["Reef Fish (Unclassified)"] = max_n_total

        return {
            "status": "success",
            "video_url": f"/uploads/{final_video_name}",
            "peak_frame_url": f"/uploads/{peak_frame_name}",
            "max_n_total": max_n_total,
            "max_n_by_family": family_breakdown,
            "peak_timestamp": peak_timestamp,
            "peak_second": peak_second,
            "duration_analyzed_sec": round(min(total_duration, float(max_duration_sec)), 1),
            "total_video_duration_sec": round(total_duration, 1),
            "frames_analyzed": processed_count,
            "sample_fps": target_sample_fps,
            "unique_tracks_count": len(tracks_registry),
            "tracks": sorted(unique_tracks_list, key=lambda x: x["track_id"]),
            "timeline": timeline,
            "is_calibrated": is_calibrated,
            "pixels_per_cm": pixels_per_cm
        }

# Global singleton
video_engine = VideoProcessingEngine()
