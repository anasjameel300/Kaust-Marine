import os
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Any

# Deterministic seed for reproducible train/val split
random.seed(42)

# Source and destination paths
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = WORKSPACE_DIR / "datasets" / "Underwater Instance Segmentation.v3i.coco-segmentation"
OUTPUT_DIR = WORKSPACE_DIR / "datasets" / "underwater_yolo_seg"

# Clean 3-Class Mapping: Focus purely on Fish, Reefs, and Human Divers
# All other noise classes (robots, wrecks, sea-floor, plants) are treated as natural background
CLASS_MAP = {
    2: 0,  # 'fish' -> Class 0 (Primary Target)
    4: 1,  # 'reefs' -> Class 1 (Prevents coral confusion and boundary bleeding)
    3: 2,  # 'human divers' -> Class 2 (Prevents human diver false positives)
}

CLASS_NAMES = [
    "fish",
    "reefs",
    "human_divers"
]

def load_coco_annotations(json_path: Path) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def convert_dataset():
    print(f"[1/4] Preparing output directory: {OUTPUT_DIR}")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "valid"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Collect all image records across source splits
    all_images = []
    
    for split_folder in ["train", "valid"]:
        split_path = SOURCE_DIR / split_folder
        json_file = split_path / "_annotations.coco.json"
        if not json_file.exists():
            continue

        print(f"[2/4] Reading {split_folder} annotations from {json_file.name}...")
        data = load_coco_annotations(json_file)

        # Index annotations by image_id
        img_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            img_to_anns.setdefault(img_id, []).append(ann)

        for img in data.get("images", []):
            src_img_file = split_path / img["file_name"]
            if src_img_file.exists():
                all_images.append({
                    "src_path": src_img_file,
                    "file_name": img["file_name"],
                    "width": img["width"],
                    "height": img["height"],
                    "annotations": img_to_anns.get(img["id"], [])
                })

    print(f"Total valid images collected: {len(all_images)}")

    # Deterministic 90% train / 10% validation split
    random.shuffle(all_images)
    val_size = max(50, int(len(all_images) * 0.10))
    val_images = all_images[:val_size]
    train_images = all_images[val_size:]

    print(f"Split: {len(train_images)} training images, {len(val_images)} validation images.")

    stats = {cls_name: 0 for cls_name in CLASS_NAMES}
    total_polygons = 0

    print("[3/4] Converting segmentation polygons with 6-decimal sub-pixel precision...")
    for split_name, img_list in [("train", train_images), ("valid", val_images)]:
        img_out_dir = OUTPUT_DIR / split_name / "images"
        lbl_out_dir = OUTPUT_DIR / split_name / "labels"

        for item in img_list:
            src_path = item["src_path"]
            w = float(item["width"])
            h = float(item["height"])
            base_stem = Path(item["file_name"]).stem
            
            # Destination image copy
            dest_img = img_out_dir / item["file_name"]
            if not dest_img.exists():
                shutil.copy2(src_path, dest_img)

            # Destination label file
            dest_lbl = lbl_out_dir / f"{base_stem}.txt"
            label_lines = []

            for ann in item["annotations"]:
                cat_id = ann.get("category_id")
                if cat_id not in CLASS_MAP:
                    continue

                yolo_cls_id = CLASS_MAP[cat_id]
                cls_name = CLASS_NAMES[yolo_cls_id]
                segmentations = ann.get("segmentation", [])

                if isinstance(segmentations, list):
                    for poly in segmentations:
                        if len(poly) < 6:
                            continue

                        # Normalize coordinates with 6 decimal places (lossless sub-pixel accuracy)
                        norm_coords = []
                        for i in range(0, len(poly), 2):
                            px = float(poly[i])
                            py = float(poly[i + 1])
                            norm_x = round(max(0.0, min(1.0, px / w)), 6)
                            norm_y = round(max(0.0, min(1.0, py / h)), 6)
                            norm_coords.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])

                        line = f"{yolo_cls_id} " + " ".join(norm_coords)
                        label_lines.append(line)
                        stats[cls_name] += 1
                        total_polygons += 1

            # Write label file (empty files act as true negative background images)
            with open(dest_lbl, "w", encoding="utf-8") as lf:
                if label_lines:
                    lf.write("\n".join(label_lines) + "\n")

    # Generate clean dataset.yaml
    print("[4/4] Writing dataset.yaml...")
    yaml_content = f"""# Ultralytics YOLO-Seg Clean 3-Class Configuration
path: {OUTPUT_DIR.as_posix()}
train: train/images
val: valid/images

names:
  0: fish
  1: reefs
  2: human_divers
"""
    yaml_file = OUTPUT_DIR / "dataset.yaml"
    with open(yaml_file, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    print("\n" + "=" * 60)
    print("CLEAN 3-CLASS DATASET READY")
    print("=" * 60)
    print(f"Total images: {len(all_images)}")
    print(f"Total polygon masks converted: {total_polygons}")
    for name, count in stats.items():
        print(f"  {name:14}: {count} masks")
    print(f"Dataset configuration saved to: {yaml_file}")

if __name__ == "__main__":
    convert_dataset()
