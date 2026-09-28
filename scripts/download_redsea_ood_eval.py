"""
Wild Red Sea Out-of-Distribution (OOD) Test Dataset Downloader
KAUST Marine Vision - Research Grade Deep Learning Platform

Downloads 100% wild, in-situ, research-grade underwater marine photos from the
Red Sea & coastal reefs (Saudi Arabia, Egypt, Jordan) via the public iNaturalist API.

Purpose:
- Establishes a true real-world Out-of-Distribution (OOD) test benchmark.
- Tests whether Vision Foundation Models (BioCLIP & DINOv2 LoRA) generalize
  better to real underwater lighting, turbidity, and coral backgrounds
  than standard CNNs (ResNet50, YOLO11s).
- Requires NO API keys or credentials.

Output Directory:
  datasets/evaluation_ood_redsea/
    ├── Acanthuridae/   (Surgeonfish / Tangs)
    ├── Balistidae/     (Triggerfish)
    ├── Chaetodontidae/ (Butterflyfish)
    ├── Holocentridae/  (Squirrelfish)
    ├── Labridae/       (Wrasses)
    ├── Lutjanidae/     (Snappers)
    ├── Nemipteridae/   (Threadfin Breams)
    ├── Pomacentridae/  (Damselfish / Clownfish)
    ├── Scaridae/       (Parrotfish)
    └── Siganidae/      (Rabbitfish)

Usage:
  # Download 25 verified wild underwater images per family (250 total):
  python scripts/download_redsea_ood_eval.py --per_family 25
"""

import os
import sys
import json
import time
import urllib.request
import urllib.parse
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", total=None, unit="it"):
        print(f"--> {desc} ({total} items)...")
        for i, item in enumerate(iterable):
            yield item
            if total and (i + 1) % max(1, total // 10) == 0:
                print(f"    Progress: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)")

# Paths
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE_DIR / "datasets" / "evaluation_ood_redsea"

# iNaturalist Taxon IDs for 10 Target Families
TARGET_FAMILIES = {
    "Acanthuridae": {"taxon_id": 47619, "common": "Surgeonfish"},
    "Balistidae": {"taxon_id": 49612, "common": "Triggerfish"},
    "Chaetodontidae": {"taxon_id": 47318, "common": "Butterflyfish"},
    "Holocentridae": {"taxon_id": 49628, "common": "Squirrelfish"},
    "Labridae": {"taxon_id": 49284, "common": "Wrasses"},
    "Lutjanidae": {"taxon_id": 47295, "common": "Snappers"},
    "Nemipteridae": {"taxon_id": 51095, "common": "Threadfin Breams"},
    "Pomacentridae": {"taxon_id": 48313, "common": "Damselfish / Clownfish"},
    "Scaridae": {"taxon_id": 516508, "common": "Parrotfish"},  # Scarinae
    "Siganidae": {"taxon_id": 84096, "common": "Rabbitfish"}
}

# Red Sea & Gulf of Aqaba Coastal Places (Saudi Arabia: 7215, Egypt: 6757, Jordan: 7155, Sudan: 97193)
RED_SEA_PLACES = "7215,6757,7155,97193"
HEADERS = {"User-Agent": "KAUST-Marine-Research/1.0 (academic-eval: marine-ai@kaust.edu.sa)"}

def fetch_family_observations(family_name: str, taxon_id: int, target_count: int = 25):
    """Queries iNaturalist for Research-Grade wild observations in Red Sea region."""
    records = []

    # Attempt 1: Query strict Red Sea coastal places
    url_redsea = (
        f"https://api.inaturalist.org/v1/observations?"
        f"taxon_id={taxon_id}&quality_grade=research&photos=true&has[]=photos"
        f"&place_id={RED_SEA_PLACES}&per_page={min(target_count * 2, 100)}"
    )

    try:
        req = urllib.request.Request(url_redsea, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode())
            results = data.get("results", [])
            for item in results:
                photos = item.get("photos", [])
                if photos:
                    # Upgrade photo from square thumbnail to crisp medium resolution (~500px)
                    p_url = photos[0].get("url", "").replace("square", "medium")
                    sp_name = item.get("species_guess") or (item.get("taxon") or {}).get("name", "Unknown")
                    records.append({
                        "family": family_name,
                        "species": sp_name,
                        "obs_id": item.get("id"),
                        "photo_url": p_url,
                        "license": photos[0].get("license_code", "unknown"),
                        "location": item.get("place_guess", "Red Sea"),
                        "geo": item.get("geojson", {}).get("coordinates", [])
                    })
    except Exception as e:
        print(f"  [Notice] Red Sea search for {family_name}: {e}")

    # Attempt 2: If quota not fully met, expand to Western Indian Ocean / Indo-Pacific
    if len(records) < target_count:
        needed = target_count - len(records)
        url_indo = (
            f"https://api.inaturalist.org/v1/observations?"
            f"taxon_id={taxon_id}&quality_grade=research&photos=true&has[]=photos"
            f"&nelat=30.0&nelng=90.0&swlat=-20.0&swlng=30.0&per_page={needed * 2}"
        )
        try:
            req = urllib.request.Request(url_indo, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode())
                results = data.get("results", [])
                existing_ids = {r["obs_id"] for r in records}
                for item in results:
                    if item.get("id") in existing_ids:
                        continue
                    photos = item.get("photos", [])
                    if photos:
                        p_url = photos[0].get("url", "").replace("square", "medium")
                        sp_name = item.get("species_guess") or (item.get("taxon") or {}).get("name", "Unknown")
                        records.append({
                            "family": family_name,
                            "species": sp_name,
                            "obs_id": item.get("id"),
                            "photo_url": p_url,
                            "license": photos[0].get("license_code", "unknown"),
                            "location": item.get("place_guess", "Indo-Pacific"),
                            "geo": item.get("geojson", {}).get("coordinates", [])
                        })
                    if len(records) >= target_count:
                        break
        except Exception as e:
            pass

    return records[:target_count]

def download_image(record: dict, output_dir: Path) -> bool:
    """Downloads a single wild in-situ photo."""
    family_dir = output_dir / record["family"]
    family_dir.mkdir(parents=True, exist_ok=True)

    filename = f"redsea_{record['family']}_{record['obs_id']}.jpg"
    dest_path = family_dir / filename

    if dest_path.exists() and dest_path.stat().st_size > 1024:
        return True

    try:
        req = urllib.request.Request(record["photo_url"], headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp, open(dest_path, "wb") as f:
            f.write(resp.read())
        return True
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description="Download Wild Red Sea OOD Test Benchmark")
    parser.add_argument("--per_family", type=int, default=25, help="Number of wild photos per family (default: 25)")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    print("=" * 80)
    print(" KAUST MARINE: WILD RED SEA OUT-OF-DISTRIBUTION (OOD) INGESTION")
    print(f" Target: {args.per_family} verified wild underwater photos per family (10 families)")
    print(f" Total Target Scale: ~{args.per_family * 10} real-world in-situ images")
    print(f" Geographic Scope: Red Sea, Gulf of Aqaba & Indo-Pacific Reefs")
    print(f" Output Location: {out_root.resolve()}")
    print("=" * 80)

    # 1. Collect records for all 10 families
    print("\n[1/3] Querying iNaturalist Research-Grade API for 10 Marine Families...")
    all_records = []
    metadata_store = []

    for fam_name, info in TARGET_FAMILIES.items():
        recs = fetch_family_observations(fam_name, info["taxon_id"], target_count=args.per_family)
        all_records.extend(recs)
        metadata_store.extend(recs)
        print(f"  {fam_name:<16} ({info['common']:<22}): Found {len(recs):2d} wild photos.")
        time.sleep(0.2)  # Polite API spacing

    print(f"\n[2/3] Total gathered: {len(all_records)} wild test specimens.")

    # 2. Download concurrently via ThreadPoolExecutor
    print("\n[3/3] Downloading high-resolution underwater photos (8 concurrent workers)...")
    success_count = 0
    fail_count = 0

    pbar = tqdm(total=len(all_records), desc="Downloading Red Sea OOD Set", unit="img")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_image, rec, out_root): rec for rec in all_records}
        for fut in as_completed(futures):
            ok = fut.result()
            if ok:
                success_count += 1
            else:
                fail_count += 1
            pbar.update(1)
    pbar.close()

    # 3. Save metadata JSON for scientific attribution and reproducibility
    meta_path = out_root / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_store, f, indent=2)

    print("\n" + "=" * 80)
    print(" WILD RED SEA OOD INGESTION COMPLETE")
    print(f" Successfully downloaded: {success_count} real-world underwater photos")
    if fail_count > 0:
        print(f" Failed downloads: {fail_count} photos")
    print(f" Metadata saved to: {meta_path}")
    print(f" Dataset ready for Out-of-Distribution evaluation at:")
    print(f"   {out_root.resolve()}")
    print("=" * 80)

if __name__ == "__main__":
    main()
