"""
Selective Fish-Vista Downloader for KAUST Marine Research
Downloads ONLY the 10 priority Red Sea ecological & commercial families from the
Hugging Face 'imageomics/fish-vista' dataset repository without pulling the full 69k images.

10 Target Families:
  1. Acanthuridae    (Surgeonfish / Tangs)
  2. Balistidae      (Triggerfish)
  3. Chaetodontidae  (Butterflyfish)
  4. Holocentridae   (Squirrelfish / Soldierfish)
  5. Labridae        (Wrasses)
  6. Lutjanidae      (Snappers)
  7. Nemipteridae    (Threadfin Breams)
  8. Pomacentridae   (Damselfish / Clownfish)
  9. Scaridae        (Parrotfish)
  10. Siganidae      (Rabbitfish)

Usage:
  # 1. Preview available images and species counts without downloading:
  python scripts/download_fish_vista_10_families.py --dry-run

  # 2. Download up to 250 balanced images per family (Train: 200, Val: 50):
  python scripts/download_fish_vista_10_families.py --max_per_family 250

  # 3. Download all available images for the 10 families:
  python scripts/download_fish_vista_10_families.py --download_all
"""

import os
import sys
import csv
import argparse
import urllib.request
from collections import defaultdict
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    # Minimal fallback progress tracker if tqdm is not installed
    def tqdm(iterable, desc="", total=None, unit="it"):
        print(f"--> {desc} ({total} items)...")
        for i, item in enumerate(iterable):
            yield item
            if total and (i + 1) % max(1, total // 10) == 0:
                print(f"    Progress: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("[ERROR] 'huggingface_hub' is required. Install via: pip install huggingface_hub")
    sys.exit(1)

TARGET_FAMILIES = {
    "acanthuridae": "Acanthuridae",
    "balistidae": "Balistidae",
    "chaetodontidae": "Chaetodontidae",
    "holocentridae": "Holocentridae",
    "labridae": "Labridae",
    "lutjanidae": "Lutjanidae",
    "nemipteridae": "Nemipteridae",
    "pomacentridae": "Pomacentridae",
    "scaridae": "Scaridae",
    "siganidae": "Siganidae",
}

HF_REPO_ID = "imageomics/fish-vista"
METADATA_SPLITS = [
    ("train", "classification_train.csv"),
    ("val", "classification_val.csv"),
    ("test", "classification_test.csv"),
]

def load_split_metadata(repo_id: str, split_name: str, csv_filename: str):
    """Downloads and parses the Fish-Vista classification split CSV."""
    print(f"[1/4] Fetching metadata CSV: {csv_filename} from {repo_id}...")
    local_csv_path = hf_hub_download(repo_id=repo_id, filename=csv_filename, repo_type="dataset")
    
    records = []
    with open(local_csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_fam = (row.get("family") or "").strip().lower()
            if raw_fam in TARGET_FAMILIES:
                canonical_fam = TARGET_FAMILIES[raw_fam]
                species_name = (row.get("standardized_species") or row.get("species") or "unknown").strip()
                hf_file_path = (row.get("file_name") or "").strip()
                original_url = (row.get("original_url") or "").strip()
                filename = (row.get("filename") or os.path.basename(hf_file_path)).strip()
                
                records.append({
                    "split": split_name,
                    "family": canonical_fam,
                    "species": species_name,
                    "file_name": hf_file_path,
                    "filename": filename,
                    "original_url": original_url,
                })
    return records

def download_single_image(repo_id: str, record: dict, dest_dir: Path) -> bool:
    """Attempts to download an image via huggingface_hub, falling back to original_url."""
    dest_path = dest_dir / record["filename"]
    if dest_path.exists() and dest_path.stat().st_size > 1024:
        return True  # Already cached / downloaded

    # Primary method: Hugging Face hub download
    if record["file_name"]:
        try:
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=record["file_name"],
                repo_type="dataset",
            )
            import shutil
            shutil.copy2(cached_path, dest_path)
            return True
        except Exception:
            pass

    # Fallback method: Direct URL download from source repository
    if record["original_url"]:
        try:
            req = urllib.request.Request(record["original_url"], headers={"User-Agent": "KAUST-Marine-Research/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp, open(dest_path, "wb") as out:
                out.write(resp.read())
            return True
        except Exception:
            pass

    return False

def main():
    parser = argparse.ArgumentParser(description="Selective Fish-Vista Downloader (10 Red Sea Families)")
    parser.add_argument("--output_dir", type=str, default="datasets/fish_vista_10_families",
                        help="Target output directory (default: datasets/fish_vista_10_families)")
    parser.add_argument("--max_per_family", type=int, default=300,
                        help="Max images per family to balance dataset (default: 300).")
    parser.add_argument("--download_all", action="store_true",
                        help="Download all available images for target families without quota cap.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Inspect species counts and image distribution without downloading images.")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    print("=" * 80)
    print(" KAUST MARINE: FISH-VISTA SELECTIVE 10-FAMILY INGESTION")
    print(f" Target Families: {', '.join(sorted(TARGET_FAMILIES.values()))}")
    print(f" Mode: {'DRY RUN (Preview Only)' if args.dry_run else 'ACTIVE DOWNLOAD'}")
    if not args.download_all and not args.dry_run:
        print(f" Quota: Up to {args.max_per_family} images per family (stratified across species)")
    print("=" * 80)

    # 1. Fetch metadata across all splits
    all_records = []
    for split_name, csv_file in METADATA_SPLITS:
        try:
            split_records = load_split_metadata(HF_REPO_ID, split_name, csv_file)
            all_records.extend(split_records)
            print(f"   Found {len(split_records):5d} target specimens in '{split_name}' split.")
        except Exception as e:
            print(f"[WARNING] Could not parse split {csv_file}: {e}")

    if not all_records:
        print("[ERROR] No matching family records were found in metadata CSVs.")
        sys.exit(1)

    # 2. Group by family and species
    fam_species = defaultdict(lambda: defaultdict(list))
    for rec in all_records:
        fam_species[rec["family"]][rec["species"]].append(rec)

    # 3. Print taxonomic census report
    print("\n" + "-" * 80)
    print(f" {'Family Name':<16} | {'Available Images':<18} | {'Distinct Species':<18} | {'Top Species Example'}")
    print("-" * 80)
    total_imgs_available = 0
    for fam in sorted(TARGET_FAMILIES.values()):
        species_dict = fam_species.get(fam, {})
        n_species = len(species_dict)
        n_images = sum(len(imgs) for imgs in species_dict.values())
        total_imgs_available += n_images
        top_sp = max(species_dict.items(), key=lambda x: len(x[1]))[0] if species_dict else "None"
        print(f" {fam:<16} | {n_images:<18d} | {n_species:<18d} | {top_sp}")
    print("-" * 80)
    print(f" TOTAL ACROSS 10 FAMILIES: {total_imgs_available:,} available specimens in Fish-Vista.")
    print("-" * 80)

    if args.dry_run:
        print("\n[DRY RUN COMPLETE] Metadata verified successfully. To start downloading, execute:")
        print(f"  python scripts/download_fish_vista_10_families.py --max_per_family {args.max_per_family}\n")
        return

    # 4. Stratified selection across species to maximize biological diversity
    selected_for_download = []
    train_ratio = 0.8  # 80% train, 20% val

    for fam, species_dict in fam_species.items():
        all_fam_records = []
        # Round-robin sampling across species within each family to avoid single-species bias
        sp_lists = [list(imgs) for imgs in species_dict.values()]
        while sp_lists:
            for sp_list in list(sp_lists):
                if sp_list:
                    all_fam_records.append(sp_list.pop(0))
                else:
                    sp_lists.remove(sp_list)

        # Apply quota cap if not --download_all
        if not args.download_all and len(all_fam_records) > args.max_per_family:
            all_fam_records = all_fam_records[:args.max_per_family]

        # Partition into train and val directories
        n_train = int(len(all_fam_records) * train_ratio)
        for i, rec in enumerate(all_fam_records):
            target_split = "train" if i < n_train else "val"
            rec["target_split"] = target_split
            selected_for_download.append(rec)

    print(f"\n[2/4] Selected {len(selected_for_download):,} images across 10 families for download.")
    print(f" Target location: {out_root.resolve()}")

    # 5. Execute parallel multi-threaded download with visible progress bar
    print(f"\n[3/4] Starting multi-threaded download (8 concurrent workers)...")
    success_count = 0
    fail_count = 0

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _task(rec):
        split_dir = out_root / rec["target_split"] / rec["family"]
        split_dir.mkdir(parents=True, exist_ok=True)
        return download_single_image(HF_REPO_ID, rec, split_dir)

    pbar = tqdm(total=len(selected_for_download), desc="Downloading Fish-Vista Crops", unit="img")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_task, rec): rec for rec in selected_for_download}
        for fut in as_completed(futures):
            ok = fut.result()
            if ok:
                success_count += 1
            else:
                fail_count += 1
            pbar.update(1)
    pbar.close()
    print(" INGESTION RUN FINISHED")
    print(f" Successfully downloaded: {success_count} images")
    if fail_count > 0:
        print(f" Failed downloads: {fail_count} images")
    print(f" Dataset saved in standard PyTorch ImageFolder format at:")
    print(f"   Train: {out_root / 'train'}")
    print(f"   Val:   {out_root / 'val'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
