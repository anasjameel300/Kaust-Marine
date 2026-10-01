"""
Download Sample Red Sea Underwater Video Clip
KAUST Marine Vision - BRUVs Video Benchmark

Downloads a verified, open-access underwater clip of Acanthurus sohal (Sohal Surgeonfish,
endemic to the Red Sea, Family: Acanthuridae) from Wikimedia Commons.
Size: ~19.6 MB
"""

import sys
import time
import urllib.request
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "app" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = SAMPLES_DIR / "sample_redsea_sohal.mp4"

VIDEO_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/transcoded/b/bf/"
    "Acanthurus_sohal_-_Arabischer_Doktorfisch_im_Roten_Meer_DSCF3888.webm/"
    "Acanthurus_sohal_-_Arabischer_Doktorfisch_im_Roten_Meer_DSCF3888.webm.360p.mpeg4.mov"
)

def download_sample_video():
    print("=" * 65)
    print("KAUST Marine Vision - Red Sea Sample Video Ingestion")
    print("=" * 65)
    print(f"Target Species: Acanthurus sohal (Sohal Surgeonfish, Red Sea)")
    print(f"Family:        Acanthuridae")
    print(f"Destination:   {OUTPUT_PATH}")
    print(f"Downloading from Wikimedia Commons...")

    headers = {"User-Agent": "KaustMarineVision/1.0 (Research Laboratory Benchmark)"}
    req = urllib.request.Request(VIDEO_URL, headers=headers)

    start_time = time.time()
    try:
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 64

            with open(OUTPUT_PATH, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    pct = (downloaded / total_size) * 100 if total_size > 0 else 0
                    mb = downloaded / (1024 * 1024)
                    sys.stdout.write(f"\rProgress: [{pct:5.1f}%] {mb:5.2f} MB downloaded...")
                    sys.stdout.flush()

        elapsed = time.time() - start_time
        print(f"\nDownload complete in {elapsed:.1f}s! File size: {downloaded / (1024 * 1024):.2f} MB")
        print(f"Saved to: {OUTPUT_PATH}")

    except Exception as e:
        print(f"\nDownload failed: {e}")

if __name__ == "__main__":
    download_sample_video()
