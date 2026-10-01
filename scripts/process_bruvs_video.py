"""
Standalone CLI Runner for In-Situ BRUVs Video Analysis & MaxN Computation
KAUST Marine Vision - Automated Video Monitoring (Dr. Renato Morais ROCKNHub)

Usage:
    python scripts/process_bruvs_video.py --video app/samples/sample_redsea_sohal.mp4 --fps 3 --duration 30
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.video_engine import video_engine

def main():
    parser = argparse.ArgumentParser(description="KAUST Marine Vision - BRUVs Video Processing & MaxN Engine")
    parser.add_argument("--video", type=str, default="app/samples/sample_redsea_sohal.mp4", help="Path to input video (.mp4/.mov)")
    parser.add_argument("--fps", type=float, default=3.0, help="Temporal sampling frame rate (default: 3.0 FPS)")
    parser.add_argument("--duration", type=float, default=30.0, help="Max duration in seconds to process (default: 30.0)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--scale", type=float, default=22.0, help="Pixels per cm for BRUVs rig (default: 22.0 px/cm)")
    parser.add_argument("--classifier", type=str, default="bioclip", help="Taxonomic classifier mode (bioclip, resnet50, ensemble)")
    parser.add_argument("--no-classify", action="store_true", help="Skip taxonomic classification on tracks")

    args = parser.parse_args()

    print("=" * 70)
    print("KAUST Marine Vision - BRUVs Video Analysis & MaxN Computation")
    print("=" * 70)
    print(f"Input Video:        {args.video}")
    print(f"Sampling Rate:      {args.fps} FPS")
    print(f"Max Duration:       {args.duration}s")
    print(f"Rig Calibration:    {args.scale} px/cm")
    print(f"Taxonomy Backbone:  {args.classifier}")
    print("-" * 70)

    results = video_engine.process_video(
        video_path=args.video,
        sample_fps=args.fps,
        max_duration_sec=args.duration,
        conf_threshold=args.conf,
        is_calibrated=True,
        pixels_per_cm=args.scale,
        run_classification=not args.no_classify,
        classifier_mode=args.classifier
    )

    print("\n" + "=" * 70)
    print("VIDEO PROCESSING & MaxN SUMMARY")
    print("=" * 70)
    print(f"Total Video Duration:   {results['total_video_duration_sec']}s")
    print(f"Duration Analyzed:      {results['duration_analyzed_sec']}s ({results['frames_analyzed']} sampled frames)")
    print(f"Global MaxN:            {results['max_n_total']} Fish (Simultaneous Peak)")
    print(f"Peak Frame Timestamp:   {results['peak_timestamp']} ({results['peak_second']}s)")
    print(f"Unique Fish Tracked:    {results['unique_tracks_count']}")
    print("-" * 70)
    print("MaxN Family Ecological Breakdown:")
    for fam, cnt in results["max_n_by_family"].items():
        print(f"  • {fam:25s}: MaxN = {cnt}")
    print("-" * 70)
    print("Unique Tracks Identified:")
    for tr in results["tracks"]:
        bio = tr.get("biomass", {})
        len_str = f"{bio.get('fork_length_cm', '--')} cm" if bio.get("fork_length_cm") else f"{bio.get('length_px', '--')} px"
        wt_str = f"{bio.get('biomass_g', '--')} g" if bio.get("biomass_g") else "--"
        peak_flag = " [In Peak Frame]" if tr.get("in_peak_frame") else ""
        print(f"  Track #{tr['track_id']:2d}: {tr['family']:20s} ({tr['confidence']*100:4.1f}%) | Time: {tr['first_seen']} -> {tr['last_seen']} ({tr['duration_sec']}s) | Len: {len_str} | W: {wt_str}{peak_flag}")
    print("-" * 70)
    print(f"Annotated Video Output: {results['video_url']}")
    print(f"Peak Frame Snapshot:    {results['peak_frame_url']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
