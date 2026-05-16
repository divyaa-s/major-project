"""
batch_hybrid_analysis.py
Fully aligned with hybrid_vid_improved.py
Optimized for large-scale batch testing (50+ videos)
"""

import sys
import argparse
import logging
import csv
import os
from datetime import datetime
from pathlib import Path

import cv2

from hybrid_vid_improved import HybridVideoAnalyzer

# Reduce console spam from deep logs
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    parser = argparse.ArgumentParser(
        description='Batch Hybrid Deepfake Detection'
    )

    parser.add_argument('--folder', type=str, required=True,
                        help='Folder containing videos')
    parser.add_argument('--recursive', action='store_true',
                        help='Search recursively')
    parser.add_argument('--keyframes', type=int, default=100)
    parser.add_argument('--temporal-frames', type=int, default=30)
    parser.add_argument('--temporal-skip', type=int, default=2)
    parser.add_argument('--save-json', action='store_true')

    args = parser.parse_args()

    folder = Path(args.folder)

    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        sys.exit(1)

    # ------------------------------------------------------------
    # Collect Videos
    # ------------------------------------------------------------
    pattern = "**/*.mp4" if args.recursive else "*.mp4"
    videos = list(folder.glob(pattern))

    if not videos:
        print("❌ No videos found.")
        sys.exit(1)

    print(f"\n🔍 Found {len(videos)} videos to process.\n")

    # ------------------------------------------------------------
    # Initialize analyzer ONCE (important)
    # ------------------------------------------------------------
    analyzer = HybridVideoAnalyzer(
        num_keyframes=args.keyframes,
        temporal_max_frames=args.temporal_frames,
        temporal_skip=args.temporal_skip
    )

    csv_file = "hybrid_detection_results.csv"

    headers = [
        "timestamp",
        "video_name",
        "true_label",

        "predicted_label",
        "confidence",
        "final_score",
        "decision_source",

        "temporal_score",
        "cnn_score",
        "quality_score",

        "temporal_decision_source",
        "blink_score",
        "blink_count",
        "blink_expected",
        "flow_score",
        "flow_avg_magnitude",
        "flow_inconsistency",
        "landmark_score",
        "landmark_jitter",
        "landmark_frames_analyzed",

        "efficientnet_b3_score",
        "xception_score",
        "vit_score",
        "convnext_score",
        "cnn_ensemble_score",
        "cnn_frames_analyzed",

        "quality_avg_mismatch",
        "quality_max_mismatch",
        "quality_interpretation",

        "faces_detected",
        "processed_frames",
        "total_frames",
        "fps",
        "duration_s",

        "needs_manual_review",
        "review_reason"
    ]

    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)

        if not file_exists:
            writer.writeheader()

        # ------------------------------------------------------------
        # Process Each Video
        # ------------------------------------------------------------
        for idx, video_path in enumerate(videos, 1):

            print(f"[{idx}/{len(videos)}] Processing: {video_path.name}")

            try:
                # Infer label from path (optional)
                lower_path = str(video_path).lower()
                if "manipulated" in lower_path:
                    true_label = "Fake"
                elif "original" in lower_path:
                    true_label = "Real"
                else:
                    true_label = ""

                # Extract metadata
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_s = total_frames / fps if fps > 0 else 0
                cap.release()

                # Run analysis
                results = analyzer.analyze_video(str(video_path))

                if "error" in results:
                    print(f"   ❌ Error: {results['error']}")
                    continue

                
                temp = results.get("temporal_analysis", {})
                cnn = results.get("cnn_analysis", {})
                quality = results.get("quality_analysis", {})
                video_info = temp.get("video_info", {})
                model_scores = cnn.get("ensemble", {}).get("per_model_avg", {})

                row = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "video_name": video_path.name,
                    "true_label": true_label,

                    "predicted_label": results.get("label", ""),
                    "confidence": results.get("confidence", ""),
                    "final_score": results.get("final_score", ""),
                    "decision_source": results.get("decision_source", ""),

                    "temporal_score": results.get("temporal_score", ""),
                    "cnn_score": results.get("cnn_score", ""),
                    "quality_score": results.get("quality_score", ""),

                    "temporal_decision_source": temp.get("decision_source", ""),
                    "blink_score": temp.get("blink_analysis", {}).get("score", ""),
                    "blink_count": temp.get("blink_analysis", {}).get("blink_count", ""),
                    "blink_expected": temp.get("blink_analysis", {}).get("expected_blinks", ""),
                    "flow_score": temp.get("optical_flow_analysis", {}).get("score", ""),
                    "flow_avg_magnitude": temp.get("optical_flow_analysis", {}).get("avg_flow_magnitude", ""),
                    "flow_inconsistency": temp.get("optical_flow_analysis", {}).get("flow_inconsistency", ""),
                    "landmark_score": temp.get("landmark_stability", {}).get("score", ""),
                    "landmark_jitter": temp.get("landmark_stability", {}).get("avg_normalized_jitter", ""),
                    "landmark_frames_analyzed": temp.get("landmark_stability", {}).get("frames_analyzed", ""),

                    "efficientnet_b3_score": model_scores.get("efficientnet_b3", ""),
                    "xception_score": model_scores.get("xception", ""),
                    "vit_score": model_scores.get("vit", ""),
                    "convnext_score": model_scores.get("convnext", ""),
                    "cnn_ensemble_score": cnn.get("ensemble", {}).get("avg_fake_probability", ""),
                    "cnn_frames_analyzed": cnn.get("ensemble", {}).get("frames_analyzed", ""),

                    "quality_avg_mismatch": quality.get("avg_quality_mismatch", ""),
                    "quality_max_mismatch": quality.get("max_quality_mismatch", ""),
                    "quality_interpretation": quality.get("interpretation", ""),

                    "faces_detected": video_info.get("faces_detected", ""),
                    "processed_frames": video_info.get("processed_frames", ""),
                    "total_frames": video_info.get("total_frames", ""),
                    "fps": video_info.get("fps", ""),
                    "duration_s": video_info.get("duration", ""),

                    "needs_manual_review": results.get("needs_manual_review", False),
                    "review_reason": results.get("review_reason", "")
                }

                writer.writerow(row)
                f.flush()
                print(f"   ✅ Done | Label: {row['predicted_label']} | Score: {row['final_score']}")

                # Optional JSON save
                if args.save_json:
                    import json
                    json_file = video_path.stem + "_results.json"
                    with open(json_file, 'w', encoding='utf-8') as jf:
                        json.dump(results, jf, indent=2)

            except Exception as e:
                print(f"   ❌ Failed: {e}")

    print(f"\n📁 All results saved to: {csv_file}\n")


if __name__ == "__main__":
    main()