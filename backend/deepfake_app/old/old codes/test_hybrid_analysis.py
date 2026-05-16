"""
test_hybrid_analysis.py - FIXED & IMPROVED VERSION WITH CSV LOGGING
Improvements:
1. Better result printing with manual review flags
2. Shows model disagreement clearly
3. Color-coded output for easier reading
4. Automatic CSV logging of all results
5. Optional --true-label argument for tagging ground truth
6. Extracts video metadata (duration, fps, total_frames) for CSV
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_results(results):
    """Pretty print hybrid analysis results with improvements"""
    print("\n" + "="*80)
    print("🎯 HYBRID DEEPFAKE DETECTION RESULTS")
    print("   (Temporal Analysis + CNN Models)")
    print("="*80)
    
    if "error" in results:
        print(f"\n❌ Error: {results['error']}")
        return
    
    # Manual review flag
    if results.get("needs_manual_review", False):
        print("\n" + "!"*80)
        print("⚠️  MANUAL REVIEW REQUIRED")
        print(f"   Reason: {results.get('review_reason', 'Unknown')}")
        print("!"*80)
    
    # Main verdict
    print(f"\n{'='*80}")
    print(f"🏆 FINAL VERDICT")
    print(f"{'='*80}")
    print(f"\n   Label: {results['label']}")
    print(f"   Confidence: {results['confidence']:.4f} ({results['confidence']*100:.2f}%)")
    print(f"   Final Score: {results['final_score']:.4f}")
    print(f"   Decision Source: {results['decision_source']}")
    
    # Component scores
    print(f"\n{'='*80}")
    print(f"📊 COMPONENT SCORES")
    print(f"{'='*80}")
    print(f"\n   Temporal Score: {results['temporal_score']:.4f}")
    print(f"   CNN Score: {results['cnn_score']:.4f}")
    print(f"   Quality Score: {results['quality_score']:.4f}")
    print(f"   Fusion Weights: Temporal {results['fusion_weights']['temporal']:.0%} | CNN {results['fusion_weights']['cnn']:.0%} | Quality {results['fusion_weights']['quality']:.0%}")
    
    # Disagreement warning
    score_diff = abs(results['temporal_score'] - results['cnn_score'])
    if score_diff > 0.30:
        print(f"\n   ⚠️  HIGH DISAGREEMENT: {score_diff:.4f} difference between temporal and CNN")
    
    # Temporal breakdown
    temp = results['temporal_analysis']
    print(f"\n{'─'*80}")
    print(f"⏱️  TEMPORAL ANALYSIS DETAILS")
    print(f"{'─'*80}")
    print(f"   Overall: {temp['temporal_consistency_score']:.4f}")
    print(f"   {temp['interpretation']}")
    print(f"   Decision: {temp.get('decision_source', 'N/A')}")
    print(f"\n   • Blink: {temp['blink_analysis']['score']:.4f} - {temp['blink_analysis']['interpretation']}")
    print(f"   • Optical Flow: {temp['optical_flow_analysis']['score']:.4f} - {temp['optical_flow_analysis']['interpretation']}")
    print(f"   • Landmarks: {temp['landmark_stability']['score']:.4f} - {temp['landmark_stability']['interpretation']}")
    
    # CNN breakdown
    cnn = results['cnn_analysis']
    print(f"\n{'─'*80}")
    print(f"🤖 CNN ENSEMBLE ANALYSIS")
    print(f"{'─'*80}")
    print(f"   Weighted Ensemble Score : {cnn['ensemble']['avg_fake_probability']:.4f}")
    print(f"   Frames Analyzed         : {cnn['ensemble']['frames_analyzed']}")

    model_scores = cnn['ensemble'].get('per_model_avg', {})

    if not model_scores:
        # ── No models loaded — show a prominent warning ─────────────────
        print(f"\n   {'!'*60}")
        print(f"   ⚠️  NO CNN MODELS LOADED")
        print(f"   All 4 models failed to load. CNN score is neutral (0.5).")
        print(f"   Check the startup log above for ❌ FAILED lines with error details.")
        print(f"   Common causes:")
        print(f"     • .pth file path is wrong (check MODEL_CONFIGS paths)")
        print(f"     • num_classes mismatch (model saved with 1 class, loaded with 2)")
        print(f"     • Architecture mismatch (wrong model_name in MODEL_CONFIGS)")
        print(f"   {'!'*60}")
    else:
        # ── Per-model score table with visual bar ────────────────────────
        print(f"\n   {'Model':<22} {'Score':>7}  {'Bar (0→1)':<32}  {'Verdict'}")
        print(f"   {'─'*22} {'─'*7}  {'─'*32}  {'─'*15}")

        # Ensemble weights (import from hybrid_vid_improved if available, else default)
        try:
            from hybrid_vid_improved import MODEL_ENSEMBLE_WEIGHTS
        except Exception:
            MODEL_ENSEMBLE_WEIGHTS = {}

        scores_list = list(model_scores.values())
        for model_name, score in model_scores.items():
            bar_len   = int(score * 30)
            bar       = "█" * bar_len + "░" * (30 - bar_len)
            weight    = MODEL_ENSEMBLE_WEIGHTS.get(model_name, 0.25)

            if score >= 0.80:
                verdict = "⚠️  FAKE signal"
            elif score >= 0.60:
                verdict = "🟡 Suspicious"
            elif score >= 0.40:
                verdict = "🔵 Borderline"
            else:
                verdict = "✅ Real signal"

            extreme = "  [EXTREME]" if score < 0.05 or score > 0.95 else ""
            print(f"   {model_name:<22} {score:>7.4f}  |{bar}|  {verdict}{extreme}")

        # Ensemble summary line
        ensemble_score = cnn['ensemble']['avg_fake_probability']
        e_bar = "█" * int(ensemble_score * 30) + "░" * (30 - int(ensemble_score * 30))
        print(f"   {'─'*22} {'─'*7}  {'─'*32}  {'─'*15}")
        print(f"   {'WEIGHTED ENSEMBLE':<22} {ensemble_score:>7.4f}  |{e_bar}|")

        # Model disagreement check
        if len(scores_list) >= 2:
            max_diff = max(scores_list) - min(scores_list)
            if max_diff > 0.50:
                print(f"\n   ⚠️  HIGH MODEL DISAGREEMENT: {max_diff:.4f} spread")
                print(f"      Highest: {max(scores_list):.4f}  |  Lowest: {min(scores_list):.4f}")
                print(f"      Models are giving conflicting signals — treat result with caution.")
            elif max_diff > 0.30:
                print(f"\n   🟡 Moderate disagreement: {max_diff:.4f} spread between models")

    # Frame-by-frame (smart truncation, shows per-model breakdown)
    if cnn['frame_predictions']:
        predictions = cnn['frame_predictions']
        print(f"\n   Frame-by-Frame Predictions  ({len(predictions)} frames, showing avg per frame):")
        print(f"   {'Frame':>7}  {'Avg':>7}  {'Per-model scores'}")
        print(f"   {'─'*7}  {'─'*7}  {'─'*50}")

        def _fmt_frame(fp):
            preds = fp['model_predictions']
            avg   = sum(preds.values()) / len(preds) if preds else 0
            per   = "  ".join(f"{k[:4]}={v:.3f}" for k, v in preds.items())
            return fp['frame_index'], avg, per

        def _print_frame(fp):
            idx, avg, per = _fmt_frame(fp)
            flag = " ⚠️" if avg > 0.80 else ""
            print(f"   {idx:>7}  {avg:>7.4f}  {per}{flag}")

        if len(predictions) <= 20:
            for fp in predictions:
                _print_frame(fp)
        else:
            for fp in predictions[:8]:
                _print_frame(fp)
            print(f"   {'':>7}  ... ({len(predictions) - 16} frames omitted) ...")
            for fp in predictions[-8:]:
                _print_frame(fp)
    
    # Quality forensics
    quality = results['quality_analysis']
    print(f"\n{'─'*80}")
    print(f"🔬 QUALITY FORENSICS")
    print(f"{'─'*80}")
    print(f"   Avg Quality Mismatch: {quality['avg_quality_mismatch']:.4f}")
    print(f"   Max Quality Mismatch: {quality['max_quality_mismatch']:.4f}")
    print(f"   Interpretation: {quality['interpretation']}")
    
    # Video info
    info = temp.get('video_info', {})
    print(f"\n{'─'*80}")
    print(f"📹 VIDEO INFORMATION")
    print(f"{'─'*80}")
    print(f"   Duration: {info.get('duration', 'N/A'):.2f}s")
    print(f"   FPS: {info.get('fps', 'N/A'):.2f}")
    print(f"   Total Frames: {info.get('total_frames', 'N/A')}")
    print(f"   Temporal Processed: {info.get('processed_frames', 'N/A')}")
    print(f"   Faces Detected: {info.get('faces_detected', 'N/A')}")
    
    # Interpretation & Recommendation
    print(f"\n{'='*80}")
    print(f"💡 INTERPRETATION & RECOMMENDATION")
    print(f"{'='*80}")
    
    score = results['final_score']
    source = results['decision_source']
    
    if source == "convnext_quality_boost_fake_strong":
        print(f"\n   🔴 STRONG FAKE SIGNAL (ConvNeXt + Quality)")
        print(f"   High ConvNeXt score and visible quality mismatch detected.")
    elif source == "temporal_quality_real_override":
        print(f"\n   🟢 STRONG REAL SIGNAL")
        print(f"   Temporal and quality indicators override CNN suspicion.")
    elif source == "disagreement_uncertain":
        print(f"\n   ⚠️  UNCERTAIN - HIGH DISAGREEMENT")
        print(f"   Temporal and CNN systems conflict significantly.")
    else:
        print(f"\n   📊 HYBRID ENSEMBLE DECISION")
        print(f"   Balanced fusion of temporal, CNN, and quality signals.")
    
    if results.get("needs_manual_review", False):
        print(f"\n   ⚠️  MANUAL REVIEW STRONGLY RECOMMENDED")
        print(f"   Edge case detected - human verification advised.")
    elif score >= 0.70:
        print(f"\n   ❌ HIGH CONFIDENCE FAKE")
        print(f"   Multiple strong indicators of manipulation.")
    elif score >= 0.50:
        print(f"\n   ⚠️  SUSPICIOUS - LIKELY FAKE")
        print(f"   Evidence of manipulation present.")
    elif score >= 0.35:
        print(f"\n   🟡 BORDERLINE - FURTHER REVIEW SUGGESTED")
    else:
        print(f"\n   ✅ LIKELY AUTHENTIC")
        print(f"   No significant manipulation signals detected.")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Hybrid video deepfake detection (Temporal + CNN + Quality)'
    )
    parser.add_argument(
        'video_path',
        type=str,
        help='Path to video file'
    )
    parser.add_argument(
        '--keyframes',
        type=int,
        default=100,
        help='Number of keyframes for CNN analysis (default: 100)'
    )
    parser.add_argument(
        '--temporal-frames',
        type=int,
        default=30,
        help='Max frames for temporal analysis (default: 30)'
    )
    parser.add_argument(
        '--temporal-skip',
        type=int,
        default=2,
        help='Skip rate for temporal analysis (default: 2)'
    )
    parser.add_argument(
        '--true-label',
        type=str,
        default="",
        help='Optional: ground truth label ("Real" or "Fake") for logging'
    )
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Save detailed results to JSON file'
    )
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print(f"\n🚀 Starting Hybrid Deepfake Detection")
    print(f"   Video: {video_path.name}")
    print(f"   CNN Keyframes: {args.keyframes}")
    print(f"   Temporal Frames: {args.temporal_frames}")
    print(f"   Temporal Skip: {args.temporal_skip}")
    if args.true_label:
        print(f"   True Label (provided): {args.true_label}")
    print()
    
    try:
        # Extract video metadata for CSV
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / fps if fps > 0 else 0
        cap.release()
        
        analyzer = HybridVideoAnalyzer(
            num_keyframes=args.keyframes,
            temporal_max_frames=args.temporal_frames,
            temporal_skip=args.temporal_skip
        )
        
        results = analyzer.analyze_video(str(video_path))
        print_results(results)
        
        # === AUTO-SAVE TO CSV ===
        csv_file = "hybrid_detection_results.csv"
        
        headers = [
            "timestamp", "video_name","true_label", "predicted_label", "confidence",
            "final_score", "decision_source", "temporal_score", "cnn_score",
            "quality_score", "needs_manual_review", "review_reason",
            "faces_detected", "duration_s", "fps", "total_frames"
        ]
        
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "timestamp": now,
            "video_name": video_path.name,
            "true_label": args.true_label,
            "predicted_label": results.get("label", ""),
            "confidence": results.get("confidence", ""),
            "final_score": results.get("final_score", ""),
            "decision_source": results.get("decision_source", ""),
            "temporal_score": results.get("temporal_score", ""),
            "cnn_score": results.get("cnn_score", ""),
            "quality_score": results.get("quality_score", ""),
            "needs_manual_review": str(results.get("needs_manual_review", False)),
            "review_reason": results.get("review_reason", ""),
            "faces_detected": results.get("temporal_analysis", {}).get("video_info", {}).get("faces_detected", "N/A"),
            "duration_s": round(duration_s, 2),
            "fps": round(fps, 2),
            "total_frames": total_frames
        }
        
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        
        print(f"Results appended to: {csv_file}")
        
        # Optional JSON save
        if args.save_json:
            import json
            output_file = video_path.stem + "_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Detailed results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\n📖 Usage: python test_hybrid_analysis.py path/to/video.mp4")
        print("\nExamples:")
        print("   python test_hybrid_analysis.py sample.mp4")
        print("   python test_hybrid_analysis.py fake_video.mp4 --true-label Fake --save-json")
        print("   python test_hybrid_analysis.py real_clip.mp4 --true-label Real")
        sys.exit(1)
    
    main()