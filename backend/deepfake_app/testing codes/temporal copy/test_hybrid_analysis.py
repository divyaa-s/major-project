"""
test_hybrid_analysis.py - FIXED VERSION
Improvements:
1. Better result printing with manual review flags
2. Shows model disagreement clearly
3. Color-coded output for easier reading
"""

import sys
import argparse
import logging
from pathlib import Path

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
    
    # NEW: Manual review flag
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
    
    # NEW: Show disagreement if present
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
    print(f"🤖 CNN MODELS ANALYSIS")
    print(f"{'─'*80}")
    print(f"   Ensemble Score: {cnn['ensemble']['avg_fake_probability']:.4f}")
    print(f"   Frames Analyzed: {cnn['ensemble']['frames_analyzed']}")
    print(f"\n   Per-Model Scores:")
    
    # NEW: Highlight extreme predictions and disagreements
    model_scores = cnn['ensemble']['per_model_avg']
    for model_name, score in model_scores.items():
        marker = ""
        if score < 0.10 or score > 0.90:
            marker = " ⚠️ EXTREME"
        print(f"      • {model_name:20s}: {score:.4f}{marker}")
    
    # NEW: Check model agreement
    scores_list = list(model_scores.values())
    if len(scores_list) >= 2:
        max_diff = max(scores_list) - min(scores_list)
        if max_diff > 0.50:
            print(f"\n   ⚠️  MODEL DISAGREEMENT: {max_diff:.4f} spread (max - min)")
            print(f"      Highest: {max(scores_list):.4f}, Lowest: {min(scores_list):.4f}")
    
    # Frame-by-frame details (show only first 10 and last 10 if many)
    if cnn['frame_predictions']:
        print(f"\n   Frame-by-Frame Predictions:")
        predictions = cnn['frame_predictions']
        
        if len(predictions) <= 20:
            # Show all
            for frame_pred in predictions:
                idx = frame_pred['frame_index']
                preds = frame_pred['model_predictions']
                avg = sum(preds.values()) / len(preds) if preds else 0
                print(f"      Frame {idx:4d}: {avg:.4f} avg")
        else:
            # Show first 10
            for frame_pred in predictions[:10]:
                idx = frame_pred['frame_index']
                preds = frame_pred['model_predictions']
                avg = sum(preds.values()) / len(preds) if preds else 0
                print(f"      Frame {idx:4d}: {avg:.4f} avg")
            
            print(f"      ... ({len(predictions) - 20} frames omitted) ...")
            
            # Show last 10
            for frame_pred in predictions[-10:]:
                idx = frame_pred['frame_index']
                preds = frame_pred['model_predictions']
                avg = sum(preds.values()) / len(preds) if preds else 0
                print(f"      Frame {idx:4d}: {avg:.4f} avg")
    
    # Quality forensics
    quality = results['quality_analysis']
    print(f"\n{'─'*80}")
    print(f"🔬 QUALITY FORENSICS")
    print(f"{'─'*80}")
    print(f"   Avg Quality Mismatch: {quality['avg_quality_mismatch']:.4f}")
    print(f"   Max Quality Mismatch: {quality['max_quality_mismatch']:.4f}")
    print(f"   Interpretation: {quality['interpretation']}")
    
    # Video info
    info = temp['video_info']
    print(f"\n{'─'*80}")
    print(f"📹 VIDEO INFORMATION")
    print(f"{'─'*80}")
    print(f"   Duration: {info['duration']:.2f}s")
    print(f"   FPS: {info['fps']:.2f}")
    print(f"   Total Frames: {info['total_frames']}")
    print(f"   Temporal Processed: {info['processed_frames']}")
    print(f"   Faces Detected: {info['faces_detected']}")  # FIXED: Should show correct count now
    
    # Interpretation
    print(f"\n{'='*80}")
    print(f"💡 INTERPRETATION")
    print(f"{'='*80}")
    
    score = results['final_score']
    source = results['decision_source']
    
    if source == "strong_agreement_fake":
        print(f"\n   🔴 STRONG CONFIDENCE: FAKE")
        print(f"   Both temporal and CNN analysis detected manipulation.")
        print(f"   This video shows clear signs of deepfake artifacts.")
    
    elif source == "strong_agreement_real":
        print(f"\n   🟢 STRONG CONFIDENCE: REAL")
        print(f"   Both temporal and CNN analysis indicate authenticity.")
        print(f"   No significant manipulation detected.")
    
    elif source == "temporal_landmark_cnn_confirm":
        print(f"\n   🟠 HIGH CONFIDENCE: FAKE (Landmark + CNN)")
        print(f"   Facial landmarks show instability (face swap signature)")
        print(f"   CNN models confirm manipulation artifacts.")
    
    elif source == "cnn_override":
        print(f"\n   🟡 CNN-DETECTED MANIPULATION")
        print(f"   CNN models detected strong spatial artifacts")
        print(f"   that temporal analysis didn't catch.")
        print(f"   Likely sophisticated GAN-generated or edited content.")
    
    elif source == "temporal_override":
        print(f"\n   🟡 TEMPORAL-DETECTED MANIPULATION")
        print(f"   Temporal patterns show manipulation")
        print(f"   that CNN models didn't catch.")
        print(f"   Likely face-swap with good texture synthesis.")
    
    elif source == "disagreement_uncertain":
        print(f"\n   ⚠️  UNCERTAIN - SYSTEMS DISAGREE")
        print(f"   Temporal score: {results['temporal_score']:.4f}")
        print(f"   CNN score: {results['cnn_score']:.4f}")
        print(f"   Recommend manual review or additional analysis.")
    
    elif source == "borderline_natural_motion":
        print(f"\n   🟡 BORDERLINE - NATURAL MOTION DETECTED")
        print(f"   High optical flow may be from person moving naturally")
        print(f"   (e.g., walking, turning head quickly)")
        print(f"   Penalty applied to reduce false positive risk.")
    
    elif source == "convnext_real_signal":
        print(f"\n   🟡 BORDERLINE - CONVNEXT DISAGREES")
        print(f"   ConvNeXt (most trusted model) indicates authenticity")
        print(f"   but other systems detected manipulation.")
        print(f"   Penalty applied due to model disagreement.")
    
    else:
        print(f"\n   📊 HYBRID ENSEMBLE DECISION")
        print(f"   Based on balanced fusion of temporal and CNN analysis.")
    
    # Recommendation
    print(f"\n{'='*80}")
    print(f"🎯 RECOMMENDATION")
    print(f"{'='*80}")
    
    if results.get("needs_manual_review", False):
        print(f"\n   ⚠️  MANUAL REVIEW STRONGLY RECOMMENDED")
        print(f"   This is an edge case where automated systems disagree.")
        print(f"   Human judgment needed for final decision.")
    elif score >= 0.70:
        print(f"\n   ❌ HIGH CONFIDENCE: This video is FAKE")
        print(f"   Multiple detection systems agree.")
        print(f"   Action: Reject/Flag this content.")
    elif score >= 0.50:
        print(f"\n   ⚠️  SUSPICIOUS: Likely manipulated")
        print(f"   Significant evidence of manipulation detected.")
        print(f"   Action: Treat with caution, recommend verification.")
    elif score >= 0.35:
        print(f"\n   🟡 BORDERLINE: Further analysis recommended")
        print(f"   Some anomalies detected but not conclusive.")
        print(f"   Action: May require human review.")
    else:
        print(f"\n   ✅ LIKELY AUTHENTIC")
        print(f"   No significant manipulation detected.")
        print(f"   Action: Content appears genuine.")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Hybrid video deepfake detection (Temporal + CNN)'
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
    
    args = parser.parse_args()
    
    # Check if file exists
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Run hybrid analysis
    print(f"\n🚀 Starting Hybrid Deepfake Detection")
    print(f"   Video: {video_path.name}")
    print(f"   CNN Keyframes: {args.keyframes}")
    print(f"   Temporal Frames: {args.temporal_frames}")
    print(f"   Temporal Skip: {args.temporal_skip}")
    print()
    
    try:
        analyzer = HybridVideoAnalyzer(
            num_keyframes=args.keyframes,
            temporal_max_frames=args.temporal_frames,
            temporal_skip=args.temporal_skip
        )
        
        results = analyzer.analyze_video(str(video_path))
        print_results(results)
        
        # NEW: Save results to JSON if requested
        if hasattr(args, 'save_json') and args.save_json:
            import json
            output_file = video_path.stem + "_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\n📖 Usage: python test_hybrid_analysis.py path/to/video.mp4")
        print("\nExample:")
        print("   python test_hybrid_analysis.py sample_video.mp4")
        print("   python test_hybrid_analysis.py sample_video.mp4 --keyframes 10")
        sys.exit(1)
    
    main()