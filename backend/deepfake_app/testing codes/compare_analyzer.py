"""
Compare Original vs Improved Hybrid Analyzer
Test which one performs better on your videos
"""

import sys
import time
from pathlib import Path

# Import both versions
from hybrid_video_analyzer import HybridVideoAnalyzer as OriginalAnalyzer
from hybrid_vid_improved import ImprovedHybridAnalyzer

def compare_analyzers(video_path, num_keyframes=5):
    """Run both analyzers and compare results"""
    
    print("\n" + "="*80)
    print(f"🔬 COMPARING ANALYZERS")
    print(f"Video: {Path(video_path).name}")
    print("="*80)
    
    # Test Original
    print("\n" + "─"*80)
    print("📊 ORIGINAL ANALYZER (Even Spacing + Old Weights)")
    print("─"*80)
    
    start = time.time()
    original = OriginalAnalyzer(num_keyframes=num_keyframes)
    result_original = original.analyze_video(video_path)
    time_original = time.time() - start
    
    if "error" not in result_original:
        print(f"\n✅ Completed in {time_original:.1f}s")
        print(f"   Label: {result_original['label']}")
        print(f"   Confidence: {result_original['confidence']:.4f}")
        print(f"   Final Score: {result_original['final_score']:.4f}")
        print(f"   Decision: {result_original['decision_source']}")
        print(f"   Temporal: {result_original['temporal_score']:.4f}")
        print(f"   CNN: {result_original['cnn_score']:.4f}")
    else:
        print(f"\n❌ Error: {result_original['error']}")
    
    # Test Improved
    print("\n" + "─"*80)
    print("🧠 IMPROVED ANALYZER (Smart Selection + New Weights + Boost)")
    print("─"*80)
    
    start = time.time()
    improved = ImprovedHybridAnalyzer(
        num_keyframes=num_keyframes,
        smart_keyframe_selection=True  # Enable smart selection
    )
    result_improved = improved.analyze_video(video_path)
    time_improved = time.time() - start
    
    if "error" not in result_improved:
        print(f"\n✅ Completed in {time_improved:.1f}s")
        print(f"   Label: {result_improved['label']}")
        print(f"   Confidence: {result_improved['confidence']:.4f}")
        print(f"   Final Score: {result_improved['final_score']:.4f}")
        print(f"   Decision: {result_improved['decision_source']}")
        print(f"   Temporal: {result_improved['temporal_score']:.4f}")
        print(f"   CNN: {result_improved['cnn_score']:.4f}")
    else:
        print(f"\n❌ Error: {result_improved['error']}")
    
    # Comparison
    print("\n" + "="*80)
    print("📊 COMPARISON")
    print("="*80)
    
    if "error" not in result_original and "error" not in result_improved:
        print(f"\n⏱️  Processing Time:")
        print(f"   Original: {time_original:.1f}s")
        print(f"   Improved: {time_improved:.1f}s")
        print(f"   Difference: {abs(time_improved - time_original):.1f}s")
        
        print(f"\n🎯 Final Scores:")
        print(f"   Original: {result_original['final_score']:.4f} ({result_original['label']})")
        print(f"   Improved: {result_improved['final_score']:.4f} ({result_improved['label']})")
        print(f"   Difference: {abs(result_improved['final_score'] - result_original['final_score']):.4f}")
        
        print(f"\n🤖 CNN Scores:")
        print(f"   Original: {result_original['cnn_score']:.4f}")
        print(f"   Improved: {result_improved['cnn_score']:.4f}")
        print(f"   Difference: {abs(result_improved['cnn_score'] - result_original['cnn_score']):.4f}")
        
        # Agreement
        if result_original['label'] == result_improved['label']:
            print(f"\n✅ BOTH AGREE: {result_original['label']}")
        else:
            print(f"\n⚠️  DISAGREEMENT!")
            print(f"   Original: {result_original['label']}")
            print(f"   Improved: {result_improved['label']}")
        
        # Key insights
        print(f"\n💡 KEY INSIGHTS:")
        
        # Smart keyframe benefit
        orig_frames = result_original['cnn_analysis']['ensemble']['frames_analyzed']
        imp_frames = result_improved['cnn_analysis']['ensemble']['frames_analyzed']
        if imp_frames > orig_frames:
            print(f"   ✅ Smart selection found {imp_frames - orig_frames} more usable frames")
        
        # Ensemble boost
        orig_cnn_raw = result_original['cnn_analysis']['ensemble']['per_model_avg']
        imp_cnn_raw = result_improved['cnn_analysis']['ensemble']['per_model_avg']
        
        orig_avg = sum(orig_cnn_raw.values()) / len(orig_cnn_raw)
        imp_avg = sum(imp_cnn_raw.values()) / len(imp_cnn_raw)
        
        if result_improved['cnn_score'] > imp_avg:
            boost = result_improved['cnn_score'] - imp_avg
            print(f"   📈 Ensemble boost applied: +{boost:.4f}")
        
        # Decision source comparison
        if result_original['decision_source'] != result_improved['decision_source']:
            print(f"   🔄 Different decision pathways:")
            print(f"      Original: {result_original['decision_source']}")
            print(f"      Improved: {result_improved['decision_source']}")
    
    print("\n" + "="*80 + "\n")
    
    return result_original, result_improved


def main():
    if len(sys.argv) < 2:
        print("\n📖 Usage: python compare_analyzers.py path/to/video.mp4 [keyframes]")
        print("\nExample:")
        print("   python compare_analyzers.py video.mp4")
        print("   python compare_analyzers.py video.mp4 10")
        sys.exit(1)
    
    video_path = sys.argv[1]
    num_keyframes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    # Check if file exists
    if not Path(video_path).exists():
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)
    
    compare_analyzers(video_path, num_keyframes)


if __name__ == "__main__":
    main()