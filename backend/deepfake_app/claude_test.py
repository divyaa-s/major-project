"""
batch_test_fakes.py
====================
Batch tests 15 randomly selected FAKE videos using the full hybrid pipeline.
Prints all component scores and saves results to CSV.
"""

import sys, os, csv, random, glob
import numpy as np
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_DIR = r"D:\Projects\Major Project\Deepfake Detection"
OUTPUT_CSV  = r"D:\Projects\Major Project\Deepfake Detection\backend\deepfake_app\batch_test_15_fakes.csv"

# ==============================================================================
# FORCE PYTHON TO LOOK IN DEEPFAKE_APP FOLDER FIRST
# ==============================================================================
GOOD_FOLDER = r"D:\Projects\Major Project\Deepfake Detection\backend\deepfake_app"
sys.path.insert(0, GOOD_FOLDER)

from hybrid_vid_improved import ImprovedHybridAnalyzer

# ==============================================================================
# RANDOMLY SELECT 15 FAKE VIDEOS
# ==============================================================================
# CHANGE THIS IF YOUR FAKE VIDEOS ARE IN A DIFFERENT FOLDER (e.g. 'altered', 'fake')
FAKE_VIDEOS_DIR = r"D:\Projects\Major Project\Deepfake Detection\datasets\dfd\manipulated"

all_fakes = glob.glob(os.path.join(FAKE_VIDEOS_DIR, "*.mp4"))

if not all_fakes:
    print(f"❌ Error: No .mp4 files found in {FAKE_VIDEOS_DIR}")
    sys.exit(1)

# Grab 15 random videos (or however many there are if less than 15)
num_to_select = min(15, len(all_fakes))
random_fakes = random.sample(all_fakes, num_to_select)

TEST_VIDEOS = [(vid_path, "Fake") for vid_path in random_fakes]

print(f"🎲 Randomly selected {num_to_select} Fake videos for batch testing.")

# ==============================================================================

def main():
    print(f"\nInitializing Hybrid Pipeline for {len(TEST_VIDEOS)} videos...")
    analyzer = ImprovedHybridAnalyzer(num_keyframes=10, temporal_max_frames=30)
    
    results = []
    
    for i, (vid_path, gt_label) in enumerate(TEST_VIDEOS, 1):
        vid_name = os.path.basename(vid_path)
        print(f"\n" + "="*80)
        print(f"[{i}/{len(TEST_VIDEOS)}] Analyzing: {vid_name} (GT: {gt_label})")
        print("="*80)
        
        try:
            res = analyzer.analyze_video(vid_path)
            
            if "error" in res:
                print(f"  ❌ ERROR processing video: {res['error']}")
                continue
                
            pred_label = res["label"]
            correct    = (pred_label == gt_label)
            
            # Flatten metrics
            flat_res = {
                "video":             vid_name,
                "gt_label":          gt_label,
                "predicted":         pred_label,
                "correct":           correct,
                "final_score":       res.get("final_score", 0.0),
                "confidence":        res.get("confidence", 0.0),
                "decision_source":   res.get("decision_source", "unknown"),
                "needs_review":      res.get("needs_manual_review", False),
                
                "bilstm_score":      res.get("bilstm_score", 0.5),
                "efficientnet_b3_score": res.get("cnn_analysis", {}).get("ensemble", {}).get("per_model_avg", {}).get("efficientnet_b3", 0.5),
                "xception_score":    res.get("cnn_analysis", {}).get("ensemble", {}).get("per_model_avg", {}).get("xception", 0.5),
                "vit_score":         res.get("cnn_analysis", {}).get("ensemble", {}).get("per_model_avg", {}).get("vit", 0.5),
                "convnext_score":    res.get("cnn_analysis", {}).get("ensemble", {}).get("per_model_avg", {}).get("convnext", 0.5),
                "cnn_score":         res.get("cnn_score", 0.5),
                
                "quality_mismatch":  res.get("quality_score", 0.0),
                "flow_score":        res.get("temporal_analysis", {}).get("flow_score", 0.5),
                "blink_score":       res.get("temporal_analysis", {}).get("blink_score", 0.5),
                "landmark_score":    res.get("temporal_analysis", {}).get("landmark_score", 0.5),
            }
            results.append(flat_res)
            
            print(f"  ✅ DONE. Pred: {pred_label} | GT: {gt_label} | Final Score: {flat_res['final_score']:.4f}")
            
        except Exception as e:
            import traceback
            print(f"  ❌ FATAL ERROR on {vid_name}: {e}")
            traceback.print_exc()

    if not results:
        print("\nNo results to save.")
        return

    # --- SAVE TO CSV ---
    fieldnames = list(results[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    print(f"\n💾 Saved detailed results to: {OUTPUT_CSV}")

    # --- PRINT SUMMARY ---
    correct_count = sum(1 for r in results if r["correct"])
    acc = correct_count / len(results)
    
    print("\n" + "#"*60)
    print(f"🏁 BATCH TEST SUMMARY ({len(results)} videos)")
    print("#"*60)
    print(f"  Accuracy     : {acc:.2%} ({correct_count}/{len(results)})")
    
    real_res = [r for r in results if r["gt_label"] == "Real"]
    fake_res = [r for r in results if r["gt_label"] == "Fake"]
    
    if real_res:
        fp = sum(1 for r in real_res if not r["correct"])
        print(f"  False Positives (Real pred Fake): {fp}/{len(real_res)}")
    
    if fake_res:
        fn = sum(1 for r in fake_res if not r["correct"])
        print(f"  False Negatives (Fake pred Real): {fn}/{len(fake_res)}")

    def avg(res_list, key):
        if not res_list: return 0.0
        return sum(r[key] for r in res_list) / len(res_list)

    if fake_res:
        print(f"\n  Avg Feature Scores — FAKE videos (higher is better):")
        for key in ["bilstm_score", "efficientnet_b3_score","xception_score","vit_score",
                    "convnext_score","quality_mismatch","flow_score",
                    "blink_score","landmark_score","final_score"]:
            print(f"    {key:<25} : {avg(fake_res, key):.4f}")

    print(f"\n  Decision sources used:")
    for src, count in Counter(r["decision_source"] for r in results).most_common():
        print(f"    {count}x  {src}")

    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\n  ❌ Wrong predictions ({len(wrong)}):")
        for r in wrong:
            print(f"\n    Video: {r['video']}")
            print(f"       GT={r['gt_label']}  Pred={r['predicted']}  "
                  f"Final={r['final_score']:.4f}  Conf={r['confidence']:.1%}")
            print(f"       bilstm={r['bilstm_score']:.4f}  "
                  f"effb3={r['efficientnet_b3_score']:.4f}  "
                  f"xception={r['xception_score']:.4f}  "
                  f"vit={r['vit_score']:.4f}  "
                  f"convnext={r['convnext_score']:.4f}  "
                  f"quality={r['quality_mismatch']:.4f}")
    else:
        print(f"\n  🎉 All videos predicted correctly!")

if __name__ == "__main__":
    main()