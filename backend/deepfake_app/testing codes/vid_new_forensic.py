"""
test_promising_forensic_features_on_videos.py

Tests the more promising features on video keyframes:
- Saliency Consistency
- Semantic Consistency
- Attention Inconsistency
- Visual Quality (skin, proportions, lighting, etc.)

Run: python test_promising_forensic_features_on_videos.py
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Import your forensic functions (adjust paths if needed)
from saliency_consistency import analyze_saliency_consistency
from semantic_consistency import analyze_semantic_consistency
from attention_inconsistency_detection import analyze_attention_inconsistency
from new_forensic_test import analyze_image_comprehensive  # combined visual quality

# =====================================================
# CONFIGURATION
# =====================================================

REAL_VIDEOS_FOLDER = r"D:\Projects\Major Project\Deepfake Detection\datasets\dfd\original"
FAKE_VIDEOS_FOLDER = r"D:\Projects\Major Project\Deepfake Detection\datasets\dfd\manipulated"

NUM_VIDEOS_PER_CLASS = 5           # keep small - these are slower
KEYFRAMES_PER_VIDEO = 20           # fewer frames to speed up

FEATURES_TO_TEST = [
    ("Saliency Consistency", analyze_saliency_consistency, "suspicion_score"),
    ("Semantic Consistency", analyze_semantic_consistency, "suspicion_score"),
    ("Attention Inconsistency", analyze_attention_inconsistency, "suspicion_score"),
    ("Visual Quality Combined", analyze_image_comprehensive, "overall_suspicion"),
]

# =====================================================
# KEYFRAME EXTRACTION
# =====================================================

def extract_keyframes(video_path, num_frames=KEYFRAMES_PER_VIDEO):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return []
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    keyframes = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        keyframes.append(frame)
    
    cap.release()
    return keyframes


# =====================================================
# FEATURE TESTING FUNCTION
# =====================================================

def test_feature_on_videos(feature_name, feature_func, score_key):
    print(f"\n{'='*90}")
    print(f"Testing: {feature_name}")
    print(f"{'='*90}")
    
    real_scores = []
    fake_scores = []
    
    # REAL videos
    real_videos = [f for f in os.listdir(REAL_VIDEOS_FOLDER) if f.lower().endswith(('.mp4', '.avi', '.mov'))][:NUM_VIDEOS_PER_CLASS]
    print(f"Processing {len(real_videos)} REAL videos...")
    
    for video_name in tqdm(real_videos, desc="REAL"):
        video_path = Path(REAL_VIDEOS_FOLDER) / video_name
        frames = extract_keyframes(video_path)
        
        for i, frame in enumerate(frames):
            temp_path = f"temp_frame_{i}.jpg"
            cv2.imwrite(temp_path, frame)
            
            try:
                result = feature_func(temp_path)
                if result and score_key in result:
                    score = result[score_key]
                    if isinstance(score, (int, float)):
                        real_scores.append(float(score))
            except Exception as e:
                print(f"  Error on {video_name} frame {i}: {e}")
            
            os.remove(temp_path)
    
    # FAKE videos
    fake_videos = [f for f in os.listdir(FAKE_VIDEOS_FOLDER) if f.lower().endswith(('.mp4', '.avi', '.mov'))][:NUM_VIDEOS_PER_CLASS]
    print(f"Processing {len(fake_videos)} FAKE videos...")
    
    for video_name in tqdm(fake_videos, desc="FAKE"):
        video_path = Path(FAKE_VIDEOS_FOLDER) / video_name
        frames = extract_keyframes(video_path)
        
        for i, frame in enumerate(frames):
            temp_path = f"temp_frame_{i}.jpg"
            cv2.imwrite(temp_path, frame)
            
            try:
                result = feature_func(temp_path)
                if result and score_key in result:
                    score = result[score_key]
                    if isinstance(score, (int, float)):
                        fake_scores.append(float(score))
            except Exception as e:
                print(f"  Error on {video_name} frame {i}: {e}")
            
            os.remove(temp_path)
    
    # Statistics
    if len(real_scores) == 0 or len(fake_scores) == 0:
        print("No valid scores collected.")
        return None
    
    real_mean = np.mean(real_scores)
    real_std = np.std(real_scores)
    fake_mean = np.mean(fake_scores)
    fake_std = np.std(fake_scores)
    
    separation = abs(fake_mean - real_mean)
    
    print(f"\nResults for {feature_name}:")
    print(f"  REAL frames (n={len(real_scores)}): {real_mean:.4f} ± {real_std:.4f}")
    print(f"  FAKE frames (n={len(fake_scores)}): {fake_mean:.4f} ± {fake_std:.4f}")
    print(f"  Separation: {separation:.4f}")
    
    threshold = (real_mean + fake_mean) / 2
    print(f"  Suggested threshold: {threshold:.4f}")
    
    real_correct = sum(1 for s in real_scores if s < threshold)
    fake_correct = sum(1 for s in fake_scores if s >= threshold)
    accuracy = (real_correct + fake_correct) / (len(real_scores) + len(fake_scores))
    print(f"  Accuracy at threshold: {accuracy:.2%}")
    
    return {
        "feature_name": feature_name,
        "real_mean": real_mean,
        "real_std": real_std,
        "fake_mean": fake_mean,
        "fake_std": fake_std,
        "separation": separation,
        "threshold": threshold,
        "accuracy": accuracy,
        "real_scores": real_scores,
        "fake_scores": fake_scores
    }


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print("\n" + "="*90)
    print("PROMISING FORENSIC FEATURES TEST ON VIDEO KEYFRAMES")
    print("="*90)
    
    all_results = []
    
    for feature_name, feature_func, score_key in FEATURES_TO_TEST:
        result = test_feature_on_videos(feature_name, feature_func, score_key)
        if result:
            all_results.append(result)
    
    # Summary table
    print("\n" + "="*90)
    print("SUMMARY TABLE")
    print("="*90)
    print(f"{'Feature':<30} {'Separation':<12} {'Accuracy':<12} {'Threshold':<12} {'Verdict'}")
    print("-"*90)
    
    for r in sorted(all_results, key=lambda x: x['separation'], reverse=True):
        verdict = "EXCELLENT" if r['separation'] > 0.30 else "GOOD" if r['separation'] > 0.15 else "WEAK" if r['separation'] > 0.05 else "POOR"
        print(f"{r['feature_name']:<30} {r['separation']:<12.4f} {r['accuracy']:<12.2%} {r['threshold']:<12.4f} {verdict}")
    
    # Boxplot
    plt.figure(figsize=(12, 6))
    positions = range(1, len(all_results)+1)
    labels = [r['feature_name'][:20] for r in all_results]
    
    plt.boxplot([r['real_scores'] for r in all_results], positions=positions, patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    plt.boxplot([r['fake_scores'] for r in all_results], positions=positions, patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    
    plt.xticks(positions, labels, rotation=45, ha='right')
    plt.title("Promising Forensic Features on Video Keyframes - Real vs Fake")
    plt.ylabel("Suspicion Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("promising_forensic_features_video_test.png", dpi=150)
    plt.show()
    
    print("\nVisualization saved as: promising_forensic_features_video_test.png")
    print("Done!")