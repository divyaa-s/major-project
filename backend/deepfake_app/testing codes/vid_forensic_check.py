"""
test_forensic_features_on_videos.py

Tests multiple forensic features (PRNU, Multi-Scale Frequency, Edge Coherence, etc.)
on keyframes extracted from REAL and FAKE videos.

Features tested:
1. PRNU (from prnu_analysis.py)
2. Multi-Scale Frequency (from multi_scale_frequency_analysis.py)
3. Edge Coherence (from edge_coherence_analysis.py)
4. Saliency Consistency (from saliency_consistency.py)  # optional, slower
5. Semantic Consistency (from semantic_consistency.py)  # optional

Run: python test_forensic_features_on_videos.py
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Import your forensic functions (adjust paths if needed)
from prnu_analysis import analyze_prnu
from multi_scale_frequency_analysis import multi_scale_frequency_analysis
from edge_coherence_analysis import analyze_edge_coherence
# from saliency_consistency import analyze_saliency_consistency      # uncomment if you want
# from semantic_consistency import analyze_semantic_consistency     # uncomment if you want

# =====================================================
# CONFIGURATION
# =====================================================

REAL_VIDEOS_FOLDER = r"D:\Projects\Major Project\Deepfake Detection\datasets\dfd\original"
FAKE_VIDEOS_FOLDER = r"D:\Projects\Major Project\Deepfake Detection\datasets\dfd\manipulated"

NUM_VIDEOS_PER_CLASS = 10          # how many videos per class to test
KEYFRAMES_PER_VIDEO = 30           # how many frames to extract per video
FEATURES_TO_TEST = [
    ("PRNU", analyze_prnu, "overall_prnu_score"),
    ("Multi-Scale Frequency", multi_scale_frequency_analysis, "suspicion_score"),
    ("Edge Coherence", analyze_edge_coherence, "suspicion_score"),
    # ("Saliency Consistency", analyze_saliency_consistency, "suspicion_score"),     # slow
    # ("Semantic Consistency", analyze_semantic_consistency, "suspicion_score"),    # slow
]

# =====================================================
# KEYFRAME EXTRACTION (same as your hybrid)
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

def test_feature_on_videos(feature_name, feature_func, score_key, real_folder, fake_folder):
    print(f"\n{'='*80}")
    print(f"Testing feature: {feature_name}")
    print(f"{'='*80}")
    
    real_scores = []
    fake_scores = []
    
    # REAL videos
    real_videos = [f for f in os.listdir(real_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))][:NUM_VIDEOS_PER_CLASS]
    print(f"Processing {len(real_videos)} REAL videos...")
    
    for video_name in tqdm(real_videos, desc="REAL videos"):
        video_path = Path(real_folder) / video_name
        frames = extract_keyframes(video_path)
        
        for frame in frames:
            # Convert frame to temp image path (some functions need path)
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            try:
                result = feature_func(temp_path)
                if result and score_key in result:
                    score = result[score_key]
                    if isinstance(score, (int, float)):
                        real_scores.append(float(score))
            except Exception as e:
                print(f"  Error on {video_name} frame: {e}")
    
    # FAKE videos
    fake_videos = [f for f in os.listdir(fake_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))][:NUM_VIDEOS_PER_CLASS]
    print(f"Processing {len(fake_videos)} FAKE videos...")
    
    for video_name in tqdm(fake_videos, desc="FAKE videos"):
        video_path = Path(fake_folder) / video_name
        frames = extract_keyframes(video_path)
        
        for frame in frames:
            temp_path = "temp_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            try:
                result = feature_func(temp_path)
                if result and score_key in result:
                    score = result[score_key]
                    if isinstance(score, (int, float)):
                        fake_scores.append(float(score))
            except Exception as e:
                print(f"  Error on {video_name} frame: {e}")
    
    # Clean up temp file
    if os.path.exists("temp_frame.jpg"):
        os.remove("temp_frame.jpg")
    
    # Statistics
    if len(real_scores) == 0 or len(fake_scores) == 0:
        print("No valid scores collected. Feature may not be working.")
        return None
    
    real_mean = np.mean(real_scores)
    real_std = np.std(real_scores)
    fake_mean = np.mean(fake_scores)
    fake_std = np.std(fake_scores)
    
    separation = abs(fake_mean - real_mean)
    
    print(f"\nResults for {feature_name}:")
    print(f"  REAL frames (n={len(real_scores)}): {real_mean:.4f} ± {real_std:.4f}")
    print(f"  FAKE frames (n={len(fake_scores)}): {fake_mean:.4f} ± {fake_std:.4f}")
    print(f"  Separation (higher is better): {separation:.4f}")
    
    # Simple threshold suggestion (midpoint)
    threshold = (real_mean + fake_mean) / 2
    print(f"  Suggested threshold: {threshold:.4f}")
    
    # Accuracy at this threshold (assuming higher score = fake)
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
# MAIN EXECUTION & VISUALIZATION
# =====================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("FORENSIC FEATURES TEST ON VIDEO KEYFRAMES")
    print("="*80)
    
    all_results = []
    
    for feature_name, feature_func, score_key in FEATURES_TO_TEST:
        result = test_feature_on_videos(
            feature_name,
            feature_func,
            score_key,
            REAL_VIDEOS_FOLDER,
            FAKE_VIDEOS_FOLDER
        )
        if result:
            all_results.append(result)
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY OF ALL FEATURES")
    print("="*80)
    print(f"{'Feature':<25} {'Separation':<12} {'Accuracy':<12} {'Threshold':<12} {'Recommendation'}")
    print("-"*80)
    
    for r in sorted(all_results, key=lambda x: x['separation'], reverse=True):
        rec = "EXCELLENT" if r['separation'] > 0.30 else \
              "GOOD" if r['separation'] > 0.15 else \
              "WEAK" if r['separation'] > 0.05 else "POOR"
        print(f"{r['feature_name']:<25} {r['separation']:<12.4f} {r['accuracy']:<12.2%} {r['threshold']:<12.4f} {rec}")
    
    # Plot boxplots
    plt.figure(figsize=(12, 6))
    scores_real = [r['real_scores'] for r in all_results]
    scores_fake = [r['fake_scores'] for r in all_results]
    labels = [r['feature_name'][:15] for r in all_results]
    
    plt.boxplot(scores_real, positions=range(1, len(all_results)+1), patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    plt.boxplot(scores_fake, positions=range(1, len(all_results)+1), patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    
    plt.xticks(range(1, len(all_results)+1), labels, rotation=45)
    plt.title("Forensic Features on Video Keyframes - Real vs Fake")
    plt.ylabel("Suspicion Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("forensic_features_video_test.png", dpi=150)
    plt.show()
    
    print("\nVisualization saved as: forensic_features_video_test.png")
    print("Done!")