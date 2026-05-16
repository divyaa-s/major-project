# quick_test_new_dataset.py

import os
import numpy as np
from tqdm import tqdm

# Update these paths
REAL_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/test/real_and_fake_face/training_real"
FAKE_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/test/real_and_fake_face/training_fake"

# Import your existing feature functions
from test_forensic_features import (
    compute_ela,
    analyze_noise_patterns,
)

from edge_coherence_analysis import analyze_edge_coherence


def quick_test(feature_name, feature_func, score_key, num_samples=50):
    """Quick test on new dataset"""
    
    print(f"\nTesting {feature_name} on NEW dataset...")
    
    real_scores = []
    fake_scores = []
    
    # Test REAL
    real_imgs = [f for f in os.listdir(REAL_FOLDER) 
                 if f.endswith(('.jpg', '.png'))][:num_samples]
    
    for img_name in tqdm(real_imgs, desc="REAL"):
        result = feature_func(os.path.join(REAL_FOLDER, img_name))
        if result and score_key in result:
            real_scores.append(result[score_key])
    
    # Test FAKE
    fake_imgs = [f for f in os.listdir(FAKE_FOLDER) 
                 if f.endswith(('.jpg', '.png'))][:num_samples]
    
    for img_name in tqdm(fake_imgs, desc="FAKE"):
        result = feature_func(os.path.join(FAKE_FOLDER, img_name))
        if result and score_key in result:
            fake_scores.append(result[score_key])
    
    # Results
    if len(real_scores) > 0 and len(fake_scores) > 0:
        sep = abs(np.mean(fake_scores) - np.mean(real_scores))
        
        print(f"Real: {np.mean(real_scores):.4f} ± {np.std(real_scores):.4f}")
        print(f"Fake: {np.mean(fake_scores):.4f} ± {np.std(fake_scores):.4f}")
        print(f"Separation: {sep:.4f}")
        
        if sep > 0.20:
            print("✅ EXCELLENT - This feature works on this dataset!")
        elif sep > 0.10:
            print("✅ GOOD - Moderate separation")
        elif sep > 0.05:
            print("⚠️ WEAK - Some separation")
        else:
            print("❌ POOR - Still doesn't work")
        
        return sep
    
    return 0.0

# Test three key features
print("="*70)
print("QUICK VALIDATION ON NEW DATASET")
print("="*70)

ela_sep = quick_test("ELA", compute_ela, "ela_score")
noise_sep = quick_test("Noise Analysis", analyze_noise_patterns, "noise_suspicion")
edge_sep = quick_test("Edge Coherence", analyze_edge_coherence, "suspicion_score")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if ela_sep > 0.15 or noise_sep > 0.15 or edge_sep > 0.15:
    print("\n🎉 SUCCESS! Features work on this dataset!")
    print("\nRecommendation: Train/test your models on THIS dataset instead.")
    print("This dataset has proper separation between real and fake.")
else:
    print("\n❌ Features still don't work well.")
    print("Dataset might still be too challenging.")
