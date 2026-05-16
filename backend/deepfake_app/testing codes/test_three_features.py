# test_three_features.py

from multi_scale_frequency_analysis import multi_scale_frequency_analysis
from edge_coherence_analysis import analyze_edge_coherence
# from attention_inconsistency import analyze_attention_inconsistency  # If using this

import os
import numpy as np
from tqdm import tqdm

REAL_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/train/real"
FAKE_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/train/fake"
NUM_SAMPLES = 100

def test_feature(feature_name, feature_func, score_key):
    """Test a feature"""
    
    print(f"\n{'='*70}")
    print(f"Testing {feature_name}")
    print('='*70)
    
    real_scores = []
    fake_scores = []
    
    # Test real
    real_images = [f for f in os.listdir(REAL_FOLDER) 
                   if f.lower().endswith(('.jpg', '.png'))][:NUM_SAMPLES]
    
    for img_name in tqdm(real_images, desc="REAL"):
        result = feature_func(os.path.join(REAL_FOLDER, img_name))
        if result and score_key in result:
            real_scores.append(result[score_key])
    
    # Test fake
    fake_images = [f for f in os.listdir(FAKE_FOLDER) 
                   if f.lower().endswith(('.jpg', '.png'))][:NUM_SAMPLES]
    
    for img_name in tqdm(fake_images, desc="FAKE"):
        result = feature_func(os.path.join(FAKE_FOLDER, img_name))
        if result and score_key in result:
            fake_scores.append(result[score_key])
    
    # Results
    if len(real_scores) > 0 and len(fake_scores) > 0:
        separation = abs(np.mean(fake_scores) - np.mean(real_scores))
        print(f"\nReal: {np.mean(real_scores):.4f} ± {np.std(real_scores):.4f}")
        print(f"Fake: {np.mean(fake_scores):.4f} ± {np.std(fake_scores):.4f}")
        print(f"Separation: {separation:.4f}")
    else:
        print("Failed")

# Test all three
test_feature("Multi-Scale Wavelet", multi_scale_frequency_analysis, "suspicion_score")
test_feature("Edge Coherence", analyze_edge_coherence, "suspicion_score")
# test_feature("Attention Inconsistency", analyze_attention_inconsistency, "suspicion_score")