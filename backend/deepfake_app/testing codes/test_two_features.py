# test_final_three_features.py

from saliency_consistency import analyze_saliency_consistency
from semantic_consistency import analyze_semantic_consistency

import os
import numpy as np
from tqdm import tqdm

REAL_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/valid/real"
FAKE_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/valid/fake"
NUM_SAMPLES = 50  # Smaller sample (these are slower)

def test_feature(name, func, key):
    print(f"\n{'='*70}")
    print(f"Testing {name}")
    print('='*70)
    
    real_scores, fake_scores = [], []
    
    real_imgs = [f for f in os.listdir(REAL_FOLDER)[:NUM_SAMPLES] 
                 if f.endswith(('.jpg', '.png'))]
    
    for img in tqdm(real_imgs, desc="REAL"):
        result = func(os.path.join(REAL_FOLDER, img))
        if result and key in result:
            real_scores.append(result[key])
    
    fake_imgs = [f for f in os.listdir(FAKE_FOLDER)[:NUM_SAMPLES]
                 if f.endswith(('.jpg', '.png'))]
    
    for img in tqdm(fake_imgs, desc="FAKE"):
        result = func(os.path.join(FAKE_FOLDER, img))
        if result and key in result:
            fake_scores.append(result[key])
    
    if real_scores and fake_scores:
        sep = abs(np.mean(fake_scores) - np.mean(real_scores))
        print(f"Real: {np.mean(real_scores):.4f} ± {np.std(real_scores):.4f}")
        print(f"Fake: {np.mean(fake_scores):.4f} ± {np.std(fake_scores):.4f}")
        print(f"Separation: {sep:.4f}")

test_feature("Saliency Consistency", analyze_saliency_consistency, "suspicion_score")
test_feature("Semantic Consistency", analyze_semantic_consistency, "suspicion_score")