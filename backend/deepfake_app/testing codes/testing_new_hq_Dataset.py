"""
Quick test to see if features work with domain-matched datasets
CelebHQ (real) vs AI-generated professional-looking faces (fake)
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================================
# SIMPLE FEATURE TEST
# ============================================================================
def extract_quick_features(img_path):
    """Extract simple features that should work with matched domains"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Texture variance (AI often smoother)
    texture_var = np.std(gray)
    
    # 2. High-frequency content (AI lacks fine details)
    kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
    high_freq = cv2.filter2D(gray, -1, kernel)
    high_freq_energy = np.mean(np.abs(high_freq))
    
    # 3. Edge coherence
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # 4. Color saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:,:,1])
    
    # 5. FFT-based artifact detection
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    h, w = magnitude.shape
    center_y, center_x = h // 2, w // 2
    
    # Check for checkerboard artifacts at specific frequencies
    artifact_score = 0
    for freq in [2, 4, 8]:
        y_sample = center_y + h // (freq * 2)
        x_sample = center_x + w // (freq * 2)
        if y_sample < h and x_sample < w:
            artifact_score += magnitude[y_sample, x_sample]
    
    artifact_score = artifact_score / magnitude[center_y, center_x]
    
    return {
        'texture_var': texture_var,
        'high_freq_energy': high_freq_energy,
        'edge_density': edge_density,
        'saturation': saturation,
        'artifact_score': artifact_score
    }

def quick_test_features(real_folder, fake_folder, num_samples=100):
    """Test features on domain-matched datasets"""
    
    print("="*70)
    print("TESTING FEATURES WITH DOMAIN-MATCHED DATASETS")
    print("="*70)
    print(f"Real: CelebHQ (professional photos)")
    print(f"Fake: AI-generated professional-looking faces")
    print()
    
    # Get images
    real_imgs = [os.path.join(real_folder, f) for f in os.listdir(real_folder) 
                 if f.endswith(('.jpg', '.png', '.jpeg'))][:num_samples]
    fake_imgs = [os.path.join(fake_folder, f) for f in os.listdir(fake_folder) 
                 if f.endswith(('.jpg', '.png', '.jpeg'))][:num_samples]
    
    # Extract features
    real_features = {k: [] for k in ['texture_var', 'high_freq_energy', 
                                     'edge_density', 'saturation', 'artifact_score']}
    fake_features = {k: [] for k in ['texture_var', 'high_freq_energy', 
                                     'edge_density', 'saturation', 'artifact_score']}
    
    print("Extracting features from REAL images...")
    for img_path in tqdm(real_imgs):
        features = extract_quick_features(img_path)
        if features:
            for k, v in features.items():
                real_features[k].append(v)
    
    print("Extracting features from FAKE images...")
    for img_path in tqdm(fake_imgs):
        features = extract_quick_features(img_path)
        if features:
            for k, v in features.items():
                fake_features[k].append(v)
    
    # Compare
    print("\n" + "="*70)
    print("FEATURE COMPARISONS")
    print("="*70)
    
    results = {}
    
    for feature_name in real_features.keys():
        real_vals = np.array(real_features[feature_name])
        fake_vals = np.array(fake_features[feature_name])
        
        real_mean = np.mean(real_vals)
        fake_mean = np.mean(fake_vals)
        separation = abs(real_mean - fake_mean)
        
        # Normalize separation by scale
        scale = max(real_mean, fake_mean, 1e-10)
        normalized_sep = separation / scale
        
        print(f"\n{feature_name}:")
        print(f"  Real: {real_mean:.4f} ± {np.std(real_vals):.4f}")
        print(f"  Fake: {fake_mean:.4f} ± {np.std(fake_vals):.4f}")
        print(f"  Separation: {separation:.4f} ({normalized_sep:.2%})")
        
        # Simple classification
        threshold = (real_mean + fake_mean) / 2
        if real_mean < fake_mean:
            real_correct = np.sum(real_vals < threshold)
            fake_correct = np.sum(fake_vals >= threshold)
        else:
            real_correct = np.sum(real_vals >= threshold)
            fake_correct = np.sum(fake_vals < threshold)
        
        accuracy = (real_correct + fake_correct) / (len(real_vals) + len(fake_vals))
        print(f"  Accuracy @ threshold: {accuracy:.2%}")
        
        if accuracy > 0.80:
            print(f"  ✅ EXCELLENT - This feature works!")
        elif accuracy > 0.70:
            print(f"  ✅ GOOD - Useful feature")
        elif accuracy > 0.60:
            print(f"  ⚠️ MODERATE - May help in combination")
        else:
            print(f"  ❌ WEAK - Not discriminative")
        
        results[feature_name] = {
            'real_mean': real_mean,
            'fake_mean': fake_mean,
            'separation': separation,
            'accuracy': accuracy,
            'real_vals': real_vals,
            'fake_vals': fake_vals
        }
    
    # Plot distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (feature_name, data) in enumerate(results.items()):
        ax = axes[idx]
        
        ax.hist(data['real_vals'], bins=30, alpha=0.5, label='Real (CelebHQ)', 
                color='green', density=True)
        ax.hist(data['fake_vals'], bins=30, alpha=0.5, label='Fake (AI)', 
                color='red', density=True)
        
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Density')
        ax.set_title(f'{feature_name}\nAcc: {data["accuracy"]:.1%}, Sep: {data["separation"]:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig('celebhq_feature_test.png', dpi=150)
    print(f"\n📊 Distribution plots saved to 'celebhq_feature_test.png'")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    best_features = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print("\n🏆 Best Features (by accuracy):")
    for i, (name, data) in enumerate(best_features[:3], 1):
        print(f"  {i}. {name}: {data['accuracy']:.2%}")
    
    max_acc = best_features[0][1]['accuracy']
    
    if max_acc > 0.80:
        print("\n🎉 SUCCESS! Features work with domain-matched datasets!")
        print("   ✅ You can now train classifiers")
        print("   ✅ Statistical features are viable")
    elif max_acc > 0.70:
        print("\n✅ GOOD! Features show promise")
        print("   → Combine multiple features")
        print("   → Use ensemble methods")
    elif max_acc > 0.60:
        print("\n⚠️ MODERATE performance")
        print("   → Still better than before (57%)")
        print("   → Deep learning recommended")
    else:
        print("\n❌ Still struggling")
        print("   → Must use deep learning")
    
    return results

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    
    CELEBHQ_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/celeba_hq_256"  # Your new CelebHQ dataset
    FAKE_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/train/fake"       # Your AI-generated fakes
    
    results = quick_test_features(CELEBHQ_FOLDER, FAKE_FOLDER, num_samples=200)
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    
    best_acc = max(r['accuracy'] for r in results.values())
    
    if best_acc > 0.75:
        print("\n1. ✅ Retrain your statistical classifier with CelebHQ")
        print("   → Should achieve 75-90% accuracy")
        print("   → Use Random Forest or SVM")
        print("\n2. ✅ Try deep learning for 85-95% accuracy")
        print("   → ResNet with transfer learning")
    elif best_acc > 0.65:
        print("\n1. ⚠️ Statistical features show some promise")
        print("   → Combine all features in ensemble")
        print("\n2. ✅ Deep learning strongly recommended")
        print("   → Should achieve 80-90%")
    else:
        print("\n1. ❌ Statistical features still insufficient")
        print("\n2. ✅ MUST use deep learning")
        print("   → Domain matching helps but isn't enough")