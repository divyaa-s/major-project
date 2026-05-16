"""
Test visual quality features on the "easy" dataset
These should work much better since the fakes are visually different
"""

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from new_forensic_test import (
    analyze_skin_texture,
    analyze_face_proportions,
    analyze_lighting_consistency,
    analyze_color_distribution,
    detect_ai_artifacts,
    analyze_image_comprehensive
)

# ============================================================================
# CONFIGURATION
# ============================================================================
REAL_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/test/real_and_fake_face/training_real"
FAKE_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/test/real_and_fake_face/training_fake"
NUM_SAMPLES = 100

# ============================================================================
# TESTING FUNCTIONS
# ============================================================================
def test_single_feature(feature_name, feature_func, score_key='suspicion_score'):
    """Test a single feature and report separation"""
    
    print(f"\n{'='*70}")
    print(f"Testing: {feature_name}")
    print('='*70)
    
    real_scores = []
    fake_scores = []
    
    # Get image paths
    real_imgs = [os.path.join(REAL_FOLDER, f) for f in os.listdir(REAL_FOLDER) 
                 if f.endswith(('.jpg', '.png', '.jpeg'))][:NUM_SAMPLES]
    fake_imgs = [os.path.join(FAKE_FOLDER, f) for f in os.listdir(FAKE_FOLDER) 
                 if f.endswith(('.jpg', '.png', '.jpeg'))][:NUM_SAMPLES]
    
    # Test REAL images
    for img_path in tqdm(real_imgs, desc="REAL"):
        try:
            result = feature_func(img_path)
            if result and score_key in result:
                real_scores.append(result[score_key])
        except Exception as e:
            print(f"Error on {img_path}: {e}")
    
    # Test FAKE images
    for img_path in tqdm(fake_imgs, desc="FAKE"):
        try:
            result = feature_func(img_path)
            if result and score_key in result:
                fake_scores.append(result[score_key])
        except Exception as e:
            print(f"Error on {img_path}: {e}")
    
    if len(real_scores) > 0 and len(fake_scores) > 0:
        real_mean = np.mean(real_scores)
        real_std = np.std(real_scores)
        fake_mean = np.mean(fake_scores)
        fake_std = np.std(fake_scores)
        separation = abs(fake_mean - real_mean)
        
        print(f"\nResults:")
        print(f"  Real: {real_mean:.4f} ± {real_std:.4f}")
        print(f"  Fake: {fake_mean:.4f} ± {fake_std:.4f}")
        print(f"  Separation: {separation:.4f}")
        
        # Calculate simple accuracy at threshold 0.5
        real_correct = sum(1 for s in real_scores if s < 0.5)
        fake_correct = sum(1 for s in fake_scores if s >= 0.5)
        accuracy = (real_correct + fake_correct) / (len(real_scores) + len(fake_scores))
        
        print(f"  Accuracy @ 0.5 threshold: {accuracy:.2%}")
        
        # Evaluation
        if separation > 0.30:
            verdict = "✅ EXCELLENT - Strong discriminative power!"
        elif separation > 0.20:
            verdict = "✅ GOOD - Should work well"
        elif separation > 0.10:
            verdict = "⚠️ MODERATE - May be useful in combination"
        else:
            verdict = "❌ WEAK - Not discriminative enough"
        
        print(f"  {verdict}")
        
        return {
            'real_scores': real_scores,
            'fake_scores': fake_scores,
            'separation': separation,
            'accuracy': accuracy
        }
    
    return None

def plot_distributions(results_dict, save_path='feature_distributions.png'):
    """Plot score distributions for all features"""
    
    n_features = len(results_dict)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (feature_name, results) in enumerate(results_dict.items()):
        if results is None:
            continue
        
        ax = axes[idx]
        
        # Plot histograms
        ax.hist(results['real_scores'], bins=30, alpha=0.5, 
                label='Real', color='green', density=True)
        ax.hist(results['fake_scores'], bins=30, alpha=0.5, 
                label='Fake', color='red', density=True)
        
        ax.axvline(0.5, color='black', linestyle='--', 
                   label='Threshold', linewidth=2)
        
        ax.set_xlabel('Suspicion Score')
        ax.set_ylabel('Density')
        ax.set_title(f'{feature_name}\nSep: {results["separation"]:.3f}, Acc: {results["accuracy"]:.1%}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Distributions saved to '{save_path}'")

# ============================================================================
# MAIN TESTING
# ============================================================================
if __name__ == "__main__":
    
    print("="*70)
    print("VISUAL QUALITY FEATURE TESTING")
    print("="*70)
    print(f"Testing on {NUM_SAMPLES} real and {NUM_SAMPLES} fake images")
    
    results = {}
    
    # Test each feature
    results['Skin Texture'] = test_single_feature(
        "Skin Texture Analysis",
        analyze_skin_texture
    )
    
    results['Face Proportions'] = test_single_feature(
        "Face Proportion Analysis",
        analyze_face_proportions
    )
    
    results['Lighting Consistency'] = test_single_feature(
        "Lighting Consistency",
        analyze_lighting_consistency
    )
    
    results['Color Distribution'] = test_single_feature(
        "Color Distribution",
        analyze_color_distribution
    )
    
    results['AI Artifacts'] = test_single_feature(
        "AI Artifact Detection",
        detect_ai_artifacts
    )
    
    # Test combined approach
    print(f"\n{'='*70}")
    print("Testing: COMBINED APPROACH")
    print('='*70)
    
    combined_results = test_single_feature(
        "Combined Analysis",
        analyze_image_comprehensive,
        score_key='overall_suspicion'
    )
    
    if combined_results:
        results['Combined'] = combined_results
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if valid_results:
        best_feature = max(valid_results.items(), 
                          key=lambda x: x[1]['separation'])
        
        print(f"\n🏆 Best performing feature: {best_feature[0]}")
        print(f"   Separation: {best_feature[1]['separation']:.4f}")
        print(f"   Accuracy: {best_feature[1]['accuracy']:.2%}")
        
        # Plot distributions
        plot_distributions(valid_results)
        
        # Recommendations
        print("\n📋 Recommendations:")
        
        excellent = [k for k, v in valid_results.items() 
                    if v['separation'] > 0.30]
        good = [k for k, v in valid_results.items() 
               if 0.20 < v['separation'] <= 0.30]
        
        if excellent:
            print(f"\n✅ Excellent features (use these!):")
            for feat in excellent:
                print(f"   - {feat}: {valid_results[feat]['accuracy']:.1%} accuracy")
        
        if good:
            print(f"\n✅ Good features (combine with others):")
            for feat in good:
                print(f"   - {feat}: {valid_results[feat]['accuracy']:.1%} accuracy")
        
        if best_feature[1]['accuracy'] > 0.85:
            print(f"\n🎉 SUCCESS! These features work well on this dataset!")
            print(f"   You can now train a simple classifier for deployment.")
        elif best_feature[1]['accuracy'] > 0.70:
            print(f"\n✅ GOOD! Features show promise.")
            print(f"   Combine multiple features for better accuracy.")
        else:
            print(f"\n⚠️ Features need improvement.")
            print(f"   Consider deep learning approaches.")
    
    else:
        print("❌ No valid results obtained. Check error messages above.")