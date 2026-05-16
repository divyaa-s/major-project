# test_prnu.py
"""
PRNU Analysis Testing & Threshold Tuning Script

Run this BEFORE deploying PRNU to production to:
1. Verify PRNU works on your dataset
2. Find optimal thresholds
3. Measure performance metrics
4. Identify failure cases
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from prnu_analysis import analyze_prnu, compute_spatial_consistency, extract_prnu_noise
import cv2

# =====================================================
# CONFIGURATION
# =====================================================

REAL_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/train/real"
FAKE_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/train/fake"

# How many images to test (use more for accurate results)
NUM_SAMPLES = 200

# =====================================================
# TEST 1: Basic Functionality Test
# =====================================================

def test_single_image():
    """Quick test to ensure PRNU code doesn't crash"""
    print("\n" + "="*60)
    print("TEST 1: Basic Functionality Check")
    print("="*60)
    
    # Test on one real image
    real_images = [f for f in os.listdir(REAL_FOLDER) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(real_images) == 0:
        print("❌ ERROR: No images found in REAL folder!")
        return False
    
    test_image = os.path.join(REAL_FOLDER, real_images[0])
    
    print(f"Testing on: {real_images[0]}")
    
    try:
        result = analyze_prnu(test_image)
        
        if result["success"]:
            print("✅ PRNU analysis completed successfully!")
            print(f"   Verdict: {result['interpretation']['verdict']}")
            print(f"   Overall Score: {result['normalized_scores']['overall_prnu_score']:.4f}")
            return True
        else:
            print(f"❌ PRNU analysis failed: {result.get('error', 'Unknown error')}")
            return False
    
    except Exception as e:
        print(f"❌ CRASH: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# =====================================================
# TEST 2: Dataset Evaluation
# =====================================================

def evaluate_dataset(real_folder, fake_folder, num_samples=100):
    """
    Run PRNU on real vs fake datasets
    Returns scores and statistics
    """
    print("\n" + "="*60)
    print("TEST 2: Dataset Evaluation")
    print("="*60)
    
    real_scores = []
    fake_scores = []
    
    real_consistency = []
    real_fb_correlation = []
    
    fake_consistency = []
    fake_fb_correlation = []
    
    failed_real = 0
    failed_fake = 0
    
    # Evaluate REAL images
    real_images = [f for f in os.listdir(real_folder) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:num_samples]
    
    print(f"\n📊 Analyzing {len(real_images)} REAL images...")
    
    for img_name in tqdm(real_images, desc="REAL"):
        img_path = os.path.join(real_folder, img_name)
        
        try:
            result = analyze_prnu(img_path)
            
            if result["success"]:
                real_scores.append(result["normalized_scores"]["overall_prnu_score"])
                
                if result["raw_metrics"]["spatial_consistency"]:
                    real_consistency.append(result["raw_metrics"]["spatial_consistency"])
                
                if result["raw_metrics"]["face_background_correlation"]:
                    real_fb_correlation.append(result["raw_metrics"]["face_background_correlation"])
            else:
                failed_real += 1
        
        except Exception as e:
            failed_real += 1
            continue
    
    # Evaluate FAKE images
    fake_images = [f for f in os.listdir(fake_folder) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:num_samples]
    
    print(f"\n📊 Analyzing {len(fake_images)} FAKE images...")
    
    for img_name in tqdm(fake_images, desc="FAKE"):
        img_path = os.path.join(fake_folder, img_name)
        
        try:
            result = analyze_prnu(img_path)
            
            if result["success"]:
                fake_scores.append(result["normalized_scores"]["overall_prnu_score"])
                
                if result["raw_metrics"]["spatial_consistency"]:
                    fake_consistency.append(result["raw_metrics"]["spatial_consistency"])
                
                if result["raw_metrics"]["face_background_correlation"]:
                    fake_fb_correlation.append(result["raw_metrics"]["face_background_correlation"])
            else:
                failed_fake += 1
        
        except Exception as e:
            failed_fake += 1
            continue
    
    # Print Statistics
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print(f"\n📈 REAL Images (n={len(real_scores)}, failed={failed_real}):")
    print(f"   Overall PRNU Score: {np.mean(real_scores):.4f} ± {np.std(real_scores):.4f}")
    print(f"   Range: [{np.min(real_scores):.4f}, {np.max(real_scores):.4f}]")
    
    if real_consistency:
        print(f"   Spatial Consistency: {np.mean(real_consistency):.6f} ± {np.std(real_consistency):.6f}")
    if real_fb_correlation:
        print(f"   Face-BG Correlation: {np.mean(real_fb_correlation):.4f} ± {np.std(real_fb_correlation):.4f}")
    
    print(f"\n📈 FAKE Images (n={len(fake_scores)}, failed={failed_fake}):")
    print(f"   Overall PRNU Score: {np.mean(fake_scores):.4f} ± {np.std(fake_scores):.4f}")
    print(f"   Range: [{np.min(fake_scores):.4f}, {np.max(fake_scores):.4f}]")
    
    if fake_consistency:
        print(f"   Spatial Consistency: {np.mean(fake_consistency):.6f} ± {np.std(fake_consistency):.6f}")
    if fake_fb_correlation:
        print(f"   Face-BG Correlation: {np.mean(fake_fb_correlation):.4f} ± {np.std(fake_fb_correlation):.4f}")
    
    # Calculate optimal threshold
    optimal_threshold = (np.mean(real_scores) + np.mean(fake_scores)) / 2
    
    print(f"\n🎯 SUGGESTED THRESHOLD: {optimal_threshold:.4f}")
    print(f"   (Midpoint between real and fake means)")
    
    # Calculate accuracy at different thresholds
    print("\n📊 ACCURACY AT DIFFERENT THRESHOLDS:")
    
    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
        real_correct = sum(1 for s in real_scores if s < threshold)
        fake_correct = sum(1 for s in fake_scores if s >= threshold)
        
        accuracy = (real_correct + fake_correct) / (len(real_scores) + len(fake_scores))
        precision = fake_correct / (fake_correct + (len(real_scores) - real_correct)) if (fake_correct + (len(real_scores) - real_correct)) > 0 else 0
        recall = fake_correct / len(fake_scores) if len(fake_scores) > 0 else 0
        
        print(f"   Threshold {threshold:.2f}: Accuracy={accuracy:.2%}, Precision={precision:.2%}, Recall={recall:.2%}")
    
    # Visualization
    plot_distributions(real_scores, fake_scores, optimal_threshold)
    
    return {
        "real_scores": real_scores,
        "fake_scores": fake_scores,
        "real_mean": np.mean(real_scores),
        "fake_mean": np.mean(fake_scores),
        "optimal_threshold": optimal_threshold,
        "separation": abs(np.mean(fake_scores) - np.mean(real_scores))
    }


# =====================================================
# TEST 3: Visualization
# =====================================================

def plot_distributions(real_scores, fake_scores, threshold):
    """Create visualization of score distributions"""
    
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(real_scores, bins=30, alpha=0.6, label='Real', color='green', edgecolor='black')
    plt.hist(fake_scores, bins=30, alpha=0.6, label='Fake', color='red', edgecolor='black')
    plt.axvline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold={threshold:.3f}')
    plt.xlabel('PRNU Score')
    plt.ylabel('Frequency')
    plt.title('PRNU Score Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot([real_scores, fake_scores], labels=['Real', 'Fake'])
    plt.axhline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold={threshold:.3f}')
    plt.ylabel('PRNU Score')
    plt.title('PRNU Score Box Plot')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prnu_evaluation_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Visualization saved as 'prnu_evaluation_results.png'")
    plt.show()


# =====================================================
# TEST 4: Failure Case Analysis
# =====================================================

def analyze_failure_cases(real_folder, fake_folder, threshold=0.50, num_samples=50):
    """Find and analyze misclassified images"""
    
    print("\n" + "="*60)
    print("TEST 3: Failure Case Analysis")
    print("="*60)
    
    false_positives = []  # Real images classified as fake
    false_negatives = []  # Fake images classified as real
    
    # Check real images
    real_images = [f for f in os.listdir(real_folder) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:num_samples]
    
    for img_name in tqdm(real_images, desc="Checking REAL"):
        img_path = os.path.join(real_folder, img_name)
        result = analyze_prnu(img_path)
        
        if result["success"]:
            score = result["normalized_scores"]["overall_prnu_score"]
            if score >= threshold:  # Misclassified as fake
                false_positives.append((img_name, score))
    
    # Check fake images
    fake_images = [f for f in os.listdir(fake_folder) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:num_samples]
    
    for img_name in tqdm(fake_images, desc="Checking FAKE"):
        img_path = os.path.join(fake_folder, img_name)
        result = analyze_prnu(img_path)
        
        if result["success"]:
            score = result["normalized_scores"]["overall_prnu_score"]
            if score < threshold:  # Misclassified as real
                false_negatives.append((img_name, score))
    
    # Report
    print(f"\n❌ FALSE POSITIVES (Real classified as Fake): {len(false_positives)}")
    if false_positives:
        print("   Top 5 worst cases:")
        for img_name, score in sorted(false_positives, key=lambda x: x[1], reverse=True)[:5]:
            print(f"      {img_name}: {score:.4f}")
    
    print(f"\n❌ FALSE NEGATIVES (Fake classified as Real): {len(false_negatives)}")
    if false_negatives:
        print("   Top 5 worst cases:")
        for img_name, score in sorted(false_negatives, key=lambda x: x[1])[:5]:
            print(f"      {img_name}: {score:.4f}")
    
    return false_positives, false_negatives


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    print("="*60)
    print("PRNU ANALYSIS TESTING SUITE")
    print("="*60)
    
    # Test 1: Basic functionality
    if not test_single_image():
        print("\n⚠️  Basic test failed! Fix errors before proceeding.")
        exit(1)
    
    # Test 2: Full dataset evaluation
    results = evaluate_dataset(REAL_FOLDER, FAKE_FOLDER, num_samples=NUM_SAMPLES)
    
    # Check if PRNU is effective
    separation = results["separation"]
    
    print("\n" + "="*60)
    print("EFFECTIVENESS ASSESSMENT")
    print("="*60)
    
    if separation < 0.10:
        print("⚠️  WARNING: Poor separation between real and fake!")
        print(f"   Difference in means: {separation:.4f}")
        print("   PRNU may not be effective on this dataset.")
        print("   Consider:")
        print("   - Using different datasets")
        print("   - Adjusting preprocessing")
        print("   - PRNU might not work for this type of deepfake")
    
    elif separation < 0.20:
        print("⚠️  MODERATE: Some separation, but not ideal")
        print(f"   Difference in means: {separation:.4f}")
        print("   PRNU provides weak evidence.")
    
    else:
        print("✅ GOOD: Clear separation between real and fake!")
        print(f"   Difference in means: {separation:.4f}")
        print("   PRNU is effective on this dataset.")
    
    # Test 3: Failure analysis
    analyze_failure_cases(REAL_FOLDER, FAKE_FOLDER, 
                         threshold=results["optimal_threshold"], 
                         num_samples=100)
    
    # Final recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    print(f"\n1. Update prnu_analysis.py threshold:")
    print(f"   Change line: if prnu_score >= 0.50")
    print(f"   To:         if prnu_score >= {results['optimal_threshold']:.4f}")
    
    print(f"\n2. Expected performance:")
    real_correct = sum(1 for s in results["real_scores"] if s < results["optimal_threshold"])
    fake_correct = sum(1 for s in results["fake_scores"] if s >= results["optimal_threshold"])
    accuracy = (real_correct + fake_correct) / (len(results["real_scores"]) + len(results["fake_scores"]))
    print(f"   Accuracy: ~{accuracy:.1%}")
    
    print("\n3. Integration:")
    if separation > 0.15:
        print("   ✅ PRNU is reliable - integrate into main pipeline")
    else:
        print("   ⚠️  PRNU is weak - use as supplementary evidence only")
    
    print("\n" + "="*60)
    print("Testing complete! Check 'prnu_evaluation_results.png' for visualizations.")
    print("="*60)
