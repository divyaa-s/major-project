"""
Comprehensive Forensic Feature Testing
Tests ELA, Noise Analysis, Specular Highlights, and Benford's Law
"""

import cv2
import numpy as np
import os
from PIL import Image
import io
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

# =====================================================
# CONFIGURATION
# =====================================================

REAL_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/train/real"
FAKE_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/train/fake"
NUM_SAMPLES = 100


# =====================================================
# FEATURE 1: ELA (Error Level Analysis)
# =====================================================

def compute_ela(image_path, quality=90):
    """
    Error Level Analysis - Detects compression inconsistencies
    Returns: Suspicion score (0-1, higher = more suspicious)
    """
    try:
        # Load original
        original = Image.open(image_path).convert('RGB')
        
        # Resave at specified quality
        temp_buffer = io.BytesIO()
        original.save(temp_buffer, format='JPEG', quality=quality)
        temp_buffer.seek(0)
        resaved = Image.open(temp_buffer)
        
        # Convert to numpy
        original_np = np.array(original).astype(np.float32)
        resaved_np = np.array(resaved).astype(np.float32)
        
        # Compute difference
        ela = np.abs(original_np - resaved_np)
        
        # Normalize
        ela_gray = cv2.cvtColor(ela.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Calculate statistics
        mean_error = np.mean(ela_gray)
        std_error = np.std(ela_gray)
        
        # High uniformity (low std) = suspicious (GAN)
        # Real photos have varied compression artifacts
        if mean_error < 1e-6:
            return None
        
        uniformity = 1 - np.clip(std_error / mean_error, 0, 1)
        
        # Combine metrics
        # Low mean error + high uniformity = likely GAN (uniform compression)
        # High mean error + low uniformity = likely real (varied compression)
        mean_score = np.clip(mean_error / 50.0, 0, 1)  # Normalize
        
        ela_score = 0.6 * uniformity + 0.4 * (1 - mean_score)
        
        return {
            "ela_score": float(np.clip(ela_score, 0, 1)),
            "mean_error": float(mean_error),
            "std_error": float(std_error),
            "uniformity": float(uniformity)
        }
    
    except Exception as e:
        print(f"ELA failed: {e}")
        return None


# =====================================================
# FEATURE 2: Noise Analysis
# =====================================================

def analyze_noise_patterns(image_path):
    """
    Analyze noise characteristics to detect GAN images
    Real photos: Gaussian-like noise, higher entropy
    GAN images: Structured noise, lower entropy
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Apply high-pass filter to isolate noise
        img_float = img.astype(np.float32)
        
        # Gaussian blur (low-pass filter)
        blurred = cv2.GaussianBlur(img_float, (5, 5), 0)
        
        # High-pass = original - blurred
        noise = img_float - blurred
        
        # Statistical analysis
        noise_std = np.std(noise)
        noise_skewness = stats.skew(noise.flatten())
        noise_kurtosis = stats.kurtosis(noise.flatten())
        
        # Entropy analysis
        # Shift noise to positive range for histogram
        noise_shifted = noise + 128
        noise_shifted = np.clip(noise_shifted, 0, 255).astype(np.uint8)
        
        hist, _ = np.histogram(noise_shifted, bins=64, range=(0, 256))
        hist = hist / (hist.sum() + 1e-8)
        noise_entropy = stats.entropy(hist)
        
        # GAN images often have:
        # - Lower entropy (more structured)
        # - Higher kurtosis (sharper peaks)
        # - Non-Gaussian distribution
        
        # Gaussian deviation (real photos closer to 0)
        gaussian_deviation = abs(noise_skewness) + abs(noise_kurtosis - 3)
        
        # Normalize to 0-1 suspicion score
        max_entropy = np.log(64)  # Maximum entropy for 64 bins
        entropy_score = np.clip(1 - (noise_entropy / max_entropy), 0, 1)  # Lower entropy = suspicious
        gaussian_score = np.clip(gaussian_deviation / 6.0, 0, 1)  # Non-Gaussian = suspicious
        
        noise_suspicion = 0.5 * entropy_score + 0.5 * gaussian_score
        
        return {
            "noise_suspicion": float(np.clip(noise_suspicion, 0, 1)),
            "entropy": float(noise_entropy),
            "skewness": float(noise_skewness),
            "kurtosis": float(noise_kurtosis),
            "gaussian_deviation": float(gaussian_deviation)
        }
    
    except Exception as e:
        print(f"Noise analysis failed: {e}")
        return None


# =====================================================
# FEATURE 3: Specular Highlight Analysis
# =====================================================

def analyze_specular_highlights(image_path):
    """
    Detect unnatural specular highlights in faces
    GANs often produce unrealistic reflections
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Detect very bright regions (potential highlights)
        _, highlights = cv2.threshold(l_channel, 200, 255, cv2.THRESH_BINARY)
        
        # Calculate highlight statistics
        num_highlights = np.sum(highlights > 0)
        total_pixels = highlights.size
        highlight_ratio = num_highlights / total_pixels
        
        # Connected components analysis
        num_components, labels = cv2.connectedComponents(highlights)
        
        # Calculate component sizes
        component_sizes = []
        for i in range(1, num_components):  # Skip background
            size = np.sum(labels == i)
            component_sizes.append(size)
        
        if len(component_sizes) > 0:
            avg_component_size = np.mean(component_sizes)
            std_component_size = np.std(component_sizes)
            
            # Uniform small highlights = suspicious (GAN artifact)
            if avg_component_size > 0:
                uniformity = 1 - np.clip(std_component_size / avg_component_size, 0, 1)
            else:
                uniformity = 0.5
        else:
            uniformity = 0.5  # No highlights
            avg_component_size = 0
        
        # Ideal highlight ratio for natural faces: 2-8%
        if highlight_ratio < 0.02:
            ratio_score = 0.7  # Too few highlights (GAN missing reflections)
        elif highlight_ratio > 0.15:
            ratio_score = 0.6  # Too many highlights (GAN artifacts)
        else:
            ratio_score = 0.3  # Normal range
        
        # Component size analysis
        # Real faces: Few large highlights (eyes, nose tip)
        # GANs: Many small scattered highlights OR no highlights
        if avg_component_size < 10 and num_components > 5:
            size_score = 0.7  # Many tiny highlights = suspicious
        elif num_components < 2:
            size_score = 0.6  # No highlights = suspicious
        else:
            size_score = 0.3  # Normal
        
        suspicion = 0.4 * ratio_score + 0.4 * size_score + 0.2 * uniformity
        
        return {
            "specular_suspicion": float(np.clip(suspicion, 0, 1)),
            "highlight_ratio": float(highlight_ratio),
            "num_components": int(num_components - 1),
            "avg_component_size": float(avg_component_size),
            "uniformity": float(uniformity)
        }
    
    except Exception as e:
        print(f"Specular analysis failed: {e}")
        return None


# =====================================================
# FEATURE 4: Benford's Law Analysis
# =====================================================

def analyze_benford_law(image_path):
    """
    Check if image follows Benford's Law
    Natural images follow this distribution
    GAN images often deviate
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        # Apply DCT
        img_float = np.float32(img) / 255.0
        dct = cv2.dct(img_float)
        
        # Get non-zero coefficients
        coeffs = dct[np.abs(dct) > 0.01].flatten()  # Filter very small values
        
        # Extract first digits
        first_digits = []
        for coeff in coeffs:
            abs_coeff = abs(coeff)
            if abs_coeff >= 0.1:  # Ignore very small coefficients
                # Convert to string and get first non-zero digit
                coeff_str = f"{abs_coeff:.10f}"
                for char in coeff_str:
                    if char.isdigit() and char != '0':
                        first_digits.append(int(char))
                        break
        
        if len(first_digits) < 100:  # Need sufficient data
            return None
        
        # Count occurrences
        observed = np.zeros(9)
        for digit in first_digits:
            if 1 <= digit <= 9:
                observed[digit - 1] += 1
        
        if observed.sum() == 0:
            return None
        
        observed = observed / observed.sum()
        
        # Expected Benford distribution
        expected = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
        
        # Calculate Chi-square statistic
        chi_square = np.sum((observed - expected) ** 2 / (expected + 1e-8))
        
        # Calculate KL divergence as alternative metric
        kl_div = stats.entropy(observed + 1e-10, expected + 1e-10)
        
        # Normalize to 0-1 suspicion score
        # Higher deviation from Benford = more suspicious
        chi_suspicion = np.clip(chi_square / 0.5, 0, 1)
        kl_suspicion = np.clip(kl_div / 1.0, 0, 1)
        
        benford_suspicion = 0.5 * chi_suspicion + 0.5 * kl_suspicion
        
        return {
            "benford_suspicion": float(np.clip(benford_suspicion, 0, 1)),
            "chi_square": float(chi_square),
            "kl_divergence": float(kl_div),
            "num_coefficients": len(first_digits)
        }
    
    except Exception as e:
        print(f"Benford analysis failed: {e}")
        return None


# =====================================================
# TESTING FUNCTIONS
# =====================================================

def test_single_feature(feature_name, feature_func, image_path):
    """Test a single feature on one image"""
    
    print(f"\n{'='*70}")
    print(f"Testing {feature_name}")
    print('='*70)
    
    result = feature_func(image_path)
    
    if result is None:
        print(f"❌ {feature_name} failed on test image")
        return False
    
    print(f"✅ {feature_name} working!")
    print(f"Result: {result}")
    return True


def evaluate_feature(feature_name, feature_func, score_key):
    """Evaluate a feature on full dataset"""
    
    print(f"\n{'='*70}")
    print(f"Evaluating {feature_name}")
    print('='*70)
    
    real_scores = []
    fake_scores = []
    real_failed = 0
    fake_failed = 0
    
    # Test REAL images
    real_images = [f for f in os.listdir(REAL_FOLDER) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:NUM_SAMPLES]
    
    print(f"\nAnalyzing {len(real_images)} REAL images...")
    for img_name in tqdm(real_images, desc="REAL", ncols=70):
        img_path = os.path.join(REAL_FOLDER, img_name)
        
        try:
            result = feature_func(img_path)
            if result and score_key in result:
                real_scores.append(result[score_key])
            else:
                real_failed += 1
        except:
            real_failed += 1
    
    # Test FAKE images
    fake_images = [f for f in os.listdir(FAKE_FOLDER) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:NUM_SAMPLES]
    
    print(f"\nAnalyzing {len(fake_images)} FAKE images...")
    for img_name in tqdm(fake_images, desc="FAKE", ncols=70):
        img_path = os.path.join(FAKE_FOLDER, img_name)
        
        try:
            result = feature_func(img_path)
            if result and score_key in result:
                fake_scores.append(result[score_key])
            else:
                fake_failed += 1
        except:
            fake_failed += 1
    
    # Results
    print(f"\n{'='*70}")
    print(f"{feature_name} RESULTS")
    print('='*70)
    
    if len(real_scores) == 0 or len(fake_scores) == 0:
        print("❌ Insufficient data - feature failed")
        return None
    
    print(f"\n📈 REAL Images (n={len(real_scores)}, failed={real_failed}):")
    print(f"   Score: {np.mean(real_scores):.4f} ± {np.std(real_scores):.4f}")
    print(f"   Range: [{np.min(real_scores):.4f}, {np.max(real_scores):.4f}]")
    
    print(f"\n📈 FAKE Images (n={len(fake_scores)}, failed={fake_failed}):")
    print(f"   Score: {np.mean(fake_scores):.4f} ± {np.std(fake_scores):.4f}")
    print(f"   Range: [{np.min(fake_scores):.4f}, {np.max(fake_scores):.4f}]")
    
    # Calculate separation
    separation = abs(np.mean(fake_scores) - np.mean(real_scores))
    print(f"\n🎯 Separation: {separation:.4f}")
    
    # Performance at threshold 0.5
    real_correct = sum(1 for s in real_scores if s < 0.5)
    fake_correct = sum(1 for s in fake_scores if s >= 0.5)
    accuracy = (real_correct + fake_correct) / (len(real_scores) + len(fake_scores))
    
    print(f"📊 Accuracy at threshold 0.5: {accuracy:.2%}")
    
    # Assessment
    if separation < 0.10:
        effectiveness = "❌ POOR"
        recommendation = "Do NOT use"
    elif separation < 0.20:
        effectiveness = "⚠️ WEAK"
        recommendation = "Use as supplementary only"
    elif separation < 0.30:
        effectiveness = "✅ MODERATE"
        recommendation = "Good addition to ensemble"
    else:
        effectiveness = "⭐ EXCELLENT"
        recommendation = "Strong standalone feature"
    
    print(f"\n{effectiveness}")
    print(f"Recommendation: {recommendation}")
    
    return {
        "feature_name": feature_name,
        "real_scores": real_scores,
        "fake_scores": fake_scores,
        "separation": separation,
        "accuracy": accuracy,
        "effectiveness": effectiveness,
        "recommendation": recommendation
    }


# =====================================================
# VISUALIZATION
# =====================================================

def plot_comparison(results_list):
    """Create comprehensive comparison visualization"""
    
    num_features = len([r for r in results_list if r is not None])
    
    if num_features == 0:
        print("No features to plot")
        return
    
    fig, axes = plt.subplots(2, num_features, figsize=(5*num_features, 10))
    
    if num_features == 1:
        axes = axes.reshape(2, 1)
    
    col = 0
    for result in results_list:
        if result is None:
            continue
        
        real_scores = result['real_scores']
        fake_scores = result['fake_scores']
        
        # Histogram
        axes[0, col].hist(real_scores, bins=20, alpha=0.7, label='Real', 
                          color='green', edgecolor='black')
        axes[0, col].hist(fake_scores, bins=20, alpha=0.7, label='Fake', 
                          color='red', edgecolor='black')
        axes[0, col].axvline(0.5, color='blue', linestyle='--', linewidth=2)
        axes[0, col].set_xlabel('Score', fontsize=11)
        axes[0, col].set_ylabel('Frequency', fontsize=11)
        axes[0, col].set_title(f"{result['feature_name']}\n"
                               f"Sep={result['separation']:.3f}, "
                               f"Acc={result['accuracy']:.1%}", 
                               fontsize=12, fontweight='bold')
        axes[0, col].legend()
        axes[0, col].grid(alpha=0.3)
        
        # Box plot
        bp = axes[1, col].boxplot([real_scores, fake_scores], 
                                   tick_labels=['Real', 'Fake'],
                                   patch_artist=True)
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')
        axes[1, col].axhline(0.5, color='blue', linestyle='--', linewidth=2)
        axes[1, col].set_ylabel('Score', fontsize=11)
        axes[1, col].grid(alpha=0.3, axis='y')
        
        col += 1
    
    plt.tight_layout()
    plt.savefig('forensic_features_comparison.png', dpi=150, bbox_inches='tight')
    print("\n📊 Comparison visualization saved as 'forensic_features_comparison.png'")
    plt.show()


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print(" "*15 + "FORENSIC FEATURES TESTING SUITE")
    print("="*70)
    
    # Get first real image for single tests
    real_images = [f for f in os.listdir(REAL_FOLDER) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(real_images) == 0:
        print("❌ No images found in REAL folder!")
        exit(1)
    
    test_image = os.path.join(REAL_FOLDER, real_images[0])
    
    # ===== TEST 1: Basic Functionality =====
    print("\n" + "="*70)
    print("TEST 1: Basic Functionality Check")
    print("="*70)
    print(f"Testing on: {real_images[0]}")
    
    features_working = []
    
    if test_single_feature("ELA", compute_ela, test_image):
        features_working.append(("ELA", compute_ela, "ela_score"))
    
    if test_single_feature("Noise Analysis", analyze_noise_patterns, test_image):
        features_working.append(("Noise Analysis", analyze_noise_patterns, "noise_suspicion"))
    
    if test_single_feature("Specular Highlights", analyze_specular_highlights, test_image):
        features_working.append(("Specular Highlights", analyze_specular_highlights, "specular_suspicion"))
    
    if test_single_feature("Benford's Law", analyze_benford_law, test_image):
        features_working.append(("Benford's Law", analyze_benford_law, "benford_suspicion"))
    
    if len(features_working) == 0:
        print("\n❌ All features failed! Check your image paths and dependencies.")
        exit(1)
    
    print(f"\n✅ {len(features_working)}/4 features working")
    
    # ===== TEST 2: Full Dataset Evaluation =====
    print("\n" + "="*70)
    print("TEST 2: Full Dataset Evaluation")
    print("="*70)
    
    results = []
    
    for feature_name, feature_func, score_key in features_working:
        result = evaluate_feature(feature_name, feature_func, score_key)
        results.append(result)
    
    # ===== FINAL COMPARISON =====
    print("\n\n" + "="*70)
    print("FINAL COMPARISON & RECOMMENDATIONS")
    print("="*70)
    
    # Sort by separation (best first)
    valid_results = [r for r in results if r is not None]
    valid_results.sort(key=lambda x: x['separation'], reverse=True)
    
    print("\n📊 Features Ranked by Effectiveness:\n")
    
    for i, result in enumerate(valid_results, 1):
        print(f"{i}. {result['feature_name']}")
        print(f"   Separation: {result['separation']:.4f}")
        print(f"   Accuracy: {result['accuracy']:.2%}")
        print(f"   {result['recommendation']}")
        print()
    
    # Recommendations
    print("="*70)
    print("INTEGRATION RECOMMENDATIONS")
    print("="*70)
    
    excellent = [r for r in valid_results if r['separation'] >= 0.30]
    moderate = [r for r in valid_results if 0.20 <= r['separation'] < 0.30]
    weak = [r for r in valid_results if 0.10 <= r['separation'] < 0.20]
    poor = [r for r in valid_results if r['separation'] < 0.10]
    
    if excellent:
        print("\n⭐ INTEGRATE IMMEDIATELY:")
        for r in excellent:
            print(f"   ✅ {r['feature_name']} (sep={r['separation']:.3f})")
    
    if moderate:
        print("\n✅ GOOD ADDITIONS:")
        for r in moderate:
            print(f"   👍 {r['feature_name']} (sep={r['separation']:.3f})")
    
    if weak:
        print("\n⚠️ USE AS SUPPLEMENTARY:")
        for r in weak:
            print(f"   ⚡ {r['feature_name']} (sep={r['separation']:.3f})")
    
    if poor:
        print("\n❌ DO NOT USE:")
        for r in poor:
            print(f"   ✗ {r['feature_name']} (sep={r['separation']:.3f})")
    
    # Visualization
    plot_comparison(results)
    
    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70)