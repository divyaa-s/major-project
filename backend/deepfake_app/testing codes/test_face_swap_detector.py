"""
PRNU Face-Swap Detection - Standalone Test
Everything in one file - no import issues
"""

import cv2
import numpy as np
import os
from PIL import Image
from facenet_pytorch import MTCNN
from scipy.signal import wiener
import matplotlib.pyplot as plt
from tqdm import tqdm

# =====================================================
# CONFIGURATION
# =====================================================

REAL_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/train/real"
FAKE_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/train/fake"
NUM_SAMPLES = 100

# Initialize face detector
mtcnn = MTCNN(keep_all=False, device='cpu')

# =====================================================
# PRNU EXTRACTION
# =====================================================

def extract_prnu_noise(image, size=128):
    """Extract PRNU noise using Wiener filter"""
    if image is None or image.size == 0:
        return None
    
    try:
        img = cv2.resize(image, (size, size))
        img = img.astype(np.float32) / 255.0
        
        # Wiener filter denoising
        denoised = wiener(img, (5, 5))
        noise = img - denoised
        
        # Zero-mean normalization
        noise = noise - np.mean(noise)
        
        # L2 normalization
        norm = np.linalg.norm(noise) + 1e-8
        noise = noise / norm
        
        return noise
    
    except Exception as e:
        print(f"ERROR: PRNU extraction failed: {e}")
        return None


# =====================================================
# FACE-SWAP DETECTION
# =====================================================

def detect_face_swap(image_path, verbose=False):
    """
    Detect face-swapped deepfakes using PRNU correlation analysis
    
    Args:
        image_path: Path to image
        verbose: Print detailed progress
    
    Returns:
        dict with detection result
    """
    
    if verbose:
        print(f"\n🔍 Analyzing: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return {"success": False, "error": "Failed to load image"}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    if verbose:
        print(f"   Image size: {w}x{h}")
    
    # Detect face
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        boxes, _ = mtcnn.detect(img_pil)
        
        if boxes is None or len(boxes) == 0:
            if verbose:
                print("   ❌ No face detected")
            return {"success": False, "error": "No face detected"}
        
        x1, y1, x2, y2 = boxes[0].astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        face_width = x2 - x1
        face_height = y2 - y1
        
        if verbose:
            print(f"   ✓ Face detected: {face_width}x{face_height} at ({x1},{y1})")
        
        if face_width < 32 or face_height < 32:
            return {"success": False, "error": "Face too small"}
        
    except Exception as e:
        return {"success": False, "error": f"Face detection failed: {e}"}
    
    # Extract face region
    face = gray[y1:y2, x1:x2]
    
    # ===== RELAXED BACKGROUND EXTRACTION =====
    # For small images (256x256), use more aggressive background extraction
    
    bg_regions = []
    margin = 5  # Reduced from 20 to allow more background
    
    # Strategy 1: Four corners (even if small)
    corner_size = 40  # Reduced from 80
    
    # Top-left corner
    if y1 > corner_size and x1 > corner_size:
        bg_regions.append(("Top-Left", gray[0:y1-margin, 0:x1-margin]))
    
    # Top-right corner
    if y1 > corner_size and (w - x2) > corner_size:
        bg_regions.append(("Top-Right", gray[0:y1-margin, x2+margin:w]))
    
    # Bottom-left corner
    if (h - y2) > corner_size and x1 > corner_size:
        bg_regions.append(("Bottom-Left", gray[y2+margin:h, 0:x1-margin]))
    
    # Bottom-right corner
    if (h - y2) > corner_size and (w - x2) > corner_size:
        bg_regions.append(("Bottom-Right", gray[y2+margin:h, x2+margin:w]))
    
    # Strategy 2: Narrow edge strips (even if small)
    edge_size = 30  # Reduced from 100
    
    # Top edge
    if y1 > edge_size:
        bg_regions.append(("Top-Edge", gray[0:y1-margin, :]))
    
    # Bottom edge
    if (h - y2) > edge_size:
        bg_regions.append(("Bottom-Edge", gray[y2+margin:h, :]))
    
    # Left edge
    if x1 > edge_size:
        bg_regions.append(("Left-Edge", gray[:, 0:x1-margin]))
    
    # Right edge
    if (w - x2) > edge_size:
        bg_regions.append(("Right-Edge", gray[:, x2+margin:w]))
    
    # Strategy 3: Thin strips around face (last resort)
    strip_width = 15
    
    # Top strip (above face)
    if y1 > strip_width:
        bg_regions.append(("Top-Strip", gray[max(0, y1-strip_width):y1, x1:x2]))
    
    # Bottom strip (below face)
    if (h - y2) > strip_width:
        bg_regions.append(("Bottom-Strip", gray[y2:min(h, y2+strip_width), x1:x2]))
    
    # Left strip (left of face)
    if x1 > strip_width:
        bg_regions.append(("Left-Strip", gray[y1:y2, max(0, x1-strip_width):x1]))
    
    # Right strip (right of face)
    if (w - x2) > strip_width:
        bg_regions.append(("Right-Strip", gray[y1:y2, x2:min(w, x2+strip_width)]))
    
    if len(bg_regions) == 0:
        if verbose:
            print("   ❌ Face occupies entire image - no background available")
        return {"success": False, "error": "Face occupies entire image"}
    
    if verbose:
        print(f"   ✓ Found {len(bg_regions)} background regions")
    
    # Extract PRNU from face
    face_noise = extract_prnu_noise(face, size=64)  # Reduced from 128 for small faces
    if face_noise is None:
        return {"success": False, "error": "PRNU extraction failed"}
    
    # Compute correlations with background regions
    correlations = []
    
    for region_name, bg_region in bg_regions:
        # Relaxed size requirement (was 10000, now 2000 pixels = ~45x45)
        if bg_region.size < 2000:
            if verbose:
                print(f"      {region_name}: Too small ({bg_region.size} pixels), skipping")
            continue
        
        bg_noise = extract_prnu_noise(bg_region, size=64)  # Reduced from 128
        if bg_noise is None:
            continue
        
        try:
            corr = np.corrcoef(face_noise.flatten(), bg_noise.flatten())[0, 1]
            
            if not np.isnan(corr):
                correlations.append(abs(corr))
                if verbose:
                    print(f"      {region_name}: {abs(corr):.4f} ({bg_region.shape[0]}x{bg_region.shape[1]})")
        
        except Exception as e:
            if verbose:
                print(f"      {region_name}: Correlation failed ({e})")
            continue
    
    if len(correlations) == 0:
        if verbose:
            print("   ❌ No valid background regions for correlation")
        return {"success": False, "error": "No valid background regions"}
    
    # Calculate metrics
    min_correlation = float(np.min(correlations))
    avg_correlation = float(np.mean(correlations))
    std_correlation = float(np.std(correlations))
    
    if verbose:
        print(f"\n   📊 PRNU Correlations:")
        print(f"      Min: {min_correlation:.4f}")
        print(f"      Avg: {avg_correlation:.4f}")
        print(f"      Std: {std_correlation:.4f}")
        print(f"      Regions used: {len(correlations)}")
    
    # Decision logic
    if min_correlation < 0.15:
        verdict = "LIKELY FACE SWAP"
        confidence = "High"
        suspicion_score = 0.85
        emoji = "🚨"
    elif min_correlation < 0.25:
        verdict = "POSSIBLE FACE SWAP"
        confidence = "Moderate"
        suspicion_score = 0.65
        emoji = "⚠️"
    elif avg_correlation < 0.35 and std_correlation > 0.15:
        verdict = "INCONSISTENT PRNU PATTERN"
        confidence = "Moderate"
        suspicion_score = 0.55
        emoji = "🤔"
    elif min_correlation < 0.40:
        verdict = "INCONCLUSIVE"
        confidence = "Low"
        suspicion_score = 0.50
        emoji = "❓"
    else:
        verdict = "LIKELY AUTHENTIC"
        confidence = "High"
        suspicion_score = 0.20
        emoji = "✅"
    
    if verbose:
        print(f"\n   {emoji} {verdict} (Confidence: {confidence})")
        print(f"   Suspicion Score: {suspicion_score:.2f}")
    
    return {
        "success": True,
        "verdict": verdict,
        "confidence": confidence,
        "suspicion_score": suspicion_score,
        "metrics": {
            "min_correlation": round(min_correlation, 4),
            "avg_correlation": round(avg_correlation, 4),
            "std_correlation": round(std_correlation, 4),
            "num_background_regions": len(correlations)
        }
    }

# =====================================================
# TESTING FUNCTIONS
# =====================================================

def test_single_image():
    """Test on one image to verify functionality"""
    
    print("="*70)
    print("TEST 1: Single Image Test")
    print("="*70)
    
    real_images = [f for f in os.listdir(REAL_FOLDER) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(real_images) == 0:
        print("❌ No images found in REAL folder!")
        return False
    
    test_image = os.path.join(REAL_FOLDER, real_images[0])
    print(f"\nTesting on: {real_images[0]}")
    print("-" * 70)
    
    result = detect_face_swap(test_image, verbose=True)
    
    print("\n" + "="*70)
    
    if result["success"]:
        print("✅ Test passed! Face-swap detector is working.")
        return True
    else:
        print(f"❌ Test failed: {result['error']}")
        return False


def evaluate_dataset():
    """Run on full dataset and calculate metrics"""
    
    print("\n" + "="*70)
    print("TEST 2: Full Dataset Evaluation")
    print("="*70)
    
    real_scores = []
    fake_scores = []
    real_failed = 0
    fake_failed = 0
    
    # Test REAL images
    real_images = [f for f in os.listdir(REAL_FOLDER) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:NUM_SAMPLES]
    
    print(f"\n📊 Analyzing {len(real_images)} REAL images...")
    print("-" * 70)
    
    for img_name in tqdm(real_images, desc="REAL", ncols=70):
        img_path = os.path.join(REAL_FOLDER, img_name)
        
        try:
            result = detect_face_swap(img_path, verbose=False)
            
            if result["success"]:
                real_scores.append(result["suspicion_score"])
            else:
                real_failed += 1
        except:
            real_failed += 1
    
    # Test FAKE images
    fake_images = [f for f in os.listdir(FAKE_FOLDER) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:NUM_SAMPLES]
    
    print(f"\n📊 Analyzing {len(fake_images)} FAKE images...")
    print("-" * 70)
    
    for img_name in tqdm(fake_images, desc="FAKE", ncols=70):
        img_path = os.path.join(FAKE_FOLDER, img_name)
        
        try:
            result = detect_face_swap(img_path, verbose=False)
            
            if result["success"]:
                fake_scores.append(result["suspicion_score"])
            else:
                fake_failed += 1
        except:
            fake_failed += 1
    
    # Print results
    print("\n\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n📈 REAL Images (n={len(real_scores)}, failed={real_failed}):")
    if len(real_scores) > 0:
        print(f"   Suspicion Score: {np.mean(real_scores):.4f} ± {np.std(real_scores):.4f}")
        print(f"   Range: [{np.min(real_scores):.4f}, {np.max(real_scores):.4f}]")
    
    print(f"\n📈 FAKE Images (n={len(fake_scores)}, failed={fake_failed}):")
    if len(fake_scores) > 0:
        print(f"   Suspicion Score: {np.mean(fake_scores):.4f} ± {np.std(fake_scores):.4f}")
        print(f"   Range: [{np.min(fake_scores):.4f}, {np.max(fake_scores):.4f}]")
    
    if len(real_scores) == 0 or len(fake_scores) == 0:
        print("\n❌ Insufficient data")
        return None
    
    # Performance metrics
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    print("\n📊 Accuracy at different thresholds:")
    
    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
        real_correct = sum(1 for s in real_scores if s < threshold)
        fake_correct = sum(1 for s in fake_scores if s >= threshold)
        accuracy = (real_correct + fake_correct) / (len(real_scores) + len(fake_scores))
        print(f"   Threshold {threshold:.2f}: Accuracy = {accuracy:.2%}")
    
    separation = abs(np.mean(fake_scores) - np.mean(real_scores))
    
    print(f"\n🎯 Score Separation: {separation:.4f}")
    
    # Assessment
    print("\n" + "="*70)
    print("EFFECTIVENESS ASSESSMENT")
    print("="*70)
    
    if separation < 0.10:
        print("\n❌ POOR: Very weak separation")
        print(f"   Difference: {separation:.4f}")
        print("\n   Your fakes are likely GAN-GENERATED, not face-swapped.")
        print("   ⚠️  DO NOT integrate this feature.")
        recommendation = "skip"
    
    elif separation < 0.20:
        print("\n⚠️  WEAK: Some separation")
        print(f"   Difference: {separation:.4f}")
        print("\n   💡 Use as SUPPLEMENTARY evidence only.")
        recommendation = "supplementary"
    
    else:
        print("\n✅ GOOD: Clear separation!")
        print(f"   Difference: {separation:.4f}")
        print("\n   ✅ Ready for integration.")
        recommendation = "integrate"
    
    # Visualization
    plot_distributions(real_scores, fake_scores)
    
    return {
        "real_scores": real_scores,
        "fake_scores": fake_scores,
        "separation": separation,
        "recommendation": recommendation
    }


def plot_distributions(real_scores, fake_scores):
    """Create visualization"""
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(real_scores, bins=20, alpha=0.7, label='Real', color='green', edgecolor='black')
    plt.hist(fake_scores, bins=20, alpha=0.7, label='Fake', color='red', edgecolor='black')
    plt.axvline(0.50, color='blue', linestyle='--', linewidth=2, label='Threshold=0.50')
    plt.xlabel('Suspicion Score', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Face-Swap Detection Score Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    box_data = [real_scores, fake_scores]
    bp = plt.boxplot(box_data, tick_labels=['Real', 'Fake'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    plt.axhline(0.50, color='blue', linestyle='--', linewidth=2, label='Threshold=0.50')
    plt.ylabel('Suspicion Score', fontsize=12)
    plt.title('Face-Swap Detection Box Plot', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('face_swap_detection_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Visualization saved as 'face_swap_detection_results.png'")
    plt.show()


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "FACE-SWAP DETECTION TEST SUITE")
    print("="*70)
    
    # Test 1
    if not test_single_image():
        print("\n⚠️  Basic test failed!")
        exit(1)
    
    # Test 2
    results = evaluate_dataset()
    
    if results is None:
        print("\n❌ Evaluation failed")
        exit(1)
    
    # Final recommendation
    print("\n" + "="*70)
    print("FINAL RECOMMENDATION")
    print("="*70)
    
    if results["recommendation"] == "skip":
        print("\n❌ DO NOT integrate face-swap detection")
        print("   Your dataset has GAN-generated fakes, not face-swaps.")
    elif results["recommendation"] == "supplementary":
        print("\n💡 USE AS SUPPLEMENTARY EVIDENCE ONLY")
    else:
        print("\n✅ INTEGRATE FACE-SWAP DETECTION")
        print("   Ready for Django integration!")
    
    print("\n" + "="*70)
    print("Testing complete!")
    print("="*70)
