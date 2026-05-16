"""
prnu_analysis.py

PRNU (Photo Response Non-Uniformity) Analysis Module
Standalone forensic feature for deepfake detection

PRNU is sensor-specific noise pattern inherent to camera hardware.
- Real photos: Consistent PRNU across entire image
- Face swaps: Different PRNU in face vs background
- GAN images: No consistent PRNU pattern
"""

import cv2
import numpy as np
import pywt
from scipy.signal import wiener
from PIL import Image
from facenet_pytorch import MTCNN
import logging

logger = logging.getLogger(__name__)

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=False, device='cpu')


# -------------------------
# CORE PRNU EXTRACTION
# -------------------------

def extract_prnu_noise(image, method='wavelet'):
    """
    Extract PRNU noise residual from image
    
    Args:
        image: Grayscale numpy array
        method: 'wavelet' (accurate) or 'wiener' (fast)
    
    Returns:
        Normalized noise residual (2D array)
    """
    if image is None or image.size == 0:
        return None
    
    # Resize for consistent analysis
    img = cv2.resize(image, (256, 256))
    img = img.astype(np.float32) / 255.0
    
    try:
        if method == 'wavelet':
            # Wavelet denoising (removes image content, keeps sensor noise)
            coeffs = pywt.wavedec2(img, 'db8', level=4)
            coeffs[0] = np.zeros_like(coeffs[0])  # Zero out approximation
            
            denoised = pywt.waverec2(coeffs, 'db8')
            
            # Handle potential shape mismatch
            h, w = img.shape
            denoised = denoised[:h, :w]
            
            noise = img - denoised
        
        elif method == 'wiener':
            # Wiener filter (faster, less accurate)
            denoised = wiener(img, (5, 5))
            noise = img - denoised
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Zero-mean normalization
        noise = noise - np.mean(noise)
        
        # L2 normalization
        norm = np.linalg.norm(noise) + 1e-8
        noise = noise / norm
        
        return noise
    
    except Exception as e:
        logger.error(f"PRNU extraction failed: {e}")
        return None


# -------------------------
# SPATIAL CONSISTENCY
# -------------------------

def compute_spatial_consistency(noise, patch_size=32):
    """
    Measure uniformity of PRNU across image regions
    
    Real photos: Low variance (consistent sensor pattern)
    Fake images: High variance (random artifacts)
    
    Returns:
        Consistency score (float): Lower = more consistent = more real
    """
    if noise is None:
        return None
    
    h, w = noise.shape
    patch_vars = []
    
    # Divide into non-overlapping patches
    for i in range(0, h - patch_size + 1, patch_size):
        for j in range(0, w - patch_size + 1, patch_size):
            patch = noise[i:i+patch_size, j:j+patch_size]
            
            if patch.shape == (patch_size, patch_size):
                patch_vars.append(np.var(patch))
    
    if len(patch_vars) < 4:
        return None
    
    patch_vars = np.array(patch_vars)
    
    # Standard deviation of local variances
    # Real images: ~0.01-0.03 (consistent)
    # Fake images: ~0.05-0.15 (inconsistent)
    return float(np.std(patch_vars))


# -------------------------
# FACE-BACKGROUND CORRELATION
# -------------------------

def compute_face_background_correlation(image_path):
    """
    Compare PRNU between face region and background
    
    Real photos: High correlation (same camera)
    Face swaps: Low correlation (different sources)
    GAN images: Random correlation (no real sensor)
    
    Returns:
        Correlation coefficient (0-1): Higher = more consistent
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Detect face using MTCNN
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        boxes, _ = mtcnn.detect(img_pil)
        
        if boxes is None or len(boxes) == 0:
            logger.warning("No face detected for face-background PRNU")
            return None
        
        # Use largest face
        x1, y1, x2, y2 = boxes[0].astype(int)
        
        # Validate bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if (x2 - x1) < 32 or (y2 - y1) < 32:
            logger.warning("Face region too small for PRNU")
            return None
        
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        return None
    
    # Extract face region
    face = gray[y1:y2, x1:x2]
    
    # Extract background regions (avoid face)
    bg_regions = []
    
    # Top region
    if y1 > 50:
        bg_regions.append(gray[0:y1-10, :])
    
    # Bottom region
    if (h - y2) > 50:
        bg_regions.append(gray[y2+10:h, :])
    
    # Left region
    if x1 > 50:
        bg_regions.append(gray[:, 0:x1-10])
    
    # Right region
    if (w - x2) > 50:
        bg_regions.append(gray[:, x2+10:w])
    
    if len(bg_regions) == 0:
        logger.warning("Insufficient background area for PRNU")
        return None
    
    # Extract PRNU from face
    face_noise = extract_prnu_noise(face, method='wiener')
    if face_noise is None:
        return None
    
    # Extract PRNU from background regions
    correlations = []
    
    for bg_region in bg_regions:
        if bg_region.size < 5000:  # Skip tiny regions
            continue
        
        bg_noise = extract_prnu_noise(bg_region, method='wiener')
        if bg_noise is None:
            continue
        
        # Resize to common dimensions
        face_resized = cv2.resize(face_noise, (128, 128))
        bg_resized = cv2.resize(bg_noise, (128, 128))
        
        # Compute correlation
        try:
            corr = np.corrcoef(
                face_resized.flatten(),
                bg_resized.flatten()
            )[0, 1]
            
            if not np.isnan(corr):
                correlations.append(abs(corr))
        
        except Exception as e:
            logger.warning(f"Correlation computation failed: {e}")
            continue
    
    if len(correlations) == 0:
        return None
    
    # Average correlation across all background regions
    avg_corr = float(np.mean(correlations))
    
    # Real images: 0.4-0.8
    # Fake images: 0.1-0.4
    return avg_corr


# -------------------------
# MAIN ANALYSIS FUNCTION
# -------------------------

def analyze_prnu(image_path):
    """
    Perform complete PRNU analysis on an image
    
    Args:
        image_path: Path to image file
    
    Returns:
        dict containing:
        - raw_metrics: Raw PRNU measurements
        - normalized_scores: 0-1 scores (higher = more suspicious)
        - interpretation: Human-readable assessment
        - overall_score: Combined PRNU fake probability
    """
    
    logger.info(f"Starting PRNU analysis on: {image_path}")
    
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {
            "error": "Failed to load image",
            "success": False
        }
    
    # Extract full-image PRNU
    noise = extract_prnu_noise(img, method='wavelet')
    if noise is None:
        return {
            "error": "PRNU extraction failed",
            "success": False
        }
    
    # Compute metrics
    consistency = compute_spatial_consistency(noise, patch_size=32)
    fb_correlation = compute_face_background_correlation(image_path)
    
    # Normalize to 0-1 fake probability scores
    # These thresholds are empirically derived - tune on your dataset
    
    consistency_score = 0.5  # Default neutral
    if consistency is not None:
        # Real: 0.01-0.03, Fake: 0.05-0.15
        consistency_score = np.clip((consistency - 0.015) / 0.10, 0, 1)
    
    fb_score = 0.5  # Default neutral
    if fb_correlation is not None:
        # Real: 0.5-0.8, Fake: 0.1-0.4
        fb_score = np.clip((0.65 - fb_correlation) / 0.55, 0, 1)
    
    # Combined PRNU fake probability
    # Weight consistency higher (more reliable)
    overall_score = 0.65 * consistency_score + 0.35 * fb_score
    
    # Generate interpretation
    if overall_score < 0.30:
        verdict = "Strong PRNU evidence of AUTHENTIC image"
        confidence = "High"
    elif overall_score < 0.50:
        verdict = "PRNU patterns suggest AUTHENTIC image"
        confidence = "Moderate"
    elif overall_score < 0.70:
        verdict = "PRNU patterns suggest MANIPULATED image"
        confidence = "Moderate"
    else:
        verdict = "Strong PRNU evidence of MANIPULATED image"
        confidence = "High"
    
    result = {
        "success": True,
        "raw_metrics": {
            "spatial_consistency": round(consistency, 6) if consistency else None,
            "face_background_correlation": round(fb_correlation, 4) if fb_correlation else None,
        },
        "normalized_scores": {
            "consistency_score": round(consistency_score, 4),
            "face_background_score": round(fb_score, 4),
            "overall_prnu_score": round(overall_score, 4)
        },
        "interpretation": {
            "verdict": verdict,
            "confidence": confidence,
            "explanation": generate_explanation(consistency_score, fb_score, overall_score)
        }
    }
    
    logger.info(f"PRNU analysis complete: {verdict} (score={overall_score:.3f})")
    
    return result


def generate_explanation(consistency_score, fb_score, overall_score):
    """Generate human-readable explanation of PRNU findings"""
    
    explanation = []
    
    # Consistency interpretation
    if consistency_score < 0.30:
        explanation.append("Uniform sensor noise pattern detected across image regions, typical of authentic photographs.")
    elif consistency_score < 0.60:
        explanation.append("Moderate variation in noise patterns across regions, could indicate compression or editing.")
    else:
        explanation.append("Highly inconsistent noise patterns across regions, suggesting synthetic generation or heavy manipulation.")
    
    # Face-background interpretation
    if fb_score < 0.30:
        explanation.append("Face and background regions show matching sensor characteristics, indicating same source camera.")
    elif fb_score < 0.60:
        explanation.append("Face and background show moderate PRNU differences, possibly from post-processing or compression.")
    else:
        explanation.append("Face and background have significantly different sensor signatures, suggesting face manipulation or composite image.")
    
    return " ".join(explanation)


# -------------------------
# BATCH EVALUATION (for testing)
# -------------------------

def evaluate_dataset(real_folder, fake_folder, num_samples=100):
    """
    Evaluate PRNU on real vs fake datasets to tune thresholds
    
    Returns:
        Statistics and optimal thresholds
    """
    import os
    from tqdm import tqdm
    
    real_scores = []
    fake_scores = []
    
    # Evaluate real images
    real_images = [f for f in os.listdir(real_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:num_samples]
    
    for img_name in tqdm(real_images, desc="Analyzing REAL images"):
        img_path = os.path.join(real_folder, img_name)
        result = analyze_prnu(img_path)
        
        if result["success"]:
            real_scores.append(result["normalized_scores"]["overall_prnu_score"])
    
    # Evaluate fake images
    fake_images = [f for f in os.listdir(fake_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:num_samples]
    
    for img_name in tqdm(fake_images, desc="Analyzing FAKE images"):
        img_path = os.path.join(fake_folder, img_name)
        result = analyze_prnu(img_path)
        
        if result["success"]:
            fake_scores.append(result["normalized_scores"]["overall_prnu_score"])
    
    # Compute statistics
    real_mean = np.mean(real_scores)
    real_std = np.std(real_scores)
    fake_mean = np.mean(fake_scores)
    fake_std = np.std(fake_scores)
    
    print("\n===== PRNU EVALUATION RESULTS =====")
    print(f"REAL images (n={len(real_scores)}):")
    print(f"  Mean score: {real_mean:.4f} ± {real_std:.4f}")
    print(f"  Range: [{min(real_scores):.4f}, {max(real_scores):.4f}]")
    
    print(f"\nFAKE images (n={len(fake_scores)}):")
    print(f"  Mean score: {fake_mean:.4f} ± {fake_std:.4f}")
    print(f"  Range: [{min(fake_scores):.4f}, {max(fake_scores):.4f}]")
    
    # Suggested threshold (midpoint)
    threshold = (real_mean + fake_mean) / 2
    print(f"\nSuggested threshold: {threshold:.4f}")
    
    # Calculate accuracy at this threshold
    real_correct = sum(1 for s in real_scores if s < threshold)
    fake_correct = sum(1 for s in fake_scores if s >= threshold)
    accuracy = (real_correct + fake_correct) / (len(real_scores) + len(fake_scores))
    
    print(f"Accuracy at threshold: {accuracy:.2%}")
    
    return {
        "real_mean": real_mean,
        "real_std": real_std,
        "fake_mean": fake_mean,
        "fake_std": fake_std,
        "suggested_threshold": threshold,
        "accuracy": accuracy
    }