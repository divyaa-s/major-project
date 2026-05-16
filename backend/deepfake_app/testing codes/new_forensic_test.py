"""
Visual Quality-Based Deepfake Detection Features
Captures what humans see: unnatural smoothness, weird proportions, lighting issues
"""

import cv2
import numpy as np
from scipy import ndimage, stats
from skimage import filters, feature, measure
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def analyze_skin_texture(img_path):
    """
    Detect unnatural smooth/waxy skin texture (major giveaway in fake faces)
    Real faces have pores, wrinkles, imperfections
    Fake faces are often too smooth or have unnatural texture patterns
    """
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to LAB color space (better for skin analysis)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # 1. Texture variance (real skin has more variance)
    texture_variance = np.std(l_channel)
    
    # 2. High-frequency content (real skin has micro-details)
    # Apply high-pass filter to detect fine details
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    high_freq = cv2.filter2D(l_channel, -1, kernel)
    high_freq_energy = np.mean(np.abs(high_freq))
    
    # 3. Local binary patterns (texture descriptor)
    # Real skin has more complex patterns
    radius = 3
    n_points = 8 * radius
    lbp = feature.local_binary_pattern(l_channel, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
    lbp_entropy = stats.entropy(lbp_hist)
    
    # 4. Gradient distribution (real skin has natural variations)
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_std = np.std(gradient_magnitude)
    
    # 5. Smoothness score (lower = more unnatural)
    # Real faces have a balance; fake faces are either too smooth or have artifacts
    smoothness_score = texture_variance * high_freq_energy * lbp_entropy
    
    # Combine into suspicion score (normalized 0-1, higher = more suspicious)
    # Fake faces tend to have: low variance, low high-freq, low entropy
    suspicion_score = 1.0 - min(1.0, smoothness_score / 100.0)
    
    return {
        'texture_variance': texture_variance,
        'high_freq_energy': high_freq_energy,
        'lbp_entropy': lbp_entropy,
        'gradient_std': gradient_std,
        'smoothness_score': smoothness_score,
        'suspicion_score': suspicion_score
    }

def analyze_face_proportions(img_path):
    """
    Detect unnatural face proportions and symmetry issues
    Fake faces sometimes have slightly off proportions
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150)
    
    # 1. Symmetry analysis
    h, w = gray.shape
    left_half = gray[:, :w//2]
    right_half = gray[:, w//2:]
    right_half_flipped = np.fliplr(right_half)
    
    # Resize to match if needed
    min_width = min(left_half.shape[1], right_half_flipped.shape[1])
    left_half = left_half[:, :min_width]
    right_half_flipped = right_half_flipped[:, :min_width]
    
    # Calculate symmetry using SSIM
    symmetry_score = ssim(left_half, right_half_flipped, data_range=255)
    
    # 2. Proportion ratios (golden ratio for faces ≈ 1.618)
    # Estimate face landmarks using simple heuristics
    edge_coords = np.argwhere(edges > 0)
    if len(edge_coords) > 0:
        y_coords = edge_coords[:, 0]
        x_coords = edge_coords[:, 1]
        
        face_height = np.max(y_coords) - np.min(y_coords)
        face_width = np.max(x_coords) - np.min(x_coords)
        
        if face_width > 0:
            aspect_ratio = face_height / face_width
            golden_ratio_deviation = abs(aspect_ratio - 1.4)  # Typical face ratio
        else:
            golden_ratio_deviation = 1.0
    else:
        golden_ratio_deviation = 1.0
    
    # 3. Edge distribution uniformity
    # Real faces have relatively uniform edge distribution
    # Fake faces may have irregular edge patterns
    edge_rows = np.sum(edges, axis=1)
    edge_cols = np.sum(edges, axis=0)
    
    row_variance = np.var(edge_rows)
    col_variance = np.var(edge_cols)
    edge_uniformity = (row_variance + col_variance) / 2
    
    # Suspicion score (normalized)
    # Lower symmetry + high golden ratio deviation = more suspicious
    suspicion_score = (1.0 - symmetry_score) * 0.6 + min(1.0, golden_ratio_deviation) * 0.4
    
    return {
        'symmetry_score': symmetry_score,
        'golden_ratio_deviation': golden_ratio_deviation,
        'edge_uniformity': edge_uniformity,
        'suspicion_score': suspicion_score
    }

def analyze_lighting_consistency(img_path):
    """
    Detect unnatural lighting and shadow inconsistencies
    Fake faces often have impossible lighting or lack proper shadows
    """
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to HSV for better lighting analysis
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value_channel = hsv[:, :, 2]  # Brightness
    
    # 1. Lighting gradient consistency
    # Real photos have consistent light direction
    # Fake images may have conflicting light sources
    
    # Calculate dominant gradient direction
    sobelx = cv2.Sobel(value_channel, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(value_channel, cv2.CV_64F, 0, 1, ksize=5)
    
    gradient_direction = np.arctan2(sobely, sobelx)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Weight directions by magnitude
    valid_gradients = gradient_magnitude > np.percentile(gradient_magnitude, 75)
    dominant_directions = gradient_direction[valid_gradients]
    
    if len(dominant_directions) > 0:
        # Calculate circular variance (how consistent is light direction)
        direction_consistency = 1.0 - np.std(np.cos(dominant_directions))
    else:
        direction_consistency = 0.5
    
    # 2. Local contrast variations
    # Real images have natural contrast falloff
    # Fake images may have flat or unnatural contrast
    
    # Divide image into regions
    h, w = value_channel.shape
    region_size = 32
    region_contrasts = []
    
    for i in range(0, h - region_size, region_size):
        for j in range(0, w - region_size, region_size):
            region = value_channel[i:i+region_size, j:j+region_size]
            region_contrast = np.std(region)
            region_contrasts.append(region_contrast)
    
    contrast_variation = np.std(region_contrasts)
    
    # 3. Highlight/shadow ratio
    # Real faces have proper highlight-to-shadow ratio
    bright_pixels = np.sum(value_channel > 200)
    dark_pixels = np.sum(value_channel < 50)
    total_pixels = value_channel.size
    
    highlight_ratio = bright_pixels / total_pixels
    shadow_ratio = dark_pixels / total_pixels
    
    # Suspicious if too flat (no highlights/shadows) or too extreme
    lighting_balance = abs(highlight_ratio - 0.1) + abs(shadow_ratio - 0.1)
    
    # Suspicion score
    # Low direction consistency + unnatural contrast = suspicious
    suspicion_score = (1.0 - direction_consistency) * 0.5 + \
                     min(1.0, lighting_balance * 5) * 0.3 + \
                     (1.0 - min(1.0, contrast_variation / 50)) * 0.2
    
    return {
        'direction_consistency': direction_consistency,
        'contrast_variation': contrast_variation,
        'highlight_ratio': highlight_ratio,
        'shadow_ratio': shadow_ratio,
        'lighting_balance': lighting_balance,
        'suspicion_score': suspicion_score
    }

def analyze_color_distribution(img_path):
    """
    Detect unnatural color distributions
    Fake faces may have oversaturated or unrealistic color palettes
    """
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 1. Saturation analysis
    saturation = hsv[:, :, 1]
    mean_saturation = np.mean(saturation)
    saturation_std = np.std(saturation)
    
    # Real faces: moderate saturation with variation
    # Fake faces: often oversaturated or too flat
    saturation_naturalness = 1.0 - abs(mean_saturation - 100) / 100
    
    # 2. Skin tone analysis (if detectable)
    # Real skin tones fall within specific ranges
    hue = hsv[:, :, 0]
    
    # Skin tone is typically in hue range 0-25 (red-orange)
    skin_mask = (hue >= 0) & (hue <= 25) & (saturation > 30)
    skin_pixels = np.sum(skin_mask)
    
    if skin_pixels > 0:
        skin_hue = hue[skin_mask]
        skin_hue_std = np.std(skin_hue)
        
        # Real skin has some variation but not too much
        skin_naturalness = 1.0 - min(1.0, skin_hue_std / 20)
    else:
        skin_naturalness = 0.5
    
    # 3. Color histogram entropy
    # Real images have more diverse color distributions
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    
    hist_h = hist_h.flatten() / hist_h.sum()
    hist_s = hist_s.flatten() / hist_s.sum()
    hist_v = hist_v.flatten() / hist_v.sum()
    
    color_entropy = (stats.entropy(hist_h) + stats.entropy(hist_s) + stats.entropy(hist_v)) / 3
    
    # Suspicion score
    suspicion_score = (1.0 - saturation_naturalness) * 0.4 + \
                     (1.0 - skin_naturalness) * 0.4 + \
                     (1.0 - min(1.0, color_entropy / 5)) * 0.2
    
    return {
        'mean_saturation': mean_saturation,
        'saturation_std': saturation_std,
        'saturation_naturalness': saturation_naturalness,
        'skin_naturalness': skin_naturalness,
        'color_entropy': color_entropy,
        'suspicion_score': suspicion_score
    }

def detect_ai_artifacts(img_path):
    """
    Detect specific AI generation artifacts
    - Checkerboard patterns (common in GANs)
    - Repeated patterns
    - Unnatural frequency peaks
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # FFT analysis for checkerboard artifacts
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    
    # Look for peaks at specific frequencies (2, 4, 8 pixels - common in GANs)
    h, w = magnitude_spectrum.shape
    center_y, center_x = h // 2, w // 2
    
    artifact_score = 0
    for freq in [2, 4, 8, 16]:
        # Sample in multiple directions
        for angle in [0, 45, 90, 135]:
            rad = np.radians(angle)
            dy = int(h // (freq * 2) * np.sin(rad))
            dx = int(w // (freq * 2) * np.cos(rad))
            
            y_sample = center_y + dy
            x_sample = center_x + dx
            
            if 0 <= y_sample < h and 0 <= x_sample < w:
                artifact_score += magnitude_spectrum[y_sample, x_sample]
    
    # Normalize
    artifact_score = artifact_score / magnitude_spectrum[center_y, center_x]
    artifact_score = min(1.0, artifact_score / 10)
    
    return {
        'artifact_score': artifact_score,
        'suspicion_score': artifact_score
    }

# ============================================================================
# COMBINED ANALYSIS
# ============================================================================
def analyze_image_comprehensive(img_path):
    """Run all visual quality checks"""
    
    results = {}
    
    try:
        results['skin_texture'] = analyze_skin_texture(img_path)
    except Exception as e:
        print(f"Skin texture analysis failed: {e}")
        results['skin_texture'] = {'suspicion_score': 0.5}
    
    try:
        results['face_proportions'] = analyze_face_proportions(img_path)
    except Exception as e:
        print(f"Proportion analysis failed: {e}")
        results['face_proportions'] = {'suspicion_score': 0.5}
    
    try:
        results['lighting'] = analyze_lighting_consistency(img_path)
    except Exception as e:
        print(f"Lighting analysis failed: {e}")
        results['lighting'] = {'suspicion_score': 0.5}
    
    try:
        results['color'] = analyze_color_distribution(img_path)
    except Exception as e:
        print(f"Color analysis failed: {e}")
        results['color'] = {'suspicion_score': 0.5}
    
    try:
        results['ai_artifacts'] = detect_ai_artifacts(img_path)
    except Exception as e:
        print(f"Artifact detection failed: {e}")
        results['ai_artifacts'] = {'suspicion_score': 0.5}
    
    # Weighted combination
    overall_suspicion = (
        results['skin_texture']['suspicion_score'] * 0.30 +
        results['face_proportions']['suspicion_score'] * 0.20 +
        results['lighting']['suspicion_score'] * 0.25 +
        results['color']['suspicion_score'] * 0.15 +
        results['ai_artifacts']['suspicion_score'] * 0.10
    )
    
    results['overall_suspicion'] = overall_suspicion
    results['prediction'] = 'FAKE' if overall_suspicion > 0.5 else 'REAL'
    
    return results