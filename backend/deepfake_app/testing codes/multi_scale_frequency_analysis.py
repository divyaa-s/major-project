# multi_scale_frequency_analysis.py

import cv2
import numpy as np
import pywt
from scipy import stats

def multi_scale_frequency_analysis(image_path):
    """
    Multi-Scale Frequency Analysis using Discrete Wavelet Transform
    
    Academic Justification:
    - Analyzes image at multiple resolution levels
    - GANs leave artifacts at different frequency scales
    - Wavelet decomposition reveals scale-dependent anomalies
    - Complements single-scale FFT analysis
    
    Returns detailed frequency characteristics across scales
    """
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Resize for consistency
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    
    # Multi-level wavelet decomposition
    coeffs = pywt.wavedec2(img, 'db4', level=4)
    
    # Analyze each decomposition level
    level_scores = []
    
    for level_idx, detail_coeffs in enumerate(coeffs[1:], 1):  # Skip approximation
        cH, cV, cD = detail_coeffs  # Horizontal, Vertical, Diagonal
        
        # Energy in each orientation
        energy_h = np.sum(cH ** 2)
        energy_v = np.sum(cV ** 2)
        energy_d = np.sum(cD ** 2)
        
        total_energy = energy_h + energy_v + energy_d
        
        # Entropy in each orientation
        entropy_h = compute_coefficient_entropy(cH)
        entropy_v = compute_coefficient_entropy(cV)
        entropy_d = compute_coefficient_entropy(cD)
        
        avg_entropy = (entropy_h + entropy_v + entropy_d) / 3
        
        # Uniformity across orientations
        energies = [energy_h, energy_v, energy_d]
        energy_std = np.std(energies)
        energy_mean = np.mean(energies)
        
        uniformity = 1 - (energy_std / (energy_mean + 1e-8))
        
        level_scores.append({
            'level': level_idx,
            'total_energy': float(total_energy),
            'avg_entropy': float(avg_entropy),
            'uniformity': float(uniformity)
        })
    
    # Cross-level analysis
    entropies = [s['avg_entropy'] for s in level_scores]
    uniformities = [s['uniformity'] for s in level_scores]
    
    # Real images: Varied entropy across scales
    # GAN images: More uniform entropy across scales
    entropy_variance = np.var(entropies)
    uniformity_mean = np.mean(uniformities)
    
    # Combined suspicion score
    # High uniformity + low entropy variance = suspicious
    suspicion_score = 0.6 * uniformity_mean + 0.4 * (1 - np.clip(entropy_variance / 2.0, 0, 1))
    
    return {
        "suspicion_score": float(np.clip(suspicion_score, 0, 1)),
        "level_details": level_scores,
        "entropy_variance": float(entropy_variance),
        "uniformity_mean": float(uniformity_mean),
        "interpretation": interpret_results(suspicion_score, entropy_variance)
    }


def compute_coefficient_entropy(coeffs):
    """Compute entropy of wavelet coefficients"""
    # Flatten and histogram
    coeffs_flat = coeffs.flatten()
    hist, _ = np.histogram(coeffs_flat, bins=50)
    hist = hist / (hist.sum() + 1e-8)
    
    return stats.entropy(hist)


def interpret_results(suspicion, variance):
    """Generate human-readable interpretation"""
    if suspicion > 0.7:
        return "High suspicion: Uniform frequency characteristics across scales suggest synthetic generation"
    elif suspicion > 0.5:
        return "Moderate suspicion: Some frequency irregularities detected"
    else:
        return "Low suspicion: Natural frequency variation across scales"