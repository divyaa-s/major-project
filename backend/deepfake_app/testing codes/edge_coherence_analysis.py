# edge_coherence_analysis.py

import cv2
import numpy as np
from scipy import stats

def analyze_edge_coherence(image_path):
    """
    Edge Coherence Analysis
    
    Academic Basis:
    - Real photos have physically consistent edges
    - GAN images may have edge discontinuities
    - Analyzes gradient magnitude and direction consistency
    - Complements texture-based features
    """
    
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (256, 256))
    
    # Compute gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradient magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # --- Analysis 1: Edge Strength Distribution ---
    edge_pixels = magnitude[edges > 0]
    
    if len(edge_pixels) < 100:
        return None
    
    edge_mean = np.mean(edge_pixels)
    edge_std = np.std(edge_pixels)
    edge_cv = edge_std / (edge_mean + 1e-8)  # Coefficient of variation
    
    # Real images: Higher variation in edge strength
    # GAN images: More uniform edges
    uniformity_score = 1 - np.clip(edge_cv / 2.0, 0, 1)
    
    # --- Analysis 2: Gradient Direction Coherence ---
    # Divide image into blocks
    block_size = 32
    h, w = gray.shape
    
    direction_coherences = []
    
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block_dir = direction[i:i+block_size, j:j+block_size]
            block_mag = magnitude[i:i+block_size, j:j+block_size]
            
            # Only consider significant gradients
            significant = block_mag > np.percentile(magnitude, 75)
            
            if np.sum(significant) > 10:
                dirs = block_dir[significant]
                
                # Circular variance of directions
                # Low variance = coherent (good for real images)
                # High variance = incoherent (suspicious)
                mean_dir = np.arctan2(np.mean(np.sin(dirs)), np.mean(np.cos(dirs)))
                circular_var = 1 - np.sqrt(np.mean(np.cos(dirs - mean_dir))**2 + 
                                          np.mean(np.sin(dirs - mean_dir))**2)
                
                direction_coherences.append(circular_var)
    
    if len(direction_coherences) == 0:
        return None
    
    avg_coherence = np.mean(direction_coherences)
    std_coherence = np.std(direction_coherences)
    
    # Real images: Consistent coherence across blocks
    # GAN images: Variable coherence (local inconsistencies)
    coherence_variability = std_coherence
    
    # --- Analysis 3: Edge Connectivity ---
    # Contour analysis
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Average contour length
        avg_contour_length = np.mean([len(c) for c in contours])
        
        # Many short contours = fragmented (suspicious)
        # Few long contours = connected (natural)
        fragmentation_score = np.clip(100 / (avg_contour_length + 1), 0, 1)
    else:
        fragmentation_score = 0.5
    
    # --- Combined Suspicion Score ---
    suspicion_score = (
        0.35 * uniformity_score +
        0.35 * coherence_variability +
        0.30 * fragmentation_score
    )
    
    return {
        "suspicion_score": float(np.clip(suspicion_score, 0, 1)),
        "edge_uniformity": float(uniformity_score),
        "avg_direction_coherence": float(avg_coherence),
        "coherence_variability": float(coherence_variability),
        "fragmentation_score": float(fragmentation_score),
        "num_edges": int(np.sum(edges > 0)),
        "interpretation": interpret_edge_analysis(suspicion_score)
    }


def interpret_edge_analysis(suspicion):
    """Interpret edge analysis"""
    if suspicion > 0.6:
        return "High suspicion: Edge patterns show irregularities inconsistent with natural images"
    elif suspicion > 0.4:
        return "Moderate suspicion: Some edge coherence issues detected"
    else:
        return "Low suspicion: Edge patterns appear natural"