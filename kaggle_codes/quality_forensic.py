"""
quality_forensic.py
"""

import cv2
import numpy as np
import logging
from scipy import ndimage

logger = logging.getLogger(__name__)


class QualityForensicsAnalyzer:
    """
    Analyzes quality mismatches between face and body regions
    """
    
    def __init__(self):
        self.min_body_size = 50  # Minimum body region size for comparison
    
    def analyze_frame(self, frame, face_box):
        """
        Complete quality forensics analysis
        
        Args:
            frame: Full video frame (BGR)
            face_box: (x1, y1, x2, y2) face bounding box
        
        Returns:
            dict: Quality mismatch scores and details
        """
        x1, y1, x2, y2 = face_box
        
        # Extract regions
        face_region = frame[y1:y2, x1:x2]
        body_region = self._extract_body_region(frame, face_box)
        
        if body_region is None or body_region.shape[0] < self.min_body_size:
            logger.debug("Body region too small for comparison")
            return {
                "quality_mismatch_score": 0.0,
                "details": {"error": "body_region_too_small"}
            }
        
        # Run all forensic tests
        sharpness_score = self._detect_sharpness_mismatch(face_region, body_region)
        frequency_score = self._detect_frequency_mismatch(face_region, body_region)
        edge_score = self._detect_edge_anomaly(frame, face_box)
        compression_score = self._detect_compression_mismatch(face_region, body_region)
        
        # Combined score (weighted average)
        combined_score = (
            0.35 * sharpness_score +      # Most reliable
            0.30 * frequency_score +      # Very reliable
            0.20 * edge_score +            # Good indicator
            0.15 * compression_score       # Can be noisy
        )
        
        return {
            "quality_mismatch_score": float(round(combined_score, 4)),
            "details": {
                "sharpness_mismatch": float(round(sharpness_score, 4)),
                "frequency_mismatch": float(round(frequency_score, 4)),
                "edge_anomaly": float(round(edge_score, 4)),
                "compression_mismatch": float(round(compression_score, 4))
            },
            "interpretation": self._interpret_score(combined_score)
        }
    
    def analyze_keyframes(self, frames_with_boxes):
        """
        Analyze multiple keyframes and aggregate results
        
        Args:
            frames_with_boxes: List of (frame, face_box) tuples
        
        Returns:
            dict: Aggregated quality forensics results
        """
        frame_scores = []
        
        for frame, face_box in frames_with_boxes:
            result = self.analyze_frame(frame, face_box)
            if "error" not in result["details"]:
                frame_scores.append(result["quality_mismatch_score"])
        
        if not frame_scores:
            return {
                "avg_quality_mismatch": 0.0,
                "max_quality_mismatch": 0.0,
                "frames_analyzed": 0
            }
        
        avg_score = float(np.mean(frame_scores))
        max_score = float(np.max(frame_scores))
        
        return {
            "avg_quality_mismatch": float(round(avg_score, 4)),
            "max_quality_mismatch": float(round(max_score, 4)),
            "frames_analyzed": len(frame_scores),
            "interpretation": self._interpret_score(avg_score)
        }
    
    def _extract_body_region(self, frame, face_box):
        """Extract body region below face"""
        x1, y1, x2, y2 = face_box
        
        face_height = y2 - y1
        face_width = x2 - x1
        
        # Body region: same width, below face
        body_y_start = y2
        body_y_end = min(frame.shape[0], y2 + face_height)
        
        # Also check shoulders/neck area (sides of face, below)
        body_x1 = max(0, x1 - int(face_width * 0.2))
        body_x2 = min(frame.shape[1], x2 + int(face_width * 0.2))
        
        body_region = frame[body_y_start:body_y_end, body_x1:body_x2]
        
        return body_region if body_region.size > 0 else None
    
    def _detect_sharpness_mismatch(self, face_region, body_region):
        """
        Detect sharpness difference between face and body
        
        Uses Laplacian variance as sharpness measure
        """
        face_sharpness = self._calculate_sharpness(face_region)
        body_sharpness = self._calculate_sharpness(body_region)
        
        if body_sharpness < 10:  # Body too blurry to compare
            return 0.0
        
        # Calculate ratio
        sharpness_ratio = face_sharpness / (body_sharpness + 1e-8)
        
        # If face is significantly less sharp than body = suspicious
        if sharpness_ratio < 0.7:
            # Face is >30% less sharp than body
            mismatch = (0.7 - sharpness_ratio) / 0.7
            return float(np.clip(mismatch, 0, 1))
        
        return 0.0
    
    def _calculate_sharpness(self, image):
        """Calculate image sharpness using Laplacian variance"""
        if image.shape[0] < 10 or image.shape[1] < 10:
            return 0.0
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        return float(sharpness)
    
    def _detect_frequency_mismatch(self, face_region, body_region):
        """
        Detect frequency domain differences
        
        Deepfake faces often lack high-frequency details
        """
        face_freq_energy = self._calculate_high_freq_energy(face_region)
        body_freq_energy = self._calculate_high_freq_energy(body_region)
        
        if body_freq_energy < 0.01:
            return 0.0
        
        # Calculate ratio
        freq_ratio = face_freq_energy / (body_freq_energy + 1e-8)
        
        # If face has significantly less high-frequency content
        if freq_ratio < 0.6:
            mismatch = (0.6 - freq_ratio) / 0.6
            return float(np.clip(mismatch, 0, 1))
        
        return 0.0
    
    def _calculate_high_freq_energy(self, image):
        """Calculate energy in high frequencies using FFT"""
        if image.shape[0] < 10 or image.shape[1] < 10:
            return 0.0
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Get high-frequency region (outer 30%)
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Mask for high frequencies
        y, x = np.ogrid[:h, :w]
        mask = ((x - center_w)**2 + (y - center_h)**2) > (min(h, w) * 0.35)**2
        
        high_freq_energy = np.mean(magnitude[mask])
        
        return float(high_freq_energy)
    
    def _detect_edge_anomaly(self, frame, face_box):
        """
        Detect unnatural edges at face boundary
        
        Face swaps often have visible seams
        """
        x1, y1, x2, y2 = face_box
        
        # Detect edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Check boundary regions (5 pixels on each side)
        margin = 5
        
        # Left boundary
        left_x1 = max(0, x1 - margin)
        left_x2 = min(frame.shape[1], x1 + margin)
        left_edges = edges[y1:y2, left_x1:left_x2]
        
        # Right boundary
        right_x1 = max(0, x2 - margin)
        right_x2 = min(frame.shape[1], x2 + margin)
        right_edges = edges[y1:y2, right_x1:right_x2]
        
        # Top boundary
        top_y1 = max(0, y1 - margin)
        top_y2 = min(frame.shape[0], y1 + margin)
        top_edges = edges[top_y1:top_y2, x1:x2]
        
        # Bottom boundary (less important, can be obscured by body)
        bottom_y1 = max(0, y2 - margin)
        bottom_y2 = min(frame.shape[0], y2 + margin)
        bottom_edges = edges[bottom_y1:bottom_y2, x1:x2]
        
        # Calculate edge density at boundaries
        total_pixels = (left_edges.size + right_edges.size + 
                       top_edges.size + bottom_edges.size)
        
        if total_pixels == 0:
            return 0.0
        
        edge_pixels = (np.sum(left_edges > 0) + np.sum(right_edges > 0) + 
                      np.sum(top_edges > 0) + np.sum(bottom_edges > 0))
        
        edge_density = edge_pixels / total_pixels
        
        # High edge density at boundary = suspicious
        # Normal images: ~5% edge density
        # Face swaps: 15-30% edge density
        if edge_density > 0.10:
            anomaly = min((edge_density - 0.10) / 0.20, 1.0)
            return float(anomaly)
        
        return 0.0
    
    def _detect_compression_mismatch(self, face_region, body_region):
        """
        Detect compression artifact differences
        
        Deepfake faces often have different JPEG compression
        """
        face_compression = self._estimate_compression_level(face_region)
        body_compression = self._estimate_compression_level(body_region)
        
        # Significant difference in compression = suspicious
        compression_diff = abs(face_compression - body_compression)
        
        if compression_diff > 0.2:
            mismatch = min(compression_diff / 0.5, 1.0)
            return float(mismatch)
        
        return 0.0
    
    def _estimate_compression_level(self, image):
        """
        Estimate JPEG compression level from blocking artifacts
        """
        if image.shape[0] < 16 or image.shape[1] < 16:
            return 0.0
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Calculate blockiness (8x8 JPEG blocks)
        h, w = gray.shape
        block_size = 8
        
        # Horizontal blocking
        h_blocks = h // block_size
        h_diffs = []
        for i in range(1, h_blocks):
            y = i * block_size
            if y < h - 1:
                diff = np.mean(np.abs(gray[y, :] - gray[y-1, :]))
                h_diffs.append(diff)
        
        # Vertical blocking
        w_blocks = w // block_size
        v_diffs = []
        for j in range(1, w_blocks):
            x = j * block_size
            if x < w - 1:
                diff = np.mean(np.abs(gray[:, x] - gray[:, x-1]))
                v_diffs.append(diff)
        
        if not h_diffs or not v_diffs:
            return 0.0
        
        # Average blocking artifact strength
        blockiness = (np.mean(h_diffs) + np.mean(v_diffs)) / 2
        
        # Normalize (typical range: 0-10)
        compression_estimate = min(blockiness / 10.0, 1.0)
        
        return float(compression_estimate)
    
    def _interpret_score(self, score):
        """Interpret quality mismatch score"""
        if score < 0.2:
            return "Minimal quality mismatch - likely authentic"
        elif score < 0.4:
            return "Slight quality mismatch - borderline"
        elif score < 0.6:
            return "Moderate quality mismatch - suspicious"
        elif score < 0.8:
            return "Significant quality mismatch - likely deepfake"
        else:
            return "Severe quality mismatch - strong deepfake indicator"


# Convenience function
def analyze_face_body_quality(frame, face_box):
    """
    Quick analysis function
    
    Args:
        frame: Video frame
        face_box: (x1, y1, x2, y2)
    
    Returns:
        float: Quality mismatch score (0-1)
    """
    analyzer = QualityForensicsAnalyzer()
    result = analyzer.analyze_frame(frame, face_box)
    return result["quality_mismatch_score"]