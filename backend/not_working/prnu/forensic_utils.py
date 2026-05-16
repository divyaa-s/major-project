import cv2
import numpy as np
import pywt

import cv2
import numpy as np
import pywt

# --- PART 1: Noise Extraction Engine ---
def extract_noise(image_path):
    """Extracts the high-frequency PRNU noise residual using Wavelets."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    img = cv2.resize(img, (512, 512)) # Higher res for better PRNU
    img = img.astype(np.float32) / 255.0

    # Wavelet denoising
    coeffs = pywt.wavedec2(img, 'db8', level=4)
    coeffs[0] = np.zeros_like(coeffs[0])
    denoised = pywt.waverec2(coeffs, 'db8')
    
    noise = img - denoised
    return noise - np.mean(noise)

# --- PART 2: Forensic Rules Engine ---
def get_forensic_integrity_score(img_path):
    """
    Evaluates Absolute Noise Power and Consistency Ratio.
    Returns a single float between 0.0 (Fake) and 1.0 (Real).
    """
    img = cv2.imread(img_path)
    if img is None: return 0.5
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = extract_noise(img_path)
    
    if noise is None:
        return 0.5

    # Face Detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    
    overall_var = np.var(noise)

    # Fallback if no face is detected
    if len(faces) == 0:
        if overall_var < 1.0e-4:
            return 0.15 # Fake Trap (Clean AI)
        return 0.85     # Real Fallback

    # Create Face Mask
    x, y, w, h = faces[0]
    scale_x, scale_y = 512 / img.shape[1], 512 / img.shape[0]
    nx, ny, nw, nh = int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)
    
    face_mask = np.zeros((512, 512), dtype=bool)
    face_mask[ny:ny+nh, nx:nx+nw] = True
    
    # Calculate Variances
    face_var = np.var(noise[face_mask])
    bg_var = np.var(noise[~face_mask])
    
    ratio = face_var / bg_var if bg_var > 0 else 0

    # ==========================================================
    # FORENSIC RULES ENGINE
    # ==========================================================
    
    # RULE 1: The Absolute Noise Trap (Fully Synthetic AI like Midjourney)
    if overall_var < 1.0e-4:
        return 0.15

    # RULE 2: The Face-Swap Anomaly (High freq GAN artifacts strictly on face)
    if ratio > 1.5 and bg_var > 0.005:
        return 0.20

    # RULE 3: The AI Smoothing Trap (Heavy filters wiping texture)
    if ratio < 0.5:
        return 0.25

    # RULE 4: Real Portrait Mode (Natural face, blurry background)
    if ratio > 1.5 and bg_var <= 0.005:
        return 0.90

    # RULE 5: Standard Real Image (Consistent natural noise profile)
    final_score = 1.0 - abs(1.0 - ratio)
    return float(np.clip(final_score, 0.6, 0.95))