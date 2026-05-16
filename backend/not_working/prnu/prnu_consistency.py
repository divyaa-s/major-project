import cv2
import numpy as np
import pywt

# --- ENGINE: The Noise Extraction Logic ---
def extract_noise(image_path):
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

# --- ANALYSIS: The Consistency Logic ---
def get_forensic_score(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = extract_noise(img_path)
    
    if noise is None:
        return 0.5, None

    # Face Detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    
    overall_var = np.var(noise)

    if len(faces) == 0:
        if overall_var < 1.0e-4:
            return 0.15, None # Fake Trap (Clean AI)
        return 0.85, None     # Real Fallback

    # Create Face Mask
    x, y, w, h = faces[0]
    scale_x, scale_y = 512 / img.shape[1], 512 / img.shape[0]
    nx, ny, nw, nh = int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)
    
    face_mask = np.zeros((512, 512), dtype=bool)
    face_mask[ny:ny+nh, nx:nx+nw] = True
    
    # Calculate Variances
    face_var = np.var(noise[face_mask])
    bg_var = np.var(noise[~face_mask])
    
    print(f"\n--- FORENSIC DIAGNOSTICS ---")
    print(f"Overall Noise Power : {overall_var:.8f}")
    print(f"Face Noise Power    : {face_var:.8f}")
    print(f"Background Power    : {bg_var:.8f}")
    
    ratio = face_var / bg_var if bg_var > 0 else 0
    print(f"Consistency Ratio   : {ratio:.4f}")

    # ==========================================================
    # FORENSIC RULES ENGINE
    # ==========================================================
    
    # RULE 1: The Absolute Noise Trap (Fully Synthetic AI like Midjourney)
    # Raised threshold slightly to catch highly compressed fakes
    if overall_var < 1.0e-4:
        print(">> TRIGGERED: Absolute Noise Trap (Lacks camera sensor grain).")
        return 0.15, (nx, ny, nw, nh)

    # RULE 2: The Face-Swap Anomaly
    # Face has way more noise than background, AND background isn't blurry.
    # (Catches the exact 1.79 and 2.53 ratios you just found!)
    if ratio > 1.5 and bg_var > 0.005:
        print(">> TRIGGERED: Face-Swap Anomaly (Face has high-frequency GAN artifacts).")
        return 0.20, (nx, ny, nw, nh)

    # RULE 3: The AI Smoothing Trap (Filters / Beautification)
    # Face is completely wiped of texture compared to the background
    if ratio < 0.5:
        print(">> TRIGGERED: AI Smoothing Anomaly (Face texture unnaturally erased).")
        return 0.25, (nx, ny, nw, nh)

    # RULE 4: Real Portrait Mode
    # Ratio is high, BUT background noise is tiny because of depth-of-field blur
    if ratio > 1.5 and bg_var <= 0.005:
        print(">> PASSED: High ratio, but valid Portrait Mode blur detected.")
        return 0.90, (nx, ny, nw, nh)

    # RULE 5: Standard Real Image (Ratio is around 1.0)
    # We score it based on how close the ratio is to perfect 1.0
    print(">> PASSED: Consistent natural noise profile detected.")
    final_score = 1.0 - abs(1.0 - ratio)
    return float(np.clip(final_score, 0.6, 0.95)), (nx, ny, nw, nh)

# --- EXECUTION ---
if __name__ == "__main__":
    path = r"D:\Projects\Major Project\Deepfake Detection\datasets\new_combined\fake\ZUIZU4WRVY.jpg"
    score, info = get_forensic_score(path)
    print(f"\n✅ Analysis Complete! Integrity Score: {score:.4f}")
    if score < 0.5:
        print(">> VERDICT: FAKE")
    else:
        print(">> VERDICT: REAL")