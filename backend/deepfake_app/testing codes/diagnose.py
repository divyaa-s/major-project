# quick_diagnose.py
import cv2
import numpy as np
from PIL import Image
import os

def diagnose_images(real_path, fake_path, num_samples=5):
    """Deep dive into what's different"""
    
    print("\n" + "="*70)
    print("IMAGE DIAGNOSTICS")
    print("="*70)
    
    real_imgs = [f for f in os.listdir(real_path) if f.endswith(('.jpg', '.png'))][:num_samples]
    fake_imgs = [f for f in os.listdir(fake_path) if f.endswith(('.jpg', '.png'))][:num_samples]
    
    # Check 1: File formats and compression
    print("\n1. FILE CHARACTERISTICS")
    print("-" * 70)
    
    for img_name in real_imgs[:3]:
        img = Image.open(os.path.join(real_path, img_name))
        print(f"REAL: {img_name}")
        print(f"  Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
        if hasattr(img, 'info'):
            if 'quality' in img.info:
                print(f"  JPEG Quality: {img.info['quality']}")
    
    for img_name in fake_imgs[:3]:
        img = Image.open(os.path.join(fake_path, img_name))
        print(f"FAKE: {img_name}")
        print(f"  Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
        if hasattr(img, 'info'):
            if 'quality' in img.info:
                print(f"  JPEG Quality: {img.info['quality']}")
    
    # Check 2: Pixel statistics
    print("\n2. PIXEL STATISTICS")
    print("-" * 70)
    
    def get_stats(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return {
            'mean': np.mean(img, axis=(0,1)),
            'std': np.std(img, axis=(0,1)),
            'range': (np.min(img), np.max(img))
        }
    
    real_stats = [get_stats(os.path.join(real_path, img)) for img in real_imgs]
    fake_stats = [get_stats(os.path.join(fake_path, img)) for img in fake_imgs]
    
    print(f"REAL mean brightness: {np.mean([s['mean'] for s in real_stats]):.2f}")
    print(f"FAKE mean brightness: {np.mean([s['mean'] for s in fake_stats]):.2f}")
    print(f"REAL std dev: {np.mean([s['std'] for s in real_stats]):.2f}")
    print(f"FAKE std dev: {np.mean([s['std'] for s in fake_stats]):.2f}")
    
    # Check 3: Frequency domain analysis
    print("\n3. FREQUENCY DOMAIN")
    print("-" * 70)
    
    def get_freq_energy(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # High frequency energy (edges)
        h, w = magnitude.shape
        center_mask = np.zeros_like(magnitude)
        center_mask[h//4:3*h//4, w//4:3*w//4] = 1
        
        low_freq = np.sum(magnitude * center_mask)
        high_freq = np.sum(magnitude * (1 - center_mask))
        
        return high_freq / (low_freq + 1e-10)
    
    real_freq = [get_freq_energy(os.path.join(real_path, img)) for img in real_imgs]
    fake_freq = [get_freq_energy(os.path.join(fake_path, img)) for img in fake_imgs]
    
    print(f"REAL high-freq ratio: {np.mean(real_freq):.4f} ± {np.std(real_freq):.4f}")
    print(f"FAKE high-freq ratio: {np.mean(fake_freq):.4f} ± {np.std(fake_freq):.4f}")
    print(f"Separation: {abs(np.mean(real_freq) - np.mean(fake_freq)):.4f}")
    
    # Check 4: Visual inspection hints
    print("\n4. RECOMMENDATIONS")
    print("-" * 70)
    
    if abs(np.mean(real_freq) - np.mean(fake_freq)) < 0.05:
        print("⚠️ Very similar frequency signatures")
        print("   → Suggests high-quality GAN/diffusion models")
        print("   → Traditional forensics may not work")
    
    real_formats = set([Image.open(os.path.join(real_path, img)).format for img in real_imgs])
    fake_formats = set([Image.open(os.path.join(fake_path, img)).format for img in fake_imgs])
    
    if 'JPEG' in real_formats and 'PNG' in fake_formats:
        print("⚠️ Format mismatch: Real=JPEG, Fake=PNG")
        print("   → Features are detecting compression, not fakeness")
    
    print("\n" + "="*70)

# Run diagnostics
REAL_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/test/real_and_fake_face/training_real"
FAKE_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/test/real_and_fake_face/training_fake"

diagnose_images(REAL_FOLDER, FAKE_FOLDER)