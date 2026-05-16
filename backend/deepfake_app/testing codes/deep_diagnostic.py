"""
Deep diagnostic to understand why visually-fake images fool all features
"""

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import os

def deep_compare_images(real_path, fake_path, output_dir='diagnostics'):
    """
    Pixel-level comparison to find what's actually different
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    real_img = cv2.imread(real_path)
    fake_img = cv2.imread(fake_path)
    
    real_rgb = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
    fake_rgb = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)
    
    print("="*70)
    print(f"Comparing:")
    print(f"  REAL: {Path(real_path).name}")
    print(f"  FAKE: {Path(fake_path).name}")
    print("="*70)
    
    # 1. Basic statistics
    print("\n1. BASIC STATISTICS")
    print("-"*70)
    print(f"REAL - Mean: {real_rgb.mean():.2f}, Std: {real_rgb.std():.2f}")
    print(f"FAKE - Mean: {fake_rgb.mean():.2f}, Std: {fake_rgb.std():.2f}")
    
    # 2. Pixel value distributions
    print("\n2. PIXEL VALUE DISTRIBUTIONS")
    print("-"*70)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Histograms
    for i, (img, label) in enumerate([(real_rgb, 'REAL'), (fake_rgb, 'FAKE')]):
        for c, color in enumerate(['Red', 'Green', 'Blue']):
            axes[i, c].hist(img[:,:,c].ravel(), bins=50, 
                          color=color.lower(), alpha=0.7)
            axes[i, c].set_title(f'{label} - {color}')
            axes[i, c].set_xlim([0, 255])
    
    # Grayscale comparison
    real_gray = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
    fake_gray = cv2.cvtColor(fake_img, cv2.COLOR_BGR2GRAY)
    
    axes[0, 3].hist(real_gray.ravel(), bins=50, color='gray', alpha=0.7)
    axes[0, 3].set_title('REAL - Grayscale')
    axes[1, 3].hist(fake_gray.ravel(), bins=50, color='gray', alpha=0.7)
    axes[1, 3].set_title('FAKE - Grayscale')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pixel_distributions.png', dpi=150)
    print(f"   Saved: {output_dir}/pixel_distributions.png")
    
    # 3. Frequency domain analysis
    print("\n3. FREQUENCY DOMAIN ANALYSIS")
    print("-"*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (img, label) in enumerate([(real_gray, 'REAL'), (fake_gray, 'FAKE')]):
        # FFT
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        magnitude_log = np.log1p(magnitude)
        
        # Display image
        axes[idx, 0].imshow(img, cmap='gray')
        axes[idx, 0].set_title(f'{label} Image')
        axes[idx, 0].axis('off')
        
        # Display FFT
        axes[idx, 1].imshow(magnitude_log, cmap='hot')
        axes[idx, 1].set_title(f'{label} FFT Magnitude')
        axes[idx, 1].axis('off')
        
        # Radial profile
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Calculate radial average
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2).astype(int)
        
        radial_mean = np.bincount(r.ravel(), magnitude.ravel()) / np.bincount(r.ravel())
        
        axes[idx, 2].plot(radial_mean[:len(radial_mean)//2])
        axes[idx, 2].set_title(f'{label} Radial Frequency Profile')
        axes[idx, 2].set_xlabel('Frequency')
        axes[idx, 2].set_ylabel('Magnitude')
        axes[idx, 2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/frequency_analysis.png', dpi=150)
    print(f"   Saved: {output_dir}/frequency_analysis.png")
    
    # 4. Edge detection comparison
    print("\n4. EDGE DETECTION")
    print("-"*70)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (img, label) in enumerate([(real_gray, 'REAL'), (fake_gray, 'FAKE')]):
        # Original
        axes[idx, 0].imshow(img, cmap='gray')
        axes[idx, 0].set_title(f'{label} Original')
        axes[idx, 0].axis('off')
        
        # Canny edges
        edges = cv2.Canny(img, 50, 150)
        axes[idx, 1].imshow(edges, cmap='gray')
        axes[idx, 1].set_title(f'{label} Canny Edges')
        axes[idx, 1].axis('off')
        
        # Sobel gradient
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)
        
        axes[idx, 2].imshow(sobel, cmap='hot')
        axes[idx, 2].set_title(f'{label} Sobel Gradient')
        axes[idx, 2].axis('off')
        
        edge_density = np.sum(edges > 0) / edges.size
        print(f"   {label} edge density: {edge_density:.4f}")
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/edge_analysis.png', dpi=150)
    print(f"   Saved: {output_dir}/edge_analysis.png")
    
    # 5. Texture analysis
    print("\n5. TEXTURE ANALYSIS")
    print("-"*70)
    
    # Local standard deviation (texture measure)
    kernel_size = 15
    
    real_local_std = cv2.blur(real_gray**2, (kernel_size, kernel_size)) - \
                     cv2.blur(real_gray, (kernel_size, kernel_size))**2
    real_local_std = np.sqrt(np.maximum(real_local_std, 0))
    
    fake_local_std = cv2.blur(fake_gray**2, (kernel_size, kernel_size)) - \
                     cv2.blur(fake_gray, (kernel_size, kernel_size))**2
    fake_local_std = np.sqrt(np.maximum(fake_local_std, 0))
    
    print(f"   REAL local texture std: {real_local_std.mean():.2f}")
    print(f"   FAKE local texture std: {fake_local_std.mean():.2f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(real_rgb)
    axes[0, 0].set_title('REAL Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(real_local_std, cmap='hot')
    axes[0, 1].set_title('REAL Local Texture')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(fake_rgb)
    axes[1, 0].set_title('FAKE Image')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(fake_local_std, cmap='hot')
    axes[1, 1].set_title('FAKE Local Texture')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/texture_analysis.png', dpi=150)
    print(f"   Saved: {output_dir}/texture_analysis.png")
    
    # 6. CRITICAL: Check if images are preprocessed/normalized
    print("\n6. PREPROCESSING CHECK")
    print("-"*70)
    
    # Check value ranges
    print(f"   REAL - Min: {real_rgb.min()}, Max: {real_rgb.max()}")
    print(f"   FAKE - Min: {fake_rgb.min()}, Max: {fake_rgb.max()}")
    
    # Check for common preprocessing
    real_variance = np.var(real_rgb, axis=(0,1))
    fake_variance = np.var(fake_rgb, axis=(0,1))
    
    print(f"   REAL channel variance: R={real_variance[0]:.1f}, G={real_variance[1]:.1f}, B={real_variance[2]:.1f}")
    print(f"   FAKE channel variance: R={fake_variance[0]:.1f}, G={fake_variance[1]:.1f}, B={fake_variance[2]:.1f}")
    
    # Check if variance is suspiciously similar (indicates normalization)
    if np.allclose(real_variance, fake_variance, rtol=0.1):
        print("   ⚠️ WARNING: Variances are very similar!")
        print("   → Images may be preprocessed/normalized identically")
        print("   → This destroys natural statistical differences")
    
    # 7. File metadata
    print("\n7. FILE METADATA")
    print("-"*70)
    
    real_pil = Image.open(real_path)
    fake_pil = Image.open(fake_path)
    
    print(f"   REAL - Format: {real_pil.format}, Size: {real_pil.size}, Mode: {real_pil.mode}")
    print(f"   FAKE - Format: {fake_pil.format}, Size: {fake_pil.size}, Mode: {fake_pil.mode}")
    
    if hasattr(real_pil, 'info'):
        print(f"   REAL info: {real_pil.info}")
    if hasattr(fake_pil, 'info'):
        print(f"   FAKE info: {fake_pil.info}")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    # Check if preprocessing is the issue
    if np.allclose(real_variance, fake_variance, rtol=0.1):
        print("\n⚠️ CRITICAL FINDING: Images appear to be preprocessed identically!")
        print("\nThis explains why features fail:")
        print("  1. Both real and fake images went through same normalization")
        print("  2. This removes natural statistical differences")
        print("  3. Traditional features measure statistics, not semantic content")
        print("\n✅ SOLUTION: Use deep learning features that capture semantic")
        print("   content, not just pixel statistics.")
    else:
        print("\n🤔 Images are statistically very similar despite visual differences")
        print("   This suggests the generator is very good at matching statistics.")
        print("\n✅ SOLUTION: Must use deep learning to detect semantic differences.")

# ============================================================================
# BATCH ANALYSIS
# ============================================================================
def analyze_entire_dataset(real_folder, fake_folder, num_samples=10):
    """Analyze multiple image pairs"""
    
    real_imgs = [os.path.join(real_folder, f) for f in os.listdir(real_folder) 
                 if f.endswith(('.jpg', '.png'))][:num_samples]
    fake_imgs = [os.path.join(fake_folder, f) for f in os.listdir(fake_folder) 
                 if f.endswith(('.jpg', '.png'))][:num_samples]
    
    print("\nBATCH ANALYSIS ACROSS MULTIPLE IMAGES")
    print("="*70)
    
    real_stats = []
    fake_stats = []
    
    for real_path in real_imgs:
        img = cv2.imread(real_path)
        real_stats.append({
            'mean': img.mean(),
            'std': img.std(),
            'min': img.min(),
            'max': img.max()
        })
    
    for fake_path in fake_imgs:
        img = cv2.imread(fake_path)
        fake_stats.append({
            'mean': img.mean(),
            'std': img.std(),
            'min': img.min(),
            'max': img.max()
        })
    
    print(f"\nREAL images ({len(real_stats)} samples):")
    print(f"  Mean: {np.mean([s['mean'] for s in real_stats]):.2f} ± {np.std([s['mean'] for s in real_stats]):.2f}")
    print(f"  Std:  {np.mean([s['std'] for s in real_stats]):.2f} ± {np.std([s['std'] for s in real_stats]):.2f}")
    
    print(f"\nFAKE images ({len(fake_stats)} samples):")
    print(f"  Mean: {np.mean([s['mean'] for s in fake_stats]):.2f} ± {np.std([s['mean'] for s in fake_stats]):.2f}")
    print(f"  Std:  {np.mean([s['std'] for s in fake_stats]):.2f} ± {np.std([s['std'] for s in fake_stats]):.2f}")
    
    # Compare
    mean_diff = abs(np.mean([s['mean'] for s in real_stats]) - 
                   np.mean([s['mean'] for s in fake_stats]))
    std_diff = abs(np.mean([s['std'] for s in real_stats]) - 
                  np.mean([s['std'] for s in fake_stats]))
    
    print(f"\nDifferences:")
    print(f"  Mean difference: {mean_diff:.2f}")
    print(f"  Std difference:  {std_diff:.2f}")
    
    if mean_diff < 5 and std_diff < 5:
        print("\n⚠️ Pixel statistics are nearly identical!")
        print("   → Traditional forensic features won't work")
        print("   → Need deep learning approaches")

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    
    REAL_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/test/real_and_fake_face/training_real"
    FAKE_FOLDER = "D:/Projects/Major Project/Deepfake Detection/datasets/test/real_and_fake_face/training_fake"
    
    # Pick first images for detailed comparison
    real_imgs = [f for f in os.listdir(REAL_FOLDER) if f.endswith(('.jpg', '.png'))]
    fake_imgs = [f for f in os.listdir(FAKE_FOLDER) if f.endswith(('.jpg', '.png'))]
    
    if real_imgs and fake_imgs:
        real_path = os.path.join(REAL_FOLDER, real_imgs[0])
        fake_path = os.path.join(FAKE_FOLDER, fake_imgs[0])
        
        deep_compare_images(real_path, fake_path)
        
        # Batch analysis
        analyze_entire_dataset(REAL_FOLDER, FAKE_FOLDER)
    else:
        print("No images found!")