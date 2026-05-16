import cv2
import numpy as np
import pywt
from scipy.signal import wiener


def extract_noise(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32)

    # Normalize intensity
    img = img / 255.0

    # Wavelet denoising (PRNU standard)
    coeffs = pywt.wavedec2(img, 'db8', level=4)

    # Zero out approximation coefficients
    coeffs[0] = np.zeros_like(coeffs[0])

    denoised = pywt.waverec2(coeffs, 'db8')

    # Noise residual
    noise = img - denoised

    # Zero-mean
    noise = noise - np.mean(noise)

    return noise

def extract_noise_from_array(img, size=256):
    try:
        img = cv2.resize(img, (size, size))
    except:
        return None

    img = img.astype(np.float32)
    denoised = wiener(img, (5, 5))
    noise = img - denoised
    noise -= np.mean(noise)

    norm = np.linalg.norm(noise) + 1e-8
    return noise / norm
