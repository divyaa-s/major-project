import cv2
import numpy as np

def extract_dct_features(image_path, block_size=8):
    """
    Extract DCT mid-frequency features + simple statistical summaries per block
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (256, 256))
    except:
        return None

    h, w = img.shape
    features = []

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size].astype(np.float32)
            dct = cv2.dct(block)

            # Mid-frequency coefficients (avoid DC and very low freq)
            mid = dct[2:6, 2:6].flatten()

            if len(mid) == 0:
                continue

            # Add block-level statistics (helps with distortion robustness)
            mid_abs = np.abs(mid)
            features.extend([
                np.mean(mid_abs),           # mean magnitude
                np.std(mid_abs),            # spread
                np.mean(mid_abs**2),        # energy proxy
                np.max(mid_abs) - np.min(mid_abs),  # dynamic range
            ])
            features.extend(mid)  # keep raw coeffs too

    if len(features) == 0:
        return None

    return np.array(features, dtype=np.float32)


def extract_frequency_statistics(image_path):
    """
    Global frequency-domain statistics (used both in training and prediction)
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (256, 256))
    except:
        return None

    dct = cv2.dct(np.float32(img))
    mid = np.abs(dct[32:96, 32:96]) + 1e-10

    energy     = np.mean(mid)
    variance   = np.var(mid)
    flatness   = np.exp(np.mean(np.log(mid))) / np.mean(mid)
    entropy    = -np.sum((mid / mid.sum()) * np.log(mid / mid.sum() + 1e-10))

    # Very simple high-frequency energy ratio
    high = np.abs(dct[96:, 96:]) + 1e-10
    high_energy_ratio = np.mean(high) / (energy + 1e-6)

    return np.array([
        energy,
        variance,
        flatness,
        entropy,
        high_energy_ratio
    ], dtype=np.float32)