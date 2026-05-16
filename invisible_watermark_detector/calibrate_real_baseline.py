import os
import cv2
import numpy as np

def compute_dct_metrics(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    dct = cv2.dct(np.float32(img))
    mid = np.abs(dct[32:96, 32:96])

    energy = np.mean(mid)
    variance = np.var(mid)

    mid = mid + 1e-8
    flatness = np.exp(np.mean(np.log(mid))) / np.mean(mid)

    return energy, variance, flatness


real_dir = "data/real"

energies, variances, flatnesses = [], [], []

for img_name in os.listdir(real_dir):
    path = os.path.join(real_dir, img_name)
    e, v, f = compute_dct_metrics(path)
    energies.append(e)
    variances.append(v)
    flatnesses.append(f)

print("=== REAL IMAGE BASELINE ===")
print(f"Energy   mean={np.mean(energies):.2f}, std={np.std(energies):.2f}")
print(f"Variance mean={np.mean(variances):.2f}, std={np.std(variances):.2f}")
print(f"Flatness mean={np.mean(flatnesses):.3f}, std={np.std(flatnesses):.3f}")
