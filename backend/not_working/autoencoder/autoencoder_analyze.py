import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from deepfake_app.autoencoder import FaceAutoEncoder

# -------------------------------
# Config
# -------------------------------
MODEL_PATH = "models/face_autoencoder_real.pth"
REAL_DIR = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/test/real"
FAKE_DIR = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/test/fake"
DEVICE = "cpu"
IMG_SIZE = 128

# -------------------------------
# Transform (same as training)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# -------------------------------
# Load model
# -------------------------------
model = FaceAutoEncoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------------
# Helper: compute reconstruction error
# -------------------------------
def compute_errors(image_dir):
    errors = []

    for img_name in tqdm(os.listdir(image_dir), desc=f"Processing {os.path.basename(image_dir)}"):
        img_path = os.path.join(image_dir, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue

        x = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            x_hat = model(x)

        # Mean Absolute Error (L1)
        error = torch.mean(torch.abs(x - x_hat)).item()
        errors.append(error)

    return errors

# -------------------------------
# Compute errors
# -------------------------------
real_errors = compute_errors(REAL_DIR)
fake_errors = compute_errors(FAKE_DIR)

# -------------------------------
# Plot distributions
# -------------------------------
plt.figure(figsize=(8, 5))

plt.hist(real_errors, bins=50, alpha=0.6, label="Real", density=True)
plt.hist(fake_errors, bins=50, alpha=0.6, label="Fake", density=True)

plt.xlabel("Reconstruction Error")
plt.ylabel("Density")
plt.title("Reconstruction Error Distribution")
plt.legend()

os.makedirs("autoencoder_results", exist_ok=True)
plt.savefig("autoencoder_results/reconstruction_error_distribution.png", dpi=300)
plt.show()

# -------------------------------
# Print summary stats (for paper)
# -------------------------------
print("Real Images:")
print(f"  Mean Error: {np.mean(real_errors):.6f}")
print(f"  Std  Error: {np.std(real_errors):.6f}")

print("\nFake Images:")
print(f"  Mean Error: {np.mean(fake_errors):.6f}")
print(f"  Std  Error: {np.std(fake_errors):.6f}")
