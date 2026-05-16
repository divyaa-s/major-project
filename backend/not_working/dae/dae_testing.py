import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from dae_model import DAE

# -------------------------------
# Config
# -------------------------------
MODEL_PATH = "models/dae_real.pth"
TEST_DIR = r"D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/real"  
# ⬆️ change to real OR fake folder

DEVICE = "cpu"
NUM_IMAGES = 5          # how many images to visualize
SAVE_RESULTS = True
SAVE_DIR = "dae/fake_res"

#os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------------
# Transform (same as training)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# -------------------------------
# Load model
# -------------------------------
model = DAE().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------------
# Load images
# -------------------------------
image_files = os.listdir(TEST_DIR)[:NUM_IMAGES]

images = []
names = []

for name in image_files:
    path = os.path.join(TEST_DIR, name)
    img = Image.open(path).convert("RGB")
    img = transform(img)
    images.append(img)
    names.append(name)

images = torch.stack(images).to(DEVICE)

# -------------------------------
# Forward pass
# -------------------------------
errors = []

with torch.no_grad():
    reconstructions = model(images)

# -------------------------------
# Visualization + Error
# -------------------------------
for i in range(len(images)):
    original = images[i].permute(1, 2, 0).cpu().numpy()
    recon = reconstructions[i].permute(1, 2, 0).cpu().numpy()
    error_map = np.abs(original - recon)
    error_score = error_map.mean()
    errors.append(error_score)

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(recon)
    plt.title("Reconstruction")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(error_map)
    plt.title(f"Error (mean={error_score:.5f})")
    plt.axis("off")

    plt.tight_layout()

    if SAVE_RESULTS:
        plt.savefig(os.path.join(SAVE_DIR, f"{names[i]}_dae.png"))
        plt.close()
    else:
        plt.show()

# -------------------------------
# Summary
# -------------------------------
print("✅ DAE Testing Complete")
print(f"Mean Reconstruction Error: {np.mean(errors):.6f}")
print(f"Std  Reconstruction Error: {np.std(errors):.6f}")
