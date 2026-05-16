import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
from deepfake_app.autoencoder import FaceAutoEncoder

# -------------------------------
# Config
# -------------------------------
MODEL_PATH = "models/face_autoencoder_real.pth"
IMAGE_DIR = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/fake"
DEVICE = "cpu"
NUM_IMAGES = 5

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
model = FaceAutoEncoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------------
# Load images
# -------------------------------
image_files = os.listdir(IMAGE_DIR)[:NUM_IMAGES]
images = []

for img_name in image_files:
    img = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
    images.append(transform(img))

images = torch.stack(images).to(DEVICE)

# -------------------------------
# Forward pass
# -------------------------------
with torch.no_grad():
    reconstructions = model(images)

# -------------------------------
# Visualization
# -------------------------------
for i in range(NUM_IMAGES):
    original = images[i].permute(1, 2, 0).cpu().numpy()
    recon = reconstructions[i].permute(1, 2, 0).cpu().numpy()
    error = np.abs(original - recon)

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Fake Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(recon)
    plt.title("Reconstruction")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(error)
    plt.title("Reconstruction Error")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
