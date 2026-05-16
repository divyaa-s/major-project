import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ===============================
# Config
# ===============================
DEVICE = "cpu"
LATENT_DIM = 128
MODEL_PATH = "models/face_vae_real.pth"
TEST_DIR = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/real"
NUM_IMAGES = 5

# ===============================
# VAE Model (same as training)
# ===============================
class FaceVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(128 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(128 * 16 * 16, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 128 * 16 * 16)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = mu
        h = self.fc_dec(z).view(-1, 128, 16, 16)
        return self.decoder(h)

# ===============================
# Load model
# ===============================
model = FaceVAE(LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ===============================
# Transform
# ===============================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ===============================
# Load images
# ===============================
image_files = os.listdir(TEST_DIR)[:NUM_IMAGES]

images = []
for name in tqdm(image_files, desc="Loading images"):
    img = Image.open(os.path.join(TEST_DIR, name)).convert("RGB")
    images.append(transform(img))

images = torch.stack(images).to(DEVICE)

# ===============================
# Reconstruction
# ===============================
with torch.no_grad():
    recon = model(images)

# ===============================
# Visualization
# ===============================
for i in range(NUM_IMAGES):
    orig = images[i].permute(1,2,0).cpu().numpy()
    rec  = recon[i].permute(1,2,0).cpu().numpy()
    err  = np.abs(orig - rec)

    plt.figure(figsize=(9,3))

    plt.subplot(1,3,1)
    plt.imshow(orig)
    plt.title("Input (Fake)")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(rec)
    plt.title("VAE Reconstruction")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(err)
    plt.title("Reconstruction Error")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
