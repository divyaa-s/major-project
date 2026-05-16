import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ===============================
# Config
# ===============================
DEVICE = "cpu"
EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-3
LATENT_DIM = 128

REAL_TRAIN_DIR = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/train"
MODEL_PATH = "models/face_vae_real.pth"

os.makedirs("models", exist_ok=True)

# ===============================
# VAE Model
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        h = self.fc_dec(z).view(-1, 128, 16, 16)
        return self.decoder(h), mu, logvar

# ===============================
# Loss
# ===============================
def vae_loss(recon, x, mu, logvar):
    recon_loss = F.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.001 * kl

# ===============================
# Dataset
# ===============================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=REAL_TRAIN_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===============================
# Training
# ===============================
model = FaceVAE(LATENT_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    epoch_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for imgs, _ in pbar:
        imgs = imgs.to(DEVICE)

        recon, mu, logvar = model(imgs)
        loss = vae_loss(recon, imgs, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    print(f"Epoch [{epoch+1}] Avg Loss: {epoch_loss/len(loader):.6f}")

torch.save(model.state_dict(), MODEL_PATH)
print("✅ VAE training complete")
