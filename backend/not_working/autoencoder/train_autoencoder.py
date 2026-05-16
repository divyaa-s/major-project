import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm  # progress bar
from deepfake_app.autoencoder import FaceAutoEncoder

# -------------------------------
# Config
# -------------------------------
DATA_DIR = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/train"
MODEL_DIR = "models"
EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-3
DEVICE = "cpu"

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# Dataset
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# -------------------------------
# Model
# -------------------------------
model = FaceAutoEncoder().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# -------------------------------
# Training loop with progress bar
# -------------------------------
for epoch in range(EPOCHS):
    total_loss = 0
    loader_tqdm = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
    
    for imgs, _ in loader_tqdm:
        imgs = imgs.to(DEVICE)

        recon = model(imgs)
        loss = criterion(recon, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loader_tqdm.set_postfix({"batch_loss": f"{loss.item():.6f}"})

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.6f}")

# -------------------------------
# Save model
# -------------------------------
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "face_autoencoder_real.pth"))
print("✅ Autoencoder training complete & saved")
