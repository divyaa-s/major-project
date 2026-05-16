import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from dae_model import DAE

# --------------------------------------------------
# Config
# --------------------------------------------------
REAL_TRAIN_DIR = r"D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/train"
MODEL_SAVE_PATH = "models/dae_real.pth"

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
NOISE_STD = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("models", exist_ok=True)


# --------------------------------------------------
# Dataset
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root=REAL_TRAIN_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# --------------------------------------------------
# Training Setup
# --------------------------------------------------
model = DAE().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# --------------------------------------------------
# Training Loop
# --------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for imgs, _ in progress:
        imgs = imgs.to(DEVICE)

        noise = torch.randn_like(imgs) * NOISE_STD
        noisy_imgs = torch.clamp(imgs + noise, 0., 1.)

        recon = model(noisy_imgs)
        loss = criterion(recon, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    print(f"Epoch [{epoch+1}] Avg Loss: {epoch_loss / len(loader):.6f}")

# --------------------------------------------------
# Save Model
# --------------------------------------------------
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("✅ DAE training complete and model saved.")
