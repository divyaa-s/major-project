import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from multiprocessing import freeze_support

from model import UNet
from diffusion_utils import Diffusion

# --------------------
# CONFIG
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 32
TIMESTEPS = 1000
LR = 1e-4
IMG_SIZE = 128

DATASET_PATH = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/train"
CHECKPOINT_DIR = "diffusion/model/checkpoints"

NUM_WORKERS = 0  # IMPORTANT for Windows

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def main():
    # --------------------
    # DATA (REAL ONLY)
    # --------------------
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    dataset = datasets.ImageFolder(
        root=DATASET_PATH,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    # --------------------
    # MODEL
    # --------------------
    model = UNet().to(DEVICE)
    diffusion = Diffusion(timesteps=TIMESTEPS, device=DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.MSELoss()

    # --------------------
    # TRAINING LOOP
    # --------------------
    model.train()

    for epoch in range(1, EPOCHS + 1):
        loop = tqdm(loader, desc=f"Epoch [{epoch}/{EPOCHS}]")
        epoch_loss = 0.0

        for imgs, _ in loop:
            imgs = imgs.to(DEVICE)

            t = torch.randint(
                0, TIMESTEPS, (imgs.size(0),), device=DEVICE
            )

            noisy_imgs, noise = diffusion.add_noise(imgs, t)
            noise_pred = model(noisy_imgs, t)

            loss = loss_fn(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(loader)
        print(f"✅ Epoch {epoch} completed | Avg Loss: {avg_loss:.4f}")

        # save checkpoint every epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss
            },
            os.path.join(CHECKPOINT_DIR, f"diffusion_epoch_{epoch}.pth")
        )

    # save final model
    torch.save(
        model.state_dict(),
        "diffusion/model/diffusion_reconstructor_final.pth"
    )

    print("🎉 Training complete. Final model saved.")


if __name__ == "__main__":
    freeze_support()
    main()
