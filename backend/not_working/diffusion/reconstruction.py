import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import UNet
from diffusion_utils import Diffusion

# --------------------
# CONFIG
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "diffusion/model/checkpoints/diffusion_epoch_4.pth"
IMAGE_PATH = "D:/Projects/Major Project/Deepfake Detection/datasets/New folder/real-vs-fake/real/justin_real.png"
IMG_SIZE = 128
TIMESTEPS = 1000
T_START = 300   # IMPORTANT

# --------------------
# TRANSFORM
# --------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --------------------
# LOAD IMAGE
# --------------------
img = Image.open(IMAGE_PATH).convert("RGB")
x0 = transform(img).unsqueeze(0).to(DEVICE)

# --------------------
# LOAD MODEL
# --------------------
model = UNet().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded diffusion checkpoint from epoch {checkpoint['epoch']}")


diffusion = Diffusion(timesteps=TIMESTEPS, device=DEVICE)

# --------------------
# FORWARD DIFFUSION
# --------------------
t_start = torch.tensor([T_START], device=DEVICE)
x, _ = diffusion.add_noise(x0, t_start)

# --------------------
# REVERSE DIFFUSION
# --------------------
with torch.no_grad():
    for t in reversed(range(1, T_START)):
        t_batch = torch.tensor([t], device=DEVICE)

        noise_pred = model(x, t_batch)

        alpha = diffusion.alphas[t]
        alpha_hat = diffusion.alpha_hat[t]
        beta = diffusion.betas[t]

        x = (1 / torch.sqrt(alpha)) * (
            x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * noise_pred
        )

        if t > 1:
            x += torch.sqrt(beta) * torch.randn_like(x)

x_recon = torch.clamp(x, -1, 1)

# --------------------
# ERROR
# --------------------
recon_error = F.mse_loss(x_recon, x0).item()
print(f"📉 Diffusion reconstruction error: {recon_error:.6f}")

# --------------------
# VISUALIZE
# --------------------
def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)

inp = denorm(x0).squeeze().permute(1, 2, 0).cpu().numpy()
rec = denorm(x_recon).squeeze().permute(1, 2, 0).cpu().numpy()

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(inp)
plt.title("Input")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(rec)
plt.title("Diffusion Reconstruction")
plt.axis("off")

plt.show()
