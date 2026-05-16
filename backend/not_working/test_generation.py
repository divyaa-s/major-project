
from torchvision.transforms.functional import to_pil_image


import os
from PIL import Image
import torch

from deepfake_app.generative_models import (
    gan_reconstructor,
    diffusion_reconstructor
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
INPUT_IMAGE_PATH = "test_images/Dwayne_Johnson.jpg"   # CHANGE THIS
OUTPUT_DIR = "test_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# Load image
# --------------------------------------------------
image = Image.open(INPUT_IMAGE_PATH).convert("RGB")

# --------------------------------------------------
# GAN / Autoencoder reconstruction
# --------------------------------------------------
print("[INFO] Running GAN / Autoencoder reconstruction...")

with torch.no_grad():
    gan_output = gan_reconstructor.reconstruct(image)

# Convert tensor → PIL
if isinstance(gan_output, torch.Tensor):
    gan_output = gan_output.squeeze(0)  # remove batch if present
    gan_output = gan_output.clamp(0, 1)
    gan_output = to_pil_image(gan_output)

gan_output_path = os.path.join(OUTPUT_DIR, "gan_reconstruction.jpg")
gan_output.save(gan_output_path)

print(f"[OK] GAN output saved to {gan_output_path}")


# --------------------------------------------------
# Diffusion reconstruction
# --------------------------------------------------
print("[INFO] Running Diffusion reconstruction (this may be slow on CPU)...")

with torch.no_grad():
    diff_output = diffusion_reconstructor.reconstruct(image)

diff_output_path = os.path.join(OUTPUT_DIR, "diffusion_reconstruction.jpg")
diff_output.save(diff_output_path)

print(f"[OK] Diffusion output saved to {diff_output_path}")

# --------------------------------------------------
# Save input for comparison
# --------------------------------------------------
input_copy_path = os.path.join(OUTPUT_DIR, "input.jpg")
image.save(input_copy_path)

print(f"[DONE] Input image copied to {input_copy_path}")

print("\nCompare these three files:")
print("1. input.jpg")
print("2. gan_reconstruction.jpg")
print("3. diffusion_reconstruction.jpg")
