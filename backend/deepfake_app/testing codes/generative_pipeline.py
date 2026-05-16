import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image

# --------------------------------------------------
# AutoEncoder wrapper (GAN replacement)
# --------------------------------------------------

class GANReconstructor:
    def __init__(self, model):
        self.model = model
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def reconstruct(self, face_img):
        if isinstance(face_img, np.ndarray):
            face_img = Image.fromarray(face_img)

        x = self.transform(face_img).unsqueeze(0)

        with torch.no_grad():
            recon = self.model(x)

        return recon.squeeze(0)


# --------------------------------------------------
# Diffusion wrapper
# --------------------------------------------------

class DiffusionReconstructor:
    def __init__(self, pipe):
        self.pipe = pipe

    def reconstruct(self, face_img):
        if isinstance(face_img, np.ndarray):
            face_img = Image.fromarray(face_img)

        result = self.pipe(
            prompt="a realistic human face",
            image=face_img,
            strength=0.35,
            guidance_scale=7.5
        ).images[0]

        return result


# --------------------------------------------------
# Divergence metric
# --------------------------------------------------

class DivergenceMetric:
    def compute(self, img1, img2):
        """
        img1: tensor (AutoEncoder output)
        img2: PIL image (Diffusion output)
        """

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        if isinstance(img2, Image.Image):
            img2 = transform(img2)

        img1 = (img1 + 1) / 2  # denormalize

        l1 = torch.mean(torch.abs(img1 - img2))
        return l1.item()


# --------------------------------------------------
# Main pipeline call
# --------------------------------------------------

def generative_consistency_check(
    face_crop,
    gan_reconstructor,
    diffusion_reconstructor,
    divergence_metric
):
    gan_img = gan_reconstructor.reconstruct(face_crop)
    diff_img = diffusion_reconstructor.reconstruct(face_crop)
    divergence = divergence_metric.compute(gan_img, diff_img)

    return gan_img, diff_img, divergence
