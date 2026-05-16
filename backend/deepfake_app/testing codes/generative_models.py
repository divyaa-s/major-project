import torch
from diffusers import StableDiffusionImg2ImgPipeline

from deepfake_app.autoencoder import FaceAutoEncoder
from deepfake_app.generative_pipeline import (
    GANReconstructor,
    DiffusionReconstructor,
    DivergenceMetric
)

DEVICE = "cpu"

# --------------------------------------------------
# Load AutoEncoder (GAN substitute)
# --------------------------------------------------

def load_autoencoder():
    model = FaceAutoEncoder()
    model.load_state_dict(
        torch.load("models/face_autoencoder_real.pth", map_location="cpu")
    )
    model.eval()
    return model


autoencoder = load_autoencoder()
gan_reconstructor = GANReconstructor(autoencoder)

# --------------------------------------------------
# Load Diffusion (CPU)
# --------------------------------------------------

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)

pipe = pipe.to("cpu")
pipe.enable_attention_slicing()

diffusion_reconstructor = DiffusionReconstructor(pipe)
divergence_metric = DivergenceMetric()
