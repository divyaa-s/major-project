import torch

def linear_beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)

class Diffusion:
    def __init__(self, timesteps=1000, device="cuda"):
        self.device = device
        self.timesteps = timesteps

        self.betas = linear_beta_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        alpha_hat_t = self.alpha_hat[t][:, None, None, None]
        noisy_x = torch.sqrt(alpha_hat_t) * x + torch.sqrt(1 - alpha_hat_t) * noise
        return noisy_x, noise
