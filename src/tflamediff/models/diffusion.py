from __future__ import annotations

import torch
from torch import nn


def extract(coefficients: torch.Tensor, timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    values = coefficients.gather(0, timesteps)
    return values.view(-1, *([1] * (reference.ndim - 1)))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        clip_denoised: bool = False,
    ) -> None:
        super().__init__()
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)

        self.timesteps = timesteps
        self.clip_denoised = clip_denoised
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0)
        )

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start) * noise
        )

    def predict_start_from_noise(
        self, x_t: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        x0 = (
            extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t) * noise
        )
        if self.clip_denoised:
            x0 = x0.clamp(-1.0, 1.0)
        return x0

    def p_mean_variance(
        self,
        model: nn.Module,
        noisy_targets: torch.Tensor,
        condition_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_noise = model(noisy_targets=noisy_targets, condition_latents=condition_latents, timesteps=timesteps)
        x_start = self.predict_start_from_noise(noisy_targets, timesteps, pred_noise)
        posterior_mean = (
            extract(self.posterior_mean_coef1, timesteps, noisy_targets) * x_start
            + extract(self.posterior_mean_coef2, timesteps, noisy_targets) * noisy_targets
        )
        posterior_variance = extract(self.posterior_variance, timesteps, noisy_targets)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, timesteps, noisy_targets)
        return posterior_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        noisy_targets: torch.Tensor,
        condition_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        posterior_mean, _, posterior_log_variance = self.p_mean_variance(
            model=model,
            noisy_targets=noisy_targets,
            condition_latents=condition_latents,
            timesteps=timesteps,
        )
        noise = torch.randn_like(noisy_targets)
        nonzero_mask = (timesteps != 0).float().view(-1, *([1] * (noisy_targets.ndim - 1)))
        return posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        condition_latents: torch.Tensor,
        target_shape: tuple[int, ...],
        device: torch.device | str,
    ) -> torch.Tensor:
        sample = torch.randn(target_shape, device=device)
        for timestep in reversed(range(self.timesteps)):
            t = torch.full((target_shape[0],), timestep, device=device, dtype=torch.long)
            sample = self.p_sample(
                model=model,
                noisy_targets=sample,
                condition_latents=condition_latents,
                timesteps=t,
            )
        return sample
