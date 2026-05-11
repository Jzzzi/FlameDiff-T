from __future__ import annotations

import torch
from torch import nn


class RectifiedFlow(nn.Module):
    def __init__(
        self,
        sample_steps: int = 100,
        train_eps: float = 1.0e-5,
        time_scale: float = 1000.0,
    ) -> None:
        super().__init__()
        if sample_steps <= 0:
            raise ValueError("sample_steps must be positive")
        self.sample_steps = int(sample_steps)
        self.train_eps = float(train_eps)
        self.time_scale = float(time_scale)

    def sample_train_t(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        if self.train_eps <= 0:
            return torch.rand(batch_size, device=device)
        return torch.rand(batch_size, device=device) * (1.0 - 2.0 * self.train_eps) + self.train_eps

    def interpolate(
        self,
        noise: torch.Tensor,
        target: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        t = times.view(-1, *([1] * (target.ndim - 1)))
        return (1.0 - t) * noise + t * target

    def target_velocity(self, noise: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return target - noise

    def model_time(self, times: torch.Tensor) -> torch.Tensor:
        return times * self.time_scale

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        condition_latents: torch.Tensor,
        target_shape: tuple[int, ...],
        device: torch.device | str,
    ) -> torch.Tensor:
        sample = torch.randn(target_shape, device=device)
        dt = 1.0 / self.sample_steps
        for step in range(self.sample_steps):
            t_value = torch.full((target_shape[0],), step * dt, device=device, dtype=torch.float32)
            velocity = model(
                noisy_targets=sample,
                condition_latents=condition_latents,
                timesteps=self.model_time(t_value),
            )
            sample = sample + dt * velocity
        return sample
