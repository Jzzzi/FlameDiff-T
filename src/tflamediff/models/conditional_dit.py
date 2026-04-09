from __future__ import annotations

import math

import torch
from torch import nn


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ConditionalLatentDiT(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        latent_size: int,
        patch_size: int = 2,
        hidden_size: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_frames: int = 10,
        target_frames: int = 8,
    ) -> None:
        super().__init__()
        if latent_size % patch_size != 0:
            raise ValueError("latent_size must be divisible by patch_size")
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_frames = num_frames
        self.target_frames = target_frames
        self.grid_size = latent_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.patch_embed = nn.Conv2d(
            latent_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.output_proj = nn.Linear(hidden_size, latent_channels * patch_size * patch_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.frame_embed = nn.Embedding(num_frames, hidden_size)
        self.role_embed = nn.Embedding(3, hidden_size)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_frames * self.num_patches, hidden_size)
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.frame_embed.weight, std=0.02)
        nn.init.normal_(self.role_embed.weight, std=0.02)

    def forward(
        self,
        noisy_targets: torch.Tensor,
        condition_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_targets.shape[1] != self.target_frames:
            raise ValueError(f"Expected {self.target_frames} target frames, got {noisy_targets.shape[1]}")
        if condition_latents.shape[1] != 2:
            raise ValueError("condition_latents must contain first and last latent frame.")

        full_sequence = torch.cat(
            [condition_latents[:, :1], noisy_targets, condition_latents[:, 1:2]], dim=1
        )
        batch_size, frame_count, channels, height, width = full_sequence.shape
        if height != self.latent_size or width != self.latent_size:
            raise ValueError(
                f"Expected latent spatial size {self.latent_size}, got {(height, width)}"
            )

        embedded = self.patch_embed(full_sequence.reshape(batch_size * frame_count, channels, height, width))
        embedded = embedded.flatten(2).transpose(1, 2)
        embedded = embedded.reshape(batch_size, frame_count, self.num_patches, self.hidden_size)

        frame_ids = torch.arange(frame_count, device=full_sequence.device)
        role_ids = torch.zeros(frame_count, device=full_sequence.device, dtype=torch.long)
        role_ids[0] = 1
        role_ids[-1] = 2

        embedded = embedded + self.frame_embed(frame_ids)[None, :, None, :]
        embedded = embedded + self.role_embed(role_ids)[None, :, None, :]
        embedded = embedded + self.pos_embed.view(1, frame_count, self.num_patches, self.hidden_size)

        time_embed = self.time_mlp(timestep_embedding(timesteps, self.hidden_size))
        embedded = embedded + time_embed[:, None, None, :]
        tokens = embedded.reshape(batch_size, frame_count * self.num_patches, self.hidden_size)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        tokens = self.output_proj(tokens)
        tokens = tokens.reshape(
            batch_size,
            frame_count,
            self.grid_size,
            self.grid_size,
            self.latent_channels,
            self.patch_size,
            self.patch_size,
        )
        tokens = tokens.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        frames = tokens.reshape(
            batch_size,
            frame_count,
            self.latent_channels,
            self.latent_size,
            self.latent_size,
        )
        return frames[:, 1:-1]

