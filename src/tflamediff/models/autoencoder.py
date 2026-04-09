from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups_in = min(8, in_channels)
        groups_out = min(8, out_channels)
        self.norm1 = nn.GroupNorm(groups_in, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups_out, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.dropout(self.act2(self.norm2(x))))
        return x + residual


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FrameAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        latent_channels: int = 8,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels = [base_channels * multiplier for multiplier in channel_multipliers]
        self.encoder_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        encoder_layers: list[nn.Module] = []
        current_channels = channels[0]
        for stage_channels in channels:
            encoder_layers.append(ResidualBlock(current_channels, stage_channels, dropout=dropout))
            encoder_layers.append(ResidualBlock(stage_channels, stage_channels, dropout=dropout))
            current_channels = stage_channels
            if stage_channels != channels[-1]:
                encoder_layers.append(Downsample(current_channels))
        encoder_layers.extend(
            [
                ResidualBlock(current_channels, current_channels, dropout=dropout),
                nn.GroupNorm(min(8, current_channels), current_channels),
                nn.SiLU(),
                nn.Conv2d(current_channels, latent_channels, kernel_size=3, padding=1),
            ]
        )
        self.encoder = nn.Sequential(*encoder_layers)

        self.decoder_in = nn.Conv2d(latent_channels, current_channels, kernel_size=3, padding=1)
        decoder_layers: list[nn.Module] = []
        reversed_channels = list(reversed(channels))
        for stage_index, stage_channels in enumerate(reversed_channels):
            decoder_layers.append(ResidualBlock(current_channels, stage_channels, dropout=dropout))
            decoder_layers.append(ResidualBlock(stage_channels, stage_channels, dropout=dropout))
            current_channels = stage_channels
            if stage_index != len(reversed_channels) - 1:
                decoder_layers.append(Upsample(current_channels))
        decoder_layers.extend(
            [
                nn.GroupNorm(min(8, current_channels), current_channels),
                nn.SiLU(),
                nn.Conv2d(current_channels, in_channels, kernel_size=3, padding=1),
            ]
        )
        self.decoder = nn.Sequential(*decoder_layers)
        self.downsample_factor = 2 ** (len(channel_multipliers) - 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_in(x)
        return self.encoder(x)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        x = self.decoder_in(latents)
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x)
        recon = self.decode(latents)
        return recon, latents

