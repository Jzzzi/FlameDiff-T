from __future__ import annotations

from typing import Any

import torch

from tflamediff.engine.checkpoint import load_model_weights
from tflamediff.models import ConditionalLatentDiT, FrameAutoencoder, RectifiedFlow


def encode_sequence(autoencoder: FrameAutoencoder, frames: torch.Tensor) -> torch.Tensor:
    batch_size, num_frames, channels, height, width = frames.shape
    latents = autoencoder.encode(frames.reshape(batch_size * num_frames, channels, height, width))
    latent_channels, latent_height, latent_width = latents.shape[1:]
    return latents.reshape(batch_size, num_frames, latent_channels, latent_height, latent_width)


def decode_sequence(autoencoder: FrameAutoencoder, latents: torch.Tensor) -> torch.Tensor:
    batch_size, num_frames, channels, height, width = latents.shape
    recon = autoencoder.decode(latents.reshape(batch_size * num_frames, channels, height, width))
    out_channels, out_height, out_width = recon.shape[1:]
    return recon.reshape(batch_size, num_frames, out_channels, out_height, out_width)


def build_autoencoder(config: dict[str, Any]) -> FrameAutoencoder:
    model_cfg = config["model"]["autoencoder"]
    return FrameAutoencoder(
        in_channels=int(model_cfg.get("in_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 64)),
        latent_channels=int(model_cfg.get("latent_channels", 8)),
        channel_multipliers=tuple(model_cfg.get("channel_multipliers", [1, 2, 4])),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )


def build_flow_model(config: dict[str, Any], latent_size: int) -> ConditionalLatentDiT:
    dit_cfg = config["model"]["dit"]
    ae_cfg = config["model"]["autoencoder"]
    return ConditionalLatentDiT(
        latent_channels=int(ae_cfg.get("latent_channels", 8)),
        latent_size=latent_size,
        patch_size=int(dit_cfg.get("patch_size", 2)),
        hidden_size=int(dit_cfg.get("hidden_size", 512)),
        depth=int(dit_cfg.get("depth", 8)),
        num_heads=int(dit_cfg.get("num_heads", 8)),
        mlp_ratio=float(dit_cfg.get("mlp_ratio", 4.0)),
        dropout=float(dit_cfg.get("dropout", 0.0)),
        num_frames=int(dit_cfg.get("num_frames", 10)),
        target_frames=int(dit_cfg.get("target_frames", 8)),
    )


def build_flow(config: dict[str, Any]) -> RectifiedFlow:
    flow_cfg = config["flow_matching"]
    return RectifiedFlow(
        sample_steps=int(flow_cfg.get("sample_steps", 100)),
        train_eps=float(flow_cfg.get("train_eps", 1e-5)),
        time_scale=float(flow_cfg.get("time_scale", 1000.0)),
    )


def load_autoencoder_checkpoint(
    config: dict[str, Any], model: FrameAutoencoder, device: torch.device | str
) -> None:
    checkpoint_path = config["model"]["autoencoder"].get("checkpoint")
    if not checkpoint_path:
        raise ValueError(
            "model.autoencoder.checkpoint must point to a trained autoencoder checkpoint "
            "before flow matching training, inference, or evaluation."
        )
    load_model_weights(checkpoint_path, model, map_location=device)


@torch.no_grad()
def sample_sequence(
    autoencoder: FrameAutoencoder,
    flow_model: ConditionalLatentDiT,
    flow: RectifiedFlow,
    condition_frames: torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    autoencoder.eval()
    flow_model.eval()
    condition_frames = condition_frames.to(device)
    condition_latents = encode_sequence(autoencoder, condition_frames)
    target_shape = (
        condition_frames.shape[0],
        8,
        condition_latents.shape[2],
        condition_latents.shape[3],
        condition_latents.shape[4],
    )
    predicted_target_latents = flow.sample(
        model=flow_model,
        condition_latents=condition_latents,
        target_shape=target_shape,
        device=device,
    )
    predicted_frames = decode_sequence(autoencoder, predicted_target_latents)
    return torch.cat([condition_frames[:, :1], predicted_frames, condition_frames[:, 1:2]], dim=1)
