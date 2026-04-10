from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tflamediff.config import load_config
from tflamediff.data import build_combustion_datasets
from tflamediff.engine.checkpoint import load_model_weights
from tflamediff.engine.interpolation import (
    build_autoencoder,
    build_diffusion_model,
    build_diffusion_scheduler,
    sample_sequence,
)
from tflamediff.utils.io import load_frame_file
from tflamediff.utils.seed import seed_everything
from tflamediff.utils.tensor import tensor_to_numpy
from tflamediff.utils.visualization import save_comparison_strip, save_gif, save_sequence_strip, save_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temporal interpolation from frame 0 and frame 9.")
    parser.add_argument("--config", type=str, required=True, help="Path to diffusion YAML config.")
    parser.add_argument("--diffusion-checkpoint", type=str, required=True, help="Path to diffusion checkpoint.")
    parser.add_argument("--autoencoder-checkpoint", type=str, default=None, help="Override autoencoder checkpoint path.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index for inference.")
    parser.add_argument("--frame0", type=str, default=None, help="Optional external frame 0 (.npy or image).")
    parser.add_argument("--frame9", type=str, default=None, help="Optional external frame 9 (.npy or image).")
    parser.add_argument("--output-dir", type=str, default="outputs/inference", help="Directory for saved outputs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-video", action="store_true", help="Skip MP4 writing.")
    parser.add_argument("--override", type=str, nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    config = load_config(args.config, overrides=args.override)
    if args.autoencoder_checkpoint:
        config["model"]["autoencoder"]["checkpoint"] = args.autoencoder_checkpoint

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bundle = build_combustion_datasets(config["data"])
    normalizer = bundle["normalizer"]
    store = bundle["store"]
    latent_size = int(store.trajectories[0].shape_h) // (
        2 ** (len(config["model"]["autoencoder"].get("channel_multipliers", [1, 2, 4])) - 1)
    )

    autoencoder = build_autoencoder(config).to(device)
    load_model_weights(config["model"]["autoencoder"]["checkpoint"], autoencoder, map_location=device)
    diffusion_model = build_diffusion_model(config, latent_size=latent_size).to(device)
    load_model_weights(args.diffusion_checkpoint, diffusion_model, map_location=device)
    diffusion = build_diffusion_scheduler(config).to(device)

    metadata = {"source": "external" if args.frame0 and args.frame9 else "dataset"}
    target = None
    if args.frame0 and args.frame9:
        frame0 = normalizer.normalize(load_frame_file(args.frame0))
        frame9 = normalizer.normalize(load_frame_file(args.frame9))
        condition = torch.from_numpy(np.stack([frame0, frame9], axis=0)).unsqueeze(0).float()
    else:
        sample = bundle["datasets"][args.split][args.sample_index]
        condition = sample["condition"].unsqueeze(0)
        target = tensor_to_numpy(sample["target"])
        metadata.update(
            {
                "trajectory_id": sample["trajectory_id"],
                "window_start": sample["window_start"],
                "split": args.split,
                "sample_index": args.sample_index,
            }
        )

    with torch.no_grad():
        full_prediction = sample_sequence(
            autoencoder=autoencoder,
            diffusion_model=diffusion_model,
            diffusion=diffusion,
            condition_frames=condition,
            device=device,
        )[0]
        full_prediction = tensor_to_numpy(full_prediction)

    denorm_prediction = normalizer.denormalize(full_prediction)
    np.save(output_dir / "prediction.npy", denorm_prediction)
    save_sequence_strip(denorm_prediction, output_dir / "prediction_strip.png", title="Predicted 10-frame sequence")
    save_gif(denorm_prediction, output_dir / "prediction.gif")
    if not args.skip_video:
        try:
            save_video(denorm_prediction, output_dir / "prediction.mp4")
        except Exception as exc:
            metadata["video_error"] = repr(exc)

    if target is not None:
        denorm_target = normalizer.denormalize(target)
        denorm_condition = normalizer.denormalize(tensor_to_numpy(condition[0]))
        save_comparison_strip(
            condition=denorm_condition,
            prediction=denorm_prediction[1:-1],
            target=denorm_target,
            path=output_dir / "comparison.png",
        )
        np.save(output_dir / "target.npy", denorm_target)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
