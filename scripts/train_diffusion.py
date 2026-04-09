from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tflamediff.config import load_config
from tflamediff.engine.train_diffusion import train
from tflamediff.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the combustion conditional latent diffusion model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=None,
        help="Optional config overrides, e.g. model.autoencoder.checkpoint=.../best.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, overrides=args.override)
    seed_everything(int(config["experiment"].get("seed", 42)))
    train(config)


if __name__ == "__main__":
    main()

