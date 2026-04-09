from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tflamediff.config import load_config
from tflamediff.data import build_combustion_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect combustion dataset metadata and split sizes.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=None,
        help="Optional config overrides, e.g. data.window_size=10",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, overrides=args.override)
    bundle = build_combustion_datasets(config["data"])
    store = bundle["store"]
    datasets = bundle["datasets"]
    first_meta = store.trajectories[0]
    summary = {
        "dataset_root": str(store.arrow_root),
        "num_trajectories": len(store.trajectories),
        "first_trajectory": {
            "sim_id": first_meta.sim_id,
            "shape_t": first_meta.shape_t,
            "shape_h": first_meta.shape_h,
            "shape_w": first_meta.shape_w,
        },
        "splits": {
            split: {
                "num_trajectories": len(bundle["splits"][split]),
                "num_windows": len(dataset),
                "sample_trajectories": bundle["splits"][split][:5],
            }
            for split, dataset in datasets.items()
        },
        "normalization": {
            "mode": bundle["normalizer"].mode,
            "min": bundle["normalizer"].min_value,
            "max": bundle["normalizer"].max_value,
            "mean": bundle["normalizer"].mean_value,
            "std": bundle["normalizer"].std_value,
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

