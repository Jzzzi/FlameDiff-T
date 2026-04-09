from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tflamediff.config import load_config
from tflamediff.data import build_combustion_datasets, create_dataloader
from tflamediff.engine.checkpoint import load_model_weights
from tflamediff.engine.interpolation import (
    build_autoencoder,
    build_diffusion_model,
    build_diffusion_scheduler,
    sample_sequence,
)
from tflamediff.utils.io import write_csv
from tflamediff.utils.metrics import average_metric_dicts, compute_sequence_metrics
from tflamediff.utils.seed import seed_everything
from tflamediff.utils.visualization import save_comparison_strip, save_gif


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate temporal interpolation metrics on a dataset split.")
    parser.add_argument("--config", type=str, required=True, help="Path to diffusion YAML config.")
    parser.add_argument("--diffusion-checkpoint", type=str, required=True, help="Path to diffusion checkpoint.")
    parser.add_argument("--autoencoder-checkpoint", type=str, default=None, help="Override autoencoder checkpoint path.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional evaluation cap.")
    parser.add_argument("--save-cases", type=int, default=5, help="Number of qualitative cases to save.")
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    parser.add_argument("--seed", type=int, default=42)
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
    qualitative_dir = output_dir / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_combustion_datasets(config["data"])
    normalizer = bundle["normalizer"]
    store = bundle["store"]
    dataset = bundle["datasets"][args.split]
    loader, _ = create_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=int(config["data"].get("num_workers", 0)),
        shuffle=False,
        distributed=False,
        drop_last=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_size = int(store.trajectories[0].shape_h) // (
        2 ** (len(config["model"]["autoencoder"].get("channel_multipliers", [1, 2, 4])) - 1)
    )
    autoencoder = build_autoencoder(config).to(device)
    load_model_weights(config["model"]["autoencoder"]["checkpoint"], autoencoder, map_location=device)
    diffusion_model = build_diffusion_model(config, latent_size=latent_size).to(device)
    load_model_weights(args.diffusion_checkpoint, diffusion_model, map_location=device)
    diffusion = build_diffusion_scheduler(config).to(device)

    metrics_rows: list[dict[str, object]] = []
    metric_values: list[dict[str, float]] = []
    qualitative_saved = 0

    for batch_index, batch in enumerate(tqdm(loader, desc=f"eval {args.split}")):
        if args.max_samples is not None and len(metrics_rows) >= args.max_samples:
            break
        condition = batch["condition"].to(device)
        full_prediction = sample_sequence(
            autoencoder=autoencoder,
            diffusion_model=diffusion_model,
            diffusion=diffusion,
            condition_frames=condition,
            device=device,
        ).detach().cpu().numpy()
        pred_target = full_prediction[:, 1:-1]
        target = batch["target"].numpy()
        condition_np = batch["condition"].numpy()

        for sample_offset in range(pred_target.shape[0]):
            if args.max_samples is not None and len(metrics_rows) >= args.max_samples:
                break
            denorm_pred = normalizer.denormalize(pred_target[sample_offset])
            denorm_tgt = normalizer.denormalize(target[sample_offset])
            denorm_condition = normalizer.denormalize(condition_np[sample_offset])
            metrics = compute_sequence_metrics(
                prediction=denorm_pred,
                target=denorm_tgt,
                data_range=normalizer.data_range,
            )
            row = {
                "trajectory_id": batch["trajectory_id"][sample_offset],
                "window_start": int(batch["window_start"][sample_offset]),
                **metrics,
            }
            metrics_rows.append(row)
            metric_values.append(metrics)

            if qualitative_saved < args.save_cases:
                case_dir = qualitative_dir / f"case_{qualitative_saved:03d}"
                case_dir.mkdir(parents=True, exist_ok=True)
                save_comparison_strip(
                    condition=denorm_condition,
                    prediction=denorm_pred,
                    target=denorm_tgt,
                    path=case_dir / "comparison.png",
                )
                full_sequence = np.concatenate(
                    [denorm_condition[:1], denorm_pred, denorm_condition[1:]], axis=0
                )
                save_gif(full_sequence, case_dir / "prediction.gif")
                with (case_dir / "metrics.json").open("w", encoding="utf-8") as handle:
                    json.dump(row, handle, indent=2, ensure_ascii=False)
                qualitative_saved += 1

    summary = average_metric_dicts(metric_values)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    write_csv(output_dir / "per_sample_metrics.csv", metrics_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
