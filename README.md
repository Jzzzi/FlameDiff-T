# Combustion Flame Temporal Super-Resolution

This repository provides a complete training, inference, and evaluation framework for combustion flame-field temporal super-resolution on RealPDEBench `combustion/hf_dataset/real`.

The task setup is fixed to 10-frame windows:

- input conditions: frame `0` and frame `9`
- targets to reconstruct: frames `1..8`
- method: frame autoencoder + conditional latent DiT/diffusion

The framework is prepared for later training, but no training jobs are launched by default.

## Quick Start

1. Check that the dataset root exists:

```bash
python scripts/inspect_combustion_data.py --config configs/autoencoder_base.yaml
```

2. Train the autoencoder later when ready:

```bash
torchrun --standalone --nproc_per_node=1 scripts/train_autoencoder.py --config configs/autoencoder_base.yaml
```

3. Train the diffusion model after filling in the autoencoder checkpoint:

```bash
torchrun --standalone --nproc_per_node=1 scripts/train_diffusion.py --config configs/diffusion_base.yaml
```

4. Run interpolation inference:

```bash
python scripts/infer_interpolation.py \
  --config configs/diffusion_base.yaml \
  --diffusion-checkpoint outputs/diffusion/checkpoints/best.pt \
  --sample-index 0
```

5. Run evaluation:

```bash
python scripts/evaluate.py \
  --config configs/diffusion_base.yaml \
  --diffusion-checkpoint outputs/diffusion/checkpoints/best.pt
```

## Cluster Entrypoints

Two cluster-style launchers are included and follow the same environment-variable pattern as the referenced GLD script:

- `entrypoints/entry_train_autoencoder.sh`
- `entrypoints/entry_train_diffusion.sh`

Supported environment variables include `EXP_NAME`, `CONFIG_PATH`, `RESULTS_DIR`, `LOG_ROOT`, `PRECISION`, `RESUME`, `CKPT_PATH`, `WANDB_ENABLED`, `PROJECT`, `ENTITY`, and distributed launch variables such as `NPROC_PER_NODE`, `NNODES`, `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT`.

Both entrypoints launch training with `torchrun`, and both also accept arbitrary CLI overrides as trailing `key=value` arguments.

## Documentation

See [docs/usage.md](docs/usage.md) for the full workflow, configuration guide, DDP launch examples, and output structure.
