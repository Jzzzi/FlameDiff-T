from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from tflamediff.config import ensure_output_structure
from tflamediff.data import build_combustion_datasets, create_dataloader
from tflamediff.engine.checkpoint import load_checkpoint, save_checkpoint
from tflamediff.engine.distributed import (
    DistributedContext,
    barrier,
    cleanup_distributed,
    init_distributed,
    is_main_process,
    reduce_scalar,
)
from tflamediff.engine.logger import JsonlLogger, format_metrics
from tflamediff.engine.train_utils import (
    build_optimizer,
    build_scheduler,
    current_learning_rate,
    maybe_build_summary_writer,
    move_batch_to_device,
    resolve_precision,
)
from tflamediff.models import FrameAutoencoder
from tflamediff.utils.tensor import tensor_to_numpy
from tflamediff.utils.visualization import save_reconstruction_comparison_strip, save_sequence_strip
from tflamediff.engine.logger import WandbLogger


def _autocast_context(device: str, enabled: bool, dtype: torch.dtype | None):
    if enabled and device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=dtype or torch.float16)
    return nullcontext()


def _build_model(config: dict[str, Any]) -> FrameAutoencoder:
    model_cfg = config["model"]["autoencoder"]
    return FrameAutoencoder(
        in_channels=int(model_cfg.get("in_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 64)),
        latent_channels=int(model_cfg.get("latent_channels", 8)),
        channel_multipliers=tuple(model_cfg.get("channel_multipliers", [1, 2, 4])),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )


def _compute_loss(
    model: FrameAutoencoder,
    batch: dict[str, Any],
    device: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    loss_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    sequence = batch["sequence"]
    batch_size, frames, channels, height, width = sequence.shape
    flat = sequence.reshape(batch_size * frames, channels, height, width)
    with _autocast_context(device, amp_enabled, amp_dtype):
        reconstruction, _ = model(flat)
        loss_mse = F.mse_loss(reconstruction, flat)
        loss_l1 = F.l1_loss(reconstruction, flat)
        loss = (
            float(loss_config.get("recon_weight", 1.0)) * loss_mse
            + float(loss_config.get("l1_weight", 0.0)) * loss_l1
        )
    recon_sequence = reconstruction.reshape(batch_size, frames, channels, height, width)
    metrics = {
        "loss": float(loss.detach().item()),
        "mse": float(loss_mse.detach().item()),
        "l1": float(loss_l1.detach().item()),
    }
    return loss, metrics, recon_sequence


def _run_validation(
    model: FrameAutoencoder,
    loader,
    device: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    loss_config: dict[str, Any],
    context: DistributedContext,
    max_batches: int | None = None,
) -> tuple[dict[str, float], dict[str, Any] | None]:
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_l1 = 0.0
    total_batches = 0
    preview = None
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            loss, metrics, recon = _compute_loss(model, batch, device, amp_enabled, amp_dtype, loss_config)
            total_loss += metrics["loss"]
            total_mse += metrics["mse"]
            total_l1 += metrics["l1"]
            total_batches += 1
            if preview is None:
                preview = {
                    "gt": tensor_to_numpy(batch["sequence"][0]),
                    "recon": tensor_to_numpy(recon[0]),
                }
            if max_batches is not None and total_batches >= max_batches:
                break
    denom = max(total_batches, 1)
    metrics = {
        "loss": reduce_scalar(total_loss / denom, context),
        "mse": reduce_scalar(total_mse / denom, context),
        "l1": reduce_scalar(total_l1 / denom, context),
    }
    model.train()
    return metrics, preview


def train(config: dict[str, Any]) -> None:
    context = init_distributed()
    try:
        _train_impl(config=config, context=context)
    finally:
        cleanup_distributed()


def _save_validation_visuals(
    *,
    preview: dict[str, Any] | None,
    normalizer,
    output_paths: dict[str, Path],
    tag: str,
) -> dict[str, Any]:
    if preview is None:
        return {}
    gt = normalizer.denormalize(preview["gt"])
    recon = normalizer.denormalize(preview["recon"])
    shared_vmin = float(min(gt.min(), recon.min()))
    shared_vmax = float(max(gt.max(), recon.max()))
    gt_image = save_sequence_strip(
        gt,
        output_paths["visuals"] / f"{tag}_gt.png",
        title=f"Autoencoder Ground Truth ({tag})",
        vmin=shared_vmin,
        vmax=shared_vmax,
    )
    recon_image = save_sequence_strip(
        recon,
        output_paths["visuals"] / f"{tag}_recon.png",
        title=f"Autoencoder Reconstruction ({tag})",
        vmin=shared_vmin,
        vmax=shared_vmax,
    )
    comparison_image = save_reconstruction_comparison_strip(
        target=gt,
        reconstruction=recon,
        path=output_paths["visuals"] / f"{tag}_comparison.png",
        vmin=shared_vmin,
        vmax=shared_vmax,
    )
    return {"gt": gt_image, "recon": recon_image, "comparison": comparison_image}


def _run_and_log_validation(
    *,
    model,
    loader,
    device: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    loss_config: dict[str, Any],
    context: DistributedContext,
    logger: JsonlLogger,
    writer,
    wandb_logger,
    optimizer,
    scheduler,
    scaler,
    output_paths: dict[str, Path],
    config: dict[str, Any],
    normalizer,
    epoch: int,
    global_step: int,
    best_val: float,
    trigger: str,
    max_batches: int | None,
    save_visuals: bool,
) -> float:
    val_metrics, preview = _run_validation(
        model=model,
        loader=loader,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        loss_config=loss_config,
        context=context,
        max_batches=max_batches,
    )
    barrier(context)
    if not is_main_process(context):
        return best_val

    payload = {
        "phase": "val",
        "trigger": trigger,
        "epoch": epoch,
        "step": global_step,
        **val_metrics,
        "lr": current_learning_rate(optimizer),
    }
    logger.log(payload)
    if writer is not None:
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, global_step)
    if wandb_logger is not None:
        wandb_logger.log(
            {
                "val/loss": val_metrics["loss"],
                "val/mse": val_metrics["mse"],
                "val/l1": val_metrics["l1"],
                "epoch": epoch,
            },
            step=global_step,
        )

    print(f"[autoencoder][{trigger}][epoch={epoch}][step={global_step}] {format_metrics(val_metrics)}")
    is_best = val_metrics["loss"] < best_val
    if is_best:
        best_val = val_metrics["loss"]
    save_checkpoint(
        output_paths["checkpoints"] / "last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        step=global_step,
        config=config,
        extra={"best_val": best_val},
    )
    if is_best:
        save_checkpoint(
            output_paths["checkpoints"] / "best.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            step=global_step,
            config=config,
            extra={"best_val": best_val},
        )
    if save_visuals:
        tag = f"ae_{trigger}_epoch_{epoch:04d}_step_{global_step:08d}"
        visual_images = _save_validation_visuals(
            preview=preview,
            normalizer=normalizer,
            output_paths=output_paths,
            tag=tag,
        )
        if wandb_logger is not None and visual_images:
            wandb_payload = {
                "val_visual/gt": wandb_logger.image(
                    visual_images["gt"],
                    caption=f"AE GT | trigger={trigger} | epoch={epoch} | step={global_step}",
                ),
                "val_visual/recon": wandb_logger.image(
                    visual_images["recon"],
                    caption=f"AE Recon | trigger={trigger} | epoch={epoch} | step={global_step}",
                ),
                "val_visual/comparison": wandb_logger.image(
                    visual_images["comparison"],
                    caption=f"AE Comparison | trigger={trigger} | epoch={epoch} | step={global_step}",
                ),
            }
            wandb_logger.log(wandb_payload, step=global_step)
    return best_val


def _train_impl(config: dict[str, Any], context: DistributedContext) -> None:
    output_paths = ensure_output_structure(config)
    datasets_bundle = build_combustion_datasets(config["data"])
    normalizer = datasets_bundle["normalizer"]
    train_dataset = datasets_bundle["datasets"]["train"]
    val_dataset = datasets_bundle["datasets"]["val"]

    batch_size = int(config["data"].get("batch_size", 8))
    num_workers = int(config["data"].get("num_workers", 0))
    train_loader, train_sampler = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        distributed=context.enabled,
        drop_last=True,
    )
    val_loader, _ = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        distributed=context.enabled,
        drop_last=False,
    )

    device = context.device
    model = _build_model(config).to(device)
    if context.enabled:
        model = DistributedDataParallel(model, device_ids=[context.local_rank] if device.startswith("cuda") else None)

    optimizer = build_optimizer(model.parameters(), config["optimizer"])
    scheduler = build_scheduler(optimizer, config.get("scheduler", {"name": "none"}), int(config["trainer"]["epochs"]))
    amp_enabled, amp_dtype = resolve_precision(config["trainer"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and device.startswith("cuda"))

    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    resume_path = config["trainer"].get("resume")
    if resume_path:
        checkpoint = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("step", 0))
        best_val = float(checkpoint.get("extra", {}).get("best_val", best_val))

    logger = JsonlLogger(output_paths["logs"] / "autoencoder_train.jsonl")
    writer = maybe_build_summary_writer(output_paths["logs"]) if is_main_process(context) else None
    wandb_logger = WandbLogger(config, output_paths["root"]) if is_main_process(context) else None
    log_every = int(config["trainer"].get("log_every_steps", 50))
    grad_clip = float(config["trainer"].get("grad_clip_norm", 0.0))
    val_every_steps = int(config["trainer"].get("val_every_steps", 0) or 0)
    interval_val_batches = config["trainer"].get("interval_val_batches")
    interval_val_batches = None if interval_val_batches in {None, 0} else int(interval_val_batches)
    full_val_at_epoch_end = bool(config["trainer"].get("full_val_at_epoch_end", True))
    save_val_visuals = bool(config["trainer"].get("save_val_visuals", True))

    for epoch in range(start_epoch, int(config["trainer"]["epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        progress = tqdm(train_loader, disable=not is_main_process(context), desc=f"AE epoch {epoch}")
        last_interval_validation_step = -1

        for batch in progress:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics, _ = _compute_loss(model, batch, device, amp_enabled, amp_dtype, config["loss"])
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            if is_main_process(context) and global_step % log_every == 0:
                payload = {
                    "phase": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "loss": metrics["loss"],
                    "mse": metrics["mse"],
                    "l1": metrics["l1"],
                    "lr": current_learning_rate(optimizer),
                }
                logger.log(payload)
                if wandb_logger is not None:
                    wandb_logger.log(
                        {
                            "train/loss": metrics["loss"],
                            "train/mse": metrics["mse"],
                            "train/l1": metrics["l1"],
                            "train/lr": current_learning_rate(optimizer),
                            "epoch": epoch,
                        },
                        step=global_step,
                    )
                if writer is not None:
                    writer.add_scalar("train/loss", metrics["loss"], global_step)
                    writer.add_scalar("train/mse", metrics["mse"], global_step)
                    writer.add_scalar("train/l1", metrics["l1"], global_step)
                progress.set_postfix(loss=f"{metrics['loss']:.6f}")

            if val_every_steps > 0 and global_step % val_every_steps == 0:
                best_val = _run_and_log_validation(
                    model=model,
                    loader=val_loader,
                    device=device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    loss_config=config["loss"],
                    context=context,
                    logger=logger,
                    writer=writer,
                    wandb_logger=wandb_logger,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    output_paths=output_paths,
                    config=config,
                    normalizer=normalizer,
                    epoch=epoch,
                    global_step=global_step,
                    best_val=best_val,
                    trigger="step",
                    max_batches=interval_val_batches,
                    save_visuals=save_val_visuals,
                )
                last_interval_validation_step = global_step
                model.train()

        if scheduler is not None:
            scheduler.step()
        if full_val_at_epoch_end and not (
            last_interval_validation_step == global_step and interval_val_batches is None
        ):
            best_val = _run_and_log_validation(
                model=model,
                loader=val_loader,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                loss_config=config["loss"],
                context=context,
                logger=logger,
                writer=writer,
                wandb_logger=wandb_logger,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                output_paths=output_paths,
                config=config,
                normalizer=normalizer,
                epoch=epoch,
                global_step=global_step,
                best_val=best_val,
                trigger="epoch",
                max_batches=None,
                save_visuals=save_val_visuals,
            )
    if writer is not None:
        writer.close()
    if wandb_logger is not None:
        wandb_logger.finish()
