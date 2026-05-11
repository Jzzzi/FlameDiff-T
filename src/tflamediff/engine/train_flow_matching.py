from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from tflamediff.config import ensure_output_structure
from tflamediff.data import build_combustion_datasets, create_dataloader
from tflamediff.engine.checkpoint import load_checkpoint, save_checkpoint, unwrap_model
from tflamediff.engine.distributed import (
    DistributedContext,
    barrier,
    cleanup_distributed,
    init_distributed,
    is_main_process,
    reduce_scalar,
)
from tflamediff.engine.interpolation import (
    build_autoencoder,
    build_flow_model,
    build_flow,
    decode_sequence,
    encode_sequence,
    load_autoencoder_checkpoint,
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
from tflamediff.engine.logger import WandbLogger
from tflamediff.utils.metrics import compute_sequence_metrics
from tflamediff.utils.tensor import tensor_to_numpy
from tflamediff.utils.visualization import save_comparison_strip, save_sequence_strip


def _autocast_context(device: str, enabled: bool, dtype: torch.dtype | None):
    if enabled and device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=dtype or torch.float16)
    return nullcontext()


def _compute_loss(
    flow_model: torch.nn.Module,
    autoencoder: torch.nn.Module,
    flow,
    batch: dict[str, Any],
    device: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
    condition = batch["condition"]
    target = batch["target"]
    with torch.no_grad():
        condition_latents = encode_sequence(autoencoder, condition)
        target_latents = encode_sequence(autoencoder, target)

    noise = torch.randn_like(target_latents)
    times = flow.sample_train_t(target_latents.shape[0], device=device)
    noisy_targets = flow.interpolate(noise, target_latents, times)
    target_velocity = flow.target_velocity(noise, target_latents)
    timesteps = flow.model_time(times)
    with _autocast_context(device, amp_enabled, amp_dtype):
        pred_velocity = flow_model(
            noisy_targets=noisy_targets,
            condition_latents=condition_latents,
            timesteps=timesteps,
        )
        loss = F.mse_loss(pred_velocity, target_velocity)
    metrics = {"loss": float(loss.detach().item())}
    extras = {
        "condition": condition,
        "target": target,
    }
    return loss, metrics, extras


def _run_validation(
    flow_model,
    autoencoder,
    flow,
    loader,
    device: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    context: DistributedContext,
    max_batches: int | None = None,
) -> tuple[dict[str, float], dict[str, Any] | None]:
    flow_model.eval()
    total_loss = 0.0
    total_batches = 0
    preview = None
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            loss, metrics, extras = _compute_loss(
                flow_model=flow_model,
                autoencoder=autoencoder,
                flow=flow,
                batch=batch,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            total_loss += metrics["loss"]
            total_batches += 1
            if preview is None:
                preview = {
                    "condition": tensor_to_numpy(extras["condition"][:1])[0],
                    "target": tensor_to_numpy(extras["target"][:1])[0],
                }
            if max_batches is not None and total_batches >= max_batches:
                break
    metrics = {"loss": reduce_scalar(total_loss / max(total_batches, 1), context)}
    flow_model.train()
    return metrics, preview


@torch.no_grad()
def _sample_validation_preview(
    *,
    flow_model,
    autoencoder,
    flow,
    preview: dict[str, Any] | None,
    device: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
) -> dict[str, Any] | None:
    if preview is None:
        return None

    was_training = flow_model.training
    flow_model.eval()
    autoencoder.eval()

    condition = torch.from_numpy(preview["condition"][None]).to(device=device, dtype=torch.float32)
    with _autocast_context(device, amp_enabled, amp_dtype):
        condition_latents = encode_sequence(autoencoder, condition)
        target_shape = (
            condition.shape[0],
            int(preview["target"].shape[0]),
            condition_latents.shape[2],
            condition_latents.shape[3],
            condition_latents.shape[4],
        )
        predicted_target_latents = flow.sample(
            model=unwrap_model(flow_model),
            condition_latents=condition_latents,
            target_shape=target_shape,
            device=device,
        )
        prediction = decode_sequence(autoencoder, predicted_target_latents)

    if was_training:
        flow_model.train()
    return {
        "condition": preview["condition"],
        "target": preview["target"],
        "prediction": tensor_to_numpy(prediction)[0],
    }


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
    output_paths,
    tag: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    if preview is None:
        return {}, {}
    condition = normalizer.denormalize(preview["condition"])
    prediction = normalizer.denormalize(preview["prediction"])
    target = normalizer.denormalize(preview["target"])
    shared_vmin = float(min(condition.min(), prediction.min(), target.min()))
    shared_vmax = float(max(condition.max(), prediction.max(), target.max()))
    full_target = np.concatenate([condition[:1], target, condition[1:]], axis=0)
    full_prediction = np.concatenate([condition[:1], prediction, condition[1:]], axis=0)
    gt_image = save_sequence_strip(
        full_target,
        output_paths["visuals"] / f"{tag}_gt.png",
        title=f"Flow Matching Ground Truth ({tag})",
        vmin=shared_vmin,
        vmax=shared_vmax,
    )
    sample_image = save_sequence_strip(
        full_prediction,
        output_paths["visuals"] / f"{tag}_sample.png",
        title=f"Flow Matching Sample ({tag})",
        vmin=shared_vmin,
        vmax=shared_vmax,
    )
    comparison_image = save_comparison_strip(
        condition,
        prediction,
        target,
        output_paths["visuals"] / f"{tag}_preview.png",
        vmin=shared_vmin,
        vmax=shared_vmax,
    )
    sample_metrics = compute_sequence_metrics(
        prediction=prediction,
        target=target,
        data_range=normalizer.data_range,
    )
    return {"gt": gt_image, "sample": sample_image, "comparison": comparison_image}, sample_metrics


def _run_and_log_validation(
    *,
    flow_model,
    autoencoder,
    flow,
    loader,
    device: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    context: DistributedContext,
    logger: JsonlLogger,
    writer,
    wandb_logger,
    optimizer,
    scheduler,
    scaler,
    output_paths,
    config: dict[str, Any],
    normalizer,
    epoch: int,
    global_step: int,
    best_val: float,
    trigger: str,
    max_batches: int | None,
    save_visuals: bool,
    sample_val_visuals: bool,
) -> float:
    val_metrics, preview = _run_validation(
        flow_model=flow_model,
        autoencoder=autoencoder,
        flow=flow,
        loader=loader,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        context=context,
        max_batches=max_batches,
    )
    barrier(context)
    visual_images: dict[str, Any] = {}
    sample_metrics: dict[str, float] = {}
    if is_main_process(context) and save_visuals and sample_val_visuals:
        tag = f"flow_{trigger}_epoch_{epoch:04d}_step_{global_step:08d}"
        sampled_preview = _sample_validation_preview(
            flow_model=flow_model,
            autoencoder=autoencoder,
            flow=flow,
            preview=preview,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        visual_images, sample_metrics = _save_validation_visuals(
            preview=sampled_preview,
            normalizer=normalizer,
            output_paths=output_paths,
            tag=tag,
        )

    if not is_main_process(context):
        barrier(context)
        return best_val

    payload = {
        "phase": "val",
        "trigger": trigger,
        "epoch": epoch,
        "step": global_step,
        **val_metrics,
        **{f"sample_{key}": value for key, value in sample_metrics.items()},
        "lr": current_learning_rate(optimizer),
    }
    logger.log(payload)
    if writer is not None:
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, global_step)
        for key, value in sample_metrics.items():
            writer.add_scalar(f"val_sample/{key}", value, global_step)
    if wandb_logger is not None:
        wandb_payload = {"val/loss": val_metrics["loss"], "epoch": epoch}
        wandb_payload.update({f"val_sample/{key}": value for key, value in sample_metrics.items()})
        wandb_logger.log(wandb_payload, step=global_step)

    print(
        f"[flow][{trigger}][epoch={epoch}][step={global_step}] "
        f"{format_metrics({**val_metrics, **{f'sample_{key}': value for key, value in sample_metrics.items()}})}"
    )
    is_best = val_metrics["loss"] < best_val
    if is_best:
        best_val = val_metrics["loss"]
    save_checkpoint(
        output_paths["checkpoints"] / "last.pt",
        model=flow_model,
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
            model=flow_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            step=global_step,
            config=config,
            extra={"best_val": best_val},
        )
    if wandb_logger is not None and visual_images:
        wandb_logger.log(
            {
                "val_visual/gt": wandb_logger.image(
                    visual_images["gt"],
                    caption=f"Flow Matching GT | trigger={trigger} | epoch={epoch} | step={global_step}",
                ),
                "val_visual/sample": wandb_logger.image(
                    visual_images["sample"],
                    caption=f"Flow Matching Sample | trigger={trigger} | epoch={epoch} | step={global_step}",
                ),
                "val_visual/comparison": wandb_logger.image(
                    visual_images["comparison"],
                    caption=f"Flow Matching Comparison | trigger={trigger} | epoch={epoch} | step={global_step}",
                ),
            },
            step=global_step,
        )
    barrier(context)
    return best_val


def _train_impl(config: dict[str, Any], context: DistributedContext) -> None:
    output_paths = ensure_output_structure(config)
    datasets_bundle = build_combustion_datasets(config["data"])
    store = datasets_bundle["store"]
    normalizer = datasets_bundle["normalizer"]
    train_dataset = datasets_bundle["datasets"]["train"]
    val_dataset = datasets_bundle["datasets"]["val"]

    batch_size = int(config["data"].get("batch_size", 4))
    num_workers = int(config["data"].get("num_workers", 0))
    prefetch_factor = config["data"].get("prefetch_factor")
    prefetch_factor = None if prefetch_factor in {None, 0} else int(prefetch_factor)
    train_loader, train_sampler = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        distributed=context.enabled,
        drop_last=True,
        prefetch_factor=prefetch_factor,
    )
    val_loader, _ = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        distributed=context.enabled,
        drop_last=False,
        prefetch_factor=prefetch_factor,
    )

    device = context.device
    autoencoder = build_autoencoder(config).to(device)
    load_autoencoder_checkpoint(config, autoencoder, device=device)
    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad = False

    latent_size = int(store.trajectories[0].shape_h) // autoencoder.downsample_factor
    flow_model = build_flow_model(config, latent_size=latent_size).to(device)
    flow = build_flow(config).to(device)
    if context.enabled:
        flow_model = DistributedDataParallel(
            flow_model,
            device_ids=[context.local_rank] if device.startswith("cuda") else None,
        )

    optimizer = build_optimizer(flow_model.parameters(), config["optimizer"])
    scheduler = build_scheduler(optimizer, config.get("scheduler", {"name": "none"}), int(config["trainer"]["epochs"]))
    amp_enabled, amp_dtype = resolve_precision(config["trainer"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.startswith("cuda"))

    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    resume_path = config["trainer"].get("resume")
    if resume_path:
        checkpoint = load_checkpoint(
            resume_path,
            model=flow_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("step", 0))
        best_val = float(checkpoint.get("extra", {}).get("best_val", best_val))

    logger = JsonlLogger(output_paths["logs"] / "flow_matching_train.jsonl")
    writer = maybe_build_summary_writer(output_paths["logs"]) if is_main_process(context) else None
    wandb_logger = WandbLogger(config, output_paths["root"]) if is_main_process(context) else None
    log_every = int(config["trainer"].get("log_every_steps", 50))
    grad_clip = float(config["trainer"].get("grad_clip_norm", 0.0))
    val_every_steps = int(config["trainer"].get("val_every_steps", 0) or 0)
    interval_val_batches = config["trainer"].get("interval_val_batches")
    interval_val_batches = None if interval_val_batches in {None, 0} else int(interval_val_batches)
    full_val_at_epoch_end = bool(config["trainer"].get("full_val_at_epoch_end", True))
    save_val_visuals = bool(config["trainer"].get("save_val_visuals", True))
    sample_val_visuals = bool(config["trainer"].get("sample_val_visuals", save_val_visuals))

    for epoch in range(start_epoch, int(config["trainer"]["epochs"])):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        flow_model.train()
        progress = tqdm(train_loader, disable=not is_main_process(context), desc=f"Flow epoch {epoch}")
        last_interval_validation_step = -1
        for batch in progress:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics, _ = _compute_loss(
                flow_model=flow_model,
                autoencoder=autoencoder,
                flow=flow,
                batch=batch,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(flow_model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            if is_main_process(context) and global_step % log_every == 0:
                payload = {
                    "phase": "train",
                    "epoch": epoch,
                    "step": global_step,
                    "loss": metrics["loss"],
                    "lr": current_learning_rate(optimizer),
                }
                logger.log(payload)
                if wandb_logger is not None:
                    wandb_logger.log(
                        {
                            "train/loss": metrics["loss"],
                            "train/lr": current_learning_rate(optimizer),
                            "epoch": epoch,
                        },
                        step=global_step,
                    )
                if writer is not None:
                    writer.add_scalar("train/loss", metrics["loss"], global_step)
                progress.set_postfix(loss=f"{metrics['loss']:.6f}")

            if val_every_steps > 0 and global_step % val_every_steps == 0:
                best_val = _run_and_log_validation(
                    flow_model=flow_model,
                    autoencoder=autoencoder,
                    flow=flow,
                    loader=val_loader,
                    device=device,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
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
                    sample_val_visuals=sample_val_visuals,
                )
                last_interval_validation_step = global_step
                flow_model.train()

        if scheduler is not None:
            scheduler.step()
        if full_val_at_epoch_end and not (
            last_interval_validation_step == global_step and interval_val_batches is None
        ):
            best_val = _run_and_log_validation(
                flow_model=flow_model,
                autoencoder=autoencoder,
                flow=flow,
                loader=val_loader,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
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
                sample_val_visuals=sample_val_visuals,
            )
    if writer is not None:
        writer.close()
    if wandb_logger is not None:
        wandb_logger.finish()
