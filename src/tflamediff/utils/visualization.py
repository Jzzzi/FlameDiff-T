from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .io import ensure_uint8


def _figure_to_rgb_array(figure) -> np.ndarray:
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    buffer = np.frombuffer(figure.canvas.buffer_rgba(), dtype=np.uint8)
    return buffer.reshape(height, width, 4)[..., :3].copy()


def render_sequence_strip(
    sequence: np.ndarray,
    title: str | None = None,
    cmap: str = "inferno",
) -> np.ndarray:
    sequence = np.asarray(sequence)
    if sequence.ndim == 4 and sequence.shape[1] == 1:
        sequence = sequence[:, 0]
    frames = sequence.shape[0]
    figure, axes = plt.subplots(1, frames, figsize=(2 * frames, 2.5))
    if frames == 1:
        axes = [axes]
    for axis, frame_index in zip(axes, range(frames)):
        axis.imshow(sequence[frame_index], cmap=cmap)
        axis.set_title(f"t={frame_index}")
        axis.axis("off")
    if title:
        figure.suptitle(title)
    figure.tight_layout()
    image = _figure_to_rgb_array(figure)
    plt.close(figure)
    return image


def save_sequence_strip(
    sequence: np.ndarray,
    path: str | Path,
    title: str | None = None,
    cmap: str = "inferno",
) -> np.ndarray:
    image = render_sequence_strip(sequence=sequence, title=title, cmap=cmap)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(target)
    return image


def render_comparison_strip(
    condition: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    cmap: str = "inferno",
) -> np.ndarray:
    condition = np.asarray(condition)
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if condition.ndim == 4 and condition.shape[1] == 1:
        condition = condition[:, 0]
    if prediction.ndim == 4 and prediction.shape[1] == 1:
        prediction = prediction[:, 0]
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]

    full_prediction = np.concatenate([condition[:1], prediction, condition[1:]], axis=0)
    full_target = np.concatenate([condition[:1], target, condition[1:]], axis=0)
    diff = np.abs(full_prediction - full_target)

    num_frames = full_prediction.shape[0]
    figure, axes = plt.subplots(3, num_frames, figsize=(1.8 * num_frames, 6))
    rows = [full_target, full_prediction, diff]
    row_titles = ["Ground Truth", "Prediction", "Absolute Error"]
    for row_index in range(3):
        for frame_index in range(num_frames):
            axis = axes[row_index, frame_index]
            axis.imshow(rows[row_index][frame_index], cmap=cmap)
            if row_index == 0:
                axis.set_title(f"t={frame_index}")
            if frame_index == 0:
                axis.set_ylabel(row_titles[row_index])
            axis.axis("off")
    figure.tight_layout()
    image = _figure_to_rgb_array(figure)
    plt.close(figure)
    return image


def save_comparison_strip(
    condition: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    path: str | Path,
    cmap: str = "inferno",
) -> np.ndarray:
    image = render_comparison_strip(
        condition=condition,
        prediction=prediction,
        target=target,
        cmap=cmap,
    )
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(target_path)
    return image


def save_gif(sequence: np.ndarray, path: str | Path, fps: int = 8) -> None:
    sequence = np.asarray(sequence)
    if sequence.ndim == 4 and sequence.shape[1] == 1:
        sequence = sequence[:, 0]
    frames = [ensure_uint8(frame) for frame in sequence]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(target, frames, fps=fps)


def save_video(sequence: np.ndarray, path: str | Path, fps: int = 8) -> None:
    sequence = np.asarray(sequence)
    if sequence.ndim == 4 and sequence.shape[1] == 1:
        sequence = sequence[:, 0]
    frames = [ensure_uint8(frame) for frame in sequence]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(target, fps=fps)
    try:
        for frame in frames:
            writer.append_data(np.stack([frame] * 3, axis=-1))
    finally:
        writer.close()
