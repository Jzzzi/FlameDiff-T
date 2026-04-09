from __future__ import annotations

from typing import Iterable

import numpy as np
from skimage.metrics import structural_similarity


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(prediction - target)))


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - target)))


def psnr(prediction: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    mse_value = mse(prediction, target)
    if mse_value <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse_value))


def ssim_sequence(prediction: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction and target shapes must match, got {prediction.shape} vs {target.shape}")
    scores = []
    for pred_frame, tgt_frame in zip(prediction, target):
        pred_img = pred_frame[0] if pred_frame.ndim == 3 else pred_frame
        tgt_img = tgt_frame[0] if tgt_frame.ndim == 3 else tgt_frame
        scores.append(
            structural_similarity(
                tgt_img,
                pred_img,
                data_range=data_range,
            )
        )
    return float(np.mean(scores))


def compute_sequence_metrics(
    prediction: np.ndarray, target: np.ndarray, data_range: float = 1.0
) -> dict[str, float]:
    return {
        "mse": mse(prediction, target),
        "mae": mae(prediction, target),
        "psnr": psnr(prediction, target, data_range=data_range),
        "ssim": ssim_sequence(prediction, target, data_range=data_range),
    }


def average_metric_dicts(metrics: Iterable[dict[str, float]]) -> dict[str, float]:
    metrics = list(metrics)
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: float(np.mean([item[key] for item in metrics])) for key in keys}

