from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ensure_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.dtype == np.uint8:
        return frame
    frame = np.clip(frame, 0.0, 1.0)
    return (frame * 255.0).round().astype(np.uint8)


def load_frame_file(path: str | Path) -> np.ndarray:
    source = Path(path)
    if source.suffix == ".npy":
        array = np.load(source)
    else:
        array = np.asarray(Image.open(source).convert("L"), dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"Expected frame array with shape [1, H, W], got {array.shape}")
    return array.astype(np.float32)


def save_frame_png(path: str | Path, frame: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ensure_uint8(frame)).save(target)
