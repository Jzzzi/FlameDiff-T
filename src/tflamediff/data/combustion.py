from __future__ import annotations

from fractions import Fraction
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


@dataclass(frozen=True)
class TrajectoryMeta:
    sim_id: str
    file_path: Path
    row_index: int
    shape_t: int
    shape_h: int
    shape_w: int


@dataclass(frozen=True)
class WindowMeta:
    sim_id: str
    start_index: int


class CombustionNormalizer:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.mode = config.get("mode", "minmax")
        self.min_value = config.get("min")
        self.max_value = config.get("max")
        self.mean_value = config.get("mean")
        self.std_value = config.get("std")
        self.eps = float(config.get("eps", 1e-6))

        if self.mode == "minmax":
            self.min_value = 0.0 if self.min_value is None else float(self.min_value)
            self.max_value = 1.0 if self.max_value is None else float(self.max_value)
        elif self.mode == "zscore":
            if self.mean_value is None or self.std_value is None:
                raise ValueError("zscore normalization requires mean and std values.")
            self.mean_value = float(self.mean_value)
            self.std_value = float(self.std_value)
        elif self.mode != "none":
            raise ValueError(f"Unsupported normalization mode: {self.mode}")

    def normalize(self, array: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return array.astype(np.float32)
        if self.mode == "minmax":
            denom = max(self.max_value - self.min_value, self.eps)
            return ((array - self.min_value) / denom).astype(np.float32)
        return ((array - self.mean_value) / max(self.std_value, self.eps)).astype(np.float32)

    def denormalize(self, array: np.ndarray) -> np.ndarray:
        if self.mode == "none":
            return array.astype(np.float32)
        if self.mode == "minmax":
            return (array * (self.max_value - self.min_value) + self.min_value).astype(np.float32)
        return (array * self.std_value + self.mean_value).astype(np.float32)

    @property
    def data_range(self) -> float:
        if self.mode == "minmax":
            return max(float(self.max_value) - float(self.min_value), self.eps)
        if self.mode == "zscore":
            return max(6.0 * self.std_value, self.eps)
        return 1.0

    @classmethod
    def fit(
        cls,
        store: "CombustionTrajectoryStore",
        sim_ids: list[str],
        mode: str = "minmax",
        eps: float = 1e-6,
    ) -> "CombustionNormalizer":
        mins: list[float] = []
        maxs: list[float] = []
        count = 0
        running_sum = 0.0
        running_sq_sum = 0.0
        for sim_id in sim_ids:
            array = store.get_trajectory(sim_id)
            mins.append(float(array.min()))
            maxs.append(float(array.max()))
            running_sum += float(array.sum())
            running_sq_sum += float(np.square(array).sum())
            count += int(array.size)
        if mode == "minmax":
            return cls({"mode": mode, "min": min(mins), "max": max(maxs), "eps": eps})
        mean = running_sum / max(count, 1)
        variance = max(running_sq_sum / max(count, 1) - mean * mean, 0.0)
        std = math.sqrt(variance)
        return cls({"mode": "zscore", "mean": mean, "std": std, "eps": eps})


class CombustionTrajectoryStore:
    def __init__(
        self,
        dataset_root: str | Path,
        subset: str = "real",
        file_cache_size: int = 1,
        trajectory_cache_size: int = 2,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.subset = subset
        self.arrow_root = self.dataset_root / "combustion" / "hf_dataset" / subset
        if not self.arrow_root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.arrow_root}")
        self._pa, self._ipc = self._import_pyarrow()
        self._file_cache_size = max(1, file_cache_size)
        self._trajectory_cache_size = max(1, trajectory_cache_size)
        self._file_table_cache: OrderedDict[str, Any] = OrderedDict()
        self._trajectory_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.trajectories = self._scan_metadata()
        self._trajectory_index = {meta.sim_id: meta for meta in self.trajectories}

    def preload_trajectories(self, sim_ids: list[str]) -> None:
        for sim_id in sim_ids:
            self.get_trajectory(sim_id)

    @staticmethod
    def _import_pyarrow():
        try:
            import pyarrow as pa
            import pyarrow.ipc as ipc
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required to read RealPDEBench Arrow shards. "
                "Install pyarrow in the runtime environment before training or evaluation."
            ) from exc
        return pa, ipc

    def _scan_metadata(self) -> list[TrajectoryMeta]:
        trajectories: list[TrajectoryMeta] = []
        for file_path in sorted(self.arrow_root.glob("data-*.arrow")):
            with self._pa.memory_map(str(file_path), "r") as source:
                table = self._ipc.RecordBatchStreamReader(source).read_all()
            for row_index, row in enumerate(table.to_pylist()):
                trajectories.append(
                    TrajectoryMeta(
                        sim_id=row["sim_id"],
                        file_path=file_path,
                        row_index=row_index,
                        shape_t=int(row["shape_t"]),
                        shape_h=int(row["shape_h"]),
                        shape_w=int(row["shape_w"]),
                    )
                )
        if not trajectories:
            raise RuntimeError(f"No trajectories found under {self.arrow_root}")
        return trajectories

    def list_sim_ids(self) -> list[str]:
        return sorted(self._trajectory_index.keys())

    def get_meta(self, sim_id: str) -> TrajectoryMeta:
        return self._trajectory_index[sim_id]

    def get_trajectory(self, sim_id: str) -> np.ndarray:
        if sim_id in self._trajectory_cache:
            self._trajectory_cache.move_to_end(sim_id)
            return self._trajectory_cache[sim_id]

        meta = self.get_meta(sim_id)
        table = self._get_file_table(meta.file_path)
        row = table.slice(meta.row_index, 1).to_pylist()[0]
        array = (
            np.frombuffer(row["observed"], dtype=np.float32)
            .copy()
            .reshape(meta.shape_t, meta.shape_h, meta.shape_w)
        )
        self._trajectory_cache[sim_id] = array
        self._trajectory_cache.move_to_end(sim_id)
        while len(self._trajectory_cache) > self._trajectory_cache_size:
            self._trajectory_cache.popitem(last=False)
        return array

    def _get_file_table(self, file_path: Path):
        cache_key = str(file_path)
        if cache_key in self._file_table_cache:
            self._file_table_cache.move_to_end(cache_key)
            return self._file_table_cache[cache_key]
        with self._pa.memory_map(str(file_path), "r") as source:
            table = self._ipc.RecordBatchStreamReader(source).read_all()
        self._file_table_cache[cache_key] = table
        self._file_table_cache.move_to_end(cache_key)
        while len(self._file_table_cache) > self._file_cache_size:
            self._file_table_cache.popitem(last=False)
        return table


class CombustionWindowDataset(Dataset):
    def __init__(
        self,
        store: CombustionTrajectoryStore,
        sim_ids: list[str],
        normalizer: CombustionNormalizer,
        window_size: int = 10,
        stride: int = 1,
        condition_indices: tuple[int, int] = (0, 9),
        target_indices: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8),
        windows: list[WindowMeta] | None = None,
    ) -> None:
        self.store = store
        self.sim_ids = sorted({window.sim_id for window in windows}) if windows is not None else sim_ids
        self.normalizer = normalizer
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.condition_indices = condition_indices
        self.target_indices = target_indices
        self.windows = list(windows) if windows is not None else self._build_windows()

    def _build_windows(self) -> list[WindowMeta]:
        windows: list[WindowMeta] = []
        for sim_id in self.sim_ids:
            meta = self.store.get_meta(sim_id)
            max_start = meta.shape_t - self.window_size
            if max_start < 0:
                continue
            for start_index in range(0, max_start + 1, self.stride):
                windows.append(WindowMeta(sim_id=sim_id, start_index=start_index))
        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window_meta = self.windows[index]
        trajectory = self.store.get_trajectory(window_meta.sim_id)
        window = trajectory[window_meta.start_index : window_meta.start_index + self.window_size]
        window = self.normalizer.normalize(window)
        window_tensor = torch.from_numpy(window[:, None, :, :]).float()
        condition = window_tensor[list(self.condition_indices)]
        target = window_tensor[list(self.target_indices)]
        return {
            "sequence": window_tensor,
            "condition": condition,
            "target": target,
            "trajectory_id": window_meta.sim_id,
            "window_start": window_meta.start_index,
        }


def _split_cycle(split_ratios: dict[str, float]) -> list[str]:
    split_names = ("train", "val", "test")
    weights = [Fraction(str(split_ratios[name])).limit_denominator(1000) for name in split_names]
    if any(weight <= 0 for weight in weights):
        raise ValueError(f"Split ratios must be positive, got: {split_ratios}")
    total_weight = sum(weights)
    normalized = [weight / total_weight for weight in weights]
    denominator_lcm = 1
    for weight in normalized:
        denominator_lcm = math.lcm(denominator_lcm, weight.denominator)
    counts = [int(weight * denominator_lcm) for weight in normalized]
    count_gcd = counts[0]
    for count in counts[1:]:
        count_gcd = math.gcd(count_gcd, count)
    counts = [count // count_gcd for count in counts]
    cycle: list[str] = []
    for split_name, count in zip(split_names, counts, strict=True):
        cycle.extend([split_name] * count)
    return cycle


def split_windows_within_trajectories(
    store: CombustionTrajectoryStore,
    sim_ids: list[str],
    split_ratios: dict[str, float],
    window_size: int,
    stride: int,
) -> dict[str, list[WindowMeta]]:
    cycle = _split_cycle(split_ratios)
    split_windows: dict[str, list[WindowMeta]] = {"train": [], "val": [], "test": []}
    for sim_id in sorted(sim_ids):
        meta = store.get_meta(sim_id)
        max_start = meta.shape_t - window_size
        if max_start < 0:
            continue
        for window_index, start_index in enumerate(range(0, max_start + 1, stride)):
            split_name = cycle[window_index % len(cycle)]
            split_windows[split_name].append(WindowMeta(sim_id=sim_id, start_index=start_index))
    return split_windows


def build_combustion_datasets(data_config: dict[str, Any]) -> dict[str, Any]:
    store = CombustionTrajectoryStore(
        dataset_root=data_config["dataset_root"],
        subset=data_config.get("subset", "real"),
        file_cache_size=data_config.get("file_cache_size", 1),
        trajectory_cache_size=data_config.get("trajectory_cache_size", 2),
    )
    sim_ids = store.list_sim_ids()
    window_splits: dict[str, list[WindowMeta]] | None = None
    if bool(data_config.get("split_sets", False)):
        window_splits = split_windows_within_trajectories(
            store=store,
            sim_ids=sim_ids,
            split_ratios=data_config.get("splits", {"train": 0.7, "val": 0.15, "test": 0.15}),
            window_size=int(data_config.get("window_size", 10)),
            stride=int(data_config.get("stride", 1)),
        )
        split_ids = {
            split: sorted({window.sim_id for window in windows})
            for split, windows in window_splits.items()
        }
    else:
        split_ids = {"train": sim_ids, "val": sim_ids, "test": sim_ids}
    preload_splits = set(data_config.get("preload_splits", []))
    preload_limit = data_config.get("preload_max_trajectories")
    preload_limit = None if preload_limit in {None, 0} else int(preload_limit)
    if preload_splits:
        preload_ids: list[str] = []
        for split_name in ("train", "val", "test"):
            if split_name not in preload_splits:
                continue
            preload_ids.extend(split_ids[split_name])
        preload_ids = list(dict.fromkeys(preload_ids))
        if preload_limit is not None:
            preload_ids = preload_ids[:preload_limit]
        store.preload_trajectories(preload_ids)

    norm_config = dict(data_config.get("normalization", {}))
    if norm_config.get("auto_compute", False):
        fitted = CombustionNormalizer.fit(
            store=store,
            sim_ids=split_ids["train"],
            mode=norm_config.get("mode", "minmax"),
            eps=float(norm_config.get("eps", 1e-6)),
        )
        normalizer = fitted
    else:
        normalizer = CombustionNormalizer(norm_config)

    datasets = {}
    for split, split_sim_ids in split_ids.items():
        datasets[split] = CombustionWindowDataset(
            store=store,
            sim_ids=split_sim_ids,
            normalizer=normalizer,
            window_size=int(data_config.get("window_size", 10)),
            stride=int(data_config.get("stride", 1)),
            condition_indices=tuple(data_config.get("condition_indices", [0, 9])),
            target_indices=tuple(data_config.get("target_indices", [1, 2, 3, 4, 5, 6, 7, 8])),
            windows=window_splits[split] if window_splits is not None else None,
        )
    return {
        "store": store,
        "normalizer": normalizer,
        "splits": split_ids,
        "window_splits": window_splits,
        "datasets": datasets,
    }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    distributed: bool,
    drop_last: bool = False,
    prefetch_factor: int | None = None,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle and sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "drop_last": drop_last,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    loader = DataLoader(
        **loader_kwargs,
    )
    return loader, sampler
