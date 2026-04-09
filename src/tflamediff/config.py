from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config is None:
        config = {}
    if overrides:
        apply_overrides(config, overrides)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def clone_config(config: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(config)


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        assign_nested_key(config, key, value)


def assign_nested_key(config: dict[str, Any], key: str, value: Any) -> None:
    cursor = config
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def get_nested(config: dict[str, Any], key: str, default: Any = None) -> Any:
    cursor: Any = config
    for part in key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def ensure_output_structure(config: dict[str, Any]) -> dict[str, Path]:
    output_root = Path(config["experiment"]["output_dir"]).resolve()
    paths = {
        "root": output_root,
        "checkpoints": output_root / "checkpoints",
        "logs": output_root / "logs",
        "visuals": output_root / "visuals",
        "eval": output_root / "eval",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    save_config(config, output_root / "resolved_config.yaml")
    return paths

