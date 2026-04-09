from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def format_metrics(metrics: dict[str, float]) -> str:
    return " | ".join(f"{key}={value:.6f}" for key, value in metrics.items())


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


class WandbLogger:
    def __init__(self, config: dict[str, Any], output_dir: str | Path, enabled: bool | None = None) -> None:
        wandb_cfg = config.get("wandb", {})
        self.enabled = enabled if enabled is not None else bool(
            wandb_cfg.get("enabled", _env_flag("WANDB_ENABLED", default=False))
        )
        self.run = None
        self._module = None
        if not self.enabled:
            return
        try:
            import wandb
        except ImportError:
            print("[warning] wandb is enabled but the package is not installed; skipping wandb logging.")
            self.enabled = False
            return

        self._module = wandb
        project = wandb_cfg.get("project") or os.getenv("PROJECT", "tflamediff")
        entity = wandb_cfg.get("entity") or os.getenv("ENTITY") or None
        run_name = wandb_cfg.get("run_name") or os.getenv("EXP_NAME") or config["experiment"]["name"]
        tags = wandb_cfg.get("tags")
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=tags,
            dir=str(output_dir),
            config=config,
            reinit=True,
        )

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        if self.enabled and self.run is not None:
            self._module.log(payload, step=step)

    def image(self, image, caption: str | None = None):
        if not self.enabled or self.run is None:
            return None
        return self._module.Image(image, caption=caption)

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            self._module.finish()
