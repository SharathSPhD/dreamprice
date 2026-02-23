"""Metrics logging: W&B and null logger for tests."""

from __future__ import annotations

from typing import Any, Protocol


class MetricsLogger(Protocol):
    """Protocol for metrics loggers."""

    def log(self, step: int, metrics: dict[str, float]) -> None: ...
    def log_image(self, step: int, key: str, image: Any) -> None: ...


class WandbLogger:
    """W&B experiment logger."""

    def __init__(self, project: str = "dreamprice", **kwargs: Any) -> None:
        import wandb

        self._run = wandb.init(project=project, **kwargs)

    def log(self, step: int, metrics: dict[str, float]) -> None:
        import wandb

        wandb.log(metrics, step=step)

    def log_image(self, step: int, key: str, image: Any) -> None:
        import wandb

        wandb.log({key: wandb.Image(image)}, step=step)


class NullLogger:
    """Discards all metrics. Used in tests."""

    def log(self, step: int, metrics: dict[str, float]) -> None:
        pass

    def log_image(self, step: int, key: str, image: Any) -> None:
        pass
