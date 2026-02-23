"""Abstract base for grocery pricing environments."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np


class BaseGroceryEnv(gym.Env):
    """Abstract base. Subclasses implement _compute_reward() and _get_obs()."""

    metadata: dict[str, Any] = {"render_modes": []}

    @abstractmethod
    def _compute_reward(
        self,
        price: np.ndarray,
        cost: np.ndarray,
        units_sold: np.ndarray,
        prev_price: np.ndarray,
    ) -> float:
        """Compute scalar reward from pricing outcomes."""
        ...

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Return current observation vector."""
        ...

    @abstractmethod
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        ...

    @abstractmethod
    def step(
        self, action: np.ndarray | int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        ...
