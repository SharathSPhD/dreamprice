"""Device utilities: autocast context, FP32 casting for SSM stability."""

from contextlib import contextmanager

import torch


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@contextmanager
def autocast_ctx(device: torch.device | None = None):
    """BF16 autocast for training. Falls back to no-op on CPU."""
    if device is None:
        device = get_device()
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def to_fp32_for_ssm(x: torch.Tensor) -> torch.Tensor:
    """Cast to FP32 for SSM state transitions (numerical stability)."""
    return x.float()
