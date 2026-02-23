"""Mamba-2 SSM backbone with GRU fallback.

Falls back to GRU when mamba-ssm is unavailable OR when tensors are on CPU
(Mamba2 CUDA kernels require device tensors). This allows the same model to
be tested on CPU and trained on GPU transparently.
"""

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba2
    from mamba_ssm.utils.generation import InferenceParams

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


class GRUFallback(nn.Module):
    """GRU fallback when mamba-ssm is not available or tensors are on CPU."""

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self._hidden: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)."""
        out, _ = self.gru(x)
        return out

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        """x_t: (B, d_model) -> (B, d_model). Recurrent single step."""
        out, self._hidden = self.gru(x_t.unsqueeze(1), self._hidden)
        return out.squeeze(1)

    def reset_state(self) -> None:
        self._hidden = None


class MambaBackbone(nn.Module):
    """Mamba-2 backbone for the RSSM.

    d_model=512, d_state=64, d_conv=4, expand=2, headdim=64, chunk_size=256.
    Training: parallel SSD scan via forward(x_BT).
    Inference: recurrent step via step(x_t, inference_params).

    Device routing: Mamba2 CUDA kernels only run on GPU tensors. When the
    input is on CPU, the backbone transparently routes through GRUFallback
    so tests and CPU-only environments work without modification.
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self._has_mamba = HAS_MAMBA

        if HAS_MAMBA:
            self.mamba = Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                chunk_size=chunk_size,
                rmsnorm=True,
                layer_idx=0,
            )

        self.gru_fallback = GRUFallback(d_model)

    def _use_mamba(self, x: torch.Tensor) -> bool:
        return self._has_mamba and x.is_cuda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model). Training mode (parallel)."""
        if self._use_mamba(x):
            return self.mamba(x)
        return self.gru_fallback(x)

    def step(self, x_t: torch.Tensor, inference_params: object | None = None) -> torch.Tensor:
        """x_t: (B, d_model) -> (B, d_model). Recurrent inference mode."""
        if self._use_mamba(x_t):
            # .contiguous() ensures strides are aligned for causal_conv1d
            out = self.mamba(x_t.unsqueeze(1).contiguous(), inference_params=inference_params)
            return out.squeeze(1)
        return self.gru_fallback.step(x_t)

    def reset_state(self) -> None:
        """Zero conv_state and ssm_state at episode boundaries."""
        self.gru_fallback.reset_state()

    def init_inference_params(self, batch_size: int, max_seqlen: int = 13) -> object | None:
        """Return InferenceParams for recurrent generation."""
        if self._has_mamba:
            return InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size)
        self.gru_fallback.reset_state()
        return None
