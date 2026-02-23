"""Mamba-2 SSM backbone with GRU fallback for environments without mamba-ssm."""

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba2
    from mamba_ssm.utils.generation import InferenceParams

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


class GRUFallback(nn.Module):
    """GRU fallback when mamba-ssm is not available."""

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self._hidden: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model)."""
        out, _ = self.gru(x)
        return out

    def step(self, x_t: torch.Tensor, inference_params: object | None = None) -> torch.Tensor:
        """x_t: (B, d_model) -> (B, d_model). Recurrent single step."""
        out, self._hidden = self.gru(x_t.unsqueeze(1), self._hidden)
        return out.squeeze(1)

    def reset_state(self) -> None:
        self._hidden = None

    def init_inference_params(self, batch_size: int, max_seqlen: int = 13) -> None:
        self._hidden = None
        return None


class MambaBackbone(nn.Module):
    """Mamba-2 backbone for the RSSM.

    d_model=512, d_state=64, d_conv=4, expand=2, headdim=64, chunk_size=256.
    Training: parallel SSD scan via forward(x_BT).
    Inference: recurrent step via step(x_t, inference_params).
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
        else:
            self.mamba = GRUFallback(d_model)

        self._use_mamba = HAS_MAMBA

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model). Training mode (parallel)."""
        return self.mamba(x)

    def step(self, x_t: torch.Tensor, inference_params: object | None = None) -> torch.Tensor:
        """x_t: (B, d_model) -> (B, d_model). Recurrent inference mode."""
        if self._use_mamba:
            # Mamba2 step expects (B, 1, d_model) and returns (B, 1, d_model)
            # .contiguous() ensures strides are aligned for causal_conv1d
            out = self.mamba(x_t.unsqueeze(1).contiguous(), inference_params=inference_params)
            return out.squeeze(1)
        else:
            return self.mamba.step(x_t, inference_params)

    def reset_state(self) -> None:
        """Zero conv_state and ssm_state at episode boundaries."""
        if not self._use_mamba:
            self.mamba.reset_state()
        # For Mamba2, state is managed via InferenceParams

    def init_inference_params(self, batch_size: int, max_seqlen: int = 13) -> object | None:
        """Return InferenceParams for recurrent generation."""
        if self._use_mamba:
            return InferenceParams(max_seqlen=max_seqlen, max_batch_size=batch_size)
        else:
            self.mamba.reset_state()
            return None
