#!/usr/bin/env bash
set -euo pipefail

export MAMBA_FORCE_BUILD=TRUE
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

# Remove any x86_64 .so stubs left by precompiled wheel downloads
rm -f /usr/local/lib/python3.12/dist-packages/selective_scan_cuda*x86_64*.so \
      /usr/local/lib/python3.12/dist-packages/causal_conv1d_cuda*x86_64*.so 2>/dev/null || true

# Install causal-conv1d + mamba-ssm with locally-compiled CUDA kernels.
# The site-packages volume persists across container restarts, so the
# expensive GPU compilation only happens on first run (~10 min).
if ! python -c "from mamba_ssm import Mamba2" 2>/dev/null; then
    echo ">>> Installing causal-conv1d + mamba-ssm (first run compiles CUDA kernels)..."
    pip uninstall -y mamba-ssm causal-conv1d 2>/dev/null || true
    pip install --no-build-isolation causal-conv1d
    pip install --no-build-isolation mamba-ssm
    python -c "from mamba_ssm import Mamba2; print('  mamba-ssm: verified OK')"
fi

# Ensure project is installed in editable mode
if ! python -c "import retail_world_model" 2>/dev/null; then
    echo ">>> Installing project in editable mode..."
    pip install -e "/workspace[dev]"
fi

exec "$@"
