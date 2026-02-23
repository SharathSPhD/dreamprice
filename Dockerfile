# DreamPrice — ARM64 Docker build for NVIDIA DGX Spark (GB10 Grace Blackwell)
# Base: nvcr.io/nvidia/pytorch:25.11-py3 (PyTorch 2.10.0, CUDA 13.0, Python 3.12)

FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WANDB_PROJECT=dreamprice

# System deps for building mamba-ssm (needs ninja, git for submodules)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ── CUDA-dependent packages ──────────────────────────────────────────────────
# mamba-ssm and causal-conv1d compile custom CUDA kernels that require GPU
# access at build time. Since `docker build` has no GPU, we install them at
# first container startup via the entrypoint. A named volume caches the
# compiled wheels so subsequent starts are instant.
ENV TORCH_CUDA_ARCH_LIST="12.0"
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# ── Project dependencies (cached unless pyproject.toml changes) ──────────────
COPY pyproject.toml README.md ./
RUN mkdir -p src/retail_world_model && \
    echo '__version__ = "0.1.0"' > src/retail_world_model/__init__.py && \
    pip install -e ".[dev]"

# ── Copy full source ─────────────────────────────────────────────────────────
COPY . .

# Re-install with full source tree
RUN pip install -e ".[dev]"

# ── Non-root user (uid 1000 to match typical host user) ──────────────────────
RUN groupadd -g 1000 dreamprice 2>/dev/null || true && \
    useradd -u 1000 -g 1000 -m -s /bin/bash dreamprice 2>/dev/null || true && \
    chown -R 1000:1000 /workspace

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import retail_world_model; import torch; assert torch.cuda.is_available()" || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["/bin/bash"]
