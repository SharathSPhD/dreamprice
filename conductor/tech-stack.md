# Tech Stack — DreamPrice

## Primary Language
Python 3.11+

## Core ML Stack
| Library | Role | Install note |
|---------|------|-------------|
| `torch` (BF16) | Neural network training | CUDA-dependent, install separately |
| `mamba-ssm` | Mamba-2 SSM backbone | CUDA-dependent, install separately |
| `causal-conv1d` | Mamba-2 dependency | CUDA-dependent, install separately |
| `triton` | Kernel optimization | CUDA-dependent, install separately |

**Install order matters:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install mamba-ssm causal-conv1d
pip install -e ".[dev]"
```

## Data & Causal Inference
| Library | Role |
|---------|------|
| `pandas`, `numpy` | Data loading and preprocessing |
| `scipy` | Statistical tests, copula transforms |
| `doubleml` | DML-PLIV causal elasticity estimation |
| `scikit-learn` | Random forest nuisance models for DML |
| `pydantic` | Data schemas and validation |

## RL & Environment
| Library | Role |
|---------|------|
| `gymnasium` | Pricing environment interface |
| `stable-baselines3` | PPO, SAC, DQN baselines |

## Training & Experiment Tracking
| Library | Role |
|---------|------|
| `hydra-core` | Config management (configs/ directory) |
| `wandb` | Experiment tracking (`WANDB_PROJECT=dreamprice`) |
| `safetensors` | Checkpoint serialization |
| `huggingface_hub` | Model upload/download |

## Serving
| Library | Role |
|---------|------|
| `fastapi` | API framework |
| `uvicorn` | ASGI server |
| `sse-starlette` | SSE streaming for `/rollout/stream` |

## Frontend (Track 7)
| Library | Role |
|---------|------|
| React + Vite | Interactive RL playground |
| Streamlit | Researcher dashboard |

## Build & Quality
| Tool | Role |
|------|------|
| `hatchling` | Build backend (pure Python, no CUDA extensions in package) |
| `ruff` | Lint + format (C901 McCabe complexity) |
| `complexipy` | Cognitive complexity (Rust-backed, PyO3) |
| `pyright` / `mypy` | Type checking |
| `pytest` | Testing |

## Hardware Target
DGX Spark — 128 GB unified memory, ~100 TFLOPS BF16. Training is **memory-bandwidth bound**. Mixed precision: BF16 forward/backward, FP32 master weights and optimizer. SSM state transitions in FP32 for numerical stability.
