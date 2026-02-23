# DreamPrice — Agent Team Instructions

> Authoritative reference for the Claude Code agent team working on the DreamPrice project.
> Last updated: 2026-02-23 (baseline Docker setup established).

---

## 1. Project Overview

DreamPrice is a **DreamerV3-style world model** for retail pricing that learns demand dynamics from Dominick's Finer Foods scanner data and optimizes pricing decisions via offline reinforcement learning.

**Key architectural decisions:**

- **Mamba-2 SSM backbone** replaces GRU for parallel training via SSD (with GRU fallback when `mamba-ssm` is unavailable)
- **DRAMA-style decoupled posterior**: `z_t = encode(x_t)` — the posterior never sees the deterministic state `h_t`
- **Causal demand decoder** with frozen Hausman IV elasticities estimated via DML-PLIV
- **MOPO-style LCB pessimism** with 5-head reward ensemble for offline RL safety
- **Competition modeled implicitly** via RSSM's stochastic state conditioned on store demographics

**Reference documents (read in this order):**

1. `docs/project-blueprint.md` — Consolidated architectural blueprint
2. `spec.md` — Full specification: design patterns, interfaces, agent topology, hooks
3. `CLAUDE.md` — Quick-start development commands and module map

---

## 2. Hardware Environment

| Property | Value |
|----------|-------|
| Machine | NVIDIA DGX Spark |
| GPU | NVIDIA GB10 (Blackwell, compute capability 12.1) |
| CUDA | 13.0 |
| CPU | 20× Grace ARM64 (aarch64) cores |
| Memory | 128GB unified LPDDR5x (~116GB free) |
| Disk | ~3.0TB free |
| OS | Linux 6.17.0 (Ubuntu Noble) |

**Memory bandwidth bound** — not compute bound. Batch sizes should saturate memory bandwidth rather than GPU SMs.

---

## 3. Docker Setup

All development, testing, and training happens inside the Docker container. **Never install dependencies or run tests on the host directly.**

### 3.1 Quick Start

```bash
cd /home/sharaths/projects/dreamprice

# Build the image (fast — ~20s, no CUDA compilation)
make build

# Enter the container (first run takes ~10 min to compile mamba-ssm CUDA kernels)
make shell

# Verify everything works
make verify-gpu

# Run tests
make test
```

### 3.2 Architecture

```
Host: /home/sharaths/projects/dreamprice/
  ↓ bind mount
Container: /workspace/
  ├── src/retail_world_model/    ← source (live edits from host)
  ├── tests/                    ← tests
  ├── scripts/                  ← train.py, evaluate.py, serve.py
  └── docs/data/                ← Dominick's CSV data (bind-mounted)
```

**Three Docker volumes persist across container restarts:**

| Volume | Mount Point | Purpose |
|--------|-------------|---------|
| `site-packages` | `/usr/local/lib/python3.12/dist-packages` | Compiled mamba-ssm/causal-conv1d CUDA kernels |
| `pip-cache` | `/root/.cache/pip` | Download cache for pip |
| Bind mount | `/workspace` | Project source code (live edits) |

### 3.3 How the Entrypoint Works

The `docker-entrypoint.sh` runs on every container start:

1. Sets `MAMBA_FORCE_BUILD=TRUE` and `CAUSAL_CONV1D_FORCE_BUILD=TRUE` to prevent downloading precompiled x86_64 wheels
2. Checks if `mamba-ssm` is importable — if not, compiles from source with GPU
3. Checks if `retail_world_model` is importable — if not, runs `pip install -e ".[dev]"`
4. Executes the passed command (`bash`, `pytest`, etc.)

### 3.4 Makefile Targets

| Target | Command | Description |
|--------|---------|-------------|
| `make build` | `docker compose build` | Build/rebuild the image |
| `make shell` | `docker compose run --rm --service-ports dreamprice bash` | Interactive shell with GPU |
| `make test` | `pytest tests/ -v --tb=short` | Run full test suite |
| `make test-collect` | `pytest tests/ --co -q` | Verify test collection |
| `make train` | `python scripts/train.py` | Run training |
| `make serve` | `python -m retail_world_model.api.serve` | Start FastAPI on :8000 |
| `make lint` | `ruff check + pyright` | Lint check |
| `make format` | `ruff format` | Auto-format |
| `make check` | Lint + format check + pyright + pytest | Full quality gate |
| `make verify-gpu` | Python one-liner | Verify CUDA + mamba-ssm + package |
| `make clean` | Remove images, volumes, caches | Full cleanup |

### 3.5 When to Rebuild

| Scenario | Action |
|----------|--------|
| Changed `pyproject.toml` dependencies | `make build` then `make shell` — entrypoint will re-install |
| Changed source code | Nothing — bind mount means changes are live |
| mamba-ssm import fails | `make clean && make build && make shell` (recompile) |
| New Docker base image | `docker pull nvcr.io/nvidia/pytorch:25.11-py3 && make clean && make build` |

### 3.6 Environment Variables

Create a `.env` file in the project root:

```bash
WANDB_API_KEY=your_key_here
HF_TOKEN=your_token_here
```

These are passed into the container via `docker-compose.yml`.

---

## 4. Codebase State (Baseline 2026-02-23)

### 4.1 Test Results

**104 tests collected, 104 passed, 0 failed, 0 warnings.**

All tests pass cleanly, including the Mamba2/RSSM tests which use GRU fallback
on CPU tensors and the full Mamba2 CUDA path on GPU.

### 4.2 Module Status

| Module | Path | Status | Notes |
|--------|------|--------|-------|
| Data loading | `data/dominicks_loader.py` | Working | All tests pass |
| Data schemas | `data/schemas.py` | Working | All tests pass |
| Data transforms | `data/transforms.py` | Working | All tests pass (symlog, Hausman IV, temporal split) |
| Copula correction | `data/copula_correction.py` | Scaffolded | Not tested |
| Encoder | `models/encoder.py` | Scaffolded | Not tested |
| Decoder | `models/decoder.py` | Working | CausalDemandDecoder tests pass |
| Posterior | `models/posterior.py` | Working | DecoupledPosterior tests pass |
| Distributions | `utils/distributions.py` | Working | Symlog, twohot, unimix, straight-through pass |
| Mamba backbone | `models/mamba_backbone.py` | Working | Device-aware routing: Mamba2 on CUDA, GRU on CPU |
| RSSM | `models/rssm.py` | Working | All 4 tests pass |
| World model | `models/world_model.py` | Working | Forward + imagine tests pass |
| Reward head | `models/reward_head.py` | Scaffolded | Not tested |
| Training losses | `training/losses.py` | Working | KL balancing, MOPO LCB, lambda returns all pass |
| Trainer | `training/trainer.py` | Scaffolded | Not tested |
| Offline utils | `training/offline_utils.py` | Scaffolded | Not tested |
| Environments | `envs/grocery.py` | Working | All 5 tests pass |
| API batching | `api/batching.py` | Working | All 6 tests pass |
| API serve | `api/serve.py` | Working | All 6 tests pass |
| Imagination | `inference/imagination.py` | Scaffolded | Not tested |
| Pricing policy | `applications/pricing_policy.py` | Scaffolded | Not tested |
| Ablations | `scripts/ablations.py` | Working | All 15 tests pass |

### 4.3 Missing Pieces

1. **Causal estimation pipeline** — DML-PLIV for Hausman IV elasticities (needed before world model training)
2. **Dataset class** — Sequence batching for training (sliding window over temporal data)
3. **Full training loop** — Wiring trainer, losses, world model, replay buffer together
4. **Evaluation pipeline** — Rollout + metrics collection
5. **Entity-factored attention** — Temporal self-attention + relational cross-attention for multi-SKU

---

## 5. Development Workflow

### 5.1 Edit-Test Cycle

```bash
# On host: edit code in your IDE (files are bind-mounted)
# In container (make shell):
pytest tests/test_models/test_rssm.py -v --tb=short   # run specific tests
ruff check src/ --fix                                    # auto-fix lint
pyright src/                                             # type check
```

### 5.2 Quality Gates

Before merging any track, ALL of these must pass:

```bash
make check    # runs: ruff check + ruff format --check + pyright + pytest
```

This is equivalent to:
```bash
ruff check src/ tests/
ruff format --check src/ tests/
pyright src/
pytest tests/ -v --tb=short
```

### 5.3 Adding Dependencies

1. Add to `pyproject.toml` under the appropriate extras group
2. Rebuild: `make build`
3. Re-enter: `make shell`

### 5.4 Running Training

```bash
make train                          # default config
make train ARGS="--epochs 10"       # with overrides
```

### 5.5 Running the API Server

```bash
make serve    # starts FastAPI on port 8000, accessible from host
```

---

## 6. Track Status & Priority

Based on `spec.md` Section 3 and the current codebase state:

| Track | Branch | Status | Blocking? | Priority |
|-------|--------|--------|-----------|----------|
| 1 — Data Pipeline | `track/data-pipeline` | Core loading works, needs sequence batching | No | **High** |
| Causal | `track/causal` | Not started | Blocks Track 2 decoder | **High** |
| 2 — World Model | `track/world-model` | RSSM partially working (4 test failures) | Needs Track 1 + Causal | **High** |
| 3 — Training Loop | `track/training-loop` | Losses working, trainer scaffolded | Needs Track 2 | Medium |
| 4 — Baselines | `track/baselines` | Not started | Needs Track 3 | Low |
| 5 — Ablations | `track/ablations` | Test infrastructure ready | Needs Track 3 | Low |
| 6 — Packaging | `track/packaging` | Docker setup complete | None | Done |
| 7 — API | `track/api` | Working (batching + serve tests pass) | None | Low |

### Recommended Next Steps (in order)

1. **Implement sequence Dataset** — `data/dataset.py` for sliding-window batching over temporal sequences
2. **Run DML-PLIV causal estimation** — Outputs frozen `theta` vector consumed by `CausalDemandDecoder`
3. **Wire the training loop** — Connect trainer, losses, world model, and replay
4. **Add entity-factored attention** — Multi-SKU temporal + relational cross-attention

---

## 7. Git Worktree Setup

Each agent works in an isolated worktree. To set up:

```bash
cd /home/sharaths/projects/dreamprice    # main worktree

# Create worktrees for each track (run once)
git worktree add ../dreamprice-track1 -b track/data-pipeline
git worktree add ../dreamprice-track2 -b track/world-model
git worktree add ../dreamprice-track3 -b track/training-loop
git worktree add ../dreamprice-track7 -b track/api
git worktree add ../dreamprice-causal -b track/causal
```

Each worktree shares the same `.git` but has its own branch and working directory. Agents never touch each other's files.

**Merge protocol (team-lead only):**

```bash
cd /home/sharaths/projects/dreamprice
git merge track/data-pipeline --no-ff -m "feat: merge track 1 — data pipeline"
git push origin main
```

---

## 8. Module Dependency Map

```
data/dominicks_loader.py
  └── data/transforms.py (symlog, Hausman IV, temporal split)
      └── data/schemas.py (MovementRow, UPCRow, etc.)

models/encoder.py ─────┐
models/posterior.py ────┼── models/rssm.py
models/mamba_backbone.py┘       │
                                ├── models/world_model.py
models/decoder.py ──────────────┘       │
models/reward_head.py ──────────────────┘
                                        │
training/losses.py ─────────── training/trainer.py
training/offline_utils.py ─────┘       │
                                       │
envs/grocery.py ───────────── inference/imagination.py
                                       │
                              applications/pricing_policy.py
                                       │
                              api/serve.py ←── api/batching.py
```

---

## 9. Known Issues & Workarounds

### 9.1 mamba-ssm on ARM64

The `mamba-ssm` and `causal-conv1d` packages ship precompiled x86_64 CUDA wheels on PyPI and GitHub Releases. On ARM64 (DGX Spark), these must be force-compiled from source:

- `MAMBA_FORCE_BUILD=TRUE` — prevents downloading x86_64 cached wheels
- `CAUSAL_CONV1D_FORCE_BUILD=TRUE` — same for causal-conv1d
- `--no-build-isolation` — uses container's PyTorch instead of pip creating a conflicting isolated environment

The `docker-entrypoint.sh` handles all of this automatically.

### 9.2 site-packages Volume

The `site-packages` Docker volume persists compiled CUDA extensions across container restarts. If you need a completely fresh environment:

```bash
make clean    # removes images + all volumes
make build    # rebuild image
make shell    # first run recompiles mamba-ssm (~10 min)
```

### 9.3 Mamba2 Device-Aware Routing (Resolved)

`MambaBackbone` automatically routes through GRU fallback when input tensors
are on CPU, and through Mamba2 when on CUDA. This means:

- Tests work on CPU without modification
- Training on GPU transparently uses the full Mamba2 CUDA path
- `layer_idx=0` is set in the `Mamba2` constructor for inference cache support

---

## 10. Environment Checklist for New Agents

When starting work on a new track:

- [ ] Confirm Docker image is built: `make build`
- [ ] Enter container: `make shell`
- [ ] Verify GPU: `make verify-gpu`
- [ ] Run tests: `make test` — confirm baseline (104 pass, 0 failures)
- [ ] Read `spec.md` for your track's design patterns and interfaces
- [ ] Create/attach your worktree branch
- [ ] Write tests FIRST (TDD), then implement
- [ ] Run `make check` before every commit
- [ ] Document progress in `docs/progress/<track-name>.md`
