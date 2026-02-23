# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DreamPrice** is a research project implementing the first learned world model for retail pricing environments. It combines DreamerV3's training recipe, a Mamba-2 SSM backbone (DRAMA-style), entity-factored multi-SKU representation, Hausman IV causal identification, and MOPO-style offline pessimism — trained on the Dominick's Finer Foods scanner dataset (1989–1997, 93 stores, ~18K UPCs).

The authoritative specification is `docs/project-blueprint.md`. Read it before making architectural decisions.

---

## Commands

### Setup

```bash
# Install CUDA-dependent packages FIRST (torch, mamba-ssm, triton require CUDA version matching)
# These are NOT listed in pyproject.toml hard dependencies by design
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install mamba-ssm causal-conv1d

# Install the package in editable mode
pip install -e ".[dev]"
```

### Development

```bash
# Lint and format
ruff check src/ tests/
ruff format src/ tests/

# Type checking
pyright src/
# or: mypy src/

# Complexity checks
ruff check --select C901 src/   # McCabe cyclomatic complexity
complexipy src/                  # Cognitive complexity (Rust-backed)

# Run tests
pytest tests/

# Run a single test file
pytest tests/test_world_model.py

# Run a single test
pytest tests/test_world_model.py::test_rssm_forward -v

# Train (entry point — config via Hydra)
python scripts/train.py experiment=configs/experiment/main.yaml

# Evaluate
python scripts/evaluate.py checkpoint=<path>

# Serve the API
python -m retail_world_model.api.serve
```

### W&B logging is standard. Set `WANDB_PROJECT=dreamprice`.

---

## Architecture

The planned package lives at `src/retail_world_model/`. See `docs/project-blueprint.md §12` for full module layout.

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `data/dominicks_loader.py` | Load movement/UPC/demo CSVs from `docs/data/` |
| `data/transforms.py` | `symlog`/`symexp`, Hausman IV construction, feature engineering |
| `data/copula_correction.py` | 2sCOPE endogeneity correction (robustness check only) |
| `models/rssm.py` | RSSM core: prior + posterior + Mamba-2 backbone |
| `models/decoder.py` | `CausalDemandDecoder` with frozen DML-PLIV elasticities |
| `models/reward_head.py` | 5-head ensemble for MOPO LCB |
| `training/trainer.py` | Three-phase loop: (A) world model, (B) actor imagination, (C) critic |
| `training/losses.py` | ELBO, KL balancing, causal regularization, twohot cross-entropy |
| `inference/imagination.py` | Latent rollouts; switches Mamba-2 to recurrent `step()` mode |
| `api/serve.py` | FastAPI app factory with lifespan |
| `api/batching.py` | Asyncio queue-based dynamic batcher (max batch 8, max wait 50ms) |
| `envs/grocery.py` | Gymnasium-compatible pricing environment |

### Key Architectural Decisions

**DRAMA-style decoupled posterior**: `z_t = encode(x_t)` — posterior depends ONLY on the observation, not on `h_t`. This breaks the sequential dependency and enables Mamba-2's parallel SSD scan during training. During imagination, switch to `mamba_block.step()` for recurrent single-step generation. `h_t` is the **output** of the Mamba-2 block (`B × d_model`), NOT the internal SSM state.

**Stochastic latent**: 32 independent categorical distributions × 32 classes = 1024-dim one-hot. Entity-factored variant: 8 categoricals × 8 classes per entity. Always apply 1% unimix: `probs = 0.99 × softmax(logits) + 0.01/32`.

**Causal constrained decoder**: Per-category price elasticities (`θ̂`) pre-estimated via DML-PLIV are **frozen** in `CausalDemandDecoder`. The MLP residual learns everything else. This prevents the world model from learning a confounded price-demand relationship. Expected elasticity range: −2.0 to −3.0 (grocery). See `docs/project-blueprint.md §9`.

**KL balancing**: `β_pred=1.0`, `β_dyn=0.5` (sg(posterior)), `β_rep=0.1` (sg(prior)), free bits=1 nat. The 5:1 asymmetry prevents posterior collapse.

**MOPO LCB pessimism**: `r_pessimistic = r_mean - λ_lcb × r_std` from 5 independent reward heads. Modify only the reward signal — no changes to critic training loop. Start with `λ_lcb=1.0`, search `{0.25, 0.5, 1.0, 2.0, 5.0}`.

**Symlog everywhere**: Apply `symlog(x) = sign(x) · ln(|x|+1)` to encoder inputs, decoder targets, reward targets, and critic targets. Never use raw log (undefined at zero) on demand/price data.

**Twohot + distributional critic**: 255 bins uniformly in symlog [−20, +20]. Reward and critic losses use categorical cross-entropy against soft twohot labels. Recover scalar as `symexp(Σ probs_i × bins_i)`.

**Straight-through gradients**: `z = one_hot + probs - probs.detach()`. No Gumbel-Softmax.

### Training Loop (Phase A / B / C)

- **Phase A** (World model): B=32 sequences × T=64 steps. Adam lr=1e-4, grad clip global norm 1000.
- **Phase B** (Actor in imagination): H=13 steps (one retail quarter). Lambda-returns with γ=0.95, λ=0.95. REINFORCE (discrete) or reparameterization (continuous) + entropy η=3e-4. Adam lr=3e-5, grad clip 100.
- **Phase C** (Critic): Twohot distributional regression. EMA slow critic decay=0.98. Adam lr=3e-5, grad clip 100.
- Return normalization: `R_norm = R^λ / max(1, P95 − P5)` with EMA decay 0.99.
- Replay: 70% uniform across quarterly strata + 30% overweighting most recent 2 years.

### Hardware Target

DGX Spark (128 GB unified memory, ~100 TFLOPS BF16). Training is **memory-bandwidth bound**. BF16 forward/backward, FP32 master weights/optimizer. SSM state transitions in FP32 for numerical stability.

---

## Dataset

Raw data is in `docs/data/`:

- `docs/data/category/wXXX.csv` — movement files (STORE, UPC, WEEK, MOVE, QTY, PRICE, SALE, PROFIT, OK, PRICE_HEX, PROFIT_HEX). Drop `PRICE_HEX`, `PROFIT_HEX`. Drop rows where `OK==0` or `PRICE<=0`.
- `docs/data/category/upcXXX.csv` — UPC metadata (COM_CODE, UPC, DESCRIP, SIZE, CASE, NITEM).
- `docs/data/demo.csv` — Store demographics (300+ columns, one row per store).
- `docs/data/ccount.csv` — Weekly store revenue by department + customer counts.

**Start with canned soup (`cso`)**: ~25 SKUs, clear substitution, ~581K training tuples after preprocessing. Beer (`ber`) and soft drinks (`sdr`) are also recommended.

**Temporal split** (strictly chronological): Train weeks 1–280, Val 281–340, Test 341–400. Never shuffle across time.

**Critical derived fields**:
- `unit_price = PRICE / QTY`
- `cost = PRICE × (1 − PROFIT/100) / QTY`
- `hausman_iv = (sum_log_price_all_stores − own_log_price) / (n_stores − 1)`
- `on_promotion`: SALE field has false negatives; flag any week where `unit_price < 0.95 × modal_price`.

---

## Public API

```python
from retail_world_model import WorldModel, RLAgent, PricingEnvironment, Trainer
```

Environments follow Gymnasium's `reset()`/`step()`. World models expose `predict()`, `rollout()`, `imagine()`. Checkpoints on HuggingFace Hub under CC-BY-NC-4.0.

---

## Code Quality

- Build backend: **Hatchling** (pure Python, no compiled CUDA extensions in the package).
- `torch`, `mamba-ssm`, `triton` are optional dependencies — document install sequence separately.
- Config management: **Hydra** (configs in `configs/`).
- Experiment tracking: **W&B**.
- Ruff with C901 (cyclomatic complexity) + complexipy (cognitive complexity).
- Pyright / mypy for type checking.
