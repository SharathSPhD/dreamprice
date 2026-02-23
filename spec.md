# DreamPrice — Full Project Specification

> This document extends `docs/project-blueprint.md` with implementation-level decisions: design patterns, module contracts, agent team topology, worktree assignments, hook wiring, and environment setup. It is the authoritative reference for the agent team.

---

## 1. Design Patterns

The following patterns are applied consistently across the codebase. Every module must respect the pattern assigned to it.

### 1.1 Repository Pattern — Data Access

Hides all file I/O, CSV parsing, and schema validation behind a typed interface.

```python
# src/retail_world_model/data/repository.py
class DominickRepository:
    def __init__(self, data_root: Path): ...
    def load_category(self, category: str) -> pd.DataFrame: ...
    def load_store_demographics(self) -> pd.DataFrame: ...
    def load_customer_counts(self) -> pd.DataFrame: ...
```

Callers never touch raw CSVs directly. All drop logic (PRICE_HEX, OK==0, PRICE<=0) lives inside the repository.

### 1.2 Strategy Pattern — Interchangeable Algorithms

Used for components with multiple valid implementations (e.g., ablation variants).

```python
# Posterior strategies (DRAMA vs. standard RSSM)
class PosteriorStrategy(Protocol):
    def encode(self, x_t: Tensor, h_t: Tensor | None) -> tuple[Tensor, Tensor]: ...

class DecoupledPosterior:     # DRAMA: z_t = encode(x_t) only — NEVER uses h_t
    def encode(self, x_t, h_t=None): ...

class CoupledPosterior:       # Standard RSSM: z_t = encode(x_t, h_t) — ablation only
    def encode(self, x_t, h_t): ...

# Backbone strategies (Mamba-2 vs. GRU)
class BackboneStrategy(Protocol):
    def forward(self, x: Tensor) -> Tensor: ...
    def step(self, x_t: Tensor, state: Any) -> tuple[Tensor, Any]: ...

# Replay strategies
class ReplaySamplerStrategy(Protocol):
    def sample(self, dataset: Dataset, batch_size: int) -> Batch: ...

class HybridReplaySampler:    # 70% uniform quarterly + 30% recent 2 years
    def sample(self, dataset, batch_size): ...
```

### 1.3 Template Method Pattern — Training Loop

The three-phase loop structure is fixed; each phase is overridable.

```python
# src/retail_world_model/training/trainer.py
class BaseTrainer(ABC):
    def train(self, n_steps: int) -> None:
        for step in range(n_steps):
            batch = self.sampler.sample(self.dataset, self.cfg.batch_size)
            self.phase_A_world_model(batch)     # Adam lr=1e-4, clip=1000
            self.phase_B_actor_imagination(batch)  # Adam lr=3e-5, clip=100, H=13
            self.phase_C_critic(batch)          # Adam lr=3e-5, clip=100, EMA=0.98
            self.logger.log(step, self.metrics)

    @abstractmethod
    def phase_A_world_model(self, batch: Batch) -> None: ...
    @abstractmethod
    def phase_B_actor_imagination(self, batch: Batch) -> None: ...
    @abstractmethod
    def phase_C_critic(self, batch: Batch) -> None: ...
```

### 1.4 Observer Pattern — Metrics Logging

W&B and any future logger plugs in without changing training code.

```python
# src/retail_world_model/utils/logging.py
class MetricsLogger(Protocol):
    def log(self, step: int, metrics: dict[str, float]) -> None: ...
    def log_image(self, step: int, key: str, image: Any) -> None: ...

class WandbLogger:
    def log(self, step, metrics): wandb.log(metrics, step=step)

class NullLogger:              # For tests — discards all metrics
    def log(self, step, metrics): pass
```

### 1.5 Builder Pattern — Model Construction

Assembles `MambaWorldModel` from Hydra config without exposing constructor complexity.

```python
# src/retail_world_model/models/builder.py
class WorldModelBuilder:
    def __init__(self, cfg: DictConfig): self.cfg = cfg

    def build(self) -> MambaWorldModel:
        backbone = self._build_backbone()         # Mamba-2 or GRU per cfg
        posterior = self._build_posterior()       # Decoupled or coupled per cfg
        decoder = self._build_decoder()           # Loads frozen elasticities
        reward = RewardEnsemble(...)
        return MambaWorldModel(backbone, posterior, decoder, reward)
```

### 1.6 Facade Pattern — Public API

The `retail_world_model` package surface hides all internals.

```python
# src/retail_world_model/__init__.py
from retail_world_model.models.world_model import MambaWorldModel as WorldModel
from retail_world_model.applications.pricing_policy import ActorCritic as RLAgent
from retail_world_model.envs.grocery import GroceryPricingEnv as PricingEnvironment
from retail_world_model.training.trainer import DreamerTrainer as Trainer

__all__ = ["WorldModel", "RLAgent", "PricingEnvironment", "Trainer"]
```

### 1.7 Chain of Responsibility — Preprocessing Pipeline

Each preprocessing step is a handler that transforms the DataFrame and passes it along.

```python
# src/retail_world_model/data/pipeline.py
class PreprocessingHandler(ABC):
    def __init__(self, next_handler: PreprocessingHandler | None = None): ...
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.process(df)
        return self.next_handler.handle(df) if self.next_handler else df

    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame: ...

# Chain: Filter → DeriveColumns → MergeMetadata → FeatureEngineer → TemporalSplit
pipeline = (
    FilterHandler()            # Drop OK==0, PRICE<=0, PRICE_HEX, PROFIT_HEX
    .set_next(DeriveHandler()) # unit_price, cost, hausman_iv
    .set_next(MergeHandler())  # Join UPC metadata + store demographics
    .set_next(FeatureHandler()) # symlog inputs, promotion flag, lag features
    .set_next(SplitHandler())  # Strict chronological train/val/test
)
```

### 1.8 Null Object Pattern — Missing Elasticity Estimates

Prevents crashes when DML-PLIV hasn't been run yet (e.g., during early development).

```python
# src/retail_world_model/models/decoder.py
class NullElasticityEstimate:
    """Used when configs/elasticities/<cat>.json doesn't exist yet.
    Initializes theta to grocery prior (-2.5) with requires_grad=False."""
    theta: float = -2.5
    is_estimated: bool = False
```

---

## 2. Module Interface Contracts

Strict input/output contracts prevent integration surprises.

### 2.1 Data Layer → Models Layer

```python
# What data/ produces, what models/ consumes
ObservationBatch = TypedDict("ObservationBatch", {
    "x_BT": Tensor,           # (B, T, obs_dim) — symlog-transformed observations
    "a_BT": Tensor,           # (B, T, act_dim) — price actions
    "r_BT": Tensor,           # (B, T) — rewards
    "done_BT": Tensor,        # (B, T) — episode terminators
    "log_price_BT": Tensor,   # (B, T, K) — log unit prices per SKU
    "category_ids": Tensor,   # (B,) — category index for decoder
    "store_features": Tensor, # (B, n_store_features) — demographics
})
```

### 2.2 Models Layer → Training Layer

```python
# What models/ produces, what training/ consumes
WorldModelOutput = TypedDict("WorldModelOutput", {
    "z_posterior_BT": Tensor,    # (B, T, n_cat*n_cls) — straight-through samples
    "z_prior_BT": Tensor,        # (B, T, n_cat*n_cls) — prior samples
    "posterior_probs_BT": Tensor, # (B, T, n_cat, n_cls) — for KL
    "prior_probs_BT": Tensor,    # (B, T, n_cat, n_cls) — for KL
    "h_BT": Tensor,              # (B, T, d_model) — Mamba-2 output states
    "x_recon_BT": Tensor,        # (B, T, obs_dim) — decoder reconstruction
    "r_pred_BT": Tensor,         # (B, T) — reward prediction (r_mean)
    "r_std_BT": Tensor,          # (B, T) — reward uncertainty (for MOPO)
    "continue_BT": Tensor,       # (B, T) — continue logits
})
```

### 2.3 Training Layer → Inference Layer

```python
# What imagination.py needs from models/
class ImagineInterface(Protocol):
    def encode_obs(self, x_t: Tensor) -> tuple[Tensor, Tensor]: ...       # z, probs
    def imagine_step(self, z_t: Tensor, a_t: Tensor,
                     h_t: Tensor, params: InferenceParams
                     ) -> tuple[Tensor, Tensor, Tensor]: ...              # h_next, z_next, r
    def reset_state(self) -> InferenceParams: ...
```

---

## 3. Agent Team Topology

### 3.1 Team Structure

```
~/.claude/teams/dreamprice/config.json    ← team member registry (managed by TeamCreate)
~/.claude/tasks/dreamprice/              ← task list with dependencies (managed by TaskCreate)
```

### 3.2 Worktree Assignment

Each agent gets an **isolated git worktree** (separate directory, separate branch). Agents never touch each other's files.

| Agent | Branch | Worktree Path | Task(s) |
|-------|--------|---------------|---------|
| team-lead | `main` | `/home/sharaths/projects/dreamprice` | Orchestration |
| data-pipeline-agent | `track/data-pipeline` | `/home/sharaths/projects/dreamprice-track1` | Task #1 |
| world-model-agent | `track/world-model` | `/home/sharaths/projects/dreamprice-track2` | Task #2 |
| training-agent | `track/training-loop` | `/home/sharaths/projects/dreamprice-track3` | Tasks #3, #4 |
| causal-estimator-agent | `track/causal` | `/home/sharaths/projects/dreamprice-causal` | Elasticity estimation |
| api-agent | `track/api` | `/home/sharaths/projects/dreamprice-track7` | Task #7 |
| experiment-tracker-agent | `track/experiments` | `/home/sharaths/projects/dreamprice-track5` | Tasks #5, #6 |

### 3.3 Merge Strategy

After each track gate passes:
1. Agent pushes branch to `origin`
2. PR opened (via `gh pr create`)
3. team-lead reviews and merges to `main`
4. Next track's worktree is created from updated `main`

### 3.4 Spawning Pattern

```python
# Each agent is spawned with:
Task(
    subagent_type="general-purpose",
    team_name="dreamprice",
    name="<agent-name>",
    isolation="worktree",           # NEW: isolated git worktree
    prompt="<full task prompt>"
)
```

---

## 4. Hook Configuration

### 4.1 Project-Level Hooks (`.claude/settings.json`)

```json
{
  "hooks": {
    "TaskCompleted": [{
      "matcher": "",
      "hooks": [{
        "type": "command",
        "command": ".claude/hooks/task-gate.sh"
      }]
    }],
    "TeammateIdle": [{
      "matcher": "",
      "hooks": [{
        "type": "command",
        "command": ".claude/hooks/idle-check.sh"
      }]
    }]
  }
}
```

### 4.2 PostToolUse Hooks (`.claude/plugins/dreamprice/hooks/`)

| Hook | Trigger | Action | Blocking |
|------|---------|--------|---------|
| `ruff-guard.sh` | Write/Edit `src/**/*.py` | `ruff check && ruff format --check` | YES |
| `pyright-guard.sh` | Write/Edit `src/**/*.py` | `pyright <file>` | YES (GREEN/REFACTOR), WARN (RED) |
| `pytest-guard.sh` | Write/Edit `src/**/*.py`, `tests/**/*.py` | `pytest tests/<module>/ -x` | YES |
| `blueprint-path-guard.sh` | Write new `src/retail_world_model/*.py` | Check against blueprint §12 paths | NO (warn only) |

---

## 5. Environment Configuration

### 5.1 Required Environment Variables

Stored in `.env` (gitignored). Load with `python-dotenv` in all scripts.

```bash
# .env — never commit this file
HF_TOKEN=hf_...          # HuggingFace Hub upload
WANDB_API_KEY=wandb_...  # W&B experiment tracking
WANDB_PROJECT=dreamprice # W&B project name
```

### 5.2 Loading Pattern

```python
# scripts/train.py and all entry points
from dotenv import load_dotenv
load_dotenv()  # Loads .env from project root
```

### 5.3 GitHub

SSH is configured. Remote: `git@github.com:SharathSPhD/dreamprice.git`
All pushes via SSH, no token needed.

---

## 6. Git Worktree Workflow

### 6.1 Branch Structure

```
main                     ← production-ready, merges in after each track gate
track/data-pipeline      ← Track 1: data-pipeline-agent works here
track/world-model        ← Track 2: world-model-agent works here (after track 1 merged)
track/training-loop      ← Track 3: training-agent works here
track/baselines          ← Track 4: training-agent continues here
track/ablations          ← Track 5: experiment-tracker-agent
track/packaging          ← Track 6: data-pipeline-agent (reused)
track/api                ← Track 7: api-agent (parallel with track 6)
track/causal             ← Causal estimation: causal-estimator-agent
```

### 6.2 Creating Worktrees

```bash
# After initial commit to main:
git worktree add ../dreamprice-track1 -b track/data-pipeline
git worktree add ../dreamprice-track2 -b track/world-model
git worktree add ../dreamprice-track3 -b track/training-loop
git worktree add ../dreamprice-track7 -b track/api
git worktree add ../dreamprice-causal -b track/causal
```

### 6.3 Agent Working Directories

When spawning an agent, set `cwd` to their worktree path. All file writes go into the isolated branch.

### 6.4 Worktree Merge Back

```bash
# team-lead runs after each track gate:
cd /home/sharaths/projects/dreamprice     # main worktree
git merge track/data-pipeline --no-ff -m "feat: merge track 1 — data pipeline complete"
git push origin main
# Then world-model worktree is created from updated main
```

---

## 7. Task Dependency DAG

```
#1 data-pipeline     [READY]
    └─► #2 world-model        [blocked by #1]
            └─► #3 training-loop    [blocked by #2]
                    ├─► #4 baselines     [blocked by #3]
                    │       └─► #5 ablations   [blocked by #4]
                    │               └─► #6 packaging  [blocked by #5]
                    └─► #7 api+demo       [blocked by #3, parallel with #6]
```

System auto-unblocks dependent tasks when predecessors complete via `TaskUpdate(status="completed")`.

---

## 8. Conductor Track Files

Each track has a corresponding file in `conductor/tracks/`:

```
conductor/
├── index.md
├── product.md
├── product-guidelines.md
├── tech-stack.md
├── workflow.md
├── tracks.md
└── code_styleguides/
    └── python.md
```

Track gate passes are recorded in `docs/progress/<track-name>.md`.

---

## 9. Quality Gates Summary

| Gate | Trigger | Check | Blocks |
|------|---------|-------|--------|
| ruff | Every file write | `ruff check + format --check` | Immediate |
| pyright | Every src write | Type errors | Immediate (GREEN/REFACTOR) |
| pytest | Every src/test write | Module test suite | Immediate |
| TaskCompleted hook | Mark task done | Full suite + pyright + ruff | Task completion |
| TeammateIdle hook | Agent going idle | Check for available tasks | Agent idle |
| Track gate | Manual (team-lead) | pytest + pyright + ruff + progress.md | Track advancement |

---

## 10. Causal Estimation Protocol

Run `causal-estimator-agent` at the end of Track 1. Output required by Track 2 decoder.

```
Input:  data/processed/cso_train.parquet
Output: configs/elasticities/cso.json

Required fields:
  theta_causal: float          # DML-PLIV estimate, expect -2.0 to -3.0
  f_stat_first_stage: float    # Must be > 10 (expect > 100)
  hausman_p_value: float       # Must be < 0.10
  cope_beta_c: float           # 2sCOPE robustness check
  n_obs: int

Gate: f_stat < 10 → raise WeakInstrumentError (block track 2)
```

---

## 11. API Serving Architecture

```
FastAPI app (serve.py)
    ├── Lifespan: load checkpoint on startup
    ├── /health                GET  <5ms
    ├── /predict               POST <50ms   → batching.py (max 8, wait 50ms)
    ├── /rollout               POST 100ms-1s
    ├── /rollout/stream        POST SSE streaming (sse-starlette)
    ├── /predict/batch         POST variable
    └── /model/info            GET  <5ms
```

Dynamic batching via asyncio queue. Max batch size 8, max wait 50ms.

---

*This spec extends `docs/project-blueprint.md`. When in conflict, the blueprint takes precedence for ML architecture decisions; this spec takes precedence for engineering/tooling decisions.*
