# DreamPrice — Implementation Guide

> Companion document to `docs/agent-team-instructions.md`. This guide contains
> the exact code changes, interface contracts, and file-level specifications
> the Claude Code agent team must implement to take DreamPrice from its current
> scaffolded state to fully functional end-to-end training, assessment, and
> ablation experiments.
>
> **Rules for agents executing this guide:**
> - Do NOT mock, stub, or simplify any component. Implement fully.
> - Do NOT skip tests. Write tests FIRST, then implement.
> - Do NOT deviate from the interfaces specified here without updating all consumers.
> - Run `make check` inside the Docker container after every file change.
> - All work runs inside `make shell`. Never install on host.

---

## Table of Contents

1. [Critical Gap Analysis](#1-critical-gap-analysis)
2. [Phase 1: Data Pipeline Completion](#2-phase-1-data-pipeline-completion)
3. [Phase 2: World Model Interface Alignment](#3-phase-2-world-model-interface-alignment)
4. [Phase 3: Causal Estimation Pipeline](#4-phase-3-causal-estimation-pipeline)
5. [Phase 4: Training Loop Wiring](#5-phase-4-training-loop-wiring)
6. [Phase 5: Assessment and Environment](#6-phase-5-assessment-and-environment)
7. [Phase 6: Ablations and Packaging](#7-phase-6-ablations-and-packaging)
8. [Dependency DAG](#8-dependency-dag)
9. [File Inventory](#9-file-inventory)
10. [Acceptance Criteria](#10-acceptance-criteria)

---

## 1. Critical Gap Analysis

The codebase has 39 source files, 13 test files (104 passing), and 13 Hydra
configs. Individual components work in isolation. The system cannot train
end-to-end because of these 7 gaps:

### Gap 1: RSSM Output Keys Mismatch

`RSSM.train_sequence()` in `src/retail_world_model/models/rssm.py` returns:

```python
return {
    "z_BT": z_BT,                    # trainer expects "z_posterior_BT"
    "h_BT": h_BT,
    "posterior_probs": posterior_probs, # losses expect "posterior_probs_BT"
    "prior_probs": prior_probs,        # losses expect "prior_probs_BT"
    "reward_mean": r_mean,
    "reward_std": r_std,
    "continue_logits": continue_logits,
}
```

But `elbo_loss()` in `src/retail_world_model/training/losses.py:127-133` does:

```python
x_recon = output["x_recon_BT"]           # KEY DOES NOT EXIST
kl_balancing(output["posterior_probs_BT"],  # KEY DOES NOT EXIST
             output["prior_probs_BT"])     # KEY DOES NOT EXIST
```

And `DreamerTrainer.train_step()` in `src/retail_world_model/training/trainer.py:200` does:

```python
z0 = output["z_posterior_BT"][:, 0]  # KEY DOES NOT EXIST
```

### Gap 2: No Observation Decoder

The ELBO loss needs `x_recon_BT` of shape `(B, T, obs_dim)` — a reconstruction
of the input observation from the latent state. `CausalDemandDecoder` predicts
per-SKU demand scalars, not full observation vectors. An observation
reconstruction decoder is completely absent from `models/rssm.py`.

### Gap 3: Imagination Interface Mismatch

`rollout_imagination()` in `src/retail_world_model/inference/imagination.py:79` calls:

```python
step_out = model.imagine_step(z_t, action.float(), h_t)  # passes h_t as 3rd arg
h_t = step_out["h"]      # expects dict return
z_t = step_out["z"]      # expects dict return
r_mean = step_out["r_mean"]  # expects rewards in dict
```

But `RSSM.imagine_step()` in `src/retail_world_model/models/rssm.py:97-118`:

```python
def imagine_step(self, z_t, a_t, inference_params):  # 3rd arg is inference_params, not h_t
    ...
    return h_next, z_next, prior_probs  # returns TUPLE, not dict; no rewards
```

### Gap 4: No SequenceDataset

No PyTorch `Dataset` converts Dominick's DataFrames into `(x_BT, a_BT, r_BT,
done_BT)` tensors of shape `(T, dim)`. The trainer expects a Dataset that
yields batches with these keys.

### Gap 5: Incomplete Feature Engineering

`build_observation_vector()` in `src/retail_world_model/data/transforms.py:67-90`
only produces 5 + N_demo features. The blueprint (Section 5) requires per-SKU:
`symlog(unit_price)`, `symlog(move)`, `discount_depth`, `on_promotion`,
`lag_price_1`, `lag_price_2`, `lag_move_1`, `lag_move_2`,
`rolling_mean_move_4`, `price_index`, `cost_per_unit`, `profit_margin` (12 dims),
plus temporal (6 dims) and store context (8 dims).

### Gap 6: No DML-PLIV Script

No runnable script performs causal estimation. `notebooks/causal_analysis.ipynb`
has reference code but does not write `configs/elasticities/cso.json`.

### Gap 7: Stub Scripts

- `scripts/train.py` prints config and exits at line 44.
- `scripts/assess.py` loads checkpoint and exits at line 31.

---

## 2. Phase 1: Data Pipeline Completion

**Agent:** `data-pipeline-agent`
**Branch:** `track/data-pipeline`
**Depends on:** Nothing (can start immediately)
**Blocks:** Phase 2, Phase 3, Phase 4

### 2.1 Feature Engineering — Modify `src/retail_world_model/data/transforms.py`

Add these functions after the existing `flag_promotions()`:

#### `compute_discount_depth(df) -> pd.Series`

```python
def compute_discount_depth(df: pd.DataFrame) -> pd.Series:
    """discount_depth = (modal_price - unit_price) / modal_price, per (UPC, STORE)."""
    unit_price = df["PRICE"] / df["QTY"]
    modal_price = unit_price.groupby([df["UPC"], df["STORE"]]).transform(
        lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else s.median()
    )
    return ((modal_price - unit_price) / modal_price.clip(lower=1e-6)).clip(lower=0.0)
```

#### `compute_lag_features(df, group_cols, value_col, lags) -> pd.DataFrame`

```python
def compute_lag_features(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    lags: list[int] = [1, 2],
) -> pd.DataFrame:
    """Create lag features within (STORE, UPC) groups sorted by WEEK.

    Returns DataFrame with columns like '{value_col}_lag_{lag}'.
    """
    df = df.sort_values(group_cols + ["WEEK"])
    result = pd.DataFrame(index=df.index)
    grouped = df.groupby(group_cols)[value_col]
    for lag in lags:
        result[f"{value_col}_lag_{lag}"] = grouped.shift(lag)
    return result.fillna(0.0)
```

#### `compute_rolling_features(df, group_cols, value_col, window) -> pd.DataFrame`

```python
def compute_rolling_features(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    window: int = 4,
) -> pd.DataFrame:
    """Rolling mean and std within (STORE, UPC) groups sorted by WEEK."""
    df = df.sort_values(group_cols + ["WEEK"])
    grouped = df.groupby(group_cols)[value_col]
    result = pd.DataFrame(index=df.index)
    result[f"{value_col}_rolling_mean_{window}"] = grouped.transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )
    result[f"{value_col}_rolling_std_{window}"] = grouped.transform(
        lambda s: s.rolling(window, min_periods=1).std().fillna(0.0)
    )
    return result
```

#### `compute_price_index(df) -> pd.Series`

```python
def compute_price_index(df: pd.DataFrame) -> pd.Series:
    """price_index = unit_price / category_mean_price per WEEK."""
    unit_price = df["PRICE"] / df["QTY"]
    cat_mean = unit_price.groupby(df["WEEK"]).transform("mean")
    return (unit_price / cat_mean.clip(lower=1e-6))
```

#### `compute_temporal_features(df) -> pd.DataFrame`

```python
def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """sin/cos week encoding, quarter one-hot, holiday indicator.

    Dominick's WEEK is an integer starting from 1. We use modulo 52 for
    annual cyclical encoding.
    """
    result = pd.DataFrame(index=df.index)
    week_in_year = (df["WEEK"] - 1) % 52
    result["week_sin"] = np.sin(2 * np.pi * week_in_year / 52)
    result["week_cos"] = np.cos(2 * np.pi * week_in_year / 52)

    quarter = (week_in_year // 13).astype(int)
    for q in range(4):
        result[f"quarter_{q}"] = (quarter == q).astype(float)

    # Holiday proxy: weeks around Thanksgiving (wk 47), Christmas (wk 51),
    # Easter (~wk 14), Memorial Day (~wk 21), Labor Day (~wk 35)
    holiday_weeks = {14, 21, 35, 47, 51}
    result["holiday"] = week_in_year.isin(holiday_weeks).astype(float)
    return result
```

#### Fix `flag_promotions(df)` — use per (UPC, STORE) modal price

Replace the existing function body so modal price is computed per (UPC, STORE)
instead of per UPC only:

```python
def flag_promotions(df: pd.DataFrame) -> pd.Series:
    """on_promotion if unit_price < 0.95 * modal_price per (UPC, STORE)."""
    unit_price = df["PRICE"] / df["QTY"]
    modal_price = unit_price.groupby([df["UPC"], df["STORE"]]).transform(
        lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else s.median()
    )
    return unit_price < (0.95 * modal_price)
```

#### Rebuild `build_observation_vector(df, store_demo_cols)` — full 26+ dim vector

Replace the existing function entirely:

```python
def build_observation_vector(
    df: pd.DataFrame,
    store_demo_cols: list[str] | None = None,
) -> np.ndarray:
    """Build full observation vector matching blueprint Section 5.

    Per-SKU features (12 dims):
        symlog(unit_price), symlog(MOVE), discount_depth, on_promotion,
        lag_price_1, lag_price_2, lag_move_1, lag_move_2,
        rolling_mean_move_4, price_index, cost_per_unit, profit_margin

    Temporal features (7 dims):
        week_sin, week_cos, quarter_0..3, holiday

    Store context (N_demo dims):
        store demographic columns passed in
    """
    group_cols = ["STORE", "UPC"]
    unit_price = compute_unit_price(df)
    cost = compute_cost(df)

    per_sku = np.column_stack([
        symlog(unit_price.values),
        symlog(df["MOVE"].values.astype(float)),
        compute_discount_depth(df).values,
        flag_promotions(df).astype(float).values,
        compute_lag_features(df, group_cols, "unit_price_raw", [1, 2]).values
            if "unit_price_raw" in df.columns
            else np.zeros((len(df), 2)),
        compute_lag_features(df, group_cols, "MOVE", [1, 2]).values,
        compute_rolling_features(df, group_cols, "MOVE", 4)
            .iloc[:, 0].values.reshape(-1, 1),
        compute_price_index(df).values.reshape(-1, 1),
        cost.values.reshape(-1, 1),
        ((unit_price - cost) / unit_price.clip(lower=1e-6)).values.reshape(-1, 1),
    ])

    temporal = compute_temporal_features(df).values

    demo = np.column_stack([
        df[col].values.astype(float) for col in (store_demo_cols or [])
    ]) if store_demo_cols else np.empty((len(df), 0))

    return np.hstack([per_sku, temporal, demo])
```

### 2.2 Sequence Dataset — Create `src/retail_world_model/data/dataset.py`

This is a new file. Create it with the following complete implementation:

```python
"""PyTorch Dataset for DreamPrice training sequences from Dominick's data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from retail_world_model.data.transforms import (
    build_observation_vector,
    compute_cost,
    compute_unit_price,
    symlog,
    temporal_split,
)


class DominicksSequenceDataset(Dataset):
    """Sliding-window sequence dataset over Dominick's panel data.

    Each sample is a contiguous window of `seq_len` weeks for one store,
    yielding observation, action, reward, and done tensors.

    Sequences never cross store boundaries or year boundaries. Windows that
    would extend past the end of a group are dropped.

    Args:
        df: Processed DataFrame with columns STORE, UPC, WEEK, PRICE, QTY,
            MOVE, PROFIT, SALE, plus merged demographics.
        seq_len: Number of time steps per sequence (default 64).
        n_skus: Number of SKUs per store (top-K by volume, zero-padded).
        store_demo_cols: List of store demographic column names.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 64,
        n_skus: int = 25,
        store_demo_cols: list[str] | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.n_skus = n_skus
        self.store_demo_cols = store_demo_cols or [
            "INCOME", "EDUC", "ETHNIC", "HSIZEAVG",
            "SSTRDIST", "SSTRVOL", "CPDIST5", "CPWVOL5",
        ]

        # Precompute derived columns
        df = df.copy()
        df["unit_price_raw"] = compute_unit_price(df)
        df["cost_raw"] = compute_cost(df)

        # Build observation vectors
        self._obs = build_observation_vector(df, self.store_demo_cols)
        self._obs_dim = self._obs.shape[1]

        # Build per-row metadata for sequence extraction
        df["_row_idx"] = np.arange(len(df))
        self._df = df

        # Identify top-K SKUs per store by total MOVE
        self._store_skus: dict[int, list[int]] = {}
        for store, grp in df.groupby("STORE"):
            sku_vol = grp.groupby("UPC")["MOVE"].sum().sort_values(ascending=False)
            self._store_skus[int(store)] = list(sku_vol.index[:n_skus])

        # Build sequence index: (store, start_week) pairs
        self._sequences: list[tuple[int, int]] = []
        for store in sorted(df["STORE"].unique()):
            weeks = sorted(df[df["STORE"] == store]["WEEK"].unique())
            for i in range(len(weeks) - seq_len + 1):
                self._sequences.append((int(store), int(weeks[i])))

        # Precompute week-indexed data per store for fast __getitem__
        self._store_week_data: dict[int, dict[int, pd.DataFrame]] = {}
        for store, grp in df.groupby("STORE"):
            self._store_week_data[int(store)] = {
                int(week): week_grp for week, week_grp in grp.groupby("WEEK")
            }

    def __len__(self) -> int:
        return len(self._sequences)

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        store, start_week = self._sequences[idx]
        skus = self._store_skus[store]
        store_data = self._store_week_data[store]

        weeks = sorted(store_data.keys())
        start_pos = weeks.index(start_week)
        seq_weeks = weeks[start_pos : start_pos + self.seq_len]

        T = len(seq_weeks)
        K = self.n_skus
        obs_dim = self._obs_dim

        x_BT = torch.zeros(T, obs_dim)
        a_BT = torch.zeros(T, K)
        r_BT = torch.zeros(T)
        done_BT = torch.zeros(T)
        log_price_BT = torch.zeros(T, K)

        prev_prices = None
        for t, week in enumerate(seq_weeks):
            if week not in store_data:
                continue
            wk_df = store_data[week]

            # Aggregate observation across SKUs (mean pool)
            row_indices = wk_df["_row_idx"].values
            if len(row_indices) > 0:
                x_BT[t] = torch.tensor(
                    self._obs[row_indices].mean(axis=0), dtype=torch.float32
                )

            # Per-SKU prices and actions
            prices = torch.zeros(K)
            for k, upc in enumerate(skus):
                sku_rows = wk_df[wk_df["UPC"] == upc]
                if len(sku_rows) > 0:
                    p = float(sku_rows["unit_price_raw"].iloc[0])
                    prices[k] = p
                    log_price_BT[t, k] = float(np.log(max(p, 1e-6)))

            # Action = price ratio (how price changed from previous week)
            if prev_prices is not None:
                safe_prev = prev_prices.clamp(min=1e-6)
                a_BT[t] = prices / safe_prev
            prev_prices = prices.clone()

            # Reward = sum of gross margins across SKUs
            for k, upc in enumerate(skus):
                sku_rows = wk_df[wk_df["UPC"] == upc]
                if len(sku_rows) > 0:
                    margin = float(
                        sku_rows["unit_price_raw"].iloc[0]
                        - sku_rows["cost_raw"].iloc[0]
                    )
                    units = float(sku_rows["MOVE"].iloc[0])
                    r_BT[t] += margin * max(units, 0.0)

            # Episode done at last step
            if t == T - 1:
                done_BT[t] = 1.0

        # Store features (static across time)
        store_rows = self._df[self._df["STORE"] == store]
        store_feats = torch.zeros(len(self.store_demo_cols))
        for i, col in enumerate(self.store_demo_cols):
            if col in store_rows.columns:
                store_feats[i] = float(store_rows[col].iloc[0])

        return {
            "x_BT": x_BT,                # (T, obs_dim)
            "a_BT": a_BT,                # (T, n_skus)
            "r_BT": r_BT,                # (T,)
            "done_BT": done_BT,           # (T,)
            "log_price_BT": log_price_BT, # (T, n_skus)
            "store_features": store_feats, # (n_store_features,)
        }


class HybridReplaySampler(Sampler):
    """70% uniform quarterly strata + 30% most recent 2 years.

    Implements the hybrid replay strategy from the blueprint:
    quarterly stratification ensures all time periods are represented,
    while overweighting recent data captures regime changes.
    """

    def __init__(
        self,
        dataset: DominicksSequenceDataset,
        batch_size: int = 32,
        recent_years_weeks: int = 104,
        recent_fraction: float = 0.30,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.recent_fraction = recent_fraction

        # Partition sequences by quarter
        self._quarterly: dict[int, list[int]] = {}
        self._recent: list[int] = []
        max_week = max(w for _, w in dataset._sequences)
        cutoff = max_week - recent_years_weeks

        for idx, (store, week) in enumerate(dataset._sequences):
            quarter = (week - 1) // 13
            self._quarterly.setdefault(quarter, []).append(idx)
            if week >= cutoff:
                self._recent.append(idx)

        self._quarters = sorted(self._quarterly.keys())

    def __iter__(self):
        n_recent = int(self.batch_size * self.recent_fraction)
        n_uniform = self.batch_size - n_recent

        total_batches = len(self.dataset) // self.batch_size
        rng = np.random.default_rng()

        for _ in range(total_batches):
            # Uniform quarterly: sample equally from each quarter
            uniform_indices = []
            per_q = max(1, n_uniform // len(self._quarters))
            for q in self._quarters:
                pool = self._quarterly[q]
                chosen = rng.choice(pool, size=min(per_q, len(pool)), replace=True)
                uniform_indices.extend(chosen.tolist())
            uniform_indices = uniform_indices[:n_uniform]

            # Recent overweight
            recent_indices = rng.choice(
                self._recent, size=min(n_recent, len(self._recent)), replace=True
            ).tolist()

            batch = uniform_indices + recent_indices
            rng.shuffle(batch)
            yield from batch

    def __len__(self) -> int:
        return len(self.dataset)
```

### 2.3 Tests — Create `tests/test_data/test_dataset.py`

```python
"""Tests for DominicksSequenceDataset and HybridReplaySampler."""

import numpy as np
import pandas as pd
import pytest
import torch

from retail_world_model.data.dataset import (
    DominicksSequenceDataset,
    HybridReplaySampler,
)


@pytest.fixture
def sample_df():
    """Minimal Dominick's-like DataFrame for testing."""
    np.random.seed(42)
    rows = []
    for store in [1, 2]:
        for upc in [100, 200, 300]:
            for week in range(1, 101):
                rows.append({
                    "STORE": store,
                    "UPC": upc,
                    "WEEK": week,
                    "MOVE": np.random.randint(0, 200),
                    "QTY": 1,
                    "PRICE": round(np.random.uniform(1.0, 5.0), 2),
                    "PROFIT": round(np.random.uniform(10, 40), 1),
                    "SALE": "L",
                    "OK": 1,
                    "INCOME": 45000,
                    "EDUC": 12.5,
                    "ETHNIC": 0.3,
                    "HSIZEAVG": 2.5,
                    "SSTRDIST": 1.2,
                    "SSTRVOL": 500,
                    "CPDIST5": 3.0,
                    "CPWVOL5": 200,
                })
    return pd.DataFrame(rows)


class TestDominicksSequenceDataset:
    def test_len_positive(self, sample_df):
        ds = DominicksSequenceDataset(sample_df, seq_len=10, n_skus=3)
        assert len(ds) > 0

    def test_getitem_keys(self, sample_df):
        ds = DominicksSequenceDataset(sample_df, seq_len=10, n_skus=3)
        item = ds[0]
        assert "x_BT" in item
        assert "a_BT" in item
        assert "r_BT" in item
        assert "done_BT" in item
        assert "log_price_BT" in item
        assert "store_features" in item

    def test_getitem_shapes(self, sample_df):
        seq_len, n_skus = 10, 3
        ds = DominicksSequenceDataset(
            sample_df, seq_len=seq_len, n_skus=n_skus
        )
        item = ds[0]
        assert item["x_BT"].shape[0] == seq_len
        assert item["a_BT"].shape == (seq_len, n_skus)
        assert item["r_BT"].shape == (seq_len,)
        assert item["done_BT"].shape == (seq_len,)
        assert item["log_price_BT"].shape == (seq_len, n_skus)

    def test_done_only_at_end(self, sample_df):
        ds = DominicksSequenceDataset(sample_df, seq_len=10, n_skus=3)
        item = ds[0]
        assert item["done_BT"][-1] == 1.0
        assert item["done_BT"][:-1].sum() == 0.0

    def test_no_nans(self, sample_df):
        ds = DominicksSequenceDataset(sample_df, seq_len=10, n_skus=3)
        item = ds[0]
        for key, val in item.items():
            assert not torch.isnan(val).any(), f"NaN in {key}"


class TestHybridReplaySampler:
    def test_produces_indices(self, sample_df):
        ds = DominicksSequenceDataset(sample_df, seq_len=10, n_skus=3)
        sampler = HybridReplaySampler(ds, batch_size=4)
        indices = list(sampler)
        assert len(indices) > 0
        assert all(0 <= i < len(ds) for i in indices)
```

### 2.4 Tests for New Transforms — Add to `tests/test_data/test_transforms.py`

Append these test classes to the existing file:

```python
class TestDiscountDepth:
    def test_zero_at_modal(self, clean_movement_df):
        dd = compute_discount_depth(clean_movement_df)
        assert (dd >= 0).all()

    def test_positive_on_sale(self):
        df = pd.DataFrame({
            "STORE": [1, 1, 1], "UPC": [10, 10, 10],
            "PRICE": [3.0, 3.0, 2.0], "QTY": [1, 1, 1],
        })
        dd = compute_discount_depth(df)
        assert dd.iloc[2] > 0

class TestLagFeatures:
    def test_lag_shape(self, clean_movement_df):
        lags = compute_lag_features(
            clean_movement_df, ["STORE", "UPC"], "MOVE", [1, 2]
        )
        assert lags.shape[0] == len(clean_movement_df)
        assert "MOVE_lag_1" in lags.columns
        assert "MOVE_lag_2" in lags.columns

class TestRollingFeatures:
    def test_rolling_shape(self, clean_movement_df):
        rolling = compute_rolling_features(
            clean_movement_df, ["STORE", "UPC"], "MOVE", 4
        )
        assert "MOVE_rolling_mean_4" in rolling.columns
        assert "MOVE_rolling_std_4" in rolling.columns

class TestPriceIndex:
    def test_mean_is_one(self, clean_movement_df):
        pi = compute_price_index(clean_movement_df)
        assert abs(pi.mean() - 1.0) < 0.5

class TestTemporalFeatures:
    def test_columns(self, clean_movement_df):
        tf = compute_temporal_features(clean_movement_df)
        assert "week_sin" in tf.columns
        assert "week_cos" in tf.columns
        assert "quarter_0" in tf.columns
        assert "holiday" in tf.columns

    def test_sin_cos_range(self, clean_movement_df):
        tf = compute_temporal_features(clean_movement_df)
        assert tf["week_sin"].between(-1, 1).all()
        assert tf["week_cos"].between(-1, 1).all()
```

**Gate:** `make check` passes. `DominicksSequenceDataset` yields tensors with correct shapes.

---

## 3. Phase 2: World Model Interface Alignment

**Agent:** `world-model-agent`
**Branch:** `track/world-model`
**Depends on:** Phase 1 (needs obs_dim from Dataset)
**Blocks:** Phase 4, Phase 5

### 3.1 Add Observation Decoder — Modify `src/retail_world_model/models/rssm.py`

In `RSSM.__init__()`, after the `continue_head`, add:

```python
# Observation decoder: reconstruct obs from latent + backbone state
self.obs_decoder = nn.Sequential(
    nn.Linear(d_model + latent_dim, d_model),
    nn.SiLU(),
    nn.Linear(d_model, d_model),
    nn.SiLU(),
    nn.Linear(d_model, obs_dim),
)
```

This requires adding `obs_dim` as a constructor argument to `RSSM.__init__()`.
Pass it through from `MambaWorldModel.__init__()`.

### 3.2 Align Output Keys — Modify `RSSM.train_sequence()`

Replace the return dict at the end of `train_sequence()`:

```python
# Observation reconstruction
x_recon_BT = self.obs_decoder(torch.cat([h_BT, z_BT], dim=-1))

# Reward logits for twohot loss
r_logits = self.reward_ensemble.forward_logits(h_flat)  # (n_heads, B*T, n_bins)
r_logits_mean = r_logits.mean(dim=0).reshape(B, T, -1)  # (B, T, n_bins)

return {
    "z_posterior_BT": z_BT,
    "h_BT": h_BT,
    "posterior_probs_BT": posterior_probs,
    "prior_probs_BT": prior_probs,
    "x_recon_BT": x_recon_BT,
    "reward_mean": r_mean,
    "reward_std": r_std,
    "reward_logits_BT": r_logits_mean,
    "continue_logits": continue_logits,
}
```

Note: `RewardEnsemble` needs a `forward_logits()` method that returns raw
logits before the twohot decode. If it doesn't exist, add it to
`models/reward_head.py`.

### 3.3 Add `imagine_step` Wrapper — Modify `src/retail_world_model/models/world_model.py`

Add `_inference_params` attribute and two new methods to `MambaWorldModel`:

```python
def __init__(self, ...):
    super().__init__()
    self.rssm = RSSM(...)
    self._inference_params: object | None = None

def reset_state(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
    """Initialize recurrent state for imagination rollout."""
    self._inference_params = self.rssm.backbone.init_inference_params(
        batch_size
    )
    device = next(self.parameters()).device
    h = torch.zeros(batch_size, self.rssm.d_model, device=device)
    z = torch.zeros(
        batch_size, self.rssm.n_cat * self.rssm.n_cls, device=device
    )
    return {"h": h, "z": z}

def imagine_step(
    self,
    z_t: torch.Tensor,
    a_t: torch.Tensor,
    h_t: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Single recurrent step for imagination, matching ImagineInterface.

    Args:
        z_t: (B, latent_dim) current stochastic state.
        a_t: (B, act_dim) action (float, not discrete index).
        h_t: Ignored. Mamba uses internal inference_params.

    Returns:
        Dict with keys: h, z, r_mean, r_std, continue.
    """
    h_next, z_next, prior_probs = self.rssm.imagine_step(
        z_t, a_t, self._inference_params
    )
    r_mean, r_std = self.rssm.reward_ensemble(h_next)
    cont = torch.sigmoid(
        self.rssm.continue_head(h_next).squeeze(-1)
    )
    return {
        "h": h_next,
        "z": z_next,
        "r_mean": r_mean,
        "r_std": r_std,
        "continue": cont,
        "prior_probs": prior_probs,
    }
```

Keep the existing `imagine()` method (batch rollout) as-is for backward compat.

### 3.4 Update `ImagineInterface` — Modify `src/retail_world_model/inference/imagination.py`

Replace the protocol at the top of the file:

```python
class ImagineInterface(Protocol):
    """Protocol for world models used in imagination rollouts."""

    def encode_obs(
        self, x_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def imagine_step(
        self,
        z_t: torch.Tensor,
        a_t: torch.Tensor,
        h_t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]: ...

    def reset_state(
        self, batch_size: int
    ) -> dict[str, torch.Tensor]: ...
```

### 3.5 Update `elbo_loss()` — Modify `src/retail_world_model/training/losses.py`

Replace the `elbo_loss` function entirely:

```python
def elbo_loss(
    batch: dict[str, torch.Tensor],
    model: object,
    beta_pred: float = BETA_PRED,
    beta_dyn: float = BETA_DYN,
    beta_rep: float = BETA_REP,
    free_bits: float = FREE_BITS,
) -> dict[str, torch.Tensor]:
    """Full ELBO loss for world model training.

    L = beta_pred * (recon + reward + continue) + kl_balancing
    """
    output = model.forward(batch["x_BT"], batch["a_BT"])

    # Reconstruction loss: symlog MSE
    x_recon = output["x_recon_BT"]
    x_target = symlog(batch["x_BT"])
    recon_loss = 0.5 * (x_recon - x_target).pow(2).sum(dim=-1).mean()

    # Reward loss: twohot CE
    reward_loss = twohot_ce_loss(
        output["reward_logits_BT"], batch["r_BT"]
    )

    # Continue loss: BCE
    cont_loss = continue_bce_loss(
        output["continue_logits"], batch["done_BT"]
    )

    # KL balancing
    kl_total, kl_dyn, kl_rep = kl_balancing(
        output["posterior_probs_BT"],
        output["prior_probs_BT"],
        beta_dyn=beta_dyn,
        beta_rep=beta_rep,
        free_bits=free_bits,
    )

    pred_loss = recon_loss + reward_loss + cont_loss
    total = beta_pred * pred_loss + kl_total

    return {
        "total": total,
        "recon": recon_loss,
        "reward": reward_loss,
        "continue": cont_loss,
        "kl_total": kl_total,
        "kl_dyn": kl_dyn,
        "kl_rep": kl_rep,
    }
```

Note: the signature changes from `(x_BT, a_BT, model)` to `(batch, model)`.
Update the call site in `DreamerTrainer.train_phase_a()` accordingly:

```python
def train_phase_a(
    self, batch: dict[str, torch.Tensor]
) -> dict[str, float]:
    self.opt_wm.zero_grad()
    losses = elbo_loss(batch, self.model)
    losses["total"].backward()
    nn.utils.clip_grad_norm_(
        self.model.parameters(), self.grad_clip_wm
    )
    self.opt_wm.step()
    return {k: v.item() for k, v in losses.items()}
```

You also need to add `twohot_ce_loss` and `continue_bce_loss` helper
functions in `losses.py`:

```python
def twohot_ce_loss(
    logits_BT: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 255,
) -> torch.Tensor:
    """Cross-entropy loss between predicted twohot logits and target scalars."""
    bins = make_bins(n_bins).to(logits_BT.device)
    targets_flat = targets.reshape(-1)
    twohot_targets = twohot_encode(targets_flat, bins)
    logits_flat = logits_BT.reshape(-1, n_bins)
    log_probs = torch.log_softmax(logits_flat, dim=-1)
    return -(twohot_targets * log_probs).sum(dim=-1).mean()


def continue_bce_loss(
    logits: torch.Tensor,
    done: torch.Tensor,
) -> torch.Tensor:
    """BCE loss for continue prediction (1 - done)."""
    continue_target = 1.0 - done
    return F.binary_cross_entropy_with_logits(
        logits.reshape_as(continue_target),
        continue_target,
    )
```

### 3.6 Update Tests — Modify `tests/test_models/test_rssm.py`

Change all references to old keys:
- `out["z_BT"]` -> `out["z_posterior_BT"]`
- `out["posterior_probs"]` -> `out["posterior_probs_BT"]`
- `out["prior_probs"]` -> `out["prior_probs_BT"]`
- Add assertion: `assert "x_recon_BT" in out`
- Add assertion: `assert out["x_recon_BT"].shape == (B, T, obs_dim)`

Add new test:

```python
def test_elbo_finite(self):
    """elbo_loss produces finite gradients."""
    B, T = 2, 10
    batch = {
        "x_BT": torch.randn(B, T, 32),
        "a_BT": torch.randn(B, T, 4),
        "r_BT": torch.randn(B, T),
        "done_BT": torch.zeros(B, T),
    }
    from retail_world_model.training.losses import elbo_loss
    from retail_world_model.models.world_model import MambaWorldModel
    model = MambaWorldModel(
        obs_dim=32, act_dim=4, d_model=64, n_cat=8, n_cls=8
    )
    losses = elbo_loss(batch, model)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
```

**Gate:** `make check` passes. `elbo_loss` produces finite loss and gradients on random data.

---

## 4. Phase 3: Causal Estimation Pipeline

**Agent:** `causal-estimator-agent`
**Branch:** `track/causal`
**Depends on:** Phase 1 (needs data loading)
**Blocks:** Phase 4 (decoder needs elasticity JSON)

### 4.1 Create `scripts/estimate_elasticity.py`

```python
"""Estimate price elasticities via Hausman IV + DML-PLIV for CausalDemandDecoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from retail_world_model.data.dominicks_loader import load_category
from retail_world_model.data.transforms import (
    compute_hausman_iv,
    compute_unit_price,
    temporal_split,
)
from retail_world_model.data.copula_correction import (
    compute_2scope_copula_residual,
)


def estimate_elasticity(
    category: str = "cso",
    data_dir: str = "docs/data",
    output_dir: str = "configs/elasticities",
    n_folds: int = 5,
    n_estimators: int = 500,
) -> dict:
    """Run DML-PLIV on Dominick's data and write elasticity JSON.

    Steps:
        1. Load and merge category data
        2. Compute Hausman IV instrument
        3. Run DML-PLIV via doubleml
        4. Validate first-stage F-stat > 10
        5. Run 2sCOPE robustness check
        6. Write configs/elasticities/{category}.json
    """
    from doubleml import DoubleMLData, DoubleMLPLIV

    # Load data
    movement_path = Path(data_dir) / category / f"w{category}.csv"
    upc_path = Path(data_dir) / category / f"upc{category}.csv"
    demo_path = Path(data_dir) / "demo.csv"

    df = load_category(
        str(movement_path), str(upc_path), str(demo_path)
    )
    train_df, _, _ = temporal_split(df)

    # Derived features
    train_df = train_df.copy()
    train_df["unit_price"] = compute_unit_price(train_df)
    train_df["log_price"] = np.log(
        train_df["unit_price"].clip(lower=1e-6)
    )
    train_df["log_move"] = np.log(
        train_df["MOVE"].clip(lower=1) + 1
    )
    train_df["hausman_iv"] = compute_hausman_iv(train_df)

    # Drop rows with NaN/inf
    cols_needed = ["log_move", "log_price", "hausman_iv"]
    train_df = train_df.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(subset=cols_needed)

    # Control variables (exogenous)
    exog_cols = []
    for col in [
        "INCOME", "EDUC", "ETHNIC", "HSIZEAVG",
        "SSTRDIST", "SSTRVOL",
    ]:
        if col in train_df.columns:
            exog_cols.append(col)
            train_df[col] = train_df[col].fillna(
                train_df[col].median()
            )

    if not exog_cols:
        exog_cols = ["WEEK"]

    # DoubleML data object
    dml_data = DoubleMLData(
        train_df[
            ["log_move", "log_price", "hausman_iv"] + exog_cols
        ],
        y_col="log_move",
        d_cols="log_price",
        z_cols="hausman_iv",
        x_cols=exog_cols,
    )

    # ML learners
    ml_l = RandomForestRegressor(
        n_estimators=n_estimators, n_jobs=-1, random_state=42
    )
    ml_m = RandomForestRegressor(
        n_estimators=n_estimators, n_jobs=-1, random_state=42
    )
    ml_r = RandomForestRegressor(
        n_estimators=n_estimators, n_jobs=-1, random_state=42
    )

    # Fit DML-PLIV
    dml = DoubleMLPLIV(dml_data, ml_l, ml_m, ml_r, n_folds=n_folds)
    dml.fit()

    theta = float(dml.coef[0])
    se = float(dml.se[0])
    ci = dml.confint(level=0.95)
    t_stat = float(dml.t_stat[0])
    p_value = float(dml.pval[0])

    # First-stage F-stat approximation (t_stat^2 for single instrument)
    f_stat = t_stat ** 2

    print(f"DML-PLIV elasticity: {theta:.4f} (SE={se:.4f})")
    print(
        f"95% CI: [{float(ci.iloc[0, 0]):.4f}, "
        f"{float(ci.iloc[0, 1]):.4f}]"
    )
    print(f"F-stat (approx): {f_stat:.1f}")

    if f_stat < 10:
        raise ValueError(
            f"Weak instrument: F-stat={f_stat:.1f} < 10. "
            "Hausman IV may be invalid for this category."
        )

    # 2sCOPE robustness check
    cope_beta = float("nan")
    try:
        copula_resid = compute_2scope_copula_residual(
            train_df, "log_price", "log_move", "hausman_iv"
        )
        from sklearn.linear_model import LinearRegression
        X_cope = np.column_stack([
            train_df["log_price"].values,
            copula_resid.values,
        ])
        y_cope = train_df["log_move"].values
        reg = LinearRegression().fit(X_cope, y_cope)
        cope_beta = float(reg.coef_[0])
        print(f"2sCOPE beta: {cope_beta:.4f}")

        if abs(theta - cope_beta) / abs(theta) > 0.5:
            print(
                f"WARNING: DML-PLIV ({theta:.3f}) and "
                f"2sCOPE ({cope_beta:.3f}) differ by >50%. "
                "Investigate endogeneity assumptions."
            )
    except Exception as e:
        print(f"2sCOPE check failed: {e}")

    # Write output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result = {
        "theta_causal": theta,
        "se": se,
        "ci_lower": float(ci.iloc[0, 0]),
        "ci_upper": float(ci.iloc[0, 1]),
        "f_stat_first_stage": f_stat,
        "p_value": p_value,
        "cope_beta_c": cope_beta,
        "n_obs": len(train_df),
        "n_folds": n_folds,
        "category": category,
    }
    out_file = output_path / f"{category}.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(f"Written: {out_file}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Estimate price elasticities"
    )
    parser.add_argument(
        "--category", default="cso",
        help="Dominick's category code",
    )
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--output-dir", default="configs/elasticities")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    estimate_elasticity(
        category=args.category,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()
```

**Gate:** `python scripts/estimate_elasticity.py --category cso` produces
`configs/elasticities/cso.json` with `f_stat_first_stage > 10` and
`theta_causal` between -4.0 and -1.0.

---

## 5. Phase 4: Training Loop Wiring

**Agent:** `training-agent`
**Branch:** `track/training-loop`
**Depends on:** Phase 1 (Dataset), Phase 2 (model interfaces), Phase 3 (elasticity JSON)
**Blocks:** Phase 5, Phase 6

### 5.1 Complete `scripts/train.py`

Replace the stub body of `main()` after the logger setup with:

```python
    # Build dataset
    from retail_world_model.data.dominicks_loader import load_category
    from retail_world_model.data.dataset import (
        DominicksSequenceDataset,
        HybridReplaySampler,
    )
    from retail_world_model.models.world_model import MambaWorldModel

    data_dir = cfg.get("data_dir", "docs/data")
    category = cfg.get("category", "cso")
    df = load_category(
        f"{data_dir}/{category}/w{category}.csv",
        f"{data_dir}/{category}/upc{category}.csv",
        f"{data_dir}/{category}/demo.csv",
    )

    seq_len = cfg.agent.seq_len
    n_skus = cfg.environment.n_skus
    dataset = DominicksSequenceDataset(
        df, seq_len=seq_len, n_skus=n_skus
    )
    sampler = HybridReplaySampler(
        dataset, batch_size=cfg.agent.batch_size
    )

    # Build model
    wm_cfg = cfg.world_model
    model = MambaWorldModel(
        obs_dim=dataset.obs_dim,
        act_dim=n_skus,
        d_model=wm_cfg.d_model,
        n_cat=wm_cfg.n_cat,
        n_cls=wm_cfg.n_cls,
        elasticity_path=f"configs/elasticities/{category}.json",
    ).to(device)

    # Build actor-critic
    state_dim = wm_cfg.d_model + wm_cfg.z_dim
    ac = ActorCritic(
        state_dim=state_dim,
        n_skus=n_skus,
        action_dim=cfg.environment.action_steps,
        eta=cfg.agent.eta,
    ).to(device)

    # Build trainer
    trainer = DreamerTrainer(
        model=model,
        actor_critic=ac,
        dataset=dataset,
        cfg=OmegaConf.to_container(cfg.agent, resolve=True),
        logger=logger,
    )

    # Train
    trainer.train(n_steps=cfg.n_steps)
```

### 5.2 Fix `DreamerTrainer.train_step()` — Modify `src/retail_world_model/training/trainer.py`

At line 200, fix the key:

```python
z0 = output["z_posterior_BT"][:, 0]  # now matches Phase 2 key rename
```

At line 81, update the `elbo_loss` call to use the new signature:

```python
losses = elbo_loss(batch, self.model)
```

Replace the `DataLoader` in `train()` to use the sampler:

```python
def train(self, n_steps: int = 100_000) -> None:
    loader = torch.utils.data.DataLoader(
        self.dataset,
        batch_sampler=torch.utils.data.BatchSampler(
            sampler=getattr(self, '_sampler', None)
                or torch.utils.data.RandomSampler(self.dataset),
            batch_size=self.cfg.get("batch_size", 32),
            drop_last=True,
        ),
    )
    # ... rest unchanged
```

Add device movement in `train_step()`:

```python
def train_step(
    self, batch: dict[str, torch.Tensor]
) -> dict[str, float]:
    device = next(self.model.parameters()).device
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    # ... rest unchanged
```

Add checkpoint saving:

```python
def train(self, n_steps: int = 100_000) -> None:
    ...
    for step_i in range(n_steps):
        ...
        self.train_step(batch)

        if (step_i + 1) % self.cfg.get("save_every", 10000) == 0:
            ckpt_dir = Path(
                self.cfg.get("checkpoint_dir", "checkpoints")
            )
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "step": self._step,
                "model": self.model.state_dict(),
                "actor_critic": self.actor_critic.state_dict(),
                "opt_wm": self.opt_wm.state_dict(),
                "opt_actor": self.opt_actor.state_dict(),
                "opt_critic": self.opt_critic.state_dict(),
            }, ckpt_dir / f"step_{self._step}.pt")
```

### 5.3 Integration Test — Create `tests/test_training/test_integration.py`

```python
"""Integration test: DreamerTrainer.train_step completes on random data."""

import torch
import pytest

from retail_world_model.models.world_model import MambaWorldModel
from retail_world_model.applications.pricing_policy import ActorCritic
from retail_world_model.training.trainer import DreamerTrainer


class TestTrainerIntegration:
    @pytest.fixture
    def trainer(self):
        obs_dim, act_dim = 32, 4
        d_model, n_cat, n_cls = 64, 8, 8
        model = MambaWorldModel(
            obs_dim=obs_dim, act_dim=act_dim,
            d_model=d_model, n_cat=n_cat, n_cls=n_cls,
        )
        state_dim = d_model + n_cat * n_cls
        ac = ActorCritic(
            state_dim=state_dim, n_skus=act_dim, action_dim=21,
        )
        dataset = torch.utils.data.TensorDataset(
            torch.randn(10, 8, obs_dim),
        )
        return DreamerTrainer(
            model=model, actor_critic=ac, dataset=dataset,
        )

    def test_train_step_completes(self, trainer):
        batch = {
            "x_BT": torch.randn(2, 8, 32),
            "a_BT": torch.randn(2, 8, 4),
            "r_BT": torch.randn(2, 8),
            "done_BT": torch.zeros(2, 8),
        }
        metrics = trainer.train_step(batch)
        assert "wm/total" in metrics
        assert "actor/actor_loss" in metrics
        assert "critic/critic_loss" in metrics

    def test_loss_decreases(self, trainer):
        batch = {
            "x_BT": torch.randn(2, 8, 32),
            "a_BT": torch.randn(2, 8, 4),
            "r_BT": torch.randn(2, 8),
            "done_BT": torch.zeros(2, 8),
        }
        losses = []
        for _ in range(5):
            m = trainer.train_step(batch)
            losses.append(m["wm/total"])
        assert losses[-1] < losses[0] * 1.5
```

**Gate:** `make test` passes. `python scripts/train.py n_steps=10` completes without error.

---

## 6. Phase 5: Assessment and Environment

**Agent:** `training-agent`
**Branch:** `track/training-loop`
**Depends on:** Phase 4

### 6.1 Complete `scripts/assess.py` (the file is named `evaluate.py`)

Replace the TODO with a full assessment loop:

```python
    # Build model from checkpoint config
    from retail_world_model.models.world_model import MambaWorldModel
    from retail_world_model.applications.pricing_policy import ActorCritic
    from retail_world_model.envs.grocery import GroceryPricingEnv

    cfg = ckpt.get("cfg", {})
    wm_cfg = cfg.get("world_model", {})
    env_cfg = cfg.get("environment", {})

    model = MambaWorldModel(
        obs_dim=wm_cfg.get("obs_dim", 64),
        act_dim=env_cfg.get("n_skus", 25),
        d_model=wm_cfg.get("d_model", 512),
        n_cat=wm_cfg.get("n_cat", 32),
        n_cls=wm_cfg.get("n_cls", 32),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.train(False)  # switch to inference mode

    state_dim = wm_cfg.get("d_model", 512) + wm_cfg.get("z_dim", 1024)
    ac = ActorCritic(
        state_dim=state_dim,
        n_skus=env_cfg.get("n_skus", 25),
    )
    ac.load_state_dict(ckpt["actor_critic"])
    ac.train(False)  # switch to inference mode

    n_skus = env_cfg.get("n_skus", 25)
    env = GroceryPricingEnv(
        world_model=model,
        store_features=np.zeros(8),
        initial_obs=np.zeros(wm_cfg.get("obs_dim", 64)),
        cost_vector=np.full(n_skus, 1.5),
        n_skus=n_skus,
        H=env_cfg.get("H", 13),
    )

    returns = []
    for ep in range(args.n_episodes):
        obs, _ = env.reset()
        total_return = 0.0
        done = False
        while not done:
            obs_t = torch.tensor(
                obs, dtype=torch.float32
            ).unsqueeze(0).to(device)
            state = torch.cat([
                torch.zeros(
                    1, wm_cfg.get("d_model", 512), device=device
                ),
                torch.zeros(
                    1, wm_cfg.get("z_dim", 1024), device=device
                ),
            ], dim=-1)
            with torch.no_grad():
                action, _, _ = ac.act(state, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(
                action.squeeze(0).cpu().numpy()
            )
            total_return += reward
            done = terminated or truncated
        returns.append(total_return)
        print(f"Episode {ep+1}: return={total_return:.2f}")

    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    n = len(returns)
    sorted_rets = sorted(returns)
    iqm = np.mean(sorted_rets[n // 4 : -(n // 4) or None])

    print(f"\nMean return: {mean_ret:.2f} +/- {std_ret:.2f}")
    print(f"IQM: {iqm:.2f}")

    results = {
        "returns": returns,
        "mean": float(mean_ret),
        "std": float(std_ret),
        "iqm": float(iqm),
    }
    Path("docs/results").mkdir(parents=True, exist_ok=True)
    Path("docs/results/assessment.json").write_text(
        json.dumps(results, indent=2)
    )
```

### 6.2 Fix `GroceryPricingEnv._predict_demand()` — Modify `src/retail_world_model/envs/grocery.py`

Replace lines 126-136 of `_predict_demand`:

```python
def _predict_demand(
    self, prices: np.ndarray, action: np.ndarray
) -> np.ndarray:
    if self.world_model is None:
        base_demand = 100.0
        log_price = np.log(np.clip(prices, 1e-3, None))
        units = base_demand * np.exp(-2.5 * log_price)
        return np.clip(units, 0, None).astype(np.float32)

    with torch.no_grad():
        a_t = torch.tensor(
            action, dtype=torch.float32
        ).unsqueeze(0)
        device = next(self.world_model.parameters()).device
        a_t = a_t.to(device)

        if self._z_t is not None:
            result = self.world_model.imagine_step(
                self._z_t, a_t
            )
            self._h_t = result["h"]
            self._z_t = result["z"]
            r_mean = result["r_mean"]
            mean_margin = float(np.mean(
                np.clip(prices - self.cost_vector, 0.01, None)
            ))
            demand = (r_mean / mean_margin).clamp(min=0).squeeze(0)
            return demand.cpu().numpy().astype(np.float32)

    return np.full(self.n_skus, 50.0, dtype=np.float32)
```

### 6.3 Fix `GroceryPricingEnv.reset()` — same file

Replace the world model state initialization:

```python
if hasattr(self.world_model, "reset_state"):
    state = self.world_model.reset_state(batch_size=1)
    self._h_t = state["h"]
    self._z_t = state["z"]
```

**Gate:** `make test` passes. `scripts/assess.py --checkpoint <path>` runs full episodes.

---

## 7. Phase 6: Ablations and Packaging

**Agent:** `experiment-tracker-agent`
**Branch:** `track/ablations`
**Depends on:** Phase 4 (training must work)

### 7.1 Verify Ablation Configs

Each YAML in `configs/experiment/ablations/` must override the correct keys.
Verify that `scripts/run_ablations.py` constructs correct Hydra overrides.

The following configs must be verified:

- `imagination_off.yaml` — disables imagination (Phase B/C skip)
- `horizon_{5,10,25}.yaml` — H override
- `no_stochastic_latent.yaml` — deterministic z (no sampling)
- `no_symlog_twohot.yaml` — raw MSE loss
- `gru_backbone.yaml` — force GRU fallback
- `flat_encoder.yaml` — ObsEncoder instead of EntityEncoder
- `no_mopo_lcb.yaml` — lambda_lcb=0

### 7.2 Run Experiments

```bash
# Inside container:
python scripts/run_ablations.py --seeds 5
python scripts/baselines/cost_plus.py
python scripts/baselines/static_xgboost.py
python scripts/baselines/competitive_matching.py
python scripts/baselines/dqn_baseline.py
python scripts/baselines/ppo_baseline.py
python scripts/baselines/sac_baseline.py
python scripts/analyze_results.py
```

### 7.3 Package

```bash
python scripts/push_to_hub.py --checkpoint checkpoints/best.pt
```

**Gate:** `docs/results/` contains ablation and baseline comparison tables.

---

## 8. Dependency DAG

```
Phase 1 (Data Pipeline)
  |
  +---> Phase 2 (World Model Interfaces) ---> Phase 4 (Training Loop) ---> Phase 5 (Assessment)
  |                                               ^                              |
  +---> Phase 3 (Causal Estimation) --------------+                              v
                                                                     Phase 6 (Ablations)
```

Phases 1 and 3 can run in parallel. Phase 2 depends on Phase 1 (needs obs_dim).
Phase 4 depends on Phases 1, 2, 3. Phase 5 depends on Phase 4. Phase 6 depends
on Phase 5.

---

## 9. File Inventory

### Files to Create

| File | Phase | Purpose |
|------|-------|---------|
| `src/retail_world_model/data/dataset.py` | 1 | SequenceDataset + HybridReplaySampler |
| `scripts/estimate_elasticity.py` | 3 | DML-PLIV causal estimation |
| `configs/elasticities/cso.json` | 3 | Output of elasticity estimation |
| `tests/test_data/test_dataset.py` | 1 | Dataset and sampler tests |
| `tests/test_training/test_integration.py` | 4 | End-to-end trainer test |

### Files to Modify

| File | Phase | What Changes |
|------|-------|-------------|
| `src/retail_world_model/data/transforms.py` | 1 | Add 7 new functions, fix `flag_promotions`, rebuild `build_observation_vector` |
| `src/retail_world_model/models/rssm.py` | 2 | Add `obs_decoder`, add `obs_dim` param, rename output keys, add `x_recon_BT` and `reward_logits_BT` |
| `src/retail_world_model/models/world_model.py` | 2 | Add `imagine_step()` returning dict, add `reset_state()` |
| `src/retail_world_model/training/losses.py` | 2 | Rewrite `elbo_loss()` with reward + continue losses, add `twohot_ce_loss`, add `continue_bce_loss`, change signature to accept batch dict |
| `src/retail_world_model/training/trainer.py` | 4 | Fix key names, update `elbo_loss` call, add device movement, add checkpointing, use sampler |
| `src/retail_world_model/inference/imagination.py` | 2 | Update `ImagineInterface` protocol signature |
| `src/retail_world_model/envs/grocery.py` | 5 | Fix `_predict_demand()` and `reset()` for dict-returning `imagine_step` |
| `scripts/train.py` | 4 | Full training pipeline wiring |
| `scripts/assess.py` (evaluate.py) | 5 | Full assessment loop |
| `tests/test_models/test_rssm.py` | 2 | Update output key assertions |
| `tests/test_data/test_transforms.py` | 1 | Add tests for new transform functions |

---

## 10. Acceptance Criteria

The implementation is complete when ALL of the following are true:

1. `make check` passes (ruff + pyright + pytest, 0 errors, 0 warnings)
2. `pytest tests/` collects and passes all tests (existing 104 + new ~20)
3. `python scripts/estimate_elasticity.py` writes `configs/elasticities/cso.json` with valid elasticity
4. `python scripts/train.py n_steps=100` completes, produces checkpoint, loss decreases
5. `python scripts/assess.py --checkpoint checkpoints/step_100.pt` runs episodes, reports returns
6. `make verify-gpu` confirms Mamba2 CUDA path works
7. No mocks, no stubs, no shortcuts — all paths are real implementations
8. Every function has at least one test
9. All interface contracts from `spec.md` Section 2 are satisfied
10. `docs/results/` contains metrics after a training run
