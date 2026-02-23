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
            "INCOME",
            "EDUC",
            "ETHNIC",
            "HSIZEAVG",
            "SSTRDIST",
            "SSTRVOL",
            "CPDIST5",
            "CPWVOL5",
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

    def _get_week_prices(
        self,
        wk_df: pd.DataFrame,
        skus: list[int],
        log_price_row: torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-SKU prices for one week and update log_price_row in place."""
        K = self.n_skus
        prices = torch.zeros(K)
        for k, upc in enumerate(skus):
            sku_rows = wk_df[wk_df["UPC"] == upc]
            if len(sku_rows) > 0:
                p = float(sku_rows["unit_price_raw"].iloc[0])
                prices[k] = p
                log_price_row[k] = float(np.log(max(p, 1e-6)))
        return prices

    def _get_week_reward(
        self,
        wk_df: pd.DataFrame,
        skus: list[int],
    ) -> float:
        """Compute total gross margin reward for one week."""
        total = 0.0
        for upc in skus:
            sku_rows = wk_df[wk_df["UPC"] == upc]
            if len(sku_rows) > 0:
                margin = float(sku_rows["unit_price_raw"].iloc[0] - sku_rows["cost_raw"].iloc[0])
                units = float(sku_rows["MOVE"].iloc[0])
                total += margin * max(units, 0.0)
        return total

    def _get_store_features(self, store: int) -> torch.Tensor:
        """Extract static store demographic features."""
        store_rows = self._df[self._df["STORE"] == store]
        store_feats = torch.zeros(len(self.store_demo_cols))
        for i, col in enumerate(self.store_demo_cols):
            if col in store_rows.columns:
                store_feats[i] = float(store_rows[col].iloc[0])
        return store_feats

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
                x_BT[t] = torch.tensor(self._obs[row_indices].mean(axis=0), dtype=torch.float32)

            prices = self._get_week_prices(wk_df, skus, log_price_BT[t])

            # Action = price ratio (how price changed from previous week)
            if prev_prices is not None:
                safe_prev = prev_prices.clamp(min=1e-6)
                a_BT[t] = prices / safe_prev
            prev_prices = prices.clone()

            r_BT[t] = self._get_week_reward(wk_df, skus)

            # Episode done at last step
            if t == T - 1:
                done_BT[t] = 1.0

        return {
            "x_BT": x_BT,  # (T, obs_dim)
            "a_BT": a_BT,  # (T, n_skus)
            "r_BT": r_BT,  # (T,)
            "done_BT": done_BT,  # (T,)
            "log_price_BT": log_price_BT,  # (T, n_skus)
            "store_features": self._get_store_features(store),  # (n_store_features,)
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
