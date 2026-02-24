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

        df = df.copy()
        df["unit_price_raw"] = compute_unit_price(df)
        df["cost_raw"] = compute_cost(df)

        self._obs = build_observation_vector(df, self.store_demo_cols)
        self._obs_dim = self._obs.shape[1]

        df["_row_idx"] = np.arange(len(df))

        store_skus: dict[int, list[int]] = {}
        for store, grp in df.groupby("STORE"):
            sku_vol = grp.groupby("UPC")["MOVE"].sum().sort_values(ascending=False)
            store_skus[int(store)] = list(sku_vol.index[:n_skus])
        self._store_skus = store_skus

        store_week_data: dict[int, dict[int, pd.DataFrame]] = {}
        for store, grp in df.groupby("STORE"):
            store_week_data[int(store)] = {
                int(week): week_grp for week, week_grp in grp.groupby("WEEK")
            }

        sequences: list[tuple[int, int]] = []
        for store in sorted(df["STORE"].unique()):
            weeks = sorted(store_week_data[int(store)].keys())
            for i in range(len(weeks) - seq_len + 1):
                sequences.append((int(store), int(weeks[i])))

        self._precomputed = self._precompute_all(sequences, store_skus, store_week_data, df)
        self._sequences = sequences
        self._df = df

    def _precompute_all(self, sequences, store_skus, store_week_data, df):  # noqa: C901
        """Precompute all sequences as tensors for fast __getitem__."""
        T = self.seq_len
        K = self.n_skus
        obs_dim = self._obs_dim
        N = len(sequences)

        all_x = np.zeros((N, T, obs_dim), dtype=np.float32)
        all_a = np.zeros((N, T, K), dtype=np.float32)
        all_r = np.zeros((N, T), dtype=np.float32)
        all_done = np.zeros((N, T), dtype=np.float32)
        all_lp = np.zeros((N, T, K), dtype=np.float32)

        store_demo_cache: dict[int, np.ndarray] = {}
        for store in df["STORE"].unique():
            store_rows = df[df["STORE"] == store]
            feats = np.zeros(len(self.store_demo_cols), dtype=np.float32)
            for ci, col in enumerate(self.store_demo_cols):
                if col in store_rows.columns:
                    feats[ci] = float(store_rows[col].iloc[0])
            store_demo_cache[int(store)] = feats

        for idx, (store, start_week) in enumerate(sequences):
            skus = store_skus[store]
            swd = store_week_data[store]
            weeks = sorted(swd.keys())
            si = weeks.index(start_week)
            seq_weeks = weeks[si : si + T]

            prev_prices = None
            for t, week in enumerate(seq_weeks):
                if week not in swd:
                    continue
                wk_df = swd[week]
                row_indices = wk_df["_row_idx"].values
                if len(row_indices) > 0:
                    all_x[idx, t] = self._obs[row_indices].mean(axis=0)

                prices = np.zeros(K, dtype=np.float32)
                for k, upc in enumerate(skus):
                    sku_rows = wk_df[wk_df["UPC"] == upc]
                    if len(sku_rows) > 0:
                        p = float(sku_rows["unit_price_raw"].iloc[0])
                        prices[k] = p
                        all_lp[idx, t, k] = float(np.log(max(p, 1e-6)))

                if prev_prices is not None:
                    safe_prev = np.maximum(prev_prices, 1e-6)
                    all_a[idx, t] = prices / safe_prev
                prev_prices = prices.copy()

                total_reward = 0.0
                for upc in skus:
                    sku_rows = wk_df[wk_df["UPC"] == upc]
                    if len(sku_rows) > 0:
                        margin = float(
                            sku_rows["unit_price_raw"].iloc[0] - sku_rows["cost_raw"].iloc[0]
                        )
                        units = float(sku_rows["MOVE"].iloc[0])
                        total_reward += margin * max(units, 0.0)
                all_r[idx, t] = total_reward

                if t == T - 1:
                    all_done[idx, t] = 1.0

            if (idx + 1) % 5000 == 0:
                print(f"  Precomputed {idx + 1}/{N} sequences")

        all_sf = np.stack([store_demo_cache[s] for s, _ in sequences])
        print(f"  Precomputation complete: {N} sequences")

        return {
            "x": torch.from_numpy(all_x),
            "a": torch.from_numpy(all_a),
            "r": torch.from_numpy(all_r),
            "done": torch.from_numpy(all_done),
            "lp": torch.from_numpy(all_lp),
            "sf": torch.from_numpy(all_sf),
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
        p = self._precomputed
        return {
            "x_BT": p["x"][idx],
            "a_BT": p["a"][idx],
            "r_BT": p["r"][idx],
            "done_BT": p["done"][idx],
            "log_price_BT": p["lp"][idx],
            "store_features": p["sf"][idx],
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
