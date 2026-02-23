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
                rows.append(
                    {
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
                    }
                )
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
        ds = DominicksSequenceDataset(sample_df, seq_len=seq_len, n_skus=n_skus)
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
