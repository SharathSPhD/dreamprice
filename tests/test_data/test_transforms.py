"""Tests for feature transforms: symlog, symexp, Hausman IV, promotions, temporal split."""

import numpy as np
import pandas as pd

from retail_world_model.data.transforms import (
    compute_discount_depth,
    compute_hausman_iv,
    compute_lag_features,
    compute_price_index,
    compute_rolling_features,
    compute_temporal_features,
    flag_promotions,
    symexp,
    symlog,
    temporal_split,
)


class TestSymlog:
    def test_symlog_zero(self):
        assert symlog(0) == 0.0

    def test_symlog_positive(self):
        assert abs(symlog(1) - np.log(2)) < 1e-10

    def test_symlog_negative(self):
        assert abs(symlog(-1) - (-np.log(2))) < 1e-10

    def test_symlog_array(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = symlog(x)
        assert result.shape == (5,)
        assert result[2] == 0.0
        assert result[3] > 0
        assert result[1] < 0


class TestSymexp:
    def test_symexp_inverse(self):
        for x in [-100, -1, 0, 1, 100]:
            result = symexp(symlog(x))
            assert abs(result - x) < 1e-8, f"symexp(symlog({x})) = {result}, expected {x}"

    def test_symexp_zero(self):
        assert symexp(0) == 0.0


class TestHausmanIV:
    def test_hausman_excludes_own_store(self):
        """Hausman IV for store A should not equal log(price_A)."""
        df = pd.DataFrame(
            {
                "STORE": [1, 2, 3, 1, 2, 3],
                "UPC": [10, 10, 10, 10, 10, 10],
                "WEEK": [1, 1, 1, 2, 2, 2],
                "PRICE": [2.0, 3.0, 4.0, 2.5, 3.5, 4.5],
                "QTY": [1, 1, 1, 1, 1, 1],
            }
        )
        iv = compute_hausman_iv(df)
        # For store 1 week 1: IV = mean(log(3), log(4)) = (log(3)+log(4))/2
        # This should NOT equal log(2) (store 1's own price)
        store1_w1_iv = iv.iloc[0]
        own_log_price = np.log(2.0)
        assert abs(store1_w1_iv - own_log_price) > 0.01

    def test_hausman_iv_formula(self):
        """Verify the IV formula: (sum_log_all - own_log) / (n - 1)."""
        df = pd.DataFrame(
            {
                "STORE": [1, 2, 3],
                "UPC": [10, 10, 10],
                "WEEK": [1, 1, 1],
                "PRICE": [2.0, 3.0, 4.0],
                "QTY": [1, 1, 1],
            }
        )
        iv = compute_hausman_iv(df)
        # Store 1: (log(2)+log(3)+log(4) - log(2)) / 2 = (log(3)+log(4)) / 2
        expected_store1 = (np.log(3.0) + np.log(4.0)) / 2
        assert abs(iv.iloc[0] - expected_store1) < 1e-10


class TestTemporalSplit:
    def test_temporal_split_no_leakage(self):
        """No week from the test set should appear in the train set."""
        df = pd.DataFrame({"WEEK": list(range(1, 401)), "val": range(400)})
        train, val, test = temporal_split(df)
        train_weeks = set(train["WEEK"])
        val_weeks = set(val["WEEK"])
        test_weeks = set(test["WEEK"])

        assert train_weeks.isdisjoint(test_weeks)
        assert train_weeks.isdisjoint(val_weeks)
        assert val_weeks.isdisjoint(test_weeks)

    def test_temporal_split_boundaries(self):
        df = pd.DataFrame({"WEEK": list(range(1, 401)), "val": range(400)})
        train, val, test = temporal_split(df)
        assert train["WEEK"].max() == 280
        assert val["WEEK"].min() == 281
        assert val["WEEK"].max() == 340
        assert test["WEEK"].min() == 341
        assert test["WEEK"].max() == 400

    def test_temporal_split_sizes(self):
        df = pd.DataFrame({"WEEK": list(range(1, 401)), "val": range(400)})
        train, val, test = temporal_split(df)
        assert len(train) == 280
        assert len(val) == 60
        assert len(test) == 60


class TestFlagPromotions:
    def test_on_promotion_modal_price(self):
        """Item below 0.95 * mode is flagged even if SALE == ''."""
        # 10 rows at modal price 2.00, 2 rows at 1.80 (< 0.95 * 2.00 = 1.90)
        df = pd.DataFrame(
            {
                "STORE": [1] * 12,
                "UPC": [1] * 12,
                "PRICE": [2.0] * 10 + [1.80, 1.80],
                "QTY": [1] * 12,
                "SALE": [""] * 12,
            }
        )
        promo = flag_promotions(df)
        # Last 2 rows should be flagged (1.80 < 0.95 * 2.0 = 1.90)
        assert promo.iloc[-1] is np.True_
        assert promo.iloc[-2] is np.True_
        # First 10 should NOT be flagged
        assert not promo.iloc[0]

    def test_not_flagged_at_modal(self):
        df = pd.DataFrame(
            {
                "STORE": [1] * 5,
                "UPC": [1] * 5,
                "PRICE": [2.0] * 5,
                "QTY": [1] * 5,
                "SALE": [""] * 5,
            }
        )
        promo = flag_promotions(df)
        assert not promo.any()


class TestDiscountDepth:
    def test_zero_at_modal(self, clean_movement_df):
        dd = compute_discount_depth(clean_movement_df)
        assert (dd >= 0).all()

    def test_positive_on_sale(self):
        df = pd.DataFrame(
            {
                "STORE": [1, 1, 1],
                "UPC": [10, 10, 10],
                "PRICE": [3.0, 3.0, 2.0],
                "QTY": [1, 1, 1],
            }
        )
        dd = compute_discount_depth(df)
        assert dd.iloc[2] > 0


class TestLagFeatures:
    def test_lag_shape(self, clean_movement_df):
        lags = compute_lag_features(clean_movement_df, ["STORE", "UPC"], "MOVE", [1, 2])
        assert lags.shape[0] == len(clean_movement_df)
        assert "MOVE_lag_1" in lags.columns
        assert "MOVE_lag_2" in lags.columns


class TestRollingFeatures:
    def test_rolling_shape(self, clean_movement_df):
        rolling = compute_rolling_features(clean_movement_df, ["STORE", "UPC"], "MOVE", 4)
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
