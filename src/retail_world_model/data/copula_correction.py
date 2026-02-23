"""2sCOPE endogeneity correction (robustness check only, not primary method)."""

from __future__ import annotations

import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def compute_2scope_copula_residual(
    df: pd.DataFrame,
    price_col: str = "unit_price",
    demand_col: str = "MOVE",
    instrument_col: str = "hausman_iv",
) -> pd.Series:
    """2sCOPE endogeneity correction.

    Step 1: Regress price on instruments (hausman_iv) -> get residuals
    Step 2: Transform residuals to uniform [0,1] via empirical CDF
    Step 3: Transform to normal via inverse CDF (scipy.stats.norm.ppf)

    Returns: copula residuals to include as control in demand regression.
    """
    # Step 1: OLS price ~ hausman_iv
    X = df[[instrument_col]].values
    y = df[price_col].values

    reg = LinearRegression()
    reg.fit(X, y)
    residuals = y - reg.predict(X)

    # Step 2: Empirical CDF -> uniform [0, 1]
    n = len(residuals)
    ranks = pd.Series(residuals).rank(method="average")
    # Winsor to avoid ppf(0) or ppf(1) = +/-inf
    uniform = ranks / (n + 1)

    # Step 3: Inverse normal CDF
    copula_residual = stats.norm.ppf(uniform.values)

    return pd.Series(copula_residual, index=df.index, name="copula_residual")
