# causal-estimator Agent

## Purpose
Estimate per-category price elasticities via DML-PLIV (primary) and 2sCOPE (robustness check). Validate Hausman instrument strength. Output a frozen elasticity config consumed by `CausalDemandDecoder`.

## When to Use
Run once per category at the end of Track 1 (data pipeline), before Track 2 (world model) begins. Re-run if data preprocessing changes.

## Input
- Path to preprocessed Parquet file: `data/processed/<category>_train.parquet`
- Category code (e.g., `cso` for canned soup)

## Output
`configs/elasticities/<category>.json`:
```json
{
  "category": "cso",
  "theta_causal": -2.45,
  "theta_ols": -1.31,
  "f_stat_first_stage": 187.3,
  "hausman_p_value": 0.001,
  "sargan_p_value": 0.42,
  "cope_beta_c": -0.18,
  "cope_significant": true,
  "n_obs": 581000,
  "estimated_at": "2026-02-23T00:00:00Z"
}
```

## Validation Gates
BLOCK and raise `WeakInstrumentError` if:
- `f_stat_first_stage < 10` — weak instrument
- `hausman_p_value > 0.10` — cannot reject OLS=IV (endogeneity not confirmed)

WARN (non-blocking) if:
- `abs(theta_causal - cope_beta_c) / abs(theta_causal) > 0.5` — DML and copula estimates diverge >50%
- `theta_causal > -1.0` — unusually low elasticity for grocery (may indicate data issue)

## Procedure

### Step 1: Load preprocessed data
```python
import pandas as pd
df = pd.read_parquet(f"data/processed/{category}_train.parquet")
# Required columns: log_move, log_price, hausman_iv, log_cost,
#                   income, educ, age60, ethnic, workwom, hsizeavg
```

### Step 2: Run DML-PLIV
```python
import doubleml as dml
from sklearn.ensemble import RandomForestRegressor

control_cols = ["income", "educ", "ethnic", "hsizeavg", "age60", "workwom",
                "sin_week", "cos_week", "q2", "q3", "q4"]

data_dml = dml.DoubleMLData(
    df, y_col="log_move", d_cols="log_price",
    x_cols=control_cols, z_cols="hausman_iv"
)
dml_pliv = dml.DoubleMLPLIV(
    data_dml,
    ml_l=RandomForestRegressor(n_estimators=100, n_jobs=-1),
    ml_m=RandomForestRegressor(n_estimators=100, n_jobs=-1),
    ml_r=RandomForestRegressor(n_estimators=100, n_jobs=-1),
    n_folds=5
)
dml_pliv.fit()
theta_causal = float(dml_pliv.coef_[0])
```

### Step 3: Compute first-stage F-stat
```python
from sklearn.linear_model import LinearRegression
from scipy import stats

X_iv = df[control_cols + ["hausman_iv"]].values
y_price = df["log_price"].values
model_first = LinearRegression().fit(X_iv, y_price)
residuals = y_price - model_first.predict(X_iv)
# F-stat for hausman_iv coefficient
# Use statsmodels OLS for proper F-stat
import statsmodels.formula.api as smf
formula = "log_price ~ hausman_iv + " + " + ".join(control_cols)
ols_first = smf.ols(formula, data=df).fit()
f_stat = ols_first.fvalue
```

### Step 4: Hausman test (OLS vs IV)
```python
import statsmodels.formula.api as smf
formula_ols = "log_move ~ log_price + " + " + ".join(control_cols)
ols_model = smf.ols(formula_ols, data=df).fit()
theta_ols = ols_model.params["log_price"]
# Hausman test: regress log_move on log_price + hausman_iv residuals
df["iv_residual"] = df["log_price"] - df[control_cols + ["hausman_iv"]].pipe(
    lambda x: LinearRegression().fit(x, df["log_price"]).predict(x)
)
hausman_formula = "log_move ~ log_price + iv_residual + " + " + ".join(control_cols)
hausman_model = smf.ols(hausman_formula, data=df).fit()
hausman_p = hausman_model.pvalues["iv_residual"]
```

### Step 5: 2sCOPE robustness check
```python
from scipy.stats import norm, rankdata
import numpy as np
from sklearn.linear_model import LinearRegression

df["price_star"] = norm.ppf(rankdata(df["log_price"]) / (len(df) + 1))
exog_cols = ["income", "educ", "age60", "ethnic", "workwom", "hsizeavg"]
W_stars = np.column_stack([
    norm.ppf(rankdata(df[c]) / (len(df) + 1)) for c in exog_cols
])
reg1 = LinearRegression().fit(W_stars, df["price_star"].values)
df["copula_resid"] = df["price_star"].values - reg1.predict(W_stars)

cope_formula = "log_move ~ log_price + copula_resid + " + " + ".join(control_cols)
cope_model = smf.ols(cope_formula, data=df).fit()
cope_beta_c = float(cope_model.params["copula_resid"])
cope_significant = cope_model.pvalues["copula_resid"] < 0.05
```

### Step 6: Validate and write output
```python
import json
from datetime import datetime, timezone

if f_stat < 10:
    raise WeakInstrumentError(f"First-stage F={f_stat:.1f} < 10. Hausman IV is weak.")
if hausman_p > 0.10:
    raise WeakInstrumentError(f"Hausman p={hausman_p:.3f} > 0.10. Endogeneity not confirmed.")

result = {
    "category": category,
    "theta_causal": theta_causal,
    "theta_ols": theta_ols,
    "f_stat_first_stage": round(f_stat, 1),
    "hausman_p_value": round(hausman_p, 4),
    "sargan_p_value": None,  # requires second instrument
    "cope_beta_c": round(cope_beta_c, 4),
    "cope_significant": cope_significant,
    "n_obs": len(df),
    "estimated_at": datetime.now(timezone.utc).isoformat()
}

output_path = f"configs/elasticities/{category}.json"
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"Elasticity estimate: theta={theta_causal:.3f} (OLS: {theta_ols:.3f})")
```

## Expected Values
- `theta_causal`: −2.0 to −3.0 for grocery categories
- `f_stat`: well above 100 with 83 stores
- `hausman_p`: < 0.001 (endogeneity strongly confirmed in retail)
- If theta_causal > −1.0, investigate data quality before proceeding

## Tools
Bash, Read, Write

## Skills
- `python-development:python-error-handling`
- `python-development:python-testing-patterns`
