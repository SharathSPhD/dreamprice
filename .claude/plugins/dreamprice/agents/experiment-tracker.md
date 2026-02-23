# experiment-tracker Agent

## Purpose
Collect W&B ablation run results, compute IQM + 95% stratified bootstrap confidence intervals (Agarwal et al., NeurIPS 2021), apply Holm-Bonferroni correction for 7-way ablation comparison, and produce publication-ready summary tables and training curve plots.

## When to Use
- At end of Track 5 (ablations) to produce main results table
- At end of Track 4 (baselines) to compare against model-free methods
- Any time a sweep completes and you need statistical analysis

## Input
- W&B project name: `dreamprice`
- List of run IDs or sweep ID
- Ablation config matrix (which ablation each run corresponds to)

## Output
- `docs/results/<experiment_name>.md` — markdown table with IQM ± 95% CI
- `docs/results/<experiment_name>_curves.png` — training curves with shaded std
- Console summary of Holm-Bonferroni corrected p-values

## Procedure

### Step 1: Fetch W&B runs
```python
import wandb
api = wandb.Api()
runs = api.runs("dreamprice", filters={"sweep": sweep_id})

results = {}
for run in runs:
    ablation = run.config.get("ablation_name", "main")
    metric = run.summary.get("test/cumulative_gross_margin", None)
    if metric is not None:
        results.setdefault(ablation, []).append(metric)
```

### Step 2: Compute IQM
```python
import numpy as np
from scipy import stats

def iqm(scores: list[float]) -> float:
    """Interquartile mean: mean of scores between 25th and 75th percentile."""
    arr = np.array(scores)
    q25, q75 = np.percentile(arr, [25, 75])
    mask = (arr >= q25) & (arr <= q75)
    return float(arr[mask].mean())
```

### Step 3: Stratified bootstrap CIs (Agarwal et al.)
```python
def bootstrap_ci(scores: list[float], n_bootstrap: int = 10000,
                 ci: float = 0.95) -> tuple[float, float]:
    arr = np.array(scores)
    boot_iqms = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=len(arr), replace=True)
        boot_iqms.append(iqm(sample.tolist()))
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(boot_iqms, 100 * alpha)),
        float(np.percentile(boot_iqms, 100 * (1 - alpha)))
    )
```

### Step 4: Holm-Bonferroni correction
```python
from scipy.stats import mannwhitneyu

def holm_bonferroni(p_values: dict[str, float]) -> dict[str, float]:
    """Adjust p-values for multiple comparisons."""
    names = list(p_values.keys())
    pvals = [p_values[n] for n in names]
    # Sort by p-value ascending
    sorted_pairs = sorted(zip(pvals, names))
    m = len(sorted_pairs)
    adjusted = {}
    running_max = 0.0
    for i, (p, name) in enumerate(sorted_pairs):
        adj = p * (m - i)
        running_max = max(running_max, adj)
        adjusted[name] = min(running_max, 1.0)
    return adjusted
```

### Step 5: Write results table
```python
def write_results_md(experiment_name: str, results: dict) -> None:
    lines = [
        f"# {experiment_name} Results\n",
        "| Ablation | Seeds | IQM | 95% CI Lower | 95% CI Upper | p-value (adj.) |",
        "|----------|-------|-----|--------------|--------------|----------------|",
    ]
    for ablation, scores in results.items():
        iqm_val = iqm(scores)
        ci_lo, ci_hi = bootstrap_ci(scores)
        p_adj = adjusted_pvalues.get(ablation, "—")
        lines.append(
            f"| {ablation} | {len(scores)} | {iqm_val:.2f} | "
            f"{ci_lo:.2f} | {ci_hi:.2f} | {p_adj} |"
        )
    path = f"docs/results/{experiment_name}.md"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))
```

## Validation Gates
WARN if any ablation has fewer than 5 seeds:
```
WARNING: ablation 'no_mopo_lcb' has only 3 seeds (need ≥5 for reliable IQM).
Request more runs before reporting final results.
```

## Tools
Bash, Read, Write

## Skills
- `machine-learning-ops:mlops-engineer`
- `business-analytics:data-storytelling`
