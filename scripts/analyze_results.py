"""
Load all ablation results, compute IQM + 95% bootstrap CI via scipy.stats,
apply Holm-Bonferroni correction for multiple comparisons.
Print markdown table.

Usage:
  python scripts/analyze_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent.parent / "docs" / "results" / "ablations"

ABLATIONS = [
    "imagination_off",
    "horizon_5",
    "horizon_10",
    "horizon_25",
    "no_stochastic_latent",
    "no_symlog_twohot",
    "gru_backbone",
    "flat_encoder",
    "no_mopo_lcb",
]

FULL_MODEL_RESULT = RESULTS_DIR / "full_model.json"


def bootstrap_iqm(
    scores: list[float], n_bootstrap: int = 10_000, seed: int = 42
) -> tuple[float, float, float]:
    """
    IQM = mean of middle 50% of scores.
    Bootstrap CI by resampling with replacement.
    Returns (iqm, ci_lower_95, ci_upper_95).
    """
    arr = np.array([s for s in scores if s is not None], dtype=np.float64)
    if len(arr) == 0:
        return (float("nan"), float("nan"), float("nan"))

    def iqm(x: np.ndarray) -> float:
        q25, q75 = np.percentile(x, [25, 75])
        mask = (x >= q25) & (x <= q75)
        return float(np.mean(x[mask])) if mask.any() else float(np.mean(x))

    point = iqm(arr)
    rng = np.random.default_rng(seed)
    boot = [iqm(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_bootstrap)]
    ci_lower, ci_upper = np.percentile(boot, [2.5, 97.5])
    return (point, float(ci_lower), float(ci_upper))


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni correction. Returns list of reject booleans."""
    m = len(p_values)
    if m == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    reject = [False] * m

    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (m - rank)
        if p <= adjusted_alpha:
            reject[orig_idx] = True
        else:
            break

    return reject


def load_results(ablation_name: str) -> dict | None:
    """Load results JSON for an ablation."""
    path = RESULTS_DIR / f"{ablation_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_full_model_scores() -> list[float]:
    """Load full model (baseline) scores for comparison."""
    if FULL_MODEL_RESULT.exists():
        with open(FULL_MODEL_RESULT) as f:
            data = json.load(f)
            return [s for s in data.get("episode_rewards", []) if s is not None]
    return []


def compute_p_value(full_scores: list[float], ablation_scores: list[float]) -> float:
    """Two-sided Welch's t-test comparing full model vs ablation."""
    if len(full_scores) < 2 or len(ablation_scores) < 2:
        return float("nan")
    _, p = stats.ttest_ind(full_scores, ablation_scores, equal_var=False)
    return float(p)


def print_ablation_table(results: dict[str, dict]) -> None:
    """Print markdown table with ablation results."""
    full_scores = load_full_model_scores()

    rows = []
    p_values = []

    for name in ABLATIONS:
        data = results.get(name)
        if data is None:
            rows.append((name, "N/A", "N/A", "N/A", "N/A"))
            p_values.append(1.0)
            continue

        scores = [s for s in data.get("episode_rewards", []) if s is not None]
        if not scores:
            rows.append((name, "pending", "N/A", "N/A", "N/A"))
            p_values.append(1.0)
            continue

        iqm_val, ci_low, ci_high = bootstrap_iqm(scores)
        p = compute_p_value(full_scores, scores)
        p_values.append(p)
        rows.append(
            (
                name,
                f"{iqm_val:.3f}",
                f"[{ci_low:.3f}, {ci_high:.3f}]",
                f"{p:.4f}" if not np.isnan(p) else "N/A",
                "",  # placeholder for significance
            )
        )

    rejects = holm_bonferroni(p_values)

    print("| Ablation | IQM | 95% CI | vs. Full (p-value) | Significant? |")
    print("|----------|-----|--------|--------------------|-------------|")
    for i, (name, iqm_str, ci_str, p_str, _) in enumerate(rows):
        sig = "Yes*" if rejects[i] else "No"
        if p_str == "N/A":
            sig = "N/A"
        print(f"| {name} | {iqm_str} | {ci_str} | {p_str} | {sig} |")

    print("\n* Significant at alpha=0.05 after Holm-Bonferroni correction")


def main() -> None:
    results = {}
    for name in ABLATIONS:
        data = load_results(name)
        if data is not None:
            results[name] = data

    print("# DreamPrice Ablation Study Results\n")
    print_ablation_table(results)


if __name__ == "__main__":
    main()
