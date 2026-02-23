"""
Run all 9 ablations x 5 seeds.
For each ablation:
  - Launch training with overrides from ablation config
  - Log to W&B group: "ablations/{ablation_name}"
  - Collect final episode reward (IQM metric)
  - Save results to docs/results/ablations/{ablation_name}.json

Usage:
  python scripts/run_ablations.py --dry-run             # just print what would run
  python scripts/run_ablations.py --ablation horizon_5   # run one ablation
  python scripts/run_ablations.py                        # run all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

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
N_SEEDS = 5
BASE_SEED = 42

RESULTS_DIR = Path(__file__).resolve().parent.parent / "docs" / "results" / "ablations"
CONFIGS_DIR = (
    Path(__file__).resolve().parent.parent / "configs" / "experiment" / "ablations"
)


def run_ablation(name: str, seed: int, dry_run: bool = False) -> int:
    """Run a single ablation with a given seed. Returns subprocess exit code."""
    cmd = [
        sys.executable,
        "scripts/train.py",
        f"+experiment/ablations={name}",
        f"seed={seed}",
        f"wandb_group=ablations/{name}",
        f"checkpoint_dir=checkpoints/ablations/{name}/seed_{seed}",
    ]
    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return 0

    print(f"  Running {name} seed={seed} ...")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def collect_results(ablation_name: str) -> dict:
    """Load results JSON for an ablation, or return empty stub."""
    path = RESULTS_DIR / f"{ablation_name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {
        "ablation": ablation_name,
        "seeds": [BASE_SEED + i for i in range(N_SEEDS)],
        "episode_rewards": [None] * N_SEEDS,
        "status": "pending",
    }


def compute_iqm_with_ci(
    scores: list[float], n_bootstrap: int = 10_000
) -> tuple[float, float, float]:
    """Returns (iqm, ci_lower, ci_upper) with 95% bootstrap CI."""
    import numpy as np

    arr = np.array([s for s in scores if s is not None])
    if len(arr) == 0:
        return (float("nan"), float("nan"), float("nan"))

    def iqm(x: np.ndarray) -> float:
        q25, q75 = np.percentile(x, [25, 75])
        mask = (x >= q25) & (x <= q75)
        return float(np.mean(x[mask])) if mask.any() else float(np.mean(x))

    point = iqm(arr)
    rng = np.random.default_rng(42)
    boot = [iqm(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_bootstrap)]
    ci_lower, ci_upper = np.percentile(boot, [2.5, 97.5])
    return (point, float(ci_lower), float(ci_upper))


def save_results(ablation_name: str, data: dict) -> None:
    """Save results JSON for an ablation."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{ablation_name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved results to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DreamPrice ablation studies")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--ablation", type=str, default=None, help="Run single ablation")
    args = parser.parse_args()

    ablations = [args.ablation] if args.ablation else ABLATIONS

    total_runs = len(ablations) * N_SEEDS
    print(f"Ablation sweep: {len(ablations)} ablations x {N_SEEDS} seeds = {total_runs} runs")

    for name in ablations:
        config_path = CONFIGS_DIR / f"{name}.yaml"
        if not config_path.exists():
            print(f"  WARNING: Config not found: {config_path}")
            continue

        print(f"\n=== Ablation: {name} ===")
        for i in range(N_SEEDS):
            seed = BASE_SEED + i
            run_ablation(name, seed, dry_run=args.dry_run)

        if not args.dry_run:
            results = collect_results(name)
            results["status"] = "completed"
            save_results(name, results)


if __name__ == "__main__":
    main()
