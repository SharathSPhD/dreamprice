"""Tests for ablation infrastructure: IQM, bootstrap CI, Holm-Bonferroni, configs, dry-run."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = ROOT / "configs" / "experiment" / "ablations"

sys.path.insert(0, str(ROOT / "scripts"))
from analyze_results import bootstrap_iqm, holm_bonferroni  # noqa: E402
from run_ablations import ABLATIONS, N_SEEDS, compute_iqm_with_ci  # noqa: E402


class TestIQMComputation:
    def test_iqm_basic(self):
        """IQM of [1,2,3,4,5,6,7,8] == mean([3,4,5,6]) = 4.5"""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        iqm, _, _ = bootstrap_iqm(scores, n_bootstrap=100)
        assert abs(iqm - 4.5) < 0.01

    def test_iqm_with_ci(self):
        """compute_iqm_with_ci matches bootstrap_iqm."""
        scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        iqm1, _, _ = compute_iqm_with_ci(scores, n_bootstrap=100)
        iqm2, _, _ = bootstrap_iqm(scores, n_bootstrap=100)
        assert abs(iqm1 - iqm2) < 0.5  # both should be ~4.5

    def test_iqm_empty(self):
        """Empty scores return NaN."""
        iqm, ci_low, ci_high = bootstrap_iqm([])
        assert np.isnan(iqm)

    def test_iqm_single(self):
        """Single score returns that score."""
        iqm, _, _ = bootstrap_iqm([5.0], n_bootstrap=100)
        assert abs(iqm - 5.0) < 0.01


class TestBootstrapCIWidth:
    def test_ci_decreases_with_n(self):
        """CI width decreases as n increases."""
        rng = np.random.default_rng(42)
        small = rng.normal(10, 2, size=10).tolist()
        large = rng.normal(10, 2, size=100).tolist()

        _, lo_s, hi_s = bootstrap_iqm(small, n_bootstrap=5000)
        _, lo_l, hi_l = bootstrap_iqm(large, n_bootstrap=5000)

        width_small = hi_s - lo_s
        width_large = hi_l - lo_l
        assert width_large < width_small


class TestHolmBonferroni:
    def test_rejects_clear_signal(self):
        """Rejects at alpha=0.05 for clear signal."""
        p_values = [0.001, 0.01, 0.8]
        rejects = holm_bonferroni(p_values, alpha=0.05)
        assert rejects[0] is True
        assert rejects[1] is True
        assert rejects[2] is False

    def test_no_rejection(self):
        """Does not reject when all p-values are large."""
        p_values = [0.5, 0.6, 0.7]
        rejects = holm_bonferroni(p_values, alpha=0.05)
        assert all(r is False for r in rejects)

    def test_all_rejected(self):
        """All rejected when all p-values are tiny."""
        p_values = [0.001, 0.002, 0.003]
        rejects = holm_bonferroni(p_values, alpha=0.05)
        assert all(r is True for r in rejects)

    def test_empty(self):
        """Empty input returns empty."""
        assert holm_bonferroni([]) == []


class TestAblationConfigsExist:
    def test_all_configs_exist(self):
        """All 9 ablation YAML files exist."""
        for name in ABLATIONS:
            config_path = CONFIGS_DIR / f"{name}.yaml"
            assert config_path.exists(), f"Missing config: {config_path}"

    def test_config_count(self):
        """Exactly 9 ablation configs."""
        yamls = list(CONFIGS_DIR.glob("*.yaml"))
        assert len(yamls) == len(ABLATIONS)


class TestRunAblationsDryRun:
    def test_dry_run_prints_all_runs(self):
        """Dry-run prints 45 runs (9 ablations x 5 seeds) without executing."""
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_ablations.py"), "--dry-run"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert result.returncode == 0
        output = result.stdout
        dry_run_lines = [line for line in output.splitlines() if "[DRY RUN]" in line]
        expected = len(ABLATIONS) * N_SEEDS
        assert len(dry_run_lines) == expected, (
            f"Expected {expected} dry-run lines, got {len(dry_run_lines)}"
        )

    def test_dry_run_single_ablation(self):
        """Dry-run with --ablation flag runs only 5 seeds."""
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_ablations.py"),
                "--dry-run",
                "--ablation",
                "horizon_5",
            ],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        assert result.returncode == 0
        dry_run_lines = [line for line in result.stdout.splitlines() if "[DRY RUN]" in line]
        assert len(dry_run_lines) == N_SEEDS


class TestStubResults:
    def test_stub_files_exist(self):
        """Stub result JSON files exist for all ablations."""
        results_dir = ROOT / "docs" / "results" / "ablations"
        for name in ABLATIONS:
            path = results_dir / f"{name}.json"
            assert path.exists(), f"Missing stub: {path}"

    def test_stub_format(self):
        """Stub files have correct format."""
        results_dir = ROOT / "docs" / "results" / "ablations"
        for name in ABLATIONS:
            path = results_dir / f"{name}.json"
            with open(path) as f:
                data = json.load(f)
            assert data["ablation"] == name
            assert len(data["seeds"]) == N_SEEDS
            assert len(data["episode_rewards"]) == N_SEEDS
            assert data["status"] == "pending"
