"""Tests for the drift-detection helpers in prefect_flow.monitoring_task.

Like test_train_features, this reimplements the helpers as pure functions
rather than importing prefect_flow (which would pull in Prefect at import
time). If you extract ks_test / psi into a `src/monitoring.py` module, swap
these locals for imports.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) < 2 or len(current) < 2:
        return float("nan"), float("nan")
    stat, p = stats.ks_2samp(reference, current)
    return float(stat), float(p)


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.where(ref_counts == 0, 1e-6, ref_counts / ref_counts.sum())
    cur_pct = np.where(cur_counts == 0, 1e-6, cur_counts / cur_counts.sum())
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


# ---------------------------------------------------------------------------
# KS test
# ---------------------------------------------------------------------------
class TestKsTest:
    def test_identical_distributions_no_drift(self, rng):
        data = rng.normal(0, 1, 500)
        stat, p = ks_test(data, data)
        assert stat == 0.0
        assert p == pytest.approx(1.0)

    def test_shifted_distribution_flags_drift(self, rng):
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(3, 1, 500)
        stat, p = ks_test(ref, cur)
        assert stat > 0.5
        assert p < 0.05

    def test_nan_inputs_are_filtered(self):
        ref = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        cur = np.array([1.0, 2.0, 3.0, 4.0])
        stat, p = ks_test(ref, cur)
        assert not np.isnan(stat)

    def test_too_few_samples_returns_nan(self):
        stat, p = ks_test(np.array([1.0]), np.array([1.0]))
        assert np.isnan(stat)
        assert np.isnan(p)


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------
class TestPsi:
    def test_identical_distributions_psi_near_zero(self, rng):
        data = rng.normal(0, 1, 1000)
        # PSI of a distribution against itself should be ~0 (not exactly 0
        # because of the 1e-6 floor in empty bins).
        assert psi(data, data) < 0.01

    def test_shifted_distribution_psi_large(self, rng):
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(5, 1, 1000)
        assert psi(ref, cur) > 0.25  # "major" by industry rule of thumb

    def test_moderate_shift_in_psi_band(self, rng):
        ref = rng.normal(0, 1, 2000)
        cur = rng.normal(0.5, 1, 2000)
        result = psi(ref, cur)
        # A 0.5σ shift should land in the moderate band, not "no drift".
        assert 0.0 < result

    def test_empty_input_returns_nan(self):
        assert np.isnan(psi(np.array([]), np.array([1.0, 2.0, 3.0])))

    def test_constant_reference_returns_zero(self):
        # All values identical → only one bin edge → no comparison possible.
        result = psi(np.array([5.0] * 100), np.array([5.0] * 100))
        assert result == 0.0
