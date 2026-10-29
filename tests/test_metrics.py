import numpy as np
import pytest

from creditrisk.metrics import auc_interval, evaluate, psi


def test_perfect_predictions():
    result = evaluate([0, 0, 1, 1], [0, 0, 1, 1])
    assert result["roc_auc"] == result["ks"] == result["average_precision"] == 1
    assert result["brier_score"] == 0
    assert result["false_positive"] == result["false_negative"] == 0


def test_ties_have_chance_auc_zero_ks():
    result = evaluate([0, 1, 0, 1], [0.5] * 4)
    assert result["roc_auc"] == 0.5
    assert result["ks"] == 0
    assert result["brier_score"] == 0.25


def test_single_class_is_explicit_not_fake_auc():
    result = evaluate([0, 0, 0], [0.1, 0.2, 0.3])
    assert result["roc_auc"] is None
    assert result["ks"] is None
    assert result["recall"] is None


def test_invalid_probabilities_fail():
    with pytest.raises(ValueError):
        evaluate([0, 1], [np.nan, 0.5])


def test_psi_identity_shift_and_missingness():
    rng = np.random.default_rng(9)
    ref = rng.normal(size=2000)
    assert psi(ref, ref) == pytest.approx(0)
    assert psi(ref, ref + 3) > 0.25
    missing = ref.copy()
    missing[:1000] = np.nan
    assert psi(ref, missing) > 0.25


def test_constant_reference_detects_drift():
    assert psi(np.ones(100), np.ones(100)) == pytest.approx(0)
    assert psi(np.ones(100), np.full(100, 2)) > 0.25


def test_bootstrap_seeded():
    y = [0, 1] * 20
    p = np.linspace(0, 1, len(y))
    assert auc_interval(y, p, repetitions=30) == auc_interval(y, p, repetitions=30)
