"""Tests for logistic-regression DIF."""
import numpy as np
import pytest

from benchdif.dif import logistic


def _sim(n=1200, k=12, dif_item=5, dif=1.0, seed=4):
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n)
    g = (np.arange(n) % 2).astype(float)
    diff = rng.normal(scale=0.7, size=k)
    X = (rng.random((n, k)) < 1 / (1 + np.exp(-(theta[:, None] - diff[None, :])))).astype(float)
    foc = g == 1
    X[foc, dif_item] = (rng.random(foc.sum()) < 1 / (1 + np.exp(-(theta[foc] - diff[dif_item] - dif)))).astype(float)
    return X, g, dif_item


def test_flags_injected_dif():
    X, g, j = _sim()
    res = logistic(X, g, kind="both")
    assert res.loc[j, "flag"]
    assert res.loc[j, "df"] == 2
    assert res.loc[j, "stat"] == max(res["stat"])  # strongest signal is the injected item


def test_kinds_have_expected_df():
    X, g, _ = _sim()
    assert (logistic(X, g, kind="both")["df"] == 2).all()
    assert (logistic(X, g, kind="uniform")["df"] == 1).all()
    assert (logistic(X, g, kind="nonuniform")["df"] == 1).all()


def test_input_validation():
    with pytest.raises(ValueError):
        logistic(np.zeros((3, 3)), np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        logistic(np.zeros((4, 3)), np.array([0, 1]))
    with pytest.raises(ValueError):
        logistic(np.zeros((3, 3)), np.array([0, 1, 0]), kind="bogus")


def test_matches_statsmodels_glm():
    sm = pytest.importorskip("statsmodels.api")
    X, g, _ = _sim(seed=11)
    S = X.sum(axis=1)
    res = logistic(X, g, kind="both")
    for j in range(X.shape[1]):
        y = X[:, j]
        d0 = sm.add_constant(np.column_stack([S]), has_constant="add")
        d2 = sm.add_constant(np.column_stack([S, g, S * g]), has_constant="add")
        ll0 = sm.GLM(y, d0, family=sm.families.Binomial()).fit().llf
        ll2 = sm.GLM(y, d2, family=sm.families.Binomial()).fit().llf
        assert res.loc[j, "stat"] == pytest.approx(2 * (ll2 - ll0), abs=1e-4)
