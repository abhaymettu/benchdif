"""Unified detect() API."""
import numpy as np
import pytest

from benchdif import detect


def _sim(seed=0, n=1400, k=10, j=4, dif=1.0):
    rng = np.random.default_rng(seed)
    th = rng.normal(size=n); g = (np.arange(n) % 2)
    diff = rng.normal(scale=0.6, size=k)
    X = (rng.random((n, k)) < 1 / (1 + np.exp(-(th[:, None] - diff[None, :])))).astype(int)
    foc = g == 1
    X[foc, j] = (rng.random(foc.sum()) < 1 / (1 + np.exp(-(th[foc] - diff[j] - dif)))).astype(int)
    return X, g, j


@pytest.mark.parametrize("method", ["mh", "logistic", "irt"])
def test_normalized_columns_and_detection(method):
    X, g, j = _sim()
    res = detect(X, g, method=method)
    for col in ("method", "stat", "p_value", "flag", "effect", "effect_class"):
        assert col in res.columns
    assert (res["method"] == method).all()
    assert bool(res.loc[j, "flag"])          # injected DIF item flagged by every method


def test_kwargs_forwarded():
    X, g, _ = _sim()
    r = detect(X, g, method="mh", purify=True)     # purify kwarg reaches MH
    assert "ets" in r.columns
    r2 = detect(X, g, method="logistic", kind="uniform")
    assert (r2["df"] == 1).all()


def test_bad_method():
    X, g, _ = _sim()
    with pytest.raises(ValueError):
        detect(X, g, method="nope")
