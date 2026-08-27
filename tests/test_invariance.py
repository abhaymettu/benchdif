"""Measurement-invariance omnibus: correct verdicts + size control under impact."""
import numpy as np
import pytest

from benchdif import invariance


def _gen(seed, kind, n=1800, J=8, impact_mu=-0.4, impact_sd=1.2):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.8, 1.6, J); b = rng.normal(0, 1, J)
    a_f, b_f = a.copy(), b.copy()
    if kind == "intercept_dif":
        for j in (2, 5): b_f[j] = b[j] + 0.9
    if kind == "loading_dif":
        for j in (2, 5): a_f[j] = a[j] + 0.7
    th0 = rng.normal(0, 1, n); th1 = rng.normal(impact_mu, impact_sd, n)
    d, d_f = -a * b, -a_f * b_f
    X0 = (rng.random((n, J)) < 1 / (1 + np.exp(-(a * th0[:, None] + d)))).astype(int)
    X1 = (rng.random((n, J)) < 1 / (1 + np.exp(-(a_f * th1[:, None] + d_f)))).astype(int)
    return np.vstack([X0, X1]), np.array([0] * n + [1] * n)


def test_full_invariance_holds_despite_impact():
    res = invariance(*_gen(0, "invariant"))
    assert res.loc["metric", "holds"]      # equal loadings not rejected
    assert res.loc["scalar", "holds"]      # equal intercepts not rejected (impact absorbed)


def test_intercept_dif_localizes_to_scalar():
    # Intercept DIF must show up strongly at the scalar level; the metric level may
    # be borderline (the MI levels are not perfectly orthogonal), so assert the
    # robust localization: scalar rejects and its signal dominates metric's.
    res = invariance(*_gen(0, "intercept_dif"))
    assert not res.loc["scalar", "holds"]                       # intercepts differ
    assert res.loc["scalar", "stat"] > res.loc["metric", "stat"]  # localized to scalar
    assert res.loc["scalar", "p_value"] < res.loc["metric", "p_value"]


def test_loading_dif_rejects_metric():
    res = invariance(*_gen(0, "loading_dif"))
    assert not res.loc["metric", "holds"]


def test_bad_group():
    with pytest.raises(ValueError):
        invariance(np.zeros((4, 3)), np.array([0, 1, 2, 0]))
