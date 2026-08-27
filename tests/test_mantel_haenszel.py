"""Tests for Mantel-Haenszel DIF.

Two levels: (1) a hand-checkable tiny case with a known-degenerate structure,
(2) numerical agreement with statsmodels' CMH on simulated data (the oracle).
"""
import numpy as np
import pytest

from benchdif import mantel_haenszel


def test_no_dif_when_groups_identical():
    # Same responses in both groups -> no DIF, alpha ~ 1.
    rng = np.random.default_rng(1)
    theta = rng.normal(size=400)
    diff = rng.normal(size=8)
    P = 1 / (1 + np.exp(-(theta[:, None] - diff[None, :])))
    X = (rng.random((400, 8)) < P).astype(int)
    Xd = np.vstack([X, X])                      # duplicate as two "groups"
    g = np.array([0] * 400 + [1] * 400)
    res = mantel_haenszel(Xd, g)
    assert (~res.flag).all()
    assert np.allclose(res.alpha_mh, 1.0, atol=1e-6)
    assert (res.ets == "A").all()


def test_flags_injected_dif():
    rng = np.random.default_rng(2)
    n, k, j = 1200, 12, 5
    theta = rng.normal(size=n)
    g = (np.arange(n) % 2).astype(int)
    diff = rng.normal(scale=0.7, size=k)
    X = (rng.random((n, k)) < 1 / (1 + np.exp(-(theta[:, None] - diff[None, :])))).astype(int)
    foc = g == 1
    X[foc, j] = (rng.random(foc.sum()) < 1 / (1 + np.exp(-(theta[foc] - diff[j] - 1.0)))).astype(int)
    res = mantel_haenszel(X, g)
    assert res.loc[j, "flag"]
    assert res.loc[j, "ets"] in ("B", "C")
    # clean items should mostly not flag
    clean = res.drop(index=j)
    assert clean.flag.sum() <= 1


def test_input_validation():
    with pytest.raises(ValueError):
        mantel_haenszel(np.zeros((3, 3)), np.array([0, 1, 2]))      # bad group codes
    with pytest.raises(ValueError):
        mantel_haenszel(np.zeros((3, 3)), np.array([0, 1]))         # length mismatch
    with pytest.raises(ValueError):
        mantel_haenszel(np.array([np.nan, 1.0]), np.array([0, 1]))  # not 2-D


def test_matches_statsmodels_cmh():
    sm = pytest.importorskip("statsmodels.stats.contingency_tables")
    rng = np.random.default_rng(7)
    n, k = 1500, 15
    theta = rng.normal(size=n)
    g = (np.arange(n) % 2).astype(int)
    diff = rng.normal(scale=0.8, size=k)
    X = (rng.random((n, k)) < 1 / (1 + np.exp(-(theta[:, None] - diff[None, :])))).astype(int)
    foc = g == 1
    X[foc, 4] = (rng.random(foc.sum()) < 1 / (1 + np.exp(-(theta[foc] - diff[4] - 0.8)))).astype(int)
    res = mantel_haenszel(X, g)
    total = X.sum(axis=1)
    for j in range(k):
        tables = []
        for lvl in np.unique(total):
            at = total == lvl
            ref, fo = at & (g == 0), at & (g == 1)
            A = int(np.sum(X[ref, j] == 1)); B = int(np.sum(X[ref, j] == 0))
            C = int(np.sum(X[fo, j] == 1)); D = int(np.sum(X[fo, j] == 0))
            if A + B + C + D:
                tables.append([[A, B], [C, D]])
        st = sm.StratifiedTable(np.array(tables).transpose(1, 2, 0))
        assert res.loc[j, "stat"] == pytest.approx(
            st.test_null_odds(correction=True).statistic, abs=1e-4)
        assert res.loc[j, "alpha_mh"] == pytest.approx(st.oddsratio_pooled, abs=1e-6)


def test_purification_converges_and_keeps_dif():
    rng = np.random.default_rng(9)
    n, k = 1500, 20
    theta = rng.normal(size=n)
    g = (np.arange(n) % 2).astype(int)
    diff = rng.normal(scale=0.7, size=k)
    X = (rng.random((n, k)) < 1 / (1 + np.exp(-(theta[:, None] - diff[None, :])))).astype(int)
    foc = g == 1
    for j in (3, 11):  # two DIF items
        X[foc, j] = (rng.random(foc.sum()) < 1 / (1 + np.exp(-(theta[foc] - diff[j] - 1.2)))).astype(int)
    res = mantel_haenszel(X, g, purify=True)
    assert "niter" in res.attrs
    assert res.loc[3, "flag"] and res.loc[11, "flag"]


def test_match_vector_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.contingency_tables")
    rng = np.random.default_rng(15)
    n, k = 1000, 10
    theta = rng.normal(size=n)
    g = (np.arange(n) % 2).astype(int)
    diff = rng.normal(size=k)
    X = (rng.random((n, k)) < 1 / (1 + np.exp(-(theta[:, None] - diff[None, :])))).astype(int)
    ext = X[:, :5].sum(axis=1)               # match on first 5 items only
    res = mantel_haenszel(X, g, match=ext)
    for j in range(k):
        tables = []
        for lvl in np.unique(ext):
            at = ext == lvl
            ref, fo = at & (g == 0), at & (g == 1)
            A = int(np.sum(X[ref, j] == 1)); B = int(np.sum(X[ref, j] == 0))
            C = int(np.sum(X[fo, j] == 1)); D = int(np.sum(X[fo, j] == 0))
            if A + B + C + D:
                tables.append([[A, B], [C, D]])
        st = sm.StratifiedTable(np.array(tables).transpose(1, 2, 0))
        assert res.loc[j, "stat"] == pytest.approx(
            st.test_null_odds(correction=True).statistic, abs=1e-4)
