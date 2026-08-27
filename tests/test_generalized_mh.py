"""Generalized (>2 group) Mantel-Haenszel DIF."""
import numpy as np
import pytest

from benchdif import generalized_mantel_haenszel as gmh


def test_reduces_to_uncorrected_cmh_when_two_groups():
    sm = pytest.importorskip("statsmodels.stats.contingency_tables")
    rng = np.random.default_rng(3)
    n, k = 1500, 10
    th = rng.normal(size=n); g = (np.arange(n) % 2)
    diff = rng.normal(size=k)
    X = (rng.random((n, k)) < 1 / (1 + np.exp(-(th[:, None] - diff[None, :])))).astype(int)
    foc = g == 1
    X[foc, 4] = (rng.random(foc.sum()) < 1 / (1 + np.exp(-(th[foc] - diff[4] - 0.8)))).astype(int)
    res = gmh(X, g)
    total = X.sum(1)
    for j in range(k):
        tabs = []
        for lv in np.unique(total):
            at = total == lv; r, f = at & (g == 0), at & (g == 1)
            A = int((X[r, j] == 1).sum()); B = int((X[r, j] == 0).sum())
            C = int((X[f, j] == 1).sum()); D = int((X[f, j] == 0).sum())
            if A + B + C + D:
                tabs.append([[A, B], [C, D]])
        st = sm.StratifiedTable(np.array(tabs).transpose(1, 2, 0))
        assert res.loc[j, "stat"] == pytest.approx(
            st.test_null_odds(correction=False).statistic, abs=1e-6)
        assert res.loc[j, "df"] == 1


def test_three_groups_flags_dif():
    rng = np.random.default_rng(0)
    n, k, j = 2400, 12, 5
    th = rng.normal(size=n); grp = (np.arange(n) % 3)
    diff = rng.normal(scale=0.6, size=k)
    X = (rng.random((n, k)) < 1 / (1 + np.exp(-(th[:, None] - diff[None, :])))).astype(int)
    m = grp == 2
    X[m, j] = (rng.random(m.sum()) < 1 / (1 + np.exp(-(th[m] - diff[j] - 1.0)))).astype(int)
    res = gmh(X, grp)
    assert res.loc[j, "df"] == 2
    assert res.loc[j, "flag"]
    assert res.loc[j, "stat"] == res["stat"].max()   # injected item is the strongest signal


def test_bad_input():
    with pytest.raises(ValueError):
        gmh(np.zeros((3, 3)), np.array([0, 0, 0]))  # only one group
    with pytest.raises(ValueError):
        gmh(np.zeros((3, 3)), np.array([0, 1]))       # length mismatch
