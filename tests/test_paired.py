"""Paired-condition DIF: impact matches conditional logit, DIF is found, slopes map by label."""
import numpy as np
import pytest

from benchdif.dif import paired_dif


def _sim(seed, K=6, I=120, T=4, shift=0.8, dif=0.0, a=None):
    rng = np.random.default_rng(seed)
    a = np.ones(I) if a is None else a
    b = rng.normal(0, 1.5, I)
    delta = np.zeros(I); delta[:12] = dif * np.where(np.arange(12) % 2, -1, 1)
    out = []
    for m in range(K):
        th, u = rng.normal(), rng.normal(0, 1, I)
        p0 = 1 / (1 + np.exp(-(a * (th - b) + u)))
        p1 = 1 / (1 + np.exp(-(a * (th + shift - b) + u + delta)))
        out.append((np.full(I, m), np.arange(I), rng.binomial(T, p0), np.full(I, T),
                    rng.binomial(T, p1), np.full(I, T)))
    return [np.concatenate(c) for c in zip(*out)]


def test_impact_matches_conditional_logit():
    cm = pytest.importorskip("statsmodels.discrete.conditional_models")
    m, it, k0, n0, k1, n1 = _sim(0, K=1, I=50)
    res = paired_dif(m, it, k0, n0, k1, n1, n_boot=5)
    y, x, grp = [], [], []
    for r in range(len(m)):
        for c, k, n in ((0, k0[r], n0[r]), (1, k1[r], n1[r])):
            y += [1] * k + [0] * (n - k); x += [c] * n; grp += [r] * n
    fit = cm.ConditionalLogit(np.array(y), np.array(x, float)[:, None],
                              groups=np.array(grp)).fit(method="newton", tol=1e-12, disp=0)
    assert res.impact.delta_hat.iloc[0] == pytest.approx(fit.params[0], abs=1e-8)


def test_null_vs_planted_dif():
    null = paired_dif(*_sim(1), n_boot=200)
    assert null.p_value > 0.05 and not null.items.flag.any()
    assert null.impact.delta_hat.mean() == pytest.approx(0.8, abs=0.35)
    hit = paired_dif(*_sim(1, dif=2.0), n_boot=200)
    assert hit.p_value < 0.01 and hit.var_ratio > 1.3


def test_slope_array_is_keyed_by_label():
    a = np.exp(np.random.default_rng(2).normal(0, 0.4, 120))
    cols = _sim(3, shift=1.5, a=a)
    by_dict = paired_dif(*cols, slope=dict(enumerate(a)), n_boot=20)
    by_array = paired_dif(*cols, slope=a, n_boot=20)
    assert by_dict.stat == pytest.approx(by_array.stat)


def test_dtf_detects_one_sided_and_cancels_balanced():
    import pandas as pd
    from benchdif.dif import paired_dtf

    def frame(one_sided):
        rng = np.random.default_rng(4)
        d = np.zeros(150); d[:30] = 2.0 if one_sided else np.where(np.arange(30) % 2, -2.0, 2.0)
        b = rng.normal(0, 1.5, 150)
        rows = []
        for m in range(4):
            eta = rng.normal() - b + rng.normal(0, 1, 150)
            rows.append(pd.DataFrame({"cell": f"m{m}", "item_id": np.arange(150),
                                      "k0": rng.binomial(8, 1 / (1 + np.exp(-eta))), "n0": 8,
                                      "k1": rng.binomial(8, 1 / (1 + np.exp(-(eta + 0.8 + d)))),
                                      "n1": 8}))
        c = pd.concat(rows, ignore_index=True)
        return pd.concat([paired_dtf(c[c.cell != m], c[c.cell == m]) for m in c.cell.unique()])

    one, bal = frame(True), frame(False)   # same draws, only the DIF signs differ
    assert one.G_obs.mean() - bal.G_obs.mean() > 0.02
    assert one.G_pred.mean() - bal.G_pred.mean() > 0.02
    assert (one.gap > one.G_obs).all()     # impact carries most of the gap
