"""Paired-condition DIF: the same model under two conditions (harness, effort).

Person-level DIF needs many persons per group. A harness or effort comparison has
a few models, each run under both conditions on the same items, often with
repeated trials. The unit here is the (model m, item i) stratum: k0 of n0 trials
succeed under condition 0, k1 of n1 under condition 1, with success logit

    eta_mi + c * (a_i * Delta_m + delta_i)

eta_mi absorbs everything the two conditions share for that model and item,
including the model-specific item quirks that dominate cross-model variance.
Conditioning on the stratum total k0 + k1 removes eta_mi exactly: k1 then follows
Fisher's noncentral hypergeometric law with log odds ratio a_i*Delta_m + delta_i.
No ability model and no anchor items are needed.

    Delta_m : impact, the condition's shift for model m (logit scale)
    delta_i : DIF, the item-specific deviation from that shift
    a_i     : slope. 1 (default) is the Rasch null; fitted 2PL slopes give the
              2PL null, under which a pure ability shift moves steep items more.

Delta is fitted by conditional ML with delta = 0. Each item's score U_i (observed
minus expected k1 summed over its strata) gets an exact p-value from the
convolution of its strata's null laws. The omnibus T = sum_i U_i^2 / V_i is
calibrated by a conditional parametric bootstrap that refits Delta per replicate.
T over its null mean is the variance ratio of FINDINGS.md section 1: how much more
the item-level gaps vary than sampling alone allows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import gammaln

from benchdif.multiplicity import adjust


@dataclass
class PairedDIF:
    impact: pd.DataFrame   # per model: delta_hat (logit), se, n_strata
    items: pd.DataFrame    # per item: u, v, delta, p_value, q_value, flag, d_dif, ets
    stat: float            # omnibus T
    p_value: float         # bootstrap p for "no DIF beyond impact"
    null_mean: float       # bootstrap mean of T under the null
    var_ratio: float       # stat / null_mean


def _lognchg_base(k0, n0, k1, n1):
    """Support grid and log base weights of k1 given the stratum total."""
    s = k0 + k1
    kmax = int(n1.max())
    x = np.arange(kmax + 1)[None, :]
    lo = np.maximum(0, s - n0)[:, None]
    hi = np.minimum(s, n1)[:, None]
    ok = (x >= lo) & (x <= hi)
    with np.errstate(invalid="ignore"):
        lb = (gammaln(n1[:, None] + 1) - gammaln(x + 1) - gammaln(n1[:, None] - x + 1)
              + gammaln(n0[:, None] + 1) - gammaln(s[:, None] - x + 1)
              - gammaln(n0[:, None] - s[:, None] + x + 1))
    return x, np.where(ok, lb, -np.inf)


def _pmf(x, lb, psi):
    lw = lb + psi[:, None] * x
    lw -= lw.max(axis=1, keepdims=True)
    w = np.exp(lw)
    return w / w.sum(axis=1, keepdims=True)


def _fit_delta(k1, x, lb, a, m, n_models, iters=50):
    """Conditional ML of Delta_m (delta = 0) by Newton, vectorized over models."""
    D = np.zeros(n_models)
    for _ in range(iters):
        p = _pmf(x, lb, a * D[m])
        e = (p * x).sum(1)
        v = (p * x * x).sum(1) - e * e
        u = np.bincount(m, a * (k1 - e), n_models)
        info = np.bincount(m, a * a * v, n_models)
        step = np.divide(u, info, out=np.zeros(n_models), where=info > 0)
        D = np.clip(D + step, -10, 10)
        if np.max(np.abs(step)) < 1e-10:
            break
    return D, info


def _scores(k1, x, lb, a, D, m, it, n_items):
    p = _pmf(x, lb, a * D[m])
    e = (p * x).sum(1)
    v = (p * x * x).sum(1) - e * e
    return (np.bincount(it, k1 - e, n_items), np.bincount(it, v, n_items), p)


def paired_dif(model, item, k0, n0, k1, n1, slope=None, n_boot=500,
               alpha=0.05, seed=0) -> PairedDIF:
    """Test whether a condition change is pure impact or also DIF.

    model, item : (S,) labels of each (model, item) stratum.
    k0, n0, k1, n1 : (S,) successes / trials under condition 0 and 1.
    slope : optional dict {item label: a_i}, or an array aligned with
        np.unique(item) (default 1 for every item: the Rasch null).
    n_boot : bootstrap replicates for the omnibus p-value.

    Strata with no information (all successes or all failures across both
    conditions) are dropped. Items are flagged at BH q < alpha; the ETS class
    uses D = 2.35 * delta (positive = easier under condition 1 than impact
    predicts): A if not flagged or |D| < 1, C if |D| >= 1.5, else B.
    """
    k0, n0, k1, n1 = (np.asarray(v, dtype=np.int64) for v in (k0, n0, k1, n1))
    model = np.asarray(model); item = np.asarray(item)
    if np.any((k0 < 0) | (k0 > n0) | (k1 < 0) | (k1 > n1)):
        raise ValueError("need 0 <= k <= n in every stratum")
    labels = np.unique(item)
    s = k0 + k1
    keep = (s > 0) & (s < n0 + n1)
    if not keep.any():
        raise ValueError("no informative strata")
    model, item, k0, n0, k1, n1 = (v[keep] for v in (model, item, k0, n0, k1, n1))
    mods, m = np.unique(model, return_inverse=True)
    its, it = np.unique(item, return_inverse=True)
    if slope is None:
        a = np.ones(len(item))
    elif isinstance(slope, dict):
        a = np.array([slope[i] for i in item], dtype=float)
    else:
        a = np.asarray(slope, dtype=float)[np.searchsorted(labels, item)]

    x, lb = _lognchg_base(k0, n0, k1, n1)
    D, info = _fit_delta(k1, x, lb, a, m, len(mods))
    u, v, p = _scores(k1, x, lb, a, D, m, it, len(its))
    ok = v > 1e-12
    T = float(np.sum(u[ok] ** 2 / v[ok]))

    # exact per-item p: convolve the strata's null laws, two-sided by probability
    pv = np.full(len(its), np.nan)
    for j in np.where(ok)[0]:
        rows = np.where(it == j)[0]
        dist = np.array([1.0]); obs = 0
        for r in rows:
            dist = np.convolve(dist, p[r]); obs += k1[r]
        pv[j] = min(1.0, dist[dist <= dist[obs] * (1 + 1e-7)].sum())
    q = adjust(pv, "bh")
    delta = np.divide(u, v, out=np.zeros_like(u), where=ok)
    d = 2.35 * delta
    flag = np.nan_to_num(q, nan=1.0) < alpha
    ets = np.where(~flag | (np.abs(d) < 1), "A", np.where(np.abs(d) >= 1.5, "C", "B"))

    rng = np.random.default_rng(seed)
    cdf = np.cumsum(p, axis=1)
    hi = np.minimum(s[keep], n1)
    Tb = np.empty(n_boot)
    for b in range(n_boot):
        kb = np.minimum((rng.random(len(k1))[:, None] > cdf).sum(1), hi)
        Db, _ = _fit_delta(kb, x, lb, a, m, len(mods))
        ub, vb, _ = _scores(kb, x, lb, a, Db, m, it, len(its))
        okb = vb > 1e-12
        Tb[b] = np.sum(ub[okb] ** 2 / vb[okb])
    null_mean = float(Tb.mean())

    impact = pd.DataFrame({"model": mods, "delta_hat": D,
                           "se": 1 / np.sqrt(np.where(info > 0, info, np.nan)),
                           "n_strata": np.bincount(m, minlength=len(mods))})
    items = pd.DataFrame({"item": its, "u": u, "v": v, "delta": delta,
                          "p_value": pv, "q_value": q, "flag": flag,
                          "d_dif": d, "ets": ets})
    return PairedDIF(impact=impact, items=items, stat=T,
                     p_value=float((1 + np.sum(Tb >= T)) / (n_boot + 1)),
                     null_mean=null_mean, var_ratio=T / null_mean)
