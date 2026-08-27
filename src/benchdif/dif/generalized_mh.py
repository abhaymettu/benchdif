"""Generalized Mantel-Haenszel DIF for more than two groups (Somes 1986; Penfield).

Real leaderboards compare many model families at once. The generalized MH tests,
for each item, whether the correct-response rate differs across G groups after
conditioning on ability (the total score) -- a single omnibus (G-1) df chi-square,
the multi-group analogue of the two-group MH.

At each score stratum k a 2 x G table is formed (correct/incorrect by group). Let
n_gk be group g's size, a_gk its correct count, m1k/m0k the stratum correct/incorrect
totals, Tk the stratum total. Using the first G-1 groups (the last is redundant):

  E[a_gk]      = n_gk m1k / Tk
  Var[a_gk]    = n_gk (Tk - n_gk) m1k m0k / (Tk^2 (Tk - 1))
  Cov[a_gk,a_hk] = -n_gk n_hk m1k m0k / (Tk^2 (Tk - 1))

  stat = (sum_k A_k - E_k)' (sum_k V_k)^{-1} (sum_k A_k - E_k)  ~  chi2(G-1)

For G = 2 this equals the uncorrected two-group MH / CMH statistic (a validation
anchor). No continuity correction is applied, matching difR::difGMH.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2 as _chi2


def generalized_mantel_haenszel(responses, group) -> pd.DataFrame:
    """Generalized MH DIF across G>=2 groups.

    responses : (n_persons x n_items) 0/1 array, no missing.
    group : (n_persons,) with G>=2 distinct labels (any hashable/orderable values).

    Returns per-item DataFrame: stat (chi-square), df (= G-1), p_value, flag (p<.05).
    """
    X = np.asarray(responses, dtype=float)
    g = np.asarray(group).ravel()
    if X.ndim != 2:
        raise ValueError("responses must be 2-D (persons x items)")
    if X.shape[0] != g.shape[0]:
        raise ValueError("responses and group disagree on n_persons")
    if np.isnan(X).any():
        raise ValueError("missing values not supported")
    levels = sorted(set(g.tolist()))
    G = len(levels)
    if G < 2:
        raise ValueError("need at least 2 groups")
    df = G - 1
    total = X.sum(axis=1)
    # boolean membership per group, in fixed order
    memb = [g == lv for lv in levels]
    n_items = X.shape[1]
    rows = []
    for j in range(n_items):
        item = X[:, j]
        diff = np.zeros(df)
        V = np.zeros((df, df))
        for k in np.unique(total):
            at = total == k
            n_g = np.array([np.sum(at & m) for m in memb], dtype=float)
            a_g = np.array([np.sum(item[at & m] == 1) for m in memb], dtype=float)
            Tk = n_g.sum()
            m1 = a_g.sum(); m0 = Tk - m1
            if Tk < 2 or m1 == 0 or m0 == 0:
                continue
            E = n_g * m1 / Tk
            diff += (a_g - E)[:df]
            fac = m1 * m0 / (Tk * Tk * (Tk - 1))
            # covariance of the first df group correct-counts
            for gi in range(df):
                V[gi, gi] += n_g[gi] * (Tk - n_g[gi]) * fac
                for hi in range(gi + 1, df):
                    cov = -n_g[gi] * n_g[hi] * fac
                    V[gi, hi] += cov
                    V[hi, gi] += cov
        try:
            stat = float(diff @ np.linalg.solve(V, diff))
        except np.linalg.LinAlgError:
            stat = float(diff @ np.linalg.pinv(V) @ diff)
        stat = max(stat, 0.0)
        p = float(_chi2.sf(stat, df))
        rows.append(dict(stat=stat, df=df, p_value=p, flag=p < 0.05))
    return pd.DataFrame(rows)
