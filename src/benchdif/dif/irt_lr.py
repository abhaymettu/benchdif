"""IRT likelihood-ratio DIF (Thissen, Steinberg & Wainer) with anchor purification.

The studied item is tested with a concurrent two-group 2PL by comparing:

  M0_j : item j shared across groups
  M1_j : item j freed across groups

    LR_j = 2 (loglik(M1_j) - loglik(M0_j))  ~  chi2(2)

Only a DIF-free *anchor* set is held shared across groups to fix the metric and
estimate the focal ability distribution (impact). Every non-anchor item other than
j is *freed* in both M0_j and M1_j, so DIF elsewhere cannot contaminate the test of
item j -- the failure mode of the naive all-items-shared baseline.

Anchor selection is a two-stage scheme (rank-based, cf. Kopf, Zeileis & Strobl):
stage 1 runs the naive all-shared test to rank items by DIF; the least-DIF items
become the anchor; stage 2 does the purified test above.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2 as _chi2

from benchdif.irt.multigroup import fit_multigroup_2pl


def _naive_stats(X, g, n_nodes):
    """Stage 1: all-shared baseline vs freeing each item; raw LR to rank anchors."""
    J = X.shape[1]
    base = fit_multigroup_2pl(X, g, np.zeros(J, bool), n_nodes=n_nodes)
    stats = np.empty(J)
    for j in range(J):
        mask = np.zeros(J, bool); mask[j] = True
        f = fit_multigroup_2pl(X, g, mask, n_nodes=n_nodes)
        stats[j] = max(2 * (f.loglik - base.loglik), 0.0)
    return stats


def irt_lr(responses, group, n_nodes=41, n_anchor=None) -> pd.DataFrame:
    """Anchor-purified IRT-LR DIF on every item (2 df test).

    n_anchor : size of the DIF-free anchor set (default max(4, J//4)). The anchor
    items are the lowest-DIF items from a stage-1 screen; they are reported with
    is_anchor=True and are not themselves tested.

    Returns per-item DataFrame: stat (LR chi-square), df, p_value, flag (p<.05),
    da (a_foc-a_ref), db (b_foc-b_ref), is_anchor.
    """
    X = np.asarray(responses, dtype=float)
    g = np.asarray(group).ravel()
    if X.ndim != 2:
        raise ValueError("responses must be 2-D (persons x items)")
    if X.shape[0] != g.shape[0]:
        raise ValueError("responses and group disagree on n_persons")
    if not set(np.unique(g).tolist()) <= {0, 1}:
        raise ValueError("group must be coded 0/1")
    J = X.shape[1]
    if n_anchor is None:
        n_anchor = max(4, J // 4)
    n_anchor = min(n_anchor, J - 1)

    raw = _naive_stats(X, g, n_nodes)
    anchor = set(np.argsort(raw)[:n_anchor].tolist())  # lowest-DIF items = anchor

    non_anchor = np.array([j for j in range(J) if j not in anchor])
    # M1 (all non-anchor freed) is the same for every tested item -> fit once
    free_all = np.zeros(J, bool); free_all[non_anchor] = True
    full = fit_multigroup_2pl(X, g, free_all, n_nodes=n_nodes)

    rows = []
    for j in range(J):
        if j in anchor:
            rows.append(dict(stat=np.nan, df=2, p_value=np.nan, flag=False,
                             da=np.nan, db=np.nan, is_anchor=True))
            continue
        mask0 = free_all.copy(); mask0[j] = False          # M0_j: item j shared
        f0 = fit_multigroup_2pl(X, g, mask0, n_nodes=n_nodes)
        stat = max(2 * (full.loglik - f0.loglik), 0.0)
        p = float(_chi2.sf(stat, 2))
        b_ref = -full.d_ref[j] / full.a_ref[j]
        b_foc = -full.d_foc[j] / full.a_foc[j]
        rows.append(dict(stat=stat, df=2, p_value=p, flag=p < 0.05,
                         da=full.a_foc[j] - full.a_ref[j], db=b_foc - b_ref,
                         is_anchor=False))
    return pd.DataFrame(rows)
