"""Multiple-comparison correction and data-adequacy checks.

A benchmark has hundreds or thousands of items, so per-item DIF testing at a
nominal 0.05 produces false positives by construction (1500 items -> ~75 expected).
Any honest DIF report over a real benchmark must correct for multiplicity; this is
the single biggest difference between a usable finding and a table of noise.

Also provides `check_adequacy`, which flags the data shapes that silently break
conditional DIF tests: too few persons for score stratification, and items with no
response variance.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def adjust(p_values, method: str = "bh") -> np.ndarray:
    """Adjust p-values for multiple testing.

    method : 'bh' (Benjamini-Hochberg FDR, the sane default for screening many
        items) or 'holm' (family-wise error, stricter).
    Returns adjusted p-values in the original order; NaNs are preserved.
    """
    p = np.asarray(p_values, dtype=float)
    ok = ~np.isnan(p)
    vals = p[ok]
    m = vals.size
    if m == 0:
        return p.copy()
    order = np.argsort(vals)
    sorted_p = vals[order]
    if method == "bh":
        ranks = np.arange(1, m + 1)
        adj_sorted = np.minimum.accumulate((sorted_p * m / ranks)[::-1])[::-1]
    elif method == "holm":
        mult = m - np.arange(m)
        adj_sorted = np.maximum.accumulate(sorted_p * mult)
    else:
        raise ValueError("method must be 'bh' or 'holm'")
    adj_sorted = np.clip(adj_sorted, 0, 1)
    adj_vals = np.empty(m)
    adj_vals[order] = adj_sorted
    out = p.copy()
    out[ok] = adj_vals
    return out


def check_adequacy(responses, group, warn: bool = True) -> dict:
    """Report data conditions that invalidate or weaken conditional DIF tests.

    Returns a dict with n_persons, n_items, per-group sizes, the number of
    zero-variance (degenerate) items, and `persons_per_stratum` -- the average
    number of people per distinct total-score level. When that is near 1, score
    stratification collapses and Mantel-Haenszel has essentially no power, which
    otherwise looks indistinguishable from "no DIF found".
    """
    X = np.asarray(responses, dtype=float)
    g = np.asarray(group).ravel()
    n, J = X.shape
    total = X.sum(axis=1)
    n_strata = len(np.unique(total))
    per_stratum = n / max(n_strata, 1)
    degenerate = int(np.sum(X.var(axis=0) == 0))
    sizes = {str(lv): int(np.sum(g == lv)) for lv in sorted(set(g.tolist()))}
    info = dict(n_persons=n, n_items=J, group_sizes=sizes,
                degenerate_items=degenerate, n_strata=n_strata,
                persons_per_stratum=float(per_stratum))
    if warn:
        if per_stratum < 3:
            warnings.warn(
                f"only {per_stratum:.1f} persons per score stratum "
                f"({n} persons, {n_strata} score levels): Mantel-Haenszel has "
                "very low power here and may report no DIF simply because the "
                "strata are empty. Prefer irt_lr, or coarsen the matching score.",
                stacklevel=2)
        if degenerate:
            warnings.warn(f"{degenerate} items have zero variance (all-correct or "
                          "all-incorrect); they carry no DIF information.",
                          stacklevel=2)
        if min(sizes.values()) < 30:
            warnings.warn(f"smallest group has {min(sizes.values())} persons; DIF "
                          "estimates are unstable below ~30 per group.",
                          stacklevel=2)
    return info
