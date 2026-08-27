"""Unified entry point: run any DIF method and get one tidy summary.

Each method returns its own native columns; `detect` normalizes them to a common
shape so results are comparable and easy to filter:

    method       which test was run
    stat         the test statistic (chi-square)
    p_value      its p-value
    flag         bool, significant DIF
    effect       the method's primary effect size
    effect_class ordinal severity where the method defines one ('A'/'B'/'C'), else NA

Native columns are kept too, so nothing is lost.
"""
from __future__ import annotations

import pandas as pd

from benchdif.dif import mantel_haenszel, logistic, irt_lr

_EFFECT = {"mh": ("mh_ddif", "ets"),
           "logistic": ("delta_r2", "jg"),
           "irt": ("db", None)}


def detect(responses, group, method: str = "mh", **kwargs) -> pd.DataFrame:
    """Run a DIF method and return a normalized summary.

    Parameters
    ----------
    responses : (n_persons x n_items) 0/1 array.
    group : (n_persons,) coded 0=reference, 1=focal.
    method : 'mh' (Mantel-Haenszel), 'logistic' (logistic regression),
        or 'irt' (anchor-purified IRT-LR).
    **kwargs : forwarded to the underlying method (e.g. purify=True for 'mh',
        kind='uniform' for 'logistic', n_anchor for 'irt').

    Returns a DataFrame per item with normalized columns [method, stat, p_value,
    flag, effect, effect_class] followed by the method's native columns.
    """
    runners = {"mh": mantel_haenszel, "logistic": logistic, "irt": irt_lr}
    if method not in runners:
        raise ValueError(f"method must be one of {sorted(runners)}, got {method!r}")
    native = runners[method](responses, group, **kwargs)
    eff_col, cls_col = _EFFECT[method]
    out = pd.DataFrame({
        "method": method,
        "stat": native["stat"],
        "p_value": native["p_value"],
        "flag": native["flag"],
        "effect": native[eff_col],
        "effect_class": native[cls_col] if cls_col else pd.NA,
    })
    # append native-only columns (avoid duplicating the normalized ones)
    keep = [c for c in native.columns if c not in {"stat", "p_value", "flag", eff_col}]
    return out.join(native[keep], rsuffix="_native")
