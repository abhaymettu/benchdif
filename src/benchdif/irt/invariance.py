"""Measurement-invariance omnibus for two groups (2PL).

A global, scale-level complement to item-level DIF. Fits a nested ladder and tests
each step by likelihood-ratio:

  configural : all item params free per group; focal ability fixed N(0,1)
  metric     : slopes equal across groups; focal variance freed
  scalar     : slopes AND intercepts equal; focal mean freed

  metric LR = 2(ll_configural - ll_metric) ~ chi2(J-1)   -> equal loadings?
  scalar LR = 2(ll_metric     - ll_scalar) ~ chi2(J-1)   -> equal intercepts?

Freeing a focal distribution parameter as each measurement constraint is added is
the standard identification for IRT invariance (cf. mirt multipleGroup): the equal
slopes identify the focal variance, the equal intercepts identify the focal mean, so
group ability differences (impact) are not confused with non-invariance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2 as _chi2

from benchdif.irt.multigroup import fit_multigroup_2pl


def invariance(responses, group, n_nodes=41, alpha=0.05) -> pd.DataFrame:
    """Run the configural -> metric -> scalar invariance ladder.

    Returns a DataFrame with rows 'metric' and 'scalar', columns:
    stat (LR chi-square), df, p_value, holds (bool, p > alpha).
    """
    X = np.asarray(responses, dtype=float)
    g = np.asarray(group).ravel()
    if not set(np.unique(g).tolist()) <= {0, 1}:
        raise ValueError("group must be coded 0/1")
    J = X.shape[1]
    free = np.array(["free"] * J, dtype=object)
    eqa = np.array(["equal_a"] * J, dtype=object)
    eqb = np.array(["equal_both"] * J, dtype=object)

    conf = fit_multigroup_2pl(X, g, constraints=free,
                              fix_focal_mean=True, fix_focal_var=True, n_nodes=n_nodes)
    metr = fit_multigroup_2pl(X, g, constraints=eqa,
                              fix_focal_mean=True, fix_focal_var=False, n_nodes=n_nodes)
    scal = fit_multigroup_2pl(X, g, constraints=eqb,
                              fix_focal_mean=False, fix_focal_var=False, n_nodes=n_nodes)

    df = J - 1
    lr_m = max(2 * (conf.loglik - metr.loglik), 0.0)
    lr_s = max(2 * (metr.loglik - scal.loglik), 0.0)
    p_m = float(_chi2.sf(lr_m, df))
    p_s = float(_chi2.sf(lr_s, df))
    return pd.DataFrame(
        [{"level": "metric", "stat": lr_m, "df": df, "p_value": p_m, "holds": p_m > alpha},
         {"level": "scalar", "stat": lr_s, "df": df, "p_value": p_s, "holds": p_s > alpha}]
    ).set_index("level")
