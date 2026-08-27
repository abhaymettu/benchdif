"""Mantel-Haenszel DIF, matching difR::difMH defaults.

The Mantel-Haenszel (MH) procedure tests whether a dichotomous item favors one
group over another *after conditioning on ability* (the total test score). For a
studied item, examinees are stratified by their total score k, and at each level
a 2x2 table is formed:

               correct   incorrect | total
  reference     A_k        B_k      | nR_k
  focal         C_k        D_k      | nF_k
  total         m1_k       m0_k     | T_k

We compute (Holland & Thayer 1988):
  alpha_MH  = sum_k(A_k D_k / T_k) / sum_k(B_k C_k / T_k)      common odds ratio
  MH D-DIF  = -2.35 * ln(alpha_MH)                            ETS delta scale
  chi2_MH   = (|sum A_k - sum E[A_k]| - 0.5)^2 / sum Var[A_k]  continuity-corrected
    E[A_k]   = nR_k m1_k / T_k
    Var[A_k] = nR_k nF_k m1_k m0_k / (T_k^2 (T_k - 1))

An item is flagged if chi2_MH > qchisq(.95, 1) = 3.841. ETS A/B/C classes follow
difR: A = not significant or |D-DIF|<1; C = significant and |D-DIF|>=1.5; B = rest.

Reference group is coded 0, focal group 1 (difR's convention with focal named).
Matching is on the total score across ALL items, studied item included (difR
default match="score"). Purification is not yet implemented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2 as _chi2

CHI2_CRIT = _chi2.ppf(0.95, 1)  # 3.8414588...


def mantel_haenszel(responses, group) -> pd.DataFrame:
    """Run MH DIF on every item.

    Parameters
    ----------
    responses : array-like, shape (n_persons, n_items)
        Dichotomous 0/1 responses. No missing values.
    group : array-like, shape (n_persons,)
        0 = reference, 1 = focal.

    Returns
    -------
    DataFrame indexed by item position with columns:
        stat      MH chi-square (continuity corrected)
        p_value   upper-tail p on 1 df
        alpha_mh  MH common odds ratio
        mh_ddif   ETS MH D-DIF (delta scale)
        flag      bool, stat > 3.841
        ets       'A' | 'B' | 'C'
    """
    X = np.asarray(responses, dtype=float)
    g = np.asarray(group).ravel()
    if X.ndim != 2:
        raise ValueError("responses must be 2-D (persons x items)")
    if X.shape[0] != g.shape[0]:
        raise ValueError("responses and group disagree on n_persons")
    if np.isnan(X).any():
        raise ValueError("missing values not supported yet (listwise-drop upstream)")
    uniq = set(np.unique(g).tolist())
    if not uniq <= {0, 1}:
        raise ValueError(f"group must be coded 0/1, got {sorted(uniq)}")

    total = X.sum(axis=1)              # matching score = sum over all items
    is_ref = g == 0
    is_foc = g == 1
    n_items = X.shape[1]
    rows = []

    for j in range(n_items):
        item = X[:, j]
        num_alpha = den_alpha = 0.0
        sum_A = sum_E = sum_V = 0.0
        for k in np.unique(total):
            at = total == k
            A = float(np.sum(item[at & is_ref] == 1))
            B = float(np.sum(item[at & is_ref] == 0))
            C = float(np.sum(item[at & is_foc] == 1))
            D = float(np.sum(item[at & is_foc] == 0))
            T = A + B + C + D
            if T == 0:
                continue
            num_alpha += A * D / T
            den_alpha += B * C / T
            nR, nF = A + B, C + D
            m1, m0 = A + C, B + D
            sum_A += A
            sum_E += nR * m1 / T
            if T > 1:
                sum_V += (nR * nF * m1 * m0) / (T * T * (T - 1))

        alpha = num_alpha / den_alpha if den_alpha > 0 else np.inf
        ddif = -2.35 * np.log(alpha) if 0 < alpha < np.inf else np.nan
        stat = (abs(sum_A - sum_E) - 0.5) ** 2 / sum_V if sum_V > 0 else 0.0
        p = float(_chi2.sf(stat, 1))
        flag = stat > CHI2_CRIT
        a = abs(ddif)
        if not flag or np.isnan(a) or a < 1.0:
            ets = "A"
        elif a >= 1.5:
            ets = "C"
        else:
            ets = "B"
        rows.append(dict(stat=stat, p_value=p, alpha_mh=alpha,
                         mh_ddif=ddif, flag=bool(flag), ets=ets))

    return pd.DataFrame(rows)
