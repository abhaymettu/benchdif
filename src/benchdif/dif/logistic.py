"""Logistic-regression DIF (Swaminathan & Rogers 1990; Zumbo 1999).

For each item, the 0/1 response is modeled from the matching score S (total test
score) and group G via nested logistic regressions:

    M0 : logit P = b0 + b1 S
    M1 : logit P = b0 + b1 S + b2 G                      (adds uniform DIF)
    M2 : logit P = b0 + b1 S + b2 G + b3 (S*G)           (adds non-uniform DIF)

Likelihood-ratio tests (matching difR::difLogistic, criterion="LRT"):
    both       : 2(ll_M2 - ll_M0) ~ chi2(2)
    uniform    : 2(ll_M1 - ll_M0) ~ chi2(1)
    nonuniform : 2(ll_M2 - ll_M1) ~ chi2(1)

Effect size is the Nagelkerke pseudo-R^2 increase from M0 to M2, classified by
Jodoin & Gierl (2001): A (<.035), B (.035-.07), C (>=.07).

The logistic fit is Newton-Raphson / IRLS, the same estimator R's glm(binomial)
uses, so coefficients and deviance agree with glm to convergence tolerance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2 as _chi2


def _irls(X, y, max_iter=100, tol=1e-10):
    """Fit logistic regression by IRLS. X includes an intercept column.

    Returns (beta, loglik). Mirrors glm(family=binomial): Newton steps on the
    binomial log-likelihood with the canonical link.
    """
    n, p = X.shape
    beta = np.zeros(p)
    ll_old = -np.inf
    for _ in range(max_iter):
        eta = X @ beta
        # clip to avoid overflow in exp; mu stays in (eps, 1-eps)
        eta = np.clip(eta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = mu * (1.0 - mu)
        W = np.clip(W, 1e-10, None)
        z = eta + (y - mu) / W
        # weighted least squares: (X' W X) beta = X' W z
        XtW = X.T * W
        try:
            beta = np.linalg.solve(XtW @ X, XtW @ z)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(XtW @ X, XtW @ z, rcond=None)[0]
        mu = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
        mu = np.clip(mu, 1e-12, 1 - 1e-12)
        ll = float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll
    return beta, ll


def _nagelkerke(ll_full, ll_null, n):
    cox = 1 - np.exp((2.0 / n) * (ll_null - ll_full))
    denom = 1 - np.exp((2.0 / n) * ll_null)
    return cox / denom if denom > 0 else np.nan


def logistic(responses, group, kind: str = "both") -> pd.DataFrame:
    """Logistic-regression DIF on every item.

    Parameters
    ----------
    responses : (n_persons x n_items) 0/1 array, no missing.
    group : (n_persons,) coded 0=reference, 1=focal.
    kind : 'both' (2 df), 'uniform' (1 df), or 'nonuniform' (1 df).

    Returns DataFrame per item: stat (LR chi-square), df, p_value,
    delta_r2 (Nagelkerke, M0->M2), flag (p<.05), jg (A/B/C).
    """
    X = np.asarray(responses, dtype=float)
    g = np.asarray(group, dtype=float).ravel()
    if X.ndim != 2:
        raise ValueError("responses must be 2-D (persons x items)")
    if X.shape[0] != g.shape[0]:
        raise ValueError("responses and group disagree on n_persons")
    if np.isnan(X).any():
        raise ValueError("missing values not supported yet")
    if not set(np.unique(g).tolist()) <= {0.0, 1.0}:
        raise ValueError("group must be coded 0/1")
    if kind not in {"both", "uniform", "nonuniform"}:
        raise ValueError("kind must be 'both', 'uniform', or 'nonuniform'")

    n, n_items = X.shape
    total = X.sum(axis=1)
    ones = np.ones(n)
    rows = []
    for j in range(n_items):
        y = X[:, j]
        S = total  # difR matches on total score including studied item
        d0 = np.column_stack([ones, S])
        d1 = np.column_stack([ones, S, g])
        d2 = np.column_stack([ones, S, g, S * g])
        _, ll0 = _irls(d0, y)
        _, ll1 = _irls(d1, y)
        _, ll2 = _irls(d2, y)
        if kind == "both":
            stat, df = 2 * (ll2 - ll0), 2
        elif kind == "uniform":
            stat, df = 2 * (ll1 - ll0), 1
        else:
            stat, df = 2 * (ll2 - ll1), 1
        stat = max(stat, 0.0)
        p = float(_chi2.sf(stat, df))
        # effect size always M0 -> M2 (Zumbo-Thomas), classified Jodoin-Gierl
        dr2 = _nagelkerke(ll2, ll0, n)
        flag = p < 0.05
        a = abs(dr2) if not np.isnan(dr2) else 0.0
        if not flag or a < 0.035:
            jg = "A"
        elif a >= 0.07:
            jg = "C"
        else:
            jg = "B"
        rows.append(dict(stat=stat, df=df, p_value=p, delta_r2=dr2,
                         flag=bool(flag), jg=jg))
    return pd.DataFrame(rows)
