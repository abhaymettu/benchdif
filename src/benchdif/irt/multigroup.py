"""Two-group 2PL calibration for IRT-based DIF (concurrent, anchor-identified).

Both groups share one ability metric. The reference group's ability is fixed
N(0,1) for identification; the focal group's ability N(mu, sigma) is estimated
(this absorbs group ability differences -- "impact" -- so it is not confused with
DIF). Item parameters are either *shared* across groups (anchors) or *free*
(group-specific), controlled by `free_mask`.

This is the machine behind the IRT-LR DIF test: fit the fully-constrained model
(all items shared) and, per studied item, a model freeing just that item, then
compare marginal log-likelihoods.

A fixed ability grid with normal-density prior weights is used (rectangular
quadrature), so the focal N(mu, sigma) prior is just a reweighting of the shared
nodes. numpy/scipy only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchdif.irt.twopl import _mstep_item, _mstep_equal_a


@dataclass
class MultiGroupFit:
    a_ref: np.ndarray
    d_ref: np.ndarray
    a_foc: np.ndarray
    d_foc: np.ndarray
    mu: float
    sigma: float
    loglik: float
    n_iter: int


def _grid(n_nodes=41, lo=-5.0, hi=5.0):
    return np.linspace(lo, hi, n_nodes)


def _normal_w(nodes, mu, sigma):
    w = np.exp(-0.5 * ((nodes - mu) / sigma) ** 2)
    return w / w.sum()


def _post(X_g, P, logw):
    """Posterior over nodes for a group. P is (Q x J), X_g is (n_g x J)."""
    P = np.clip(P, 1e-12, 1 - 1e-12)
    LL = X_g @ np.log(P).T + (1 - X_g) @ np.log(1 - P).T + logw[None, :]
    m = LL.max(axis=1, keepdims=True)
    e = np.exp(LL - m)
    denom = e.sum(axis=1, keepdims=True)
    ll = float(np.sum(m.ravel() + np.log(denom.ravel())))
    return e / denom, ll


def fit_multigroup_2pl(responses, group, free_mask=None, constraints=None,
                       fix_focal_mean=False, fix_focal_var=False,
                       n_nodes=41, max_iter=500, tol=1e-5) -> MultiGroupFit:
    """Concurrent two-group 2PL EM.

    responses : (n x J) 0/1. group : (n,) 0=reference, 1=focal.
    free_mask : (J,) bool; True = item free across groups, False = shared (both
        a and d). Back-compatible shortcut for `constraints`.
    constraints : (J,) of {'free','equal_a','equal_both'}; overrides free_mask.
        'equal_a' shares the slope but frees intercepts (the metric level).
    fix_focal_mean, fix_focal_var : hold the focal ability mean at 0 / variance at 1
        instead of estimating them (used to identify the configural/metric levels).

    Reference ability is fixed N(0,1); the focal ability N(mu,sigma) is estimated
    unless fixed.
    """
    X = np.asarray(responses, dtype=float)
    g = np.asarray(group).ravel()
    J = X.shape[1]
    if constraints is None:
        if free_mask is None:
            free_mask = np.zeros(J, bool)
        free = np.asarray(free_mask, dtype=bool)
        constraints = np.where(free, "free", "equal_both")
    constraints = np.asarray(constraints, dtype=object)
    X0, X1 = X[g == 0], X[g == 1]
    theta = _grid(n_nodes)
    w0 = _normal_w(theta, 0.0, 1.0)
    logw0 = np.log(w0)

    a_r = np.ones(J); d_r = np.zeros(J)
    a_f = np.ones(J); d_f = np.zeros(J)
    mu, sigma = 0.0, 1.0
    ll_old = -np.inf
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        w1 = _normal_w(theta, mu, sigma)
        logw1 = np.log(w1)
        Pr = 1 / (1 + np.exp(-np.clip(theta[:, None] * a_r + d_r, -30, 30)))
        Pf = 1 / (1 + np.exp(-np.clip(theta[:, None] * a_f + d_f, -30, 30)))
        r0, ll0 = _post(X0, Pr, logw0)
        r1, ll1 = _post(X1, Pf, logw1)
        ll = ll0 + ll1
        n0 = r0.sum(axis=0); n1 = r1.sum(axis=0)
        c0 = r0.T @ X0; c1 = r1.T @ X1
        for j in range(J):
            mode = constraints[j]
            if mode == "free":
                a_r[j], d_r[j] = _mstep_item(theta, n0, c0[:, j], a_r[j], d_r[j])
                a_f[j], d_f[j] = _mstep_item(theta, n1, c1[:, j], a_f[j], d_f[j])
            elif mode == "equal_a":
                a, d0, d1 = _mstep_equal_a(theta, n0, c0[:, j], n1, c1[:, j],
                                           a_r[j], d_r[j], d_f[j])
                a_r[j] = a_f[j] = a
                d_r[j] = d0; d_f[j] = d1
            else:  # equal_both
                a, d = _mstep_item(theta, n0 + n1, c0[:, j] + c1[:, j],
                                   a_r[j], d_r[j])
                a_r[j] = a_f[j] = a
                d_r[j] = d_f[j] = d
        N1 = n1.sum()
        if not fix_focal_mean:
            mu = float((n1 * theta).sum() / N1)
        if not fix_focal_var:
            sigma = max(float(np.sqrt((n1 * (theta - mu) ** 2).sum() / N1)), 0.2)
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll

    return MultiGroupFit(a_ref=a_r, d_ref=d_r, a_foc=a_f, d_foc=d_f,
                         mu=mu, sigma=sigma, loglik=ll, n_iter=n_iter)
