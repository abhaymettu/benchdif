"""Two-parameter logistic (2PL) IRT via marginal maximum likelihood (Bock-Aitkin EM).

The 2PL models P(correct | theta) = 1 / (1 + exp(-(a*theta + d))) for a latent
ability theta ~ N(0,1), with slope a (discrimination) and intercept d; the usual
difficulty is b = -d / a. Ability is integrated out over Gauss-Hermite quadrature
nodes (Bock & Aitkin 1981):

  E-step: posterior of each person over ability nodes given current item params.
  M-step: per item, a weighted (grouped-binomial) logistic regression of the
          expected correct-counts on the quadrature abilities -> new (a, d).

Runtime deps are numpy/scipy only. Correctness is checked by parameter recovery:
simulate from known (a, b), refit, and confirm the estimates track the truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite_e import hermegauss


@dataclass
class TwoPLFit:
    a: np.ndarray        # discriminations, one per item
    b: np.ndarray        # difficulties (b = -d/a)
    d: np.ndarray        # intercepts
    loglik: float
    n_iter: int
    nodes: np.ndarray
    weights: np.ndarray


def _std_normal_quadrature(n_nodes):
    x, w = hermegauss(n_nodes)          # integrates f(x) exp(-x^2/2) dx
    w = w / np.sqrt(2 * np.pi)          # normalize so weights sum to 1 under N(0,1)
    return x, w


def _mstep_item(theta, n_q, r1_q, a0, d0, iters=50, tol=1e-9):
    """Grouped-binomial logistic fit for one item: successes r1_q of n_q at theta.

    Newton-Raphson on (a, d) with logit = a*theta + d. Returns (a, d).
    """
    a, d = a0, d0
    for _ in range(iters):
        eta = np.clip(a * theta + d, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = n_q * p * (1 - p)
        grad = np.array([np.sum(theta * (r1_q - n_q * p)),
                         np.sum(r1_q - n_q * p)])
        H = np.array([[np.sum(W * theta * theta), np.sum(W * theta)],
                      [np.sum(W * theta),         np.sum(W)]])
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        a += step[0]; d += step[1]
        if np.max(np.abs(step)) < tol:
            break
    return a, d


def fit_2pl(responses, n_nodes=41, max_iter=500, tol=1e-5) -> TwoPLFit:
    """Fit a 2PL model by marginal maximum likelihood (EM).

    Parameters
    ----------
    responses : (n_persons x n_items) 0/1 array, no missing.
    n_nodes : number of Gauss-Hermite quadrature points for the N(0,1) ability.
    max_iter, tol : EM stopping on change in marginal log-likelihood.
    """
    X = np.asarray(responses, dtype=float)
    if X.ndim != 2:
        raise ValueError("responses must be 2-D (persons x items)")
    if np.isnan(X).any():
        raise ValueError("missing values not supported")
    n, J = X.shape
    theta, wq = _std_normal_quadrature(n_nodes)
    Q = n_nodes

    a = np.ones(J)
    d = np.zeros(J)
    ll_old = -np.inf
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        # item probabilities at each node: P[q, j]
        eta = np.clip(theta[:, None] * a[None, :] + d[None, :], -30, 30)
        P = 1.0 / (1.0 + np.exp(-eta))
        P = np.clip(P, 1e-12, 1 - 1e-12)
        # log-lik of each person at each node: LL[p, q]
        logP, log1mP = np.log(P), np.log(1 - P)
        LL = X @ logP.T + (1 - X) @ log1mP.T          # (n x Q)
        LL += np.log(wq)[None, :]
        m = LL.max(axis=1, keepdims=True)
        post = np.exp(LL - m)
        denom = post.sum(axis=1, keepdims=True)
        ll = float(np.sum(m.ravel() + np.log(denom.ravel())))
        post /= denom                                  # posterior r[p, q]
        # expected counts per node/item
        n_q = post.sum(axis=0)                          # (Q,)
        r1 = post.T @ X                                 # (Q x J) expected correct
        for j in range(J):
            a[j], d[j] = _mstep_item(theta, n_q, r1[:, j], a[j], d[j])
        if abs(ll - ll_old) < tol:
            break
        ll_old = ll

    b = -d / a
    return TwoPLFit(a=a, b=b, d=d, loglik=ll, n_iter=n_iter,
                    nodes=theta, weights=wq)
