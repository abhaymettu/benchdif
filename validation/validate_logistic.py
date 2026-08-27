"""Validate benchdif's logistic DIF against statsmodels GLM (independent glm).

For a studied item we fit the same nested models with statsmodels' Binomial GLM
and compare the log-likelihoods and the LR chi-square. Agreement to ~1e-6 shows
the IRLS fitter reproduces glm, so the DIF statistics are correct.
"""
import numpy as np
import statsmodels.api as sm

from benchdif.dif.logistic import logistic, _irls


def _sim(n=1500, k=15, dif_item=4, seed=11):
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n)
    g = (np.arange(n) % 2).astype(float)
    diff = rng.normal(scale=0.8, size=k)
    X = (rng.random((n, k)) < 1 / (1 + np.exp(-(theta[:, None] - diff[None, :])))).astype(float)
    foc = g == 1
    lo = theta[foc] - diff[dif_item] - 0.9
    X[foc, dif_item] = (rng.random(foc.sum()) < 1 / (1 + np.exp(-lo))).astype(float)
    return X, g


def _sm_ll(y, cols):
    design = sm.add_constant(np.column_stack(cols), has_constant="add")
    res = sm.GLM(y, design, family=sm.families.Binomial()).fit()
    return res.llf


def main():
    X, g = _sim()
    S = X.sum(axis=1)
    res = logistic(X, g, kind="both")
    max_dstat = 0.0
    print(f"{'item':>4} {'bd_stat':>10} {'sm_stat':>10} {'diff':>10}")
    for j in range(X.shape[1]):
        y = X[:, j]
        ll0 = _sm_ll(y, [S])
        ll2 = _sm_ll(y, [S, g, S * g])
        sm_stat = 2 * (ll2 - ll0)
        bd_stat = res.loc[j, "stat"]
        d = abs(bd_stat - sm_stat)
        max_dstat = max(max_dstat, d)
        print(f"{j:>4} {bd_stat:>10.5f} {sm_stat:>10.5f} {d:>10.2e}")
    print(f"\nmax |dstat| = {max_dstat:.2e}")
    assert max_dstat < 1e-4, "logistic LR stat disagrees with statsmodels GLM"
    print("VALIDATED: benchdif logistic == statsmodels GLM")


if __name__ == "__main__":
    main()
