"""Validate the measurement-invariance omnibus across scenarios and seeds."""
import numpy as np
from benchdif import invariance


def gen(seed, kind, n=2500, J=12, mu=-0.4, sd=1.2):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.8, 1.6, J); b = rng.normal(0, 1, J)
    a_f, b_f = a.copy(), b.copy()
    if kind == "intercept_dif":
        for j in (2, 5, 8): b_f[j] = b[j] + 0.8
    if kind == "loading_dif":
        for j in (2, 5, 8): a_f[j] = a[j] + 0.6
    th0 = rng.normal(0, 1, n); th1 = rng.normal(mu, sd, n)
    d, d_f = -a * b, -a_f * b_f
    X0 = (rng.random((n, J)) < 1 / (1 + np.exp(-(a * th0[:, None] + d)))).astype(int)
    X1 = (rng.random((n, J)) < 1 / (1 + np.exp(-(a_f * th1[:, None] + d_f)))).astype(int)
    return np.vstack([X0, X1]), np.array([0] * n + [1] * n)


print(f"{'scenario':14s} {'metric_rej':>11s} {'scalar_rej':>11s}  (over 8 seeds)")
summary = {}
for kind in ["invariant", "intercept_dif", "loading_dif"]:
    m = s = 0
    for seed in range(8):
        r = invariance(*gen(seed, kind))
        m += not r.loc["metric", "holds"]; s += not r.loc["scalar", "holds"]
    summary[kind] = (m, s)
    print(f"{kind:14s} {m:>9}/8 {s:>9}/8")

assert summary["invariant"][0] <= 1 and summary["invariant"][1] <= 1, "size too high"
assert summary["intercept_dif"][1] >= 7 and summary["intercept_dif"][0] <= 2, "scalar power/localization off"
assert summary["loading_dif"][0] >= 7, "metric power off"
print("\nVALIDATED: metric/scalar tests have power on the right violations, "
      "hold under full invariance despite impact, and localize intercept vs loading DIF")
