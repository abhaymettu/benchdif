"""Validate generalized MH: G=2 reduces to uncorrected CMH; G=3 power + size."""
import numpy as np
from statsmodels.stats.contingency_tables import StratifiedTable

from benchdif import generalized_mantel_haenszel as gmh


def g2_vs_cmh():
    rng = np.random.default_rng(3); n, k = 1500, 10
    th = rng.normal(size=n); g = (np.arange(n) % 2); diff = rng.normal(size=k)
    X = (rng.random((n, k)) < 1 / (1 + np.exp(-(th[:, None] - diff[None, :])))).astype(int)
    foc = g == 1
    X[foc, 4] = (rng.random(foc.sum()) < 1 / (1 + np.exp(-(th[foc] - diff[4] - 0.8)))).astype(int)
    res = gmh(X, g); total = X.sum(1); maxd = 0.0
    for j in range(k):
        tabs = []
        for lv in np.unique(total):
            at = total == lv; r, f = at & (g == 0), at & (g == 1)
            A = int((X[r, j] == 1).sum()); B = int((X[r, j] == 0).sum())
            C = int((X[f, j] == 1).sum()); D = int((X[f, j] == 0).sum())
            if A + B + C + D:
                tabs.append([[A, B], [C, D]])
        st = StratifiedTable(np.array(tabs).transpose(1, 2, 0))
        maxd = max(maxd, abs(res.loc[j, "stat"] - st.test_null_odds(correction=False).statistic))
    return maxd


def sim3(seed, n=2400, k=12, j=5, null=False):
    rng = np.random.default_rng(seed)
    th = rng.normal(size=n); grp = (np.arange(n) % 3); diff = rng.normal(scale=0.6, size=k)
    X = (rng.random((n, k)) < 1 / (1 + np.exp(-(th[:, None] - diff[None, :])))).astype(int)
    if not null:
        m = grp == 2
        X[m, j] = (rng.random(m.sum()) < 1 / (1 + np.exp(-(th[m] - diff[j] - 1.0)))).astype(int)
    return X, grp


maxd = g2_vs_cmh()
print(f"G=2 vs uncorrected CMH: max |diff| = {maxd:.2e}")
hits = sum(gmh(*sim3(s)).loc[5, "flag"] for s in range(6))
fp = tot = 0
for s in range(8):
    r = gmh(*sim3(s, null=True)); fp += int(r.flag.sum()); tot += len(r)
print(f"G=3 power on injected item: {hits}/6 seeds")
print(f"G=3 null FPR: {fp}/{tot} = {fp/tot:.3f} (nominal 0.05)")
assert maxd < 1e-6 and hits == 6 and fp / tot < 0.10
print("VALIDATED: generalized MH matches CMH at G=2, powered at G=3, size controlled")
