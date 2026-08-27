"""Validate benchdif's Mantel-Haenszel against statsmodels' StratifiedTable.

statsmodels.stats.contingency_tables.StratifiedTable implements the
Cochran-Mantel-Haenszel test and pooled odds ratio from an independent codebase.
For dichotomous DIF these are the identical estimands as MH D-DIF's chi-square
(continuity corrected) and alpha_MH. Agreement to ~1e-6 is the correctness proof.
"""
import numpy as np
from statsmodels.stats.contingency_tables import StratifiedTable

from benchdif import mantel_haenszel


def _simulate(n=1500, n_items=15, dif_item=4, dif_size=0.8, seed=7):
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=n)
    group = (np.arange(n) % 2).astype(int)  # 0 ref, 1 focal
    diff = rng.normal(scale=0.8, size=n_items)
    logits = theta[:, None] - diff[None, :]
    X = (rng.random((n, n_items)) < 1 / (1 + np.exp(-logits))).astype(int)
    foc = group == 1
    lo = theta[foc] - diff[dif_item] - dif_size  # item harder for focal
    X[foc, dif_item] = (rng.random(foc.sum()) < 1 / (1 + np.exp(-lo))).astype(int)
    return X, group


def _oracle_item(X, group, j):
    """CMH chi2 (corrected) and pooled OR for item j via statsmodels."""
    total = X.sum(axis=1)
    tables = []
    for k in np.unique(total):
        at = total == k
        ref = at & (group == 0)
        foc = at & (group == 1)
        A = int(np.sum(X[ref, j] == 1)); B = int(np.sum(X[ref, j] == 0))
        C = int(np.sum(X[foc, j] == 1)); D = int(np.sum(X[foc, j] == 0))
        if A + B + C + D == 0:
            continue
        tables.append([[A, B], [C, D]])
    arr = np.array(tables).transpose(1, 2, 0)  # 2 x 2 x K
    st = StratifiedTable(arr)
    chi2 = st.test_null_odds(correction=True).statistic
    return float(chi2), float(st.oddsratio_pooled)


def main():
    X, group = _simulate()
    res = mantel_haenszel(X, group)
    print(f"{'item':>4} {'bd_chi2':>10} {'sm_chi2':>10} {'d_chi2':>9} "
          f"{'bd_alpha':>9} {'sm_alpha':>9} {'d_alpha':>9}")
    max_dc = max_da = 0.0
    for j in range(X.shape[1]):
        sm_chi2, sm_or = _oracle_item(X, group, j)
        bd_chi2 = res.loc[j, "stat"]; bd_or = res.loc[j, "alpha_mh"]
        dc = abs(bd_chi2 - sm_chi2); da = abs(bd_or - sm_or)
        max_dc = max(max_dc, dc); max_da = max(max_da, da)
        print(f"{j:>4} {bd_chi2:>10.5f} {sm_chi2:>10.5f} {dc:>9.2e} "
              f"{bd_or:>9.5f} {sm_or:>9.5f} {da:>9.2e}")
    print(f"\nmax |dchi2| = {max_dc:.2e}   max |dalpha| = {max_da:.2e}")
    assert max_dc < 1e-4, "chi-square disagrees with statsmodels CMH"
    assert max_da < 1e-6, "alpha_MH disagrees with statsmodels pooled OR"
    print("VALIDATED: benchdif MH == statsmodels CMH")


if __name__ == "__main__":
    main()
