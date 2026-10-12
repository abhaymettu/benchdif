"""Validate the test-level theory check: held-out item DIF predicts the score effect.

For each model, three numbers should agree when DIF does not cancel:
  G_true : the DIF part of its condition-1 score, from the generating parameters
           (mean over items of P(theta + shift + delta_i) - P(theta + shift))
  G_obs  : paired_dtf's observed part (successes beyond anchor-defined impact)
  G_pred : paired_dtf's prediction from item DIF fitted on the OTHER models
and all three should sit near zero when DIF is balanced.

Known bias, measured here: impact and one-sided DIF cannot be fully separated
without an outside anchor, and the purified anchor settles slightly above the
true impact. Both G estimates therefore understate one-sided DIF's score effect
(conservative); the table reports by how much.

Also checks the odd-vs-even trial split used as the negative control: two halves
of one subject under one condition must show no DIF at the nominal rate.
"""
import numpy as np
import pandas as pd

from benchdif.dif.paired import paired_dif, paired_dtf


def counts(seed, K=6, I=200, T=8, dif=2.0, frac=0.2, one_sided=True):
    rng = np.random.default_rng(seed)
    b = rng.normal(-0.4, 1.5, I)
    d = np.zeros(I)
    nd = int(frac * I)
    d[:nd] = dif if one_sided else dif * np.where(np.arange(nd) % 2, -1, 1)
    rows, true = [], {}
    for m in range(K):
        eta = rng.normal() - b + rng.normal(0, 1, I)
        p0, p1 = 1 / (1 + np.exp(-eta)), 1 / (1 + np.exp(-(eta + 0.8)))
        p1d = 1 / (1 + np.exp(-(eta + 0.8 + d)))
        true[f"m{m}"] = float(np.mean(p1d - p1))
        rows.append(pd.DataFrame({"cell": f"m{m}", "item_id": np.arange(I),
                                  "k0": rng.binomial(T, p0), "n0": T,
                                  "k1": rng.binomial(T, p1d), "n1": T}))
    return pd.concat(rows, ignore_index=True), pd.Series(true)


print("| DIF | mean G_true | mean G_obs | mean G_pred | obs/true | pred/true | r(pred, obs) |")
print("|---|---:|---:|---:|---:|---:|---:|")
for one_sided in (True, False):
    t, o, p = [], [], []
    for s in range(10):
        c, true = counts(s, one_sided=one_sided)
        G = pd.concat([paired_dtf(c[c.cell != m], c[c.cell == m])
                       for m in sorted(c.cell.unique())])
        t += list(true[G.index]); o += list(G.G_obs); p += list(G.G_pred)
    t, o, p = map(np.array, (t, o, p))
    print(f"| {'one-sided +2.0 on 20%' if one_sided else 'balanced +/-2.0 on 20%'} | "
          f"{t.mean():+.3f} | {o.mean():+.3f} | {p.mean():+.3f} | "
          f"{o.mean() / t.mean():.2f} | {p.mean() / t.mean():.2f} | {np.corrcoef(p, o)[0, 1]:.2f} |")
    if one_sided:
        assert 0.8 < o.mean() / t.mean() < 1.1 and 0.6 < p.mean() / t.mean() < 1.1
    else:
        assert abs(p.mean()) < 0.01 and abs(o.mean()) < 0.01

# negative control: odd vs even trials (2 vs 2) of one subject, three models
rej = 0
for s in range(100):
    rng = np.random.default_rng(1000 + s)
    m, it, k0, k1 = [], [], [], []
    for j in range(3):
        p = 1 / (1 + np.exp(-(rng.normal() - rng.normal(-0.4, 1.5, 115) + rng.normal(0, 1, 115))))
        m += [j] * 115; it += list(range(115))
        k0 += list(rng.binomial(2, p)); k1 += list(rng.binomial(2, p))
    rej += paired_dif(m, it, k0, [2] * len(m), k1, [2] * len(m), n_boot=200, seed=s).p_value < 0.05
print(f"\nnegative control (odd vs even trials, 2 vs 2): {rej}/100 rejections at .05")
assert rej <= 10
print("VALIDATED: held-out DTF predicts the score effect of DIF; the trial split holds size")
