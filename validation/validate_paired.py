"""Validate paired-condition DIF.

1. ORACLE. The conditional-ML impact Delta_m must equal statsmodels'
   ConditionalLogit fit on the same trials expanded to one row each, grouped by
   (model, item) stratum.
2. SIZE. Pure impact (Rasch shift) with large model-specific item effects: the
   omnibus and the per-item BH flags must hold their nominal rate.
3. POWER. Planted item DIF must be found, in both directions.
4. SLOPES. If items really differ in discrimination, a pure ability shift is not a
   constant log-odds shift. The Rasch null then reads it as DIF; the 2PL null with
   the true slopes does not.
"""
import numpy as np
from statsmodels.discrete.conditional_models import ConditionalLogit

from benchdif.dif.paired import paired_dif


def sim(seed, K=3, I=150, T=4, shift=0.8, dif_frac=0.0, dif_size=1.0, slope_sd=0.0):
    rng = np.random.default_rng(seed)
    a = np.exp(rng.normal(0, slope_sd, I))
    b = rng.normal(0, 1.5, I)
    delta = np.zeros(I)
    nd = int(round(dif_frac * I))
    delta[:nd] = dif_size * np.where(np.arange(nd) % 2 == 0, 1, -1)
    rows = []
    for m in range(K):
        th = rng.normal(0, 1)
        u = rng.normal(0, 1, I)
        p0 = 1 / (1 + np.exp(-(a * (th - b) + u)))
        p1 = 1 / (1 + np.exp(-(a * (th + shift - b) + u + delta)))
        rows.append((np.full(I, m), np.arange(I), rng.binomial(T, p0), np.full(I, T),
                     rng.binomial(T, p1), np.full(I, T)))
    cols = [np.concatenate(c) for c in zip(*rows)]
    return cols, a, delta


# 1. oracle
(m, it, k0, n0, k1, n1), _, _ = sim(0, K=2, I=60)
res = paired_dif(m, it, k0, n0, k1, n1, n_boot=10)
y, X, grp = [], [], []
for r in range(len(m)):
    for c, k, n in ((0, k0[r], n0[r]), (1, k1[r], n1[r])):
        y += [1] * k + [0] * (n - k)
        X += [[c * (m[r] == j) for j in range(2)]] * n
        grp += [r] * n
sm = ConditionalLogit(np.array(y), np.array(X, float), groups=np.array(grp)).fit(method="newton", tol=1e-12, maxiter=200, disp=0)
diff = np.max(np.abs(sm.params - res.impact.delta_hat.to_numpy()))
print(f"oracle: benchdif {res.impact.delta_hat.round(6).tolist()} "
      f"statsmodels {np.round(sm.params, 6).tolist()}  max|diff| {diff:.1e}")
assert diff < 1e-6

# 2. size under pure impact
R = 40
omni = flags = tested = 0
for s in range(R):
    cols, _, _ = sim(100 + s)
    r = paired_dif(*cols, n_boot=200, seed=s)
    omni += r.p_value < 0.05
    flags += int(r.items.flag.sum()); tested += int(r.items.p_value.notna().sum())
print(f"size: omnibus {omni}/{R} rejections, item flags {flags}/{tested}")
assert omni / R <= 0.125 and flags / tested < 0.01

# 3. power: 6 models, 10% of items, |delta| = 1.5
pw = hit = planted = fpos = clean = 0
for s in range(20):
    cols, _, delta = sim(200 + s, K=6, dif_frac=0.1, dif_size=1.5)
    r = paired_dif(*cols, n_boot=200, seed=s)
    pw += r.p_value < 0.05
    f = r.items.set_index("item").flag.reindex(range(len(delta)), fill_value=False).to_numpy()
    hit += int(f[delta != 0].sum()); planted += int((delta != 0).sum())
    fpos += int(f[delta == 0].sum()); clean += int((delta == 0).sum())
print(f"power: omnibus {pw}/20, items found {hit}/{planted}, false flags {fpos}/{clean}")
assert pw >= 18

# 4. slopes: true a_i vary (sd 0.4 on log scale), no DIF
rasch = twopl = 0
for s in range(20):
    cols, a, _ = sim(300 + s, slope_sd=0.4, shift=1.5)
    rasch += paired_dif(*cols, n_boot=200, seed=s).p_value < 0.05
    twopl += paired_dif(*cols, slope=a, n_boot=200, seed=s).p_value < 0.05
print(f"slopes: Rasch-null rejections {rasch}/20, 2PL-null rejections {twopl}/20")
assert twopl / 20 <= 0.15
print("VALIDATED: paired DIF matches the oracle, holds size, finds planted DIF")
