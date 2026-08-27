"""Three robustness checks the real-data DIF finding has to survive.

1. LINEAGE-PRESERVING PERMUTATION. Plain label-shuffling is anti-conservative if
   models cluster within family. Permute at the lineage level instead.
2. HELD-OUT-MODEL REPLICATION. Flag DIF items on half the models, then test the
   SAME items on the other half. If item identity generalizes to models never used
   to find it, the effect is a property of the items, not of a few models.
3. RANKING IMPACT. Does dropping DIF items move the leaderboard? A DIF finding
   that cannot move a ranking is a curiosity; one that can is a problem.
"""
import pickle
import sys

import numpy as np

from benchdif import logistic, adjust

PATH = sys.argv[1]
KEY = sys.argv[2] if len(sys.argv) > 2 else "harness_hendrycksTest_professional_law_5"


def arch(s):
    n = str(s).lower()
    for p, l in [("mixtral", "mixtral"), ("mistral", "mistral"),
                 ("llama-2", "llama2"), ("llama2", "llama2")]:
        if p in n:
            return l
    return "other"


def lineages(Xg, thresh=0.90):
    C = np.corrcoef(Xg)
    n = len(Xg)
    lab = -np.ones(n, int)
    cur = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack, lab[i] = [i], cur
        while stack:
            a = stack.pop()
            for b in range(n):
                if lab[b] < 0 and C[a, b] > thresh:
                    lab[b] = cur
                    stack.append(b)
        cur += 1
    return lab


obj = pickle.load(open(PATH, "rb"))
models = [str(m) for m in obj["models"]]
fams = np.array([arch(m) for m in models])
X = (np.asarray(obj["data"][KEY]["correctness"]).T >= 0.5).astype(int)
sel = np.isin(fams, ["llama2", "mistral"])
Xs, f = X[sel], fams[sel]
g = (f == "mistral").astype(int)
n_items = Xs.shape[1]
print(f"{KEY}\n{Xs.shape[0]} models x {n_items} items\n")


def flagged(Xm, gm):
    q = adjust(logistic(Xm, gm).p_value, "bh")
    return q < 0.05


# --- 1. lineage-preserving permutation ---
lin = np.empty(len(f), int)
off = 0
for fam in ["llama2", "mistral"]:
    m = f == fam
    lin[m] = lineages(Xs[m]) + off
    off = lin[m].max() + 1
uniq_lin = np.unique(lin)
lin_group = {L: g[lin == L][0] for L in uniq_lin}   # each lineage's true family

real = int(flagged(Xs, g).sum())
rng = np.random.default_rng(0)
perm_counts = []
for _ in range(5):
    shuffled = rng.permutation([lin_group[L] for L in uniq_lin])
    gp = np.empty(len(f), int)
    for L, val in zip(uniq_lin, shuffled):
        gp[lin == L] = val
    if len(np.unique(gp)) < 2:
        continue
    perm_counts.append(int(flagged(Xs, gp).sum()))
print("1. LINEAGE-PRESERVING PERMUTATION")
print(f"   real labels      : {real}/{n_items} items flagged")
print(f"   permuted lineages: {perm_counts} (median {int(np.median(perm_counts))})")

# --- 2. held-out-model replication ---
print("\n2. HELD-OUT-MODEL REPLICATION")
agree = tot = 0
for seed in range(4):
    r = np.random.default_rng(seed)
    idxA, idxB = [], []
    for fam in [0, 1]:
        idx = np.where(g == fam)[0]
        r.shuffle(idx)
        h = len(idx) // 2
        idxA += list(idx[:h]); idxB += list(idx[h:])
    idxA, idxB = np.array(idxA), np.array(idxB)
    fa = flagged(Xs[idxA], g[idxA])
    fb = flagged(Xs[idxB], g[idxB])
    if fa.sum() == 0:
        continue
    # of items flagged in half A, what fraction also flag in the unseen half B?
    rep = float(fb[fa].mean())
    base = float(fb.mean())               # base rate in half B
    agree += rep; tot += 1
    print(f"   split {seed}: {int(fa.sum())} flagged in A -> {rep:.1%} also flag "
          f"in held-out B (base rate {base:.1%})")
if tot:
    print(f"   mean replication {agree/tot:.1%}")

# --- 3. ranking impact ---
print("\n3. RANKING IMPACT (does dropping DIF items move the leaderboard?)")
fl = flagged(Xs, g)
full = Xs.mean(1)
clean = Xs[:, ~fl].mean(1)
rank_full = np.argsort(np.argsort(-full))
rank_clean = np.argsort(np.argsort(-clean))
shift = np.abs(rank_full - rank_clean)
print(f"   dropped {int(fl.sum())}/{n_items} DIF items")
print(f"   mean |rank change| {shift.mean():.1f} places, max {shift.max()}")
print(f"   models moving >5 places: {int((shift > 5).sum())}/{len(shift)}")
top10_before = set(np.argsort(-full)[:10])
top10_after = set(np.argsort(-clean)[:10])
print(f"   top-10 turnover: {10 - len(top10_before & top10_after)}/10 models change")
gap_before = full[g == 1].mean() - full[g == 0].mean()
gap_after = clean[g == 1].mean() - clean[g == 0].mean()
print(f"   mistral-minus-llama2 accuracy gap: {gap_before:+.4f} -> {gap_after:+.4f}")
