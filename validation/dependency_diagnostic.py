"""How non-independent are the models? The validity threat for benchmark DIF.

DIF assumes persons are exchangeable draws. Leaderboard models are not: a
"family" is one base model plus finetunes, DPO variants and merges of each other.
If so, one quirk of the base model is replicated N times and reads as an
overwhelming group effect, and a label-shuffling permutation is anti-conservative
because shuffling destroys clustering the real data has.

This measures the dependency directly from the response vectors.
"""
import pickle
import sys

import numpy as np

PATH = sys.argv[1]


def arch(s):
    n = str(s).lower()
    for pat, lab in [("mixtral", "mixtral"), ("mistral", "mistral"),
                     ("llama-2", "llama2"), ("llama2", "llama2")]:
        if pat in n:
            return lab
    return "other"


obj = pickle.load(open(PATH, "rb"))
models = [str(m) for m in obj["models"]]
fams = np.array([arch(m) for m in models])
key = "harness_hendrycksTest_professional_law_5"
X = (np.asarray(obj["data"][key]["correctness"]).T >= 0.5).astype(int)

sel = np.isin(fams, ["llama2", "mistral"])
Xs, f = X[sel], fams[sel]
print(f"{Xs.shape[0]} models, {Xs.shape[1]} items\n")

# pairwise correlation of response vectors
C = np.corrcoef(Xs)
np.fill_diagonal(C, np.nan)
within, between = [], []
for i in range(len(f)):
    for j in range(i + 1, len(f)):
        (within if f[i] == f[j] else between).append(C[i, j])
within, between = np.array(within), np.array(between)
print(f"mean pairwise correlation WITHIN family : {within.mean():.3f}")
print(f"mean pairwise correlation BETWEEN family: {between.mean():.3f}")
print(f"near-duplicate pairs (r > 0.90): {int((within > 0.90).sum())} within, "
      f"{int((between > 0.90).sum())} between")

# cluster models into lineages by response similarity (single-linkage at r>0.9)
def lineages(Xg, thresh=0.90):
    Cg = np.corrcoef(Xg)
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
                if lab[b] < 0 and Cg[a, b] > thresh:
                    lab[b] = cur
                    stack.append(b)
        cur += 1
    return lab, cur


print()
for fam in ["llama2", "mistral"]:
    m = f == fam
    lab, k = lineages(Xs[m])
    sizes = np.bincount(lab)
    print(f"{fam}: {int(m.sum())} models -> {k} distinct lineages at r>0.90 "
          f"(largest lineage has {sizes.max()} models)")
    print(f"   effective n is ~{k}, not {int(m.sum())}")
