"""Two checks the subsampling claim must survive.

A. Is the per-item family difference REAL, or just binomial noise? Compare the
   observed variance of per-item family differences against the variance expected
   if models differed only in overall ability (no DIF). If they match, subset
   instability is ordinary sampling noise and DIF adds nothing.
B. Is the adversarial swing REAL or circular? Selecting items by the same
   quantity you then measure, on the same models, is in-sample. Select the items
   using one half of the models and measure the gap on the HELD-OUT half.
"""
import pickle
import sys

import numpy as np

PATH = sys.argv[1]
KEY = "harness_hendrycksTest_professional_law_5"
K = 100


def arch(s):
    n = str(s).lower()
    for p, l in [("mixtral", "mixtral"), ("mistral", "mistral"),
                 ("llama-2", "llama2"), ("llama2", "llama2")]:
        if p in n:
            return l
    return "other"


obj = pickle.load(open(PATH, "rb"))
fams = np.array([arch(m) for m in obj["models"]])
X = (np.asarray(obj["data"][KEY]["correctness"]).T >= 0.5).astype(int)
sel = np.isin(fams, ["llama2", "mistral"])
Xs = X[sel]
g = (fams[sel] == "mistral").astype(int)
A, B = Xs[g == 1], Xs[g == 0]          # mistral, llama2
nA, nB = len(A), len(B)

# --- A. observed vs null variance of per-item family differences ---
d = A.mean(0) - B.mean(0)
pA, pB = A.mean(0), B.mean(0)
# binomial sampling variance of the difference, per item, under no item-specific effect
null_var = pA * (1 - pA) / nA + pB * (1 - pB) / nB
print("A. IS THE PER-ITEM FAMILY DIFFERENCE REAL?")
print(f"   observed var(per-item gap) : {d.var():.5f}")
print(f"   expected under no DIF      : {null_var.mean():.5f}")
ratio = d.var() / null_var.mean()
print(f"   ratio {ratio:.2f}x  -> excess variance attributable to item-specific "
      f"family effects: {100*(1-1/ratio):.0f}%")

# --- B. out-of-sample adversarial selection ---
print(f"\nB. ADVERSARIAL {K}-ITEM SUBSET, SELECTED AND SCORED ON DIFFERENT MODELS")
rng = np.random.default_rng(0)
swings_in, swings_out = [], []
for seed in range(5):
    r = np.random.default_rng(seed)
    ia = r.permutation(nA); ib = r.permutation(nB)
    A1, A2 = A[ia[:nA//2]], A[ia[nA//2:]]
    B1, B2 = B[ib[:nB//2]], B[ib[nB//2:]]
    d1 = A1.mean(0) - B1.mean(0)               # select on half 1
    order = np.argsort(d1)
    lo, hi = order[:K], order[-K:]
    # in-sample swing (same half used to select)
    s_in = (A1[:, hi].mean() - B1[:, hi].mean()) - (A1[:, lo].mean() - B1[:, lo].mean())
    # out-of-sample swing (held-out half 2)
    s_out = (A2[:, hi].mean() - B2[:, hi].mean()) - (A2[:, lo].mean() - B2[:, lo].mean())
    swings_in.append(s_in); swings_out.append(s_out)
    print(f"   split {seed}: in-sample swing {s_in:+.3f} | held-out swing {s_out:+.3f}")
print(f"   mean in-sample {np.mean(swings_in):+.3f}, mean HELD-OUT "
      f"{np.mean(swings_out):+.3f}")
full_gap = A.mean() - B.mean()
print(f"   true full-benchmark gap {full_gap:+.4f}")
print(f"   -> even out-of-sample, a curated subset shifts the family gap by "
      f"{np.mean(swings_out):.3f}, {abs(np.mean(swings_out)/full_gap):.0f}x the real gap")
