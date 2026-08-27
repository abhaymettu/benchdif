"""Item-level DIF cancels at the test level -- but not under subsampling.

The full-benchmark result was a null: dropping DIF items barely moved the
leaderboard, because pro-Mistral and pro-Llama2 items cancel in aggregate. That
cancellation is what protects a full-benchmark score.

Benchmark *subsampling* (tinyBenchmarks, adaptive testing, "lite" splits) breaks
the cancellation: a 100-item subset need not be balanced. This measures how much
the family gap moves under random subsets, and how far it can be pushed by an
adversary choosing items.
"""
import pickle
import sys

import numpy as np

from benchdif import logistic, adjust

PATH = sys.argv[1]
KEY = "harness_hendrycksTest_professional_law_5"
K = 100  # tinyBenchmarks' subset size


def arch(s):
    n = str(s).lower()
    for p, l in [("mixtral", "mixtral"), ("mistral", "mistral"),
                 ("llama-2", "llama2"), ("llama2", "llama2")]:
        if p in n:
            return l
    return "other"


obj = pickle.load(open(PATH, "rb"))
models = [str(m) for m in obj["models"]]
fams = np.array([arch(m) for m in models])
X = (np.asarray(obj["data"][KEY]["correctness"]).T >= 0.5).astype(int)
sel = np.isin(fams, ["llama2", "mistral"])
Xs = X[sel]
g = (fams[sel] == "mistral").astype(int)

full_gap = Xs[g == 1].mean() - Xs[g == 0].mean()
print(f"{Xs.shape[0]} models x {Xs.shape[1]} items")
print(f"full-benchmark mistral-minus-llama2 gap: {full_gap:+.4f}\n")

res = logistic(Xs, g)
q = adjust(res.p_value, "bh")
# signed direction: positive db-like effect = favors mistral (focal)
mist = Xs[g == 1].mean(0) - Xs[g == 0].mean(0)     # raw per-item family difference
dif = q < 0.05

# --- random subsets ---
rng = np.random.default_rng(0)
gaps = []
for _ in range(400):
    idx = rng.choice(Xs.shape[1], K, replace=False)
    gaps.append(Xs[g == 1][:, idx].mean() - Xs[g == 0][:, idx].mean())
gaps = np.array(gaps)
print(f"RANDOM {K}-item subsets (n=400):")
print(f"   gap mean {gaps.mean():+.4f}, sd {gaps.std():.4f}")
print(f"   2.5-97.5 pct: {np.percentile(gaps,2.5):+.4f} to {np.percentile(gaps,97.5):+.4f}")
print(f"   subsets that FLIP the sign of the gap: "
      f"{int((np.sign(gaps) != np.sign(full_gap)).sum())}/400")

# --- adversarial subsets: pick the most pro-one-family DIF items ---
order = np.argsort(mist)
pro_llama = order[:K]
pro_mistral = order[-K:]
gl = Xs[g == 1][:, pro_llama].mean() - Xs[g == 0][:, pro_llama].mean()
gm = Xs[g == 1][:, pro_mistral].mean() - Xs[g == 0][:, pro_mistral].mean()
print(f"\nADVERSARIAL {K}-item subsets (same benchmark, same models):")
print(f"   chosen to favour llama2 : gap {gl:+.4f}")
print(f"   chosen to favour mistral: gap {gm:+.4f}")
print(f"   swing {gm-gl:+.4f} vs a true full-benchmark gap of {full_gap:+.4f}")
print(f"   -> a curated 'lite' split can show either family ahead by "
      f"{max(abs(gl),abs(gm)):.1%}")
print(f"\nDIF items flagged: {int(dif.sum())}/{len(dif)}; of these "
      f"{int((mist[dif]>0).sum())} favour mistral, {int((mist[dif]<0).sum())} favour llama2 "
      f"(cancellation is why the full benchmark looks fine)")
