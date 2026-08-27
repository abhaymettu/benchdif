"""Quickstart: every public entry point on one simulated benchmark.

Run:  uv run python examples/quickstart.py
"""
import numpy as np
import pandas as pd

import benchdif
from benchdif import (mantel_haenszel, logistic, irt_lr,
                      generalized_mantel_haenszel, invariance, fit_2pl,
                      from_long, from_matrix, detect)

print("benchdif", benchdif.__version__, "-- public API:", benchdif.__all__)

# --- simulate a benchmark: 300 models, 20 questions, 2 families; q7 leaks to fam B ---
rng = np.random.default_rng(0)
N, K, LEAK = 300, 20, 7
ability = rng.normal(size=N)
family = np.where(np.arange(N) % 2 == 0, "A", "B")
difficulty = rng.normal(scale=0.7, size=K)
rows = []
for m in range(N):
    for q in range(K):
        logit = ability[m] - difficulty[q] + (1.4 if q == LEAK and family[m] == "B" else 0)
        rows.append((f"model_{m}", f"q{q}", int(rng.random() < 1 / (1 + np.exp(-logit))), family[m]))
df = pd.DataFrame(rows, columns=["model", "question", "correct", "family"])

# --- adapters: long -> matrix ---
X, g, models, items = from_long(df, person="model", item="question",
                                score="correct", group="family")
print(f"\nmatrix: {X.shape[0]} models x {X.shape[1]} items, groups {np.bincount(g)}")

# --- two-group DIF, three ways + the unified API ---
print("\nMantel-Haenszel flags:", [items[i] for i in np.where(mantel_haenszel(X, g).flag)[0]])
print("Logistic flags:       ", [items[i] for i in np.where(logistic(X, g).flag)[0]])
print("IRT-LR flags:         ", [items[i] for i in np.where(irt_lr(X, g).flag & ~irt_lr(X, g).is_anchor)[0]])
print("detect(method='mh'):  ", [items[i] for i in np.where(detect(X, g, method="mh").flag)[0]])

# --- >2 groups: split family B into two size buckets to make 3 groups ---
size3 = np.where(np.arange(N) % 3 == 0, "small", np.where(np.arange(N) % 3 == 1, "mid", "big"))
res3 = generalized_mantel_haenszel(X, size3)
print("\nGeneralized MH (3 groups) top item:", items[res3.stat.idxmax()], f"(df={res3.df.iloc[0]})")

# --- IRT backend + measurement invariance ---
fit = fit_2pl(X)
print(f"\n2PL fit: {len(fit.a)} items, mean discrimination {fit.a.mean():.2f}")
inv = invariance(X, g)
print("Measurement invariance:")
print(inv.to_string())

# --- from_matrix round-trip ---
wide = pd.DataFrame(X, index=models, columns=items)
X2, g2, _, _ = from_matrix(wide, pd.Series(g, index=models))
assert np.array_equal(X, X2)
print("\nfrom_matrix round-trip OK. All public functions exercised.")
