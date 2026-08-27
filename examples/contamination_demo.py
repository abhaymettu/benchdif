"""Demo: DIF as a benchmark-contamination probe.

Story: a benchmark of 25 questions is run against 300 models from two families,
A and B. Family B was (secretly) trained on question q13, so B answers it far
better than its *overall ability* predicts. Overall-accuracy leaderboards can't
see this -- it just makes B look a bit better. DIF sees it: conditioning on total
score, q13 functions differently for B, and gets flagged large (ETS C).

Run:  uv run python examples/contamination_demo.py
"""
import numpy as np
import pandas as pd

from benchdif import from_long, mantel_haenszel, logistic

rng = np.random.default_rng(20)
N, K, LEAK = 300, 25, 13

ability = rng.normal(size=N)
family = np.where(np.arange(N) % 2 == 0, "A", "B")
difficulty = rng.normal(scale=0.7, size=K)

rows = []
for m in range(N):
    for q in range(K):
        logit = ability[m] - difficulty[q]
        if q == LEAK and family[m] == "B":
            logit += 1.5                      # leaked item: easy for B beyond ability
        correct = int(rng.random() < 1 / (1 + np.exp(-logit)))
        rows.append((f"model_{m:03d}", f"q{q:02d}", correct, family[m]))
df = pd.DataFrame(rows, columns=["model", "question", "correct", "family"])

# Overall accuracy: the leak is nearly invisible.
acc = df.groupby("family")["correct"].mean()
print("Overall accuracy by family (leak hidden in the average):")
print(acc.round(4).to_string(), "\n")

# DIF: reshape and test every item.
X, g, models, items = from_long(
    df, person="model", item="question", score="correct", group="family")
mh = mantel_haenszel(X, g); mh.index = items
lr = logistic(X, g, kind="both"); lr.index = items

flagged = mh[mh.flag].sort_values("stat", ascending=False)
print("Items flagged by Mantel-Haenszel DIF (focal = family B):")
print(flagged[["stat", "p_value", "mh_ddif", "ets"]].round(3).to_string(), "\n")

print(f"Leaked item was q{LEAK:02d}.")
print(f"  MH:       chi2={mh.loc[f'q{LEAK}', 'stat']:.1f}  ETS={mh.loc[f'q{LEAK}', 'ets']}")
print(f"  logistic: chi2={lr.loc[f'q{LEAK}', 'stat']:.1f}  JG={lr.loc[f'q{LEAK}', 'jg']}")
