"""Planted-DIF simulation at the measurement-db harness/effort design sizes.

Design shapes come from the public catalog (subject names of the 270 benchmarks
in aims-foundations/measurement-db, GitHub revision a06b94d): how many models
share a contrast, how many items, how many trials. Each design is simulated with
model-specific item effects (sd 1 logit, the size FINDINGS.md section 1 implies)
and a real condition shift (impact, +0.8 logit), then:

  - size: no DIF (Rasch truth), and no DIF with slopes varying (log-sd 0.3)
    tested under the Rasch null, the case where impact alone can look like DIF
  - power: 10% of items get DIF of +/-delta (balanced, so it cancels in the score)
  - MDE: smallest delta in the grid with omnibus power >= 0.8

Part 2 runs benchdif's person-level methods (logistic + BH, invariance ladder)
on the runs-as-persons layout for the one design with enough runs per group,
uncalibrated and with the lineage-preserving permutation (condition labels
flipped within model).

Usage: python validation/simulate_harness_dif.py [reps]
"""
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from benchdif import adjust, invariance, logistic
from benchdif.dif.paired import paired_dif

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
DESIGNS = [  # name, models K, items I, trials T, base rate
    ("matharena effort (6 models x 4 trials)", 6, 120, 4, 0.65),
    ("matharena_platform harness (1 model x 4 trials)", 1, 100, 4, 0.45),
    ("swebench_live harness (4 models x 1 trial)", 4, 300, 1, 0.40),
    ("mtbbench tools (9 models x 1 trial)", 9, 573, 1, 0.50),
    ("weavebench harness (1 model x 1 trial)", 1, 114, 1, 0.50),
]
DELTAS = [0.5, 1.0, 1.5, 2.0, 3.0]


def draw(rng, K, I, T, rate, delta=0.0, slope_sd=0.0, shift=0.8, trials=False):
    a = np.exp(rng.normal(0, slope_sd, I))
    b = rng.normal(-np.log(rate / (1 - rate)), 1.5, I)
    d = np.zeros(I)
    nd = int(round(0.1 * I))
    d[:nd] = delta * np.where(np.arange(nd) % 2 == 0, 1, -1)
    out = []
    for m in range(K):
        th, u = rng.normal(), rng.normal(0, 1, I)
        p0 = 1 / (1 + np.exp(-(a * (th - b) + u)))
        p1 = 1 / (1 + np.exp(-(a * (th + shift - b) + u + d)))
        if trials:
            out.append((rng.random((T, I)) < p0, rng.random((T, I)) < p1))
        else:
            out.append((np.full(I, m), np.arange(I), rng.binomial(T, p0), np.full(I, T),
                        rng.binomial(T, p1), np.full(I, T)))
    if trials:
        return out
    return [np.concatenate(c) for c in zip(*out)], d


def rate(fn, n):
    return sum(fn(r) for r in range(n)) / n


def paired_row(design):
    name, K, I, T, br = design

    def rej(r, **kw):
        rng = np.random.default_rng(10_000 * r + K * 100 + T)
        cols, _ = draw(rng, K, I, T, br, **kw)
        return paired_dif(*cols, n_boot=200, seed=r).p_value < 0.05
    size = rate(lambda r: rej(r), REPS)
    size_s = rate(lambda r: rej(r, slope_sd=0.3), REPS)
    pw = [rate(lambda r: rej(r, delta=x), REPS) for x in DELTAS]
    mde = next((f"{x}" for x, p in zip(DELTAS, pw) if p >= 0.8), f">{DELTAS[-1]}")
    hit = planted = false = flagged = 0
    for r in range(min(REPS, 30)):
        cols, d = draw(np.random.default_rng(777 + r), K, I, T, br, delta=2.0)
        res = paired_dif(*cols, n_boot=20, seed=r).items.set_index("item")
        f = res.flag.reindex(range(I), fill_value=False).to_numpy()
        hit += int(f[d != 0].sum()); planted += int((d != 0).sum())
        false += int(f[d == 0].sum()); flagged += int(f.sum())
    fdr = f"{false / flagged:.2f}" if flagged else "n/a"
    return (f"| {name} | {size:.2f} | {size_s:.2f} | " + " | ".join(f"{p:.2f}" for p in pw)
            + f" | {mde} | {hit / planted:.2f} | {fdr} |")


# Part 2: person-level methods on runs (matharena effort design only: 24 runs/group)
K2, I2, T2, BR2 = 6, 120, 4, 0.65
R2 = max(REPS // 5, 10)
FLIPS = np.array(list(itertools.product([0, 1], repeat=K2 - 1)))


def person_level(X, g):
    q = adjust(logistic(X, g)["p_value"].to_numpy(), "bh")
    inv = invariance(X, g, n_nodes=21)
    return (float(np.mean(q < 0.05)), inv.loc["metric", "stat"] + inv.loc["scalar", "stat"],
            bool(not inv.holds.all()))


def person_rep(args):
    delta, r = args
    data = draw(np.random.default_rng(55 + r), K2, I2, T2, BR2, delta=delta, trials=True)
    X = np.vstack([np.vstack(pair) for pair in data]).astype(int)
    g = np.tile(np.repeat([0, 1], T2), K2)
    lin = np.repeat(np.arange(K2), 2 * T2)
    X = X[:, X.std(axis=0) > 0]
    share, stat, rejected = person_level(X, g)
    null = []
    for f in FLIPS:  # all 2^(K-1) within-model relabelings, identity included
        flip = np.concatenate([[0], f])[lin]
        null.append(person_level(X, np.where(flip == 1, 1 - g, g))[1])
    return share, rejected, np.mean([s >= stat - 1e-9 for s in null]) < 0.05


if __name__ == "__main__":
    with ProcessPoolExecutor() as pool:
        rows = list(pool.map(paired_row, DESIGNS))
        person = list(pool.map(person_rep, [(d, r) for d in (0.0, 2.0) for r in range(R2)]))
    print(f"# Paired-condition DIF at measurement-db design sizes ({REPS} reps per cell)\n")
    print("| design | size (Rasch) | size (slopes vary, Rasch null) | "
          + " | ".join(f"power d={x}" for x in DELTAS)
          + " | MDE | item recall d=2 | item FDR d=2 |")
    print("|---|---:|---:|" + "---:|" * len(DELTAS) + "---:|---:|---:|")
    print("\n".join(rows))
    print(f"\n# Person-level methods, runs as persons ({K2} models x {T2} trials per group, "
          f"{R2} reps, 21 quadrature nodes)\n")
    print("| truth | logistic share flagged (BH) | ladder rejects (nominal) | "
          "ladder rejects (lineage permutation, 32 relabelings) |")
    print("|---|---:|---:|---:|")
    for i, delta in enumerate((0.0, 2.0)):
        part = person[i * R2:(i + 1) * R2]
        print(f"| {'no DIF' if delta == 0 else f'DIF d={delta}'} | "
              f"{np.mean([p[0] for p in part]):.3f} | {sum(p[1] for p in part)}/{R2} | "
              f"{sum(p[2] for p in part)}/{R2} |")
