"""Does a harness or effort change only shift ability, or change which items a model solves?

    python examples/harness_dif.py ROOT [--min-items 30] [--top 3]

ROOT holds measurement-db benchmarks as ROOT/<benchmark>/formatted_tables/
{subjects,responses}.parquet. Steps, in order:

1. Eligible cells: one model under two conditions that differ only in harness or
   only in reasoning effort, on >= min-items shared binary items. The count is
   printed before anything else; with none, the script stops there.
2. Each contrast (benchmark, factor, cond_0 -> cond_1, pooled over its models):
   paired_dif under the Rasch null, and under the 2PL null when >= 30 subjects
   answer every item; per-item BH flags, direction balance, ETS classes.
3. Controls: lineage-preserving permutation (condition labels flipped within
   model, all 2^(K-1) relabelings up to 64) and a negative control (odd vs even
   trials of the same subject and condition, which should show neither impact nor
   DIF). Person-level logistic + invariance ladder on runs when >= 20 runs/group.
4. Test-level consequence and theory check. Impact is fixed on anchor items (the
   lowest-DIF half, chosen on held-out data). G_obs = observed successes minus
   their impact-only expectation, per model, as a share of trials. G_pred = Raju's
   DTF in conditional form: the sum over items of the gap between the response
   law with and without item DIF, at the model's own matched level, with item
   DIF estimated on held-out data (the other models, or the other half of the
   trials for a single-model contrast). Agreement means the score effect of DIF
   is explained by the item parameters. Reordering compares the models' order
   under cond_1 with the impact-only order.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from benchdif import (adjust, condition_pairs, eligible_cells, fit_2pl, invariance,
                      logistic, paired_counts, runs_matrix)
from benchdif.dif.paired import paired_dif, paired_dtf

FACTORS = ("harness", "reasoning_effort")


def flip(c, cells):
    f = c.cell.isin(cells)
    out = c.copy()
    out.loc[f, ["k0", "n0", "k1", "n1"]] = c.loc[f, ["k1", "n1", "k0", "n0"]].to_numpy()
    return out


def analyse(name, factor, c0, c1, cells, subjects, responses, min_items):
    pairs = condition_pairs(subjects, factor)
    pairs = pairs[(pairs.cond_0 == c0) & (pairs.cond_1 == c1)
                  & pairs.subject_0.isin(cells.subject_0)]
    c = paired_counts(responses, pairs, min_items)
    K = c.cell.nunique()
    T = int(cells[["trials_0", "trials_1"]].min().min())
    print(f"\n## {name} / {factor}: {c0} -> {c1}  ({K} models, "
          f"{c.item_id.nunique()} items, {T} trials)\n")
    res = paired_dif(c.cell, c.item_id, c.k0, c.n0, c.k1, c.n1, n_boot=1000)
    it = res.items
    fl = it[it.flag]
    print(f"- impact (logit, per model): "
          + ", ".join(f"{r.model.split('|')[0]} {r.delta_hat:+.2f}"
                      for r in res.impact.itertuples()))
    print(f"- omnibus DIF beyond Rasch impact: T = {res.stat:.1f}, null mean "
          f"{res.null_mean:.1f}, variance ratio {res.var_ratio:.2f}, p = {res.p_value:.4f}")
    print(f"- items flagged (BH q<.05): {len(fl)}/{len(it)} = {len(fl) / len(it):.1%}; "
          f"favour {c1}: {(fl.delta > 0).sum()}, favour {c0}: {(fl.delta < 0).sum()}; "
          f"ETS A/B/C: {(it.ets == 'A').sum()}/{(it.ets == 'B').sum()}/{(it.ets == 'C').sum()}")

    # 2PL null: slopes from all subjects that answered every contrast item (trial 1)
    X, persons = runs_matrix(responses, subjects.subject_id, items=sorted(it.item))
    X = X[(persons.trial == persons.trial.min()).to_numpy()]
    if len(X) >= 30:
        a = fit_2pl(X).a
        r2 = paired_dif(c.cell, c.item_id, c.k0, c.n0, c.k1, c.n1,
                        slope=dict(zip(sorted(it.item), a)), n_boot=1000)
        print(f"- omnibus under the 2PL null ({len(X)} subjects calibrate slopes): "
              f"variance ratio {r2.var_ratio:.2f}, p = {r2.p_value:.4f}")
    else:
        print(f"- 2PL null: n/a ({len(X)} subjects answer every item; need 30)")

    # control 1: lineage-preserving permutation
    if K >= 2:
        ids = sorted(c.cell.unique())
        subsets = [s for s in itertools.product([0, 1], repeat=K - 1)]
        if len(subsets) > 64:
            pick = np.random.default_rng(0).choice(len(subsets), 64, replace=False)
            subsets = [subsets[i] for i in pick]
        null = []
        for s in subsets:
            cf = flip(c, [ids[j + 1] for j, b in enumerate(s) if b])
            null.append(paired_dif(cf.cell, cf.item_id, cf.k0, cf.n0, cf.k1, cf.n1,
                                   n_boot=0).stat)
        print(f"- control, lineage-preserving permutation ({len(null)} relabelings): "
              f"real T {res.stat:.1f} vs permuted median {np.median(null):.1f}, "
              f"p = {np.mean(np.array(null) >= res.stat - 1e-9):.3f}")
    else:
        print("- control, lineage-preserving permutation: n/a with one model")

    # control 2: odd vs even trials of the same subject and condition
    if T >= 2:
        nc = paired_counts(responses[responses.trial % 2 == 1],
                           pairs.assign(subject_1=pairs.subject_0), min_items=1)
        ev = paired_counts(responses[responses.trial % 2 == 0],
                           pairs.assign(subject_1=pairs.subject_0), min_items=1)
        nc = nc.drop(columns=["k1", "n1"]).merge(
            ev[["cell", "item_id", "k0", "n0"]].rename(columns={"k0": "k1", "n0": "n1"}),
            on=["cell", "item_id"])
        rn = paired_dif(nc.cell, nc.item_id, nc.k0, nc.n0, nc.k1, nc.n1, n_boot=1000)
        print(f"- control, odd vs even trials of the same subject: impact "
              f"{rn.impact.delta_hat.abs().max():.2f} max |logit|, variance ratio "
              f"{rn.var_ratio:.2f}, p = {rn.p_value:.4f}, flagged {int(rn.items.flag.sum())}")
    else:
        print("- control, odd vs even trials: n/a with one trial")

    # person-level methods on runs
    subj = list(pairs.subject_0) + list(pairs.subject_1)
    X, persons = runs_matrix(responses[responses.item_id.isin(it.item)], subj)
    g = persons.subject_id.isin(pairs.subject_1).to_numpy().astype(int)
    if min(np.bincount(g, minlength=2)) >= 20:
        keep = X.std(axis=0) > 0
        q = adjust(logistic(X[:, keep], g)["p_value"].to_numpy(), "bh")
        inv = invariance(X[:, keep], g)
        print(f"- person-level (runs): logistic flags {np.mean(q < .05):.1%} of items; "
              f"ladder metric p = {inv.loc['metric', 'p_value']:.3g}, scalar p = "
              f"{inv.loc['scalar', 'p_value']:.3g} (nominal; see simulation for size)")
    else:
        print(f"- person-level (runs): n/a ({min(np.bincount(g, minlength=2))} runs "
              "in the smaller group; need 20)")

    # test-level consequence and theory check, on held-out folds
    if K >= 2:
        folds = [(c[c.cell != m], c[c.cell == m]) for m in sorted(c.cell.unique())]
    elif T >= 2:
        odd = paired_counts(responses[responses.trial % 2 == 1], pairs, min_items=1)
        even = paired_counts(responses[responses.trial % 2 == 0], pairs, min_items=1)
        folds = [(odd, even), (even, odd)]
    else:
        print("- test-level consequence: n/a (one model, one trial: no held-out fold)")
        return
    G = pd.concat([paired_dtf(tr, te) for tr, te in folds]).groupby(level=0).mean()
    gap = (c.groupby("cell").k1.sum() / c.groupby("cell").n1.sum()
           - c.groupby("cell").k0.sum() / c.groupby("cell").n0.sum())
    G = G.join(gap.rename("score_gap"))
    print("\n| model | score gap (cond_1 - cond_0) | DIF part, observed | DIF part, predicted |")
    print("|---|---:|---:|---:|")
    for cell, r in G.iterrows():
        print(f"| {cell.split('|')[0]} | {r.score_gap:+.3f} | {r.G_obs:+.3f} | {r.G_pred:+.3f} |")
    if K >= 3:
        print(f"\n- predicted vs observed DIF part across models: r = "
              f"{np.corrcoef(G.G_pred, G.G_obs)[0, 1]:.2f}")
    common = set.intersection(*(set(g.item_id) for _, g in c.groupby("cell")))
    if K >= 2 and len(common) >= min_items:
        cc = c[c.item_id.isin(common)]
        s1 = cc.groupby("cell").k1.sum() / cc.groupby("cell").n1.sum()
        Gc = pd.concat([paired_dtf(tr[tr.item_id.isin(common)], te[te.item_id.isin(common)])
                        for tr, te in folds]).groupby(level=0).mean()
        imp, pred = s1 - Gc.G_obs, s1 - Gc.G_obs + Gc.G_pred
        pairs_ = list(itertools.combinations(s1.index, 2))
        dis = lambda u, w: sum(np.sign(u[i] - u[j]) != np.sign(w[i] - w[j]) for i, j in pairs_)
        print(f"- reordering on {len(common)} common items: observed cond_1 order vs "
              f"impact-only order disagree on {dis(s1, imp)}/{len(pairs_)} model pairs; "
              f"the DTF-predicted order disagrees with the observed on {dis(s1, pred)}")
    else:
        print(f"- reordering: n/a ({len(common)} items common to all models)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--min-items", type=int, default=30)
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()
    data, rows = {}, []
    for t in sorted(Path(args.root).glob("*/formatted_tables/subjects.parquet")):
        name = t.parent.parent.name
        subjects = pd.read_parquet(t)
        responses = pd.read_parquet(t.parent / "responses.parquet",
                                    columns=["subject_id", "item_id", "trial", "response"])
        data[name] = subjects, responses
        for f in FACTORS:
            e = eligible_cells(responses, subjects, f, args.min_items)
            if len(e):
                rows.append(e.assign(benchmark=name, factor=f))
    print("# Eligible cells (one model, two conditions differing in one factor, "
          f">= {args.min_items} shared binary items)\n")
    if not rows:
        print("None. Stopping: the count is the result.")
        return
    cells = pd.concat(rows, ignore_index=True)
    summary = (cells.groupby(["benchmark", "factor", "cond_0", "cond_1"])
               .agg(models=("model", "nunique"), items=("n_items", "median"),
                    trials=("trials_0", "median"))
               .assign(power=lambda d: d.models * d["items"] * d.trials)
               .sort_values("power", ascending=False))
    print(f"{len(cells)} cells in {cells.benchmark.nunique()} benchmarks, "
          f"{len(summary)} contrasts.\n")
    print(summary.drop(columns="power").to_string())
    print(f"\n# Analysis of the {args.top} best-powered contrasts "
          "(models x median items x median trials)")
    for (b, f, c0, c1), _ in summary.head(args.top).iterrows():
        sub = cells[(cells.benchmark == b) & (cells.factor == f)
                    & (cells.cond_0 == c0) & (cells.cond_1 == c1)]
        analyse(b, f, c0, c1, sub, *data[b], args.min_items)


if __name__ == "__main__":
    main()
