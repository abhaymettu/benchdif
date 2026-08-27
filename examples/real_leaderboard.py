"""Real-data demo: DIF and measurement invariance on actual leaderboard results.

Data: the model-by-example correctness matrices released with efficbench /
tinyBenchmarks (Polo et al., ICML 2024), which are the real Open LLM Leaderboard
(and HELM) per-item results for hundreds of models. Ungated, MIT-licensed.

    correctness[example, model] in {0,1}  +  a list of model names

We treat person = model, item = benchmark question, group = model family (parsed
from the model name), then ask: conditioning on overall ability, does any item
favor one family? That is DIF; at scale it is a contamination / training-artifact
probe that overall accuracy cannot see.

Usage:
    python examples/real_leaderboard.py path/to/lb.pickle [benchmark_key]
"""
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd

from benchdif import (from_matrix, detect, invariance, adjust,
                      check_adequacy, generalized_mantel_haenszel)


_ARCH = [("mixtral", "mixtral"), ("mistral", "mistral"), ("llama-3", "llama3"),
         ("llama3", "llama3"), ("llama-2", "llama2"), ("llama2", "llama2"),
         ("qwen", "qwen"), ("yi-", "yi"), ("solar", "solar"), ("gemma", "gemma"),
         ("falcon", "falcon"), ("phi", "phi"), ("mpt", "mpt")]


def family(name: str) -> str:
    """Base architecture family from an Open LLM Leaderboard model id.

    Grouping by base architecture (Mistral / Llama-2 / ...) rather than by
    uploading org: it is the scientifically meaningful family (shared pretraining
    corpus and tokenizer) and it yields groups large enough to power DIF. Models
    whose architecture is not identifiable from the name are excluded.
    """
    n = str(name).lower()
    for pat, lab in _ARCH:
        if pat in n:
            return lab
    return "other"


def main(path, bench_key=None):
    obj = pickle.load(open(path, "rb"))
    data, models = obj["data"], [str(m) for m in obj["models"]]
    fams = [family(m) for m in models]
    print(f"{len(models)} models; families: {dict(Counter(fams))}")

    # pick a benchmark (largest by #items if not given)
    if bench_key is None:
        bench_key = max(data, key=lambda k: data[k]["correctness"].shape[0])
    corr = np.asarray(data[bench_key]["correctness"])
    # orient to (models x items): 'correctness' is (items x models) in this release
    if corr.shape[1] == len(models):
        X = corr.T
    else:
        X = corr
    X = (X >= 0.5).astype(int)
    print(f"\nbenchmark: {bench_key}\nmatrix: {X.shape[0]} models x {X.shape[1]} items")

    wide = pd.DataFrame(X, index=models)
    fam_series = pd.Series(fams, index=models)

    # --- two biggest identifiable architectures -> two-group DIF ---
    named = Counter(f for f in fams if f != "other")
    top2 = sorted([f for f, _ in named.most_common(2)])   # sorted => reference is top2[0]
    keep = fam_series.isin(top2)
    wide2 = wide[keep.values]
    g2 = fam_series[keep.values]
    Xg, gc, persons, items = from_matrix(wide2, g2)
    print(f"\nTwo-group DIF: reference={top2[0]} (n={int((gc==0).sum())}) vs "
          f"focal={top2[1]} (n={int((gc==1).sum())})")
    info = check_adequacy(Xg, gc)          # warns on thin strata / degenerate items
    print(f"  adequacy: {info['persons_per_stratum']:.1f} persons/score-stratum, "
          f"{info['degenerate_items']} zero-variance items")
    for m in ("mh", "logistic"):
        res = detect(Xg, gc, method=m)
        raw = int(res.flag.sum())
        q = adjust(res.p_value, "bh")       # FDR: mandatory with this many items
        print(f"  {m:9s}: {raw}/{len(res)} raw p<.05  ->  "
              f"{int((q < 0.05).sum())}/{len(res)} after BH-FDR")

    # --- all families at once -> generalized MH ---
    fam_arr = np.array(fams)
    big = [f for f, n in Counter(fams).items() if f != "other" and n >= 6]
    sel = np.isin(fam_arr, big)
    gmh = generalized_mantel_haenszel(X[sel], fam_arr[sel])
    q_g = adjust(gmh.p_value, "bh")
    print(f"\nGeneralized MH across {len(big)} architectures {sorted(big)} "
          f"(n={int(sel.sum())} models): {int(gmh.flag.sum())}/{len(gmh)} raw p<.05 "
          f"-> {int((q_g < 0.05).sum())}/{len(gmh)} after BH-FDR")
    top = gmh.sort_values("stat", ascending=False).head(5)
    print("  strongest DIF items (index : chi2):",
          {int(i): round(s, 1) for i, s in zip(top.index, top.stat)})

    # --- scale-level invariance between the two families ---
    if Xg.shape[0] >= 60 and Xg.shape[1] <= 60:
        inv = invariance(Xg, gc)
        print("\nMeasurement invariance (two families):")
        print(inv.to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
