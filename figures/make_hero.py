"""Hero figure: DIF is large per item, small in the full score, large again in a subset.

Recomputes the numbers in FINDINGS.md section 3 with the same splits and seeds as
validation/subsample_honest.py, then draws two panels:

  left   variance of the per-item family gap over its binomial null, per benchmark
  right  family accuracy gap on the full benchmark vs the held-out swing of a
         100-item subset chosen on the other half of the models

    uv run --with matplotlib python figures/make_hero.py data/lb.pickle
"""
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCH = [
    ("MMLU prof. law", "harness_hendrycksTest_professional_law_5"),
    ("ARC-Challenge", "harness_arc_challenge_25"),
    ("Winogrande", "harness_winogrande_5"),
    ("GSM8K", "harness_gsm8k_5"),
    ("HellaSwag", "harness_hellaswag_10"),
]
K = 100
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def arch(s):
    n = str(s).lower()
    for p, l in [("mixtral", "mixtral"), ("mistral", "mistral"),
                 ("llama-2", "llama2"), ("llama2", "llama2")]:
        if p in n:
            return l
    return "other"


def stats(obj, key):
    fams = np.array([arch(m) for m in obj["models"]])
    X = (np.asarray(obj["data"][key]["correctness"]).T >= 0.5).astype(int)
    sel = np.isin(fams, ["llama2", "mistral"])
    g = (fams[sel] == "mistral").astype(int)
    A, B = X[sel][g == 1], X[sel][g == 0]
    nA, nB = len(A), len(B)
    pA, pB = A.mean(0), B.mean(0)
    ratio = (pA - pB).var() / (pA * (1 - pA) / nA + pB * (1 - pB) / nB).mean()
    swings = []
    for seed in range(5):
        r = np.random.default_rng(seed)
        ia, ib = r.permutation(nA), r.permutation(nB)
        A1, A2 = A[ia[:nA // 2]], A[ia[nA // 2:]]
        B1, B2 = B[ib[:nB // 2]], B[ib[nB // 2:]]
        order = np.argsort(A1.mean(0) - B1.mean(0))
        lo, hi = order[:K], order[-K:]
        swings.append((A2[:, hi].mean() - B2[:, hi].mean())
                      - (A2[:, lo].mean() - B2[:, lo].mean()))
    return ratio, A.mean() - B.mean(), float(np.mean(swings))


def main(path):
    obj = pickle.load(open(path, "rb"))
    rows = [(name, *stats(obj, key)) for name, key in BENCH]
    for name, ratio, gap, swing in rows:
        print(f"{name:16s} var ratio {ratio:5.2f}x  full gap {gap:+.4f}  "
              f"held-out {K}-item swing {swing:+.3f}")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": INK})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=200,
                                 gridspec_kw={"width_ratios": [1, 1.3]})
    fig.patch.set_facecolor(SURFACE)
    names = [r[0] for r in rows][::-1]
    y = np.arange(len(rows))

    ratios = [r[1] for r in rows][::-1]
    a1.barh(y, ratios, height=0.5, color=BLUE)
    a1.axvline(1, color=MUTED, lw=1, ls="--")
    a1.text(1.12, -0.62, "no DIF = 1x", color=MUTED, fontsize=8.5)
    for yi, v in zip(y, ratios):
        a1.text(v + 0.15, yi, f"{v:.1f}x", va="center", color=INK, fontsize=9)
    a1.set_yticks(y, names)
    a1.set_ylim(-0.75, len(rows) - 0.5)
    a1.set_xlim(0, 11.5)
    a1.set_xlabel("variance of per-item family gap / binomial null")
    a1.set_title("Per item: 6 to 10x sampling noise", loc="left", color=INK,
                 fontsize=11, fontweight="bold")

    gaps = [abs(r[2]) for r in rows][::-1]
    swings = [r[3] for r in rows][::-1]
    for yi, gp, sw in zip(y, gaps, swings):
        a2.plot([gp, sw], [yi, yi], color=GRID, lw=2, zorder=1)
    a2.scatter(gaps, y, s=46, color=MUTED, zorder=2, label="full benchmark")
    a2.scatter(swings, y, s=46, color=ORANGE, zorder=2,
               label=f"{K}-item subset, held-out models")
    for yi, sw in zip(y, swings):
        a2.text(sw + 0.025, yi, f"{sw:.2f}", va="center", color=INK, fontsize=9)
    a2.set_yticks(y, [""] * len(rows))
    a2.set_xlim(0, 1.1)
    a2.set_ylim(-0.75, len(rows) - 0.5)
    a2.set_xlabel("Mistral minus Llama-2 accuracy gap (absolute)")
    a2.set_title("Cancels in the full score, not in a subset", loc="left",
                 color=INK, fontsize=11, fontweight="bold")
    fig.legend(loc="lower right", ncol=2, frameon=False, fontsize=8.5,
               labelcolor=MUTED, bbox_to_anchor=(0.995, -0.01))

    for ax in (a1, a2):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
    fig.text(0.01, 0.01, "Llama-2 n=38 vs Mistral n=41 of 395 Open LLM Leaderboard models. "
             "Swing: mean of 5 splits.",
             color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = os.path.join(os.path.dirname(__file__), "hero.png")
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/lb.pickle")
