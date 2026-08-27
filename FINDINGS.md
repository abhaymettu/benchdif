# Findings: item-level DIF on the real Open LLM Leaderboard

Data: `efficbench` / tinyBenchmarks release (Polo et al., ICML 2024) — real per-item
correctness for **395 Open LLM Leaderboard models**. Ungated, MIT-licensed.
Groups: base architecture parsed from model names, **Llama-2 (n=38)** vs
**Mistral (n=41)**. Person = model, item = benchmark question.

## 1. Item-level DIF is pervasive and far larger than sampling noise

Logistic-regression DIF with BH-FDR (q<.05) flags 103–571 items per benchmark. The
effect is not an artifact of multiplicity, small n, or model non-independence:

| check | result |
|---|---|
| label-permutation control | real 571 flagged vs permuted 0–7 |
| **lineage-preserving** permutation (see §4) | real 571 vs permuted median **2** |
| variance of per-item family gap vs binomial null | **6.1x – 9.7x** across 5 benchmarks |

That last row is the cleanest statement: **~85–90% of the variance in per-item family
differences is item-specific, not sampling noise.**

## 2. But it cancels at the test level — the full-benchmark score is safe

Of 571 flagged items on MMLU professional_law, **278 favour Mistral and 293 favour
Llama-2**. Dropping every flagged item barely moves anything:

- mean |rank change| 3.6 places out of 79 models; top-10 turnover 1/10
- family accuracy gap +0.0216 → +0.0221

**This is a null result and it matters.** Aggregate leaderboard scores are robust to
DIF because pro-A and pro-B items cancel almost exactly.

## 3. Subsampling destroys the cancellation — the actionable finding

Benchmark subsetting (tinyBenchmarks, adaptive testing, "lite" splits) assumes items
are exchangeable across models. DIF violates that, and the protection in §2 disappears.

Selecting 100 items on one half of the models and scoring the gap on the **held-out**
half (so the selection is not circular):

| benchmark | items | var ratio | held-out swing | true full gap |
|---|---:|---:|---:|---:|
| MMLU professional_law | 1534 | 8.8x | **+0.875** | +0.0216 |
| ARC-Challenge | 1172 | 7.8x | +0.634 | +0.0272 |
| Winogrande | 1267 | 9.7x | +0.759 | +0.0090 |
| GSM8K | 1319 | 6.1x | +0.500 | +0.1885 |
| HellaSwag | 10042 | 7.3x | **+0.954** | +0.0005 |

A curated 100-item subset moves the family gap by **0.5–0.95**, against true gaps of
0.0005–0.19 — up to ~40x the real difference on professional_law, and on HellaSwag a
genuinely null gap becomes a 95-point one. Even *random* 100-item subsets flip the
sign of the family ordering 65/400 times.

**Implication:** benchmark subset selection must be DIF-aware. A "lite" split can show
either architecture ahead without containing a single mislabelled item.

## 4. Robustness: are the models independent?

DIF assumes exchangeable persons; leaderboard "families" are finetunes and merges.
Measured directly from response vectors: mean pairwise correlation **0.469 within**
family vs **0.365 between**, but only 6 near-duplicate pairs (r>0.90). Single-linkage
clustering gives **36 distinct lineages of 38** Llama-2 models and **38 of 41** Mistral.
Effective n is close to nominal n, and the finding survives permuting **lineage** labels
rather than model labels (real 571, permuted median 2).

## 5. What this does NOT show

- **Not contamination.** DIF is a symptom. Tokenizer, prompt-format sensitivity,
  instruction-tuning style and training-mix differences all produce it. Attribution
  needs separate evidence.
- **Individual item flags are noisy.** Items flagged on half the models replicate on
  held-out models only ~40% of the time (base rate 13–22%). The aggregate effect is
  solid; "item 466 has DIF" is not a reliable claim.
- **Era-bound.** These are 2023–24 leaderboard models. Generalization to current
  frontier models is untested.
- **Mantel-Haenszel is inapplicable here** and returns 0 everywhere: 79 models over
  ~1500 score levels is ~1.2 models per stratum, so its conditional tables are empty.
  Benchmark matrices are *wide* (few persons, many items), the transpose of the
  psychometric norm. `check_adequacy()` warns about this.

Reproduce: `validation/robustness_real_data.py`, `validation/subsample_honest.py`,
`validation/dependency_diagnostic.py`, each taking the path to `lb.pickle`.
