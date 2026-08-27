# Findings: DIF on the real Open LLM Leaderboard

Data: `efficbench` / tinyBenchmarks release (Polo et al., ICML 2024) — real
per-item correctness for **395 Open LLM Leaderboard models** across MMLU, ARC,
HellaSwag, Winogrande and GSM8K. Ungated, MIT-licensed.

Grouping: base architecture parsed from model names — **Llama-2 (n=38)** vs
**Mistral (n=41)**. Person = model, item = benchmark question.

## Result: benchmark items function differently across architectures

Logistic-regression DIF, Benjamini-Hochberg FDR at q<.05, with a **permutation
control** (group labels shuffled, where no true DIF can exist):

| benchmark | items | flagged (real labels) | flagged (permuted) |
|---|---:|---:|---:|
| MMLU professional_law | 1534 | **571** | 0 |
| Winogrande | 1267 | **286** | 7 |
| ARC-Challenge | 1172 | **206** | 5 |
| GSM8K | 1319 | **154** | 0 |
| HellaSwag | 2200 | **132** | 0 |
| MMLU moral_scenarios | 895 | **103** | 0 |

The real-vs-permuted gap (up to ~50x) is the finding: a large share of items are
answered differently by Mistral-derived and Llama-2-derived models **after
conditioning on overall ability**. It replicates across six independent benchmarks.

## What this does and does not mean

- **Does mean:** these benchmarks are not strictly score-comparable across model
  families. Two models with the same total score are not answering the same items.
  Item-level composition matters when ranking across architectures.
- **Does NOT mean "contamination."** DIF is a *symptom*. Plausible causes include
  training-corpus overlap, tokenizer differences, prompt-format sensitivity, and
  instruction-tuning style. Attributing it to contamination needs separate evidence.
- **Sample caveat:** 38 + 41 models. Adequate for logistic DIF, and the permutation
  control guards against small-n artifacts, but effect sizes are not precise.

## Method note: Mantel-Haenszel fails on this data shape, by design

MH flagged **0** items everywhere. That is not "no DIF" — with 79 models and a
score range spanning ~1500 items there are ~1.2 models per score stratum, so MH's
conditional tables are empty. `check_adequacy()` now warns about exactly this.
The lesson generalizes: benchmark matrices are wide (few "persons", many items),
the transpose of the psychometric norm (many persons, few items), so stratification-
based methods need a coarsened matching score and regression/IRT methods are
preferred. This was invisible in simulation and only appeared on real data.

Reproduce: `python examples/real_leaderboard.py path/to/lb.pickle <benchmark_key>`
