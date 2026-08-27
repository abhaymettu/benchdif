# benchdif

**Differential item functioning (DIF) and measurement invariance for AI evaluations.**

Benchmark scores are treated as if a question means the same thing to every model.
Often it doesn't — an item can be systematically easier for one model family,
language, or prompt style at equal overall ability. That's *differential item
functioning*, and psychometrics has rigorous tests for it. `benchdif` brings those
tests to LLM/AI benchmarks.

Item-difficulty estimators for benchmarks already exist (`py-irt`, `IRTorch`,
`PSN-IRT`); fairness scanners exist (`LangFair`). None test invariance or DIF.
This does, and it is validated to numerical agreement against R's `difR`.

## Status

Working and validated, with a real result.

Four DIF methods (Mantel-Haenszel, logistic regression, anchor-purified IRT-LR,
generalized MH for >2 groups) plus a measurement-invariance omnibus, on a
hand-written 2PL MML-EM backend. The two closed-form tests match statsmodels to
machine precision; the IRT pieces are validated by parameter recovery and by
power/size simulation. 37 tests.

**Real-data finding.** Run on 395 Open LLM Leaderboard models, item-level DIF
between Mistral- and Llama-2-derived models is pervasive (6-10x the variance of
sampling noise across five benchmarks) but *cancels* at the test level, so full
benchmark scores are robust. Benchmark **subsets** are not: a 100-item subset
selected on one set of models shifts the family accuracy gap by 0.5-0.95 on
held-out models, against true gaps of 0.0005-0.19. Subset selection methods
(tinyBenchmarks, adaptive testing, "lite" splits) assume an item exchangeability
that does not hold. See **[FINDINGS.md](FINDINGS.md)**, including what it does not
show (this is differential functioning, not proven contamination).

## Install

```bash
uv sync
```

## Use

```python
import numpy as np
from benchdif import mantel_haenszel

# responses: (n_persons x n_items) 0/1 ; group: 0=reference, 1=focal
res = mantel_haenszel(responses, group)
print(res[res.flag])          # items with significant DIF
res.loc[3, ["stat", "mh_ddif", "ets"]]

# score purification (drop DIF items from the matching score, iteratively)
res = mantel_haenszel(responses, group, purify=True)

# logistic-regression DIF (uniform + non-uniform), matches R glm
from benchdif import logistic
lr = logistic(responses, group, kind="both")

# IRT-based DIF (anchor-purified likelihood-ratio test, separates impact from DIF)
from benchdif import irt_lr
irt = irt_lr(responses, group)      # per item: stat, p_value, flag, da, db, is_anchor

# or the unified entry point (normalized summary across methods)
from benchdif import detect
summary = detect(responses, group, method="irt")   # 'mh' | 'logistic' | 'irt'
summary[summary.flag]                                # items with DIF

# more than two groups (e.g. many model families) -- generalized MH
from benchdif import generalized_mantel_haenszel
generalized_mantel_haenszel(responses, group_labels)   # group_labels has >=2 levels

# measurement-invariance omnibus (scale-level: configural -> metric -> scalar)
from benchdif import invariance
inv = invariance(responses, group)   # rows metric/scalar: stat, df, p_value, holds
```

Returns a DataFrame per item: MH chi-square (`stat`), `p_value`, `alpha_mh`,
ETS `mh_ddif`, `flag`, and ETS class `ets` (A/B/C).

## Example

`examples/quickstart.py` exercises every public function on a simulated benchmark
(300 models, a leaked item), and `examples/contamination_demo.py` shows a leak
invisible to accuracy caught by DIF. Run either with `uv run python examples/<file>`.

## Validation

Every method is checked against `difR` on shared datasets; reference values are
frozen as fixtures in `tests/`, so `uv run pytest` needs no R. Regenerate the
oracle with the scripts in `validation/`.

For the real-data analysis, `python data/fetch_leaderboard.py` downloads the
leaderboard matrices, then run `validation/robustness_real_data.py`,
`validation/subsample_honest.py` and `validation/dependency_diagnostic.py` with the
path to `data/lb.pickle`.
