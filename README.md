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

Early. Mantel-Haenszel DIF implemented and self-tested; difR cross-validation and
logistic-regression DIF in progress. See `DECISIONS.md` for the build order.

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
```

Returns a DataFrame per item: MH chi-square (`stat`), `p_value`, `alpha_mh`,
ETS `mh_ddif`, `flag`, and ETS class `ets` (A/B/C).

## Validation

Every method is checked against `difR` on shared datasets; reference values are
frozen as fixtures in `tests/`, so `uv run pytest` needs no R. Regenerate the
oracle with the scripts in `validation/` (requires R + difR).
