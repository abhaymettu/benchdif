# AGENTS.md

Notes for coding agents working in this repo.

## Layout

```
src/benchdif/
  dif/            mantel_haenszel, logistic, irt_lr, generalized_mh, paired (paired_dif, paired_dtf)
  irt/            twopl (Bock-Aitkin MML-EM), multigroup, invariance
  adapters.py     long/matrix/lm-eval/measurement-db input shapes
  detect.py       one entry point across methods
  multiplicity.py adjust (BH or Holm) and check_adequacy
validation/       one validate_*.py per method (oracle or simulation), plus real-data scripts
tests/            pytest; cross-checks against statsmodels where an oracle exists
examples/         runnable demos; quickstart.py touches every public function
figures/          make_hero.py regenerates hero.png from data/lb.pickle
data/             fetch_leaderboard.py; the pickles it downloads are gitignored
```

Runtime dependencies stay at numpy, scipy and pandas. statsmodels, pytest and pyarrow are dev only.

## Run

```bash
uv sync
uv run pytest -q                                   # about 90 s
uv run python examples/contamination_demo.py       # 2 s smoke test
uv run python data/fetch_leaderboard.py            # 90 MB, needed for the real-data scripts
```

## Adding or changing a method

A method is not done until an independent check agrees with it.

1. If a closed form exists in statsmodels or scipy, write `validation/validate_<method>.py` that fits the same model with that library on simulated data and prints the max absolute difference. `validate_mh.py` and `validate_logistic.py` are the pattern: they end with a `VALIDATED:` line and agree to about 1e-13.
2. If no oracle exists (IRT, invariance, paired DIF), validate by simulation: parameter recovery for estimators, size under a no-DIF truth and power under planted DIF for tests. `validate_irt_lr.py` and `simulate_harness_dif.py` are the pattern.
3. Add a test in `tests/` that runs a small version of the same check, so a regression fails the test suite.
4. Record the choice and why in `DECISIONS.md`, newest first.

## Conventions

- Every number in README.md, FINDINGS*.md or a figure comes from a script in this repo with a fixed seed. If you change a script, rerun it and update the docs from its output. Do not type a number by hand.
- Matrices are persons x items, 0/1, persons are models (or runs), `group` is coded 0 = reference, 1 = focal.
- Leaderboard matrices are wide (few models, many items). Call `check_adequacy()` before trusting Mantel-Haenszel on them.
- No em dashes or en dashes anywhere: prose, comments, docstrings or commit messages.
- Commit messages are short and plain, written the way a person would write them. No co-author or tool attribution trailers.
- Say what a result does not show. DIF is not contamination, and per-item flags replicate only about 40% of the time.
