# Next

State: v0.1. Two DIF methods (Mantel-Haenszel, logistic regression), both
validated to machine precision against independent implementations (statsmodels
CMH and GLM). Adapters reshape leaderboard results into DIF matrices. Contamination
demo works. 10 tests green. Runtime deps: numpy/scipy/pandas only.

## Do next (in order)

1. **Real-data adapters + demo (highest value).** Pull an actual per-item,
   per-model correctness matrix and run DIF on it. Candidate sources:
   Open LLM Leaderboard v2 `details_*` datasets on HF, lm-eval-harness sample
   outputs, HELM instance-level results. Write `adapters.from_lm_eval(...)` and
   `examples/real_leaderboard.py`. This is the "generational" proof — DIF on a
   real benchmark, flagging real contamination/translation artifacts.
2. **IRT-based DIF** (Lord's chi-square, Raju's area). Needs a validated 2PL
   backend — use `py-irt` (add as dep) or implement MML 2PL and validate against
   `mirt`/`py-irt`. Do NOT ship unvalidated IRT.
3. **Logistic purification** (mirror the MH purify loop).
4. **Freeze fixtures**: store statsmodels reference values as CSV in tests/ so the
   cross-check tests need no statsmodels at runtime (currently importorskip).
5. **Measurement-invariance omnibus** (the broader framing): configural/metric/
   scalar invariance via multi-group models — the CFA/SEM angle beyond item-level
   DIF. Larger; scope its own design.
6. **>2 groups**: generalized MH and IRT-based multi-group DIF.
7. **Polytomous items** (ordinal/graded responses).
8. Package: docs, `report()` convenience API, PyPI publish, CI.

## Guardrails
- Every new method gets an independent-oracle validation before it's "done".
- Keep runtime deps minimal; heavy stats libs stay dev-only.
- difR would be the ideal oracle but CRAN is unreachable here; statsmodels and
  py-irt are the working substitutes.
