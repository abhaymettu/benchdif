# Next

State: v0.1. Two DIF methods (Mantel-Haenszel, logistic regression), both
validated to machine precision against independent implementations (statsmodels
CMH and GLM). Adapters reshape leaderboard results into DIF matrices. Contamination
demo works. 10 tests green. Runtime deps: numpy/scipy/pandas only.

## Do next (in order)

1. **Real-data demo — PARTIALLY DONE, blocked on data access.**
   `adapters.from_lm_eval(...)` is built and tested (reads lm-eval-harness
   `--log_samples` JSONL into the long shape). BLOCKER: every compact per-item,
   per-model source found is gated. Open LLM Leaderboard v1/v2 `*-details` repos
   (996 of them) return HTTP 401 via the HF datasets-server without a token +
   accepting each repo's terms; no ungated multi-model per-question dataset turned
   up in search. To finish: (a) drop a HF token at ~/.cache/huggingface/token and
   accept ToS, then pull ~40 models across 2 families (e.g. Qwen vs Llama) on one
   task (arc_challenge/mmlu) via datasets-server /rows, pivot to a matrix, run DIF;
   or (b) run lm-eval-harness locally on a few small models with --log_samples and
   feed the dir to from_lm_eval. Then write examples/real_leaderboard.py. This is
   the "generational" proof and the top priority once a token exists.
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
