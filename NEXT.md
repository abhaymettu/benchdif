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
2. **IRT-based DIF** — 2PL BACKEND DONE. `benchdif.fit_2pl` is a numpy/scipy
   Bock-Aitkin MML-EM estimator, validated by parameter recovery (corr_a>0.99,
   corr_b>0.999, RMSE~0.05 across seeds; see validation/validate_twopl.py). STILL
   TODO: the DIF test on top. Cleanest validated route is IRT-LR (Thissen): a
   concurrent multi-group 2PL with anchor items, comparing the studied item
   constrained-equal vs freed across groups via an LR chi-square. Alternative:
   Lord's Wald chi-square / Raju's signed+unsigned area, which need cross-group
   linking (mean-mean or Stocking-Lord on anchors) — the linking is the crux and
   must be validated (simulate group-specific params, confirm recovery + correct
   flags). Do NOT ship the DIF test until it's validated the same way.
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
