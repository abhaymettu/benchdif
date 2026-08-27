# Next

**STATUS 2026-08-27: core feature-complete and validated.** Four DIF methods
(Mantel-Haenszel, logistic, IRT-LR, generalized MH for >2 groups), a measurement-
invariance omnibus, a 2PL IRT backend, three adapters, a unified detect() API, a
quickstart, and full packaging (MIT LICENSE, CHANGELOG, pyproject metadata). 32
tests green; every statistical method validated (DIF vs statsmodels to machine
precision; IRT pieces by parameter recovery and power/size simulation). 13 commits.

The only substantive remaining work is BLOCKED on data access (the real-leaderboard
demo needs a HF token). Everything else below is optional polish. The loop cadence
has been slowed accordingly -- it now mainly watches for the token.

---

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
2. **IRT-based DIF** — DONE (IRT-LR). 2PL MML-EM backend + anchor-purified
   IRT-LR test (`benchdif.irt_lr`), validated: full power, ~0% FPR with DIF
   present, size ~0.03 under pure impact (validation/validate_irt_lr.py). The
   naive all-shared baseline over-flagged (35% FPR from anchor contamination);
   the two-stage anchor fix (rank-then-purify) solved it. Remaining IRT extras
   (optional): Lord's Wald / Raju's area with explicit linking; iterative anchor
   refinement. Original note kept below.
   OLD: `benchdif.fit_2pl` is a numpy/scipy
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
5. **Measurement-invariance omnibus** — DONE. `benchdif.invariance` runs the
   configural -> metric -> scalar LR ladder on the multi-group 2PL, validated:
   invariant data never rejects despite impact (0/8), intercept-DIF localizes to
   scalar (0/8 metric, 8/8 scalar), loading-DIF triggers metric (8/8). See
   validation/validate_invariance.py.
6. **>2 groups**: generalized MH and IRT-based multi-group DIF.
7. **Polytomous items** (ordinal/graded responses).
8. Package: docs, `report()` convenience API, PyPI publish, CI.

## Guardrails
- Every new method gets an independent-oracle validation before it's "done".
- Keep runtime deps minimal; heavy stats libs stay dev-only.
- difR would be the ideal oracle but CRAN is unreachable here; statsmodels and
  py-irt are the working substitutes.
