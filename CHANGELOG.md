# Changelog

## Unreleased (branch harness-dif)

- **Paired-condition DIF** (`paired_dif`): one model under two conditions
  (harness, reasoning effort) on shared items with repeated trials. Conditions on
  each (model, item) total so model-specific item effects cancel; impact is the
  conditional-ML log odds shift, DIF the item-level deviation, with exact per-item
  p-values, a bootstrap omnibus and optional 2PL slopes. Validated against
  statsmodels ConditionalLogit and by size/power simulation.
- **Held-out DTF** (`paired_dtf`): splits each model's score gap into impact and
  DIF parts and predicts the DIF part from item DIF fitted on other models or
  trials (Raju's DTF in conditional form).
- **measurement-db adapters**: `condition_pairs`, `paired_counts`,
  `eligible_cells`, `runs_matrix`.
- `examples/harness_dif.py` runs the full analysis on measurement-db tables;
  `examples/synthetic_mdb.py` writes schema-identical synthetic tables.

## 0.1.0 (unreleased)

First working version.

- **Mantel-Haenszel DIF** (`mantel_haenszel`) with ETS A/B/C classification, score
  purification, and external match-vector support. Validated to machine precision
  against statsmodels' CMH.
- **Logistic-regression DIF** (`logistic`): uniform / non-uniform / both, Nagelkerke
  effect size with Jodoin-Gierl classes, score purification. Validated against
  statsmodels GLM.
- **IRT-LR DIF** (`irt_lr`): anchor-purified likelihood-ratio test on a hand-written
  2PL MML-EM backend (`fit_2pl`). Full power, controlled false positives, separates
  group impact from DIF. 2PL validated by parameter recovery.
- **Adapters**: `from_long`, `from_matrix`, `from_lm_eval` reshape benchmark results
  into DIF matrices (person=model, item=question, group=family/language/condition).
- **Unified API**: `detect(responses, group, method=...)` with a normalized summary.
- **Generalized Mantel-Haenszel** (`generalized_mantel_haenszel`) for >2 groups
  (many model families at once); reduces to the uncorrected CMH at 2 groups.
- **Measurement-invariance omnibus** (`invariance`): configural -> metric ->
  scalar likelihood-ratio ladder on the multi-group 2PL, separating group impact
  from non-invariance. Validated by scenario + size simulation.
- Runtime dependencies: numpy, scipy, pandas only.
