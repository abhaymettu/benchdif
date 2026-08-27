# Changelog

## 0.1.0 (unreleased)

First working version.

- **Mantel-Haenszel DIF** (`mantel_haenszel`) with ETS A/B/C classification, score
  purification, and external match-vector support. Validated to machine precision
  against statsmodels' CMH.
- **Logistic-regression DIF** (`logistic`) — uniform / non-uniform / both, Nagelkerke
  effect size with Jodoin-Gierl classes, score purification. Validated against
  statsmodels GLM.
- **IRT-LR DIF** (`irt_lr`) — anchor-purified likelihood-ratio test on a hand-written
  2PL MML-EM backend (`fit_2pl`). Full power, controlled false positives, separates
  group impact from DIF. 2PL validated by parameter recovery.
- **Adapters**: `from_long`, `from_matrix`, `from_lm_eval` reshape benchmark results
  into DIF matrices (person=model, item=question, group=family/language/condition).
- **Unified API**: `detect(responses, group, method=...)` with a normalized summary.
- **Measurement-invariance omnibus** (`invariance`) — configural -> metric ->
  scalar likelihood-ratio ladder on the multi-group 2PL, separating group impact
  from non-invariance. Validated by scenario + size simulation.
- Runtime dependencies: numpy, scipy, pandas only.
