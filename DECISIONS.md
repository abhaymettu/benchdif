# Decisions

Analysis/design choices, written as they are made. Newest first.

## 2026-08-26 — Project start

**What this is.** `benchdif`: a Python library for differential item functioning
(DIF) and measurement invariance, aimed at AI/LLM evaluations. The bet: the field
is realizing benchmarks lack validity (NeurIPS 2025 reviewed 445 benchmarks on
construct validity) but no open tool operationalizes DIF/invariance for evals.
`py-irt`/`IRTorch`/`PSN-IRT` estimate item difficulty; `LangFair` does fairness —
none do DIF. This fills that hole.

**Correctness before features.** Every method is cross-checked numerically against
R's `difR` on shared data. Reference values are frozen as fixtures so tests run
without R. A method is not "done" until it matches difR to tolerance.

**Method order (most-validated first).** Mantel-Haenszel → logistic regression →
IRT-based (Lord, Raju). MH and logistic are exact and have a difR oracle; IRT comes
after an IRT backend is chosen.

**MH conventions (match difR::difMH defaults).**
- Matching variable = total score over ALL items, studied item included (`match="score"`).
- Continuity-corrected Holland-Thayer chi-square; flag at qchisq(.95,1)=3.841.
- ETS A/B/C via MH D-DIF = -2.35*ln(alpha_MH): A if not sig or |D-DIF|<1; C if sig
  and |D-DIF|>=1.5; B otherwise.
- Reference=0, focal=1. Purification not yet implemented.

**The AI-eval reframing (the novel contribution).** Classic DIF groups are
demographic. Here "group" = model family / language / prompt condition, "person" =
a model (or a run), "item" = a benchmark question, "response" = correct/incorrect.
Adapters reshape benchmark result files into the (person x item, group) matrix DIF
needs. This is what makes it a benchmarking tool, not a re-implementation of difR.

**Stack.** Python 3.12, numpy/scipy/pandas, statsmodels for logistic (matches R glm).
uv for env. Not R — the point is a Python-native tool for the ML world.

## 2026-08-27 — IRT backend + real-data adapter

**2PL by hand, not py-irt.** Implemented Bock-Aitkin MML-EM in numpy/scipy rather
than depend on py-irt (torch/pyro). Keeps runtime deps light (the adoption bet) and
the M-step reuses the logistic IRLS already written. Validated by parameter recovery
(the standard for an estimator) instead of an external oracle: simulate known (a,b),
refit, corr_a>0.99 / corr_b>0.999. This is the foundation for IRT-based DIF.

**Real-data blocked on gating, adapter shipped anyway.** Open LLM Leaderboard
details repos (996) are gated (HTTP 401 without a token + per-repo ToS), and no
ungated multi-model per-question dataset surfaced. Built and tested `from_lm_eval`
(reads lm-eval-harness --log_samples JSONL) so the pipeline is ready the moment
real logs exist; documented the token/local-run paths in NEXT.md.
