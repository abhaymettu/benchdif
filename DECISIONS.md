# Decisions

Analysis/design choices, written as they are made. Newest first.

## 2026-08-26: Project start

**What this is.** `benchdif`: a Python library for differential item functioning
(DIF) and measurement invariance, aimed at AI/LLM evaluations. The bet: the field
is realizing benchmarks lack validity (NeurIPS 2025 reviewed 445 benchmarks on
construct validity) but no open tool operationalizes DIF/invariance for evals.
`py-irt`/`IRTorch`/`PSN-IRT` estimate item difficulty; `LangFair` does fairness;
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
uv for env. Not R: the point is a Python-native tool for the ML world.

## 2026-08-27: IRT backend + real-data adapter

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

## 2026-08-27: IRT-LR DIF, and the anchor-contamination fix

Added IRT likelihood-ratio DIF on the 2PL backend, with a freed focal ability
distribution so group impact is not read as DIF. First cut used an all-items-shared
baseline (all-other-anchor) and over-flagged badly: full power but 35% false-positive
rate, because forcing the true DIF items equal biases the focal mean and distorts
every item's test. Fix = two-stage anchor purification: stage 1 ranks items by a
naive DIF screen, the lowest-DIF items become a clean anchor, stage 2 frees all other
non-anchor items in both nested models so only the anchor identifies the metric.
Result: full power, ~0% FPR with DIF present, size ~0.03 under pure impact. Lesson
logged: IRT DIF is not valid without anchor handling; validate size, not just power.

## 2026-10-12: Harness and effort DIF on measurement-db

**Question.** When one model runs under a different agent harness or reasoning
effort, is the change pure impact (an ability shift) or DIF (it changes which items
the model solves)? Data: aims-foundations/measurement-db, whose subjects table
records harness, reasoning_effort and harness_version per subject.

**Paired design, not person-level DIF.** In every candidate benchmark, 1-9 models
share a contrast. Person-level DIF (persons = models, group = harness) needs far
more. The unit with power is one model under two conditions on shared items, with
repeated trials. `paired_dif` conditions on each (model, item) stratum's total
successes, so the model-specific item quirks that dominate cross-model variance
(FINDINGS.md section 1) cancel exactly. What remains is a common log odds shift
(impact) plus item deviations (DIF). Noncentral hypergeometric likelihood, exact
per-item p-values, bootstrap omnibus. Oracle: statsmodels ConditionalLogit (agrees
to 2e-16 once statsmodels is run with Newton; its default BFGS stops 5e-4 short).

**Two nulls.** If items differ in discrimination, a pure ability shift moves steep
items more, and the Rasch null reads that as DIF (8/20 rejections with log-slope
sd 0.4 and a 1.5-logit shift; 1/20 under the 2PL null with the true slopes). Both
are reported; the 2PL null needs >= 30 subjects answering every item to calibrate
slopes.

**Controls.** Lineage-preserving permutation = flip condition labels within
model, which keeps every model's pairing intact. Negative control = odd vs even
trials of the same subject and condition (5/100 rejections at .05 in simulation).

**Test-level consequence.** Fitted impact makes item scores sum to zero, so
"does DIF cancel in the score" needs an anchor-defined impact, the same
identification irt_lr uses. Holding each item's total fixed, the score gap
splits exactly into an impact part and a DIF part. The theory check predicts each
model's DIF part from item DIF fitted on held-out data (other models, or the other
half of the trials) and compares it with the observed part.

**Known bias, accepted.** Impact and one-sided DIF cannot be fully separated
without an outside anchor. The purified anchor (lowest-|z| half, iterated to a
stable set) settles slightly above the true impact, so both DIF parts understate
one-sided DIF's score effect: 0.88 observed and 0.71 predicted, relative to the
truth (validation/validate_dtf.py). Ranking anchors by effect size instead of |z|
barely moved this (anchor impact 0.91 vs 0.93, true 0.84 after small-sample bias).

**Data access.** The dataset is gated. Accepting the gate means filling a form as
the account owner, which is held for Abhay. No gated file has been read. The
eligible-cell count, power and pipeline come from the public catalog (GitHub
revision a06b94d) and synthetic tables in the same schema.
