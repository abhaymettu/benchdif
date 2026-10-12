# Spec: harness and effort DIF on measurement-db

Status: draft 2026-10-12. Lightweight spec for a feature in an existing codebase.

## Question

When the same model runs under a different agent harness or reasoning-effort
setting, does the change only shift overall ability (impact), or does it change
which items the model solves (DIF)?

## Data

`aims-foundations/measurement-db` (Hugging Face, CC-BY-SA-4.0, gated). Tables
`subjects` (harness, reasoning_effort, harness_version, normalized_name),
`responses` (subject_id, item_id, trial, response), `items`, `benchmarks`.
Access is pending; until it lands everything runs on synthetic tables in the
same schema.

## Design decision: the paired within-model design

Person-level DIF (persons = models, group = harness) has 1-6 models per
harness on every candidate benchmark, far below what benchdif's person-level
methods need. The unit with power is one model under two conditions on shared
items, with repeated trials (or several models sharing the contrast).

For model m, item i, condition c in {0, 1}, the success logit is
`eta_mi + c * (a_i * Delta_m + delta_i)`. `eta_mi` absorbs everything about the
model-item pair that both conditions share, including the item-specific quirks
that dominate cross-model variance (FINDINGS.md section 1). `Delta_m` is impact.
`delta_i` is DIF. Conditioning on each (m, i) stratum's total successes removes
`eta_mi` exactly (Fisher noncentral hypergeometric), so the test needs no
ability model and no anchor items. `a_i = 1` is the Rasch null; fitted 2PL
slopes give the 2PL null, under which a pure ability shift moves
high-discrimination items more.

- Omnibus: score statistic for `delta = 0`, calibrated by a conditional
  parametric bootstrap at the fitted `Delta`.
- Per item: one-step `delta_i`, Monte Carlo p from the same bootstrap, BH-FDR,
  ETS-style class from D = -2.35 * delta.
- Variance ratio: observed item-gap dispersion over its null expectation
  (FINDINGS.md section 1 statistic).

benchdif's existing logistic, IRT-LR and invariance ladder also run on a
runs-as-persons layout (person = model x condition x trial). The simulation
decides whether their size holds there.

## Requirements

1. Eligible-cell count first: (benchmark, model, condition pair) with both
   conditions on at least 30 shared binary items. Report the count before any
   analysis; stop if none qualify.
2. Simulation at the real design sizes: Type I error and power for every
   method used, including model-specific item effects and varying slopes.
3. Per eligible benchmark: share of items flagged (BH q < .05), direction
   balance, effect class, invariance-ladder verdict, a lineage-preserving
   label permutation (condition labels flipped within model) and a negative
   control (trial split within one subject-condition).
4. Test-level consequence: net score change attributable to DIF, and subject
   reordering.
5. Theory check: predict each model's DIF-driven score change and rank shift
   (Raju DTF, the sum of item-level response-function gaps at matched
   ability) from item DIF estimated on the other models, before reading the
   observed values.

## Done when

The simulation table, the adapter tested on the synthetic schema, and the
analysis script are committed. Either the real-data results with both
controls and the quantified consequence are in FINDINGS-harness.md, or the
write-up states that data access is the blocker.
