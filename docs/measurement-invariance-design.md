# Design note: measurement-invariance omnibus (not yet implemented)

Item-level DIF (`mh`, `logistic`, `irt_lr`) localizes *which* items misbehave. A
measurement-invariance (MI) omnibus asks the complementary global question: at what
level does the whole instrument stay comparable across groups? This note scopes it;
nothing here is implemented until it is validated the same way IRT-LR was.

## The hierarchy (binary items, multi-group 2PL)

For two groups sharing one latent ability metric, fit a nested sequence and compare
by likelihood-ratio chi-square:

| Level        | Constraint across groups                | Interpretation                     |
|--------------|-----------------------------------------|------------------------------------|
| Configural   | same items, all a and d free per group  | same construct, no scale sharing   |
| Metric (weak)| equal discriminations a, intercepts d free | equal item-construct relations   |
| Scalar (strong)| equal a AND d                         | scores directly comparable         |

Omnibus tests:
- Metric holds if 2(ll_configural - ll_metric) is n.s. on df = (#items - #anchors).
- Scalar holds if 2(ll_metric - ll_scalar) is n.s. on the added df.

Partial invariance: free the worst items (guided by `irt_lr`) and retest, reporting
the largest invariant subset.

## What must be built

1. Extend `fit_multigroup_2pl` to share *only the slope* a (metric level); today it
   shares both a and d or neither. Add a per-item constraint mode {free, equal_a,
   equal_both}.
2. Identification: reference ability N(0,1); focal N(mu, sigma) free, as now.
3. An `invariance(responses, group)` entry returning the LR ladder (configural →
   metric → scalar), each with stat, df, p, and a verdict, plus effect sizes
   (e.g. ΔCFI analogue / M2-based fit if added later).

## Validation plan (required before shipping)

- Simulate strictly metric-invariant data (equal a, unequal d via impact only) and
  confirm the metric test does NOT reject while scalar may; simulate scalar-invariant
  data and confirm neither rejects; simulate loading DIF and confirm metric rejects.
- Check size under the null across seeds (~0.05) and power under injected violations.
- Only then export and document. Same bar as IRT-LR: validate size, not just power.

## Why it matters for AI evals

"Is this benchmark measuring the same construct across model families / languages?"
is a scale-level claim. MI answers it globally; DIF then localizes the offending
items. Together they turn "the leaderboard is unfair" from a vibe into a test.
