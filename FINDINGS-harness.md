# Findings: does a harness or effort setting change which items a model solves?

Question: when the same model runs under a different agent harness or reasoning-effort
setting, does the change only shift overall ability (impact), or does it change *which*
items the model solves (differential item functioning, DIF)?

Data: [aims-foundations/measurement-db](https://huggingface.co/datasets/aims-foundations/measurement-db)
(Stanford AIMS, CC-BY-SA-4.0). Its `subjects` table records harness, reasoning effort
and harness version per subject, so one model under two settings is two subjects.
**The response tables are gated, and this run has not read them.** Accepting the gate means
submitting a form as the account owner, and that decision is with the owner. Everything below
comes from the public catalog (GitHub revision `a06b94d`), simulation at the catalog's
design sizes, and a schema-identical synthetic dry run. The real-data answer is one command
away (section 6) and is not claimed here.

## 1. Eligible cells: about 68 candidates in 8 benchmarks (upper bound)

A cell is one model under two conditions that differ in exactly one factor (harness, or
effort), with at least 30 shared binary items. Counted from the public subject names,
because the gated `subjects` table holds the authoritative fields:

| benchmark | contrast | models sharing it | cells | trials | notes |
|---|---|---:|---:|---:|---|
| swebench_live (lite) | Agentless / OpenHands / SWE-agent | 4 | 12 | 1 | Claude 3.7 Sonnet, DeepSeek V3, GPT-4.1, GPT-4o |
| swebench_live (multilang) | Claude Code / OpenHands / SWE-agent | 5 | 9 | 1 | |
| mtbbench | with vs without tools | 9 | 9 | 1 | 573 items |
| matharena_platform | default / Kimi Code / OpenCode / Qwen Code; direct vs Codex | 2 | 7 | repeated | Gemini 3.8 Flash, GPT-5.6-Sol |
| matharena | direct vs agent | 3 | 3 | repeated | GPT-5, Gemini 2.5 Pro, DeepSeek-v3.2-Speciale |
| weavebench | Claude Code vs Codex | 1 | 1 | 1 | 114 items |
| matharena | effort low / medium / high / xhigh | 6 | 14 | repeated | GPT-5.2, GPT-5-nano, Grok 3 Mini, o3-mini, o4-mini, Gemini 3.1 Pro |
| matharena_platform | effort | 4 | 8 | repeated | GPT-5.2, Gemini 3.1 Pro, Claude-Fable-5.1, Claude-Opus-4.7 |
| engdesign | effort / thinking | 3 | 3 | 1 | 101 items |
| helm_bbq | thinking-10k vs off | 2 | 2 | 1 | |

Total: roughly 41 harness cells and 27 effort cells. Excluded: benchmarks whose responses
are not binary (livebench, programbench, the continuous HELM safety sets, rakuda, tengu),
and name variants that are not harness or effort (image-detail `_HIGH`, best-of-N,
context-length variants, fine-tunes). Two things can shrink the count on real data:
shared-item coverage below 30, and `normalized_name` not merging a model's variants.

**What this means for design.** No contrast has more than 9 models. benchdif's person-level
methods (persons = models, group = harness) cannot work at 1-9 persons per group.

## 2. The design that has power: one model, two conditions, conditional on the item

`paired_dif` treats each (model, item) pair as a stratum: k0 of n0 trials succeed under
condition 0, k1 of n1 under condition 1, with success logit
`eta_mi + c * (a_i * Delta_m + delta_i)`. Conditioning on the stratum's total successes
removes `eta_mi` exactly. That is everything the two conditions share for that model and
item, including the model-specific item quirks that made up 85-90% of cross-model variance
in FINDINGS.md section 1. What is left is a common log odds shift `Delta_m` (impact) and
item deviations `delta_i` (DIF). This needs no ability model and no anchor items to test.

| check | result |
|---|---|
| impact vs statsmodels ConditionalLogit (Newton) | max abs diff 2.2e-16 |
| size, pure impact plus model-specific item effects | omnibus 3/40, item flags 0/5750 |
| power, 6 models, 10% items at +/-1.5 logits | omnibus 18/20 |
| slopes vary (log-sd 0.4), no DIF: Rasch null vs 2PL null | 8/20 vs 1/20 rejections |
| negative control, odd vs even trials of one subject | 5/100 rejections at .05 |

The slope row matters for interpretation. If steep items move more under a pure ability
shift, the Rasch null calls that DIF. Real-data claims therefore need the 2PL null
alongside.

## 3. Power at the real design sizes

Simulated at each design's shape: model-specific item effects (sd 1 logit), impact +0.8
logit, DIF on 10% of items at +/-d logits (balanced). 100 replicates per cell, 200 bootstrap
draws each (`validation/simulate_harness_dif.py`, output in `validation/results/`).

| design | size | size, slopes vary (Rasch null) | power d=1 | d=1.5 | d=2 | d=3 | detectable d (80%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| matharena effort (6 models x 4 trials, 120 items) | 0.04 | 0.15 | 0.28 | 0.68 | 0.96 | 1.00 | **2.0** |
| mtbbench tools (9 models x 1 trial, 573 items) | 0.05 | 0.06 | 0.19 | 0.50 | 0.88 | 1.00 | **2.0** |
| swebench_live harness (4 models x 1 trial, 300 items) | 0.05 | 0.03 | 0.12 | 0.16 | 0.30 | 0.58 | >3 |
| matharena_platform harness (1 model x 4 trials, 100 items) | 0.04 | 0.05 | 0.07 | 0.08 | 0.10 | 0.31 | >3 |
| weavebench harness (1 model x 1 trial, 114 items) | 0.01 | 0.01 | 0.03 | 0.03 | 0.03 | 0.00 | none |

- **Size holds everywhere** under the Rasch truth. When slopes vary and the test still
  assumes Rasch, matharena's size rises to 0.15. That is the slope effect from section 2,
  and the reason the 2PL null is reported next to it.
- **Only two designs can see DIF of plausible size**: matharena effort and mtbbench tools,
  both at about 2 logits on 10% of items. For scale, the Llama-2 vs Mistral DIF in FINDINGS.md
  section 1 was 6-10x sampling variance. The four other harness designs cannot detect even
  3-logit DIF. A single model with a single trial cannot detect it at all, because each item
  gives one discordant pair and item-level dispersion is not identifiable. A null result from
  those designs says nothing.
- **Per-item flags do not survive** BH at these sizes: recall 0.09 at d=2 in the best design,
  0 elsewhere. Any real-data answer rests on the omnibus test and the variance ratio.

Person-level methods on runs (person = model x trial, group = condition; matharena effort
shape, 24 runs per group, 20 replicates):

| truth | logistic share flagged (BH) | invariance ladder rejects, nominal | ladder rejects, lineage permutation |
|---|---:|---:|---:|
| no DIF | 0.005 | **13/20** | 2/20 |
| DIF d=2.0 | 0.015 | 20/20 | 4/20 |

Run nominally, the invariance ladder rejects 65% of the time with no DIF present, because
runs of one model are not exchangeable persons. The lineage-preserving permutation brings
size back (2/20) but leaves power at 4/20, where the paired test has 0.96. The ladder is
therefore reported on real data only with its permutation p-value, and the paired test
carries the verdict.

## 4. Test-level consequence and the theory check

The fitted impact makes the item scores sum to zero, so whether DIF cancels in the score
has to be asked against an anchor-defined impact. Per item, the score gap is
`k1/n1 - k0/n0 = k1 (1/n1 + 1/n0) - s/n0`, with `s` the item's total. Holding `s` fixed,
the gap splits exactly into an impact part and a DIF part. `paired_dtf` estimates item DIF on
held-out data (the other models, or the other half of the trials), then predicts each
model's DIF part as the sum over items of the gap between the response law with and
without that item's DIF, at the model's own matched level. This is Raju's DTF in
conditional form. It is compared with the observed DIF part:

| planted DIF (6 models, 200 items, 8 trials) | true DIF part | observed | predicted (held-out) |
|---|---:|---:|---:|
| one-sided +2.0 on 20% of items | +0.045 | +0.040 (0.88x) | +0.032 (0.71x) |
| balanced +/-2.0 on 20% of items | -0.007 | -0.007 | -0.008 |

When DIF is balanced, the score effect cancels, as in FINDINGS.md section 2, and the
prediction says so in advance. When DIF is one-sided, the prediction recovers most of the
effect, but it errs low. Impact and one-sided DIF cannot be fully separated without
an outside anchor. The purified anchor settles slightly above the true impact, which
makes both estimates conservative (DECISIONS.md, 2026-10-12).

Reordering: the script compares each model's order under the new condition with the order
its impact-only score implies, and with the order the DTF prediction implies.

## 5. Synthetic dry run

`examples/synthetic_mdb.py` writes two benchmarks in the measurement-db schema. Each has 6
models under low and high effort with no DIF, and 3 of them under a second harness with
DIF on 10% of items at +/-1 logit (one benchmark balanced, one one-sided +1.5).
`examples/harness_dif.py` runs end to end on them. The effort contrasts show no DIF
(omnibus p 0.30 and 0.79; lineage permutation p 0.41 and 0.78; 0 items flagged). The
harness DIF is not detected (p 0.14 and 0.17), which section 3 predicts at 3 models
and 4 trials. On no-DIF data the person-level invariance ladder on runs rejected metric
invariance at a nominal p = 0.009. That is the dependence problem section 3 measures.

## 6. Reproduce, and what is pending

```bash
uv run python validation/validate_paired.py
uv run python validation/validate_dtf.py
uv run python validation/simulate_harness_dif.py 100   # parallel; took 85 min on 12 shared cores
uv run python examples/synthetic_mdb.py /tmp/synth && uv run python examples/harness_dif.py /tmp/synth --top 4
# with access: download <benchmark>/formatted_tables/{subjects,responses}.parquet for the
# 8 benchmarks in section 1 into ROOT, then
uv run python examples/harness_dif.py ROOT
```

## 7. What this does NOT show

- **No real-data finding yet.** No claim is made here about whether harnesses change which
  items models solve. The count in section 1 is an upper bound from names.
- **Power is limited by design, not by method.** Single-trial contrasts with few models
  (weavebench, matharena_platform's single-model harness pairs) can only see large DIF.
  A null there means "no DIF above the detectable size in section 3", not "no DIF".
- **Per-item flags are weak.** At 4 trials per condition an item needs a very large
  shift to survive BH. The omnibus and the variance ratio carry the answer, as in
  FINDINGS.md section 5.
- **DIF is not a mechanism.** A harness that makes tool-heavy items easier and one that
  changes answer extraction produce the same signature.
