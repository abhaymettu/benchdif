"""Synthetic stand-in for one benchmark of aims-foundations/measurement-db.

Same table schema as the real parquet files (subjects / responses / items), so the
condition-pair pipeline can be built and tested before gated access arrives.
Run `python synthetic_mdb.py OUTDIR` to write OUTDIR/{synthbench,synthbench_onesided}/formatted_tables/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic(seed=0, n_items=120, trials=4, dif_size=1.0, one_sided=False,
                   name="synthbench"):
    """Return dict(subjects, responses, items, planted_dif_items).

    Planted DIF (agentB only) is +/-dif_size, or +dif_size on every planted item
    when one_sided, so it does not cancel in the score.
    """
    rng = np.random.default_rng(seed)
    items_ids = [f"item{i:03d}" for i in range(n_items)]
    b = rng.normal(0, 1.5, n_items)
    planted = rng.choice(n_items, size=n_items // 10, replace=False)
    dif = np.zeros(n_items)
    dif[planted] = dif_size * (1.0 if one_sided else rng.choice([-1.0, 1.0], size=len(planted)))
    frac = set(range(n_items - 5, n_items))  # fractional-graded items

    conds = []  # (model, effort, harness)
    for m in range(6):
        conds += [(f"m{m}", "low", "agentA"), (f"m{m}", "high", "agentA")]
    conds += [(f"m{m}", "high", "agentB") for m in range(3)]
    conds += [("m6", "low", "agentA")]
    theta = {f"m{m}": rng.normal() for m in range(7)}
    u = {f"m{m}": rng.normal(size=n_items) for m in range(7)}

    subj, resp = [], []
    for model, effort, harness in conds:
        sid = f"{model}__{effort}__{harness}"
        subj.append(dict(subject_id=sid, display_name=sid, normalized_name=model,
                         provider="synth", release_date="2026-01-01",
                         access_date="2026-10-01", harness=harness,
                         reasoning_effort=effort, harness_version=None,
                         subject_features_extra=None))
        shift = (0.8 if effort == "high" else 0.0) + (0.3 if harness == "agentB" else 0.0)
        d = dif if harness == "agentB" else np.zeros(n_items)
        p = 1 / (1 + np.exp(-(theta[model] + u[model] - b + shift + d)))
        for t in range(1, trials + 1):
            y = (rng.random(n_items) < p).astype(float)
            for i in frac:
                y[i] = 0.5
            y[rng.random(n_items) < 0.02] = np.nan
            resp.append(pd.DataFrame(dict(
                subject_id=sid, item_id=items_ids, response_id=f"{sid}-{t}-" + pd.Series(items_ids),
                benchmark_id=name, trial=t, test_condition=None,
                interactors=None, response=y)))
    items = pd.DataFrame(dict(item_id=items_ids, benchmark_id=name,
                              raw_item_id=items_ids, content=None))
    return dict(subjects=pd.DataFrame(subj),
                responses=pd.concat(resp, ignore_index=True), items=items,
                planted_dif_items=[items_ids[i] for i in sorted(planted)])


if __name__ == "__main__":
    for bench, kw in (("synthbench", {}),
                      ("synthbench_onesided", dict(seed=1, dif_size=1.5, one_sided=True))):
        out = Path(sys.argv[1]) / bench / "formatted_tables"
        out.mkdir(parents=True, exist_ok=True)
        d = make_synthetic(name=bench, **kw)
        for table in ("subjects", "responses", "items"):
            d[table].to_parquet(out / f"{table}.parquet", index=False)
