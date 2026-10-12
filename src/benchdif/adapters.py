"""Reshape benchmark results into the (persons x items) response matrix DIF needs.

The reframing that makes this a benchmarking tool rather than a re-implementation
of difR:

    person   = a model (checkpoint / leaderboard submission)
    item     = a benchmark question
    response = 1 if that model answered that question correctly, else 0
    group    = a model attribute -- family (Llama/Qwen/...), open-vs-closed,
               size bucket, base-vs-instruct, or the language the item was shown in

With hundreds of models on a leaderboard this is well-powered. DIF here flags
items a group answers correctly *beyond its overall ability* -- a signal of
training-distribution artifacts or benchmark contamination for that family.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def from_long(df: pd.DataFrame, *, person: str, item: str, score: str,
              group: str, threshold: float = 0.5):
    """Pivot a long results table into (X, group_codes, persons, items).

    Parameters
    ----------
    df : long table with one row per (person, item).
    person, item, score, group : column names. `score` is coerced to 0/1 by
        `>= threshold`; already-binary scores are unaffected. `group` must have
        exactly two distinct values; the first in sorted order is the reference (0).
    threshold : cutoff turning a numeric score into correct/incorrect.

    Returns
    -------
    X : ndarray (n_persons x n_items) of 0/1
    group_codes : ndarray (n_persons,) of 0/1  (0 = reference)
    persons : Index of person labels (row order of X)
    items : Index of item labels (column order of X)

    Raises on duplicate (person, item) rows, on any missing cell (MH needs a
    complete matrix), or on a person whose group label is not unique.
    """
    dup = df.duplicated(subset=[person, item]).sum()
    if dup:
        raise ValueError(f"{dup} duplicate (person, item) rows; aggregate first")

    wide = df.pivot(index=person, columns=item, values=score)
    if wide.isna().any().any():
        n = int(wide.isna().sum().sum())
        raise ValueError(f"{n} missing (person, item) cells; MH needs a complete "
                         "matrix (subset to shared items or impute upstream)")
    X = (wide.to_numpy() >= threshold).astype(int)

    gmap = df[[person, group]].drop_duplicates()
    if gmap[person].duplicated().any():
        raise ValueError("a person maps to more than one group label")
    gmap = gmap.set_index(person).loc[wide.index, group]
    levels = sorted(gmap.unique())
    if len(levels) != 2:
        raise ValueError(f"group must have exactly 2 levels, got {levels}")
    codes = (gmap.to_numpy() == levels[1]).astype(int)  # ref=levels[0]=0
    return X, codes, wide.index, wide.columns


def from_matrix(wide: pd.DataFrame, group: pd.Series, threshold: float = 0.5):
    """Same as from_long but from an already-wide (persons x items) frame.

    `group` is a Series indexed by person label. Reference = sorted-first level.
    """
    g = group.reindex(wide.index)
    if g.isna().any():
        raise ValueError("group missing for some persons in the matrix")
    if wide.isna().any().any():
        raise ValueError("missing cells; MH needs a complete matrix")
    X = (wide.to_numpy() >= threshold).astype(int)
    levels = sorted(g.unique())
    if len(levels) != 2:
        raise ValueError(f"group must have exactly 2 levels, got {levels}")
    codes = (g.to_numpy() == levels[1]).astype(int)
    return X, codes, wide.index, wide.columns


def from_lm_eval(sample_files: dict, metric: str = "acc", threshold: float = 0.5):
    """Build a long results table from lm-evaluation-harness sample logs.

    lm-eval-harness with `--log_samples` writes one JSONL per (model, task), each
    line a scored document. This reads several models' logs into the long shape
    `from_long` expects: one row per (model, doc_id) with the metric as `correct`.

    Parameters
    ----------
    sample_files : dict[str, str]
        Maps a model label -> path to that model's JSONL samples file (a single
        task). All files must cover the same doc_id set for a complete matrix.
    metric : str
        Which per-sample metric key to read (e.g. 'acc', 'exact_match',
        'acc_norm'). Values are coerced 0/1 by `>= threshold` downstream.
    threshold : float
        Passed through to callers; kept here for documentation symmetry.

    Returns
    -------
    DataFrame with columns [model, doc_id, correct] ready for `from_long`
    (person='model', item='doc_id', score='correct'). Add a group column by
    joining on `model`.

    Notes
    -----
    Reads only stdlib json; no dependency on the lm-eval package. Each JSONL line
    is expected to contain a document id ('doc_id' or 'doc_hash') and the metric
    key at top level (lm-eval writes metrics per sample there).
    """
    import json

    rows = []
    for model, path in sample_files.items():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if "doc_id" in rec:
                    did = rec["doc_id"]
                elif "doc_hash" in rec:
                    did = rec["doc_hash"]
                else:
                    raise ValueError(f"{path}: line lacks doc_id/doc_hash")
                if metric not in rec:
                    raise ValueError(f"{path}: metric {metric!r} not in sample "
                                     f"(have {sorted(rec)[:8]}...)")
                val = rec[metric]
                val = float(val[0] if isinstance(val, (list, tuple)) else val)
                rows.append({"model": model, "doc_id": did, "correct": val})
    return pd.DataFrame(rows)


_FACTORS = ("harness", "reasoning_effort", "harness_version")
_EFFORT = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "xhigh": 4, "max": 5}


def condition_pairs(subjects: pd.DataFrame, factor: str,
                    model_key: str = "normalized_name") -> pd.DataFrame:
    """Pairs of subjects (same model) differing in `factor` and equal in the others.

    `factor` is one of harness / reasoning_effort / harness_version. Null factor
    values count as the string "default". Each pair is ordered so cond_0 < cond_1:
    reasoning effort by level (minimal < low < ... < max), anything else as
    strings, ties by subject_id. Returns [model, subject_0, subject_1, cond_0, cond_1].
    """
    if factor not in _FACTORS:
        raise ValueError(f"factor must be one of {_FACTORS}, got {factor!r}")
    others = [f for f in _FACTORS if f != factor]
    s = subjects.dropna(subset=[model_key])
    s = s.assign(**{f: s[f].fillna("default").astype(str) for f in _FACTORS})
    rows = []
    for model, g in s.groupby(model_key, sort=True):
        recs = g[["subject_id", factor] + others].to_dict("records")
        for i, a in enumerate(recs):
            for b in recs[i + 1:]:
                if a[factor] == b[factor] or any(a[o] != b[o] for o in others):
                    continue
                a2, b2 = sorted((a, b), key=lambda r: (
                    _EFFORT.get(r[factor].lower(), 99) if factor == "reasoning_effort" else 0,
                    r[factor], r["subject_id"]))
                rows.append((model, a2["subject_id"], b2["subject_id"],
                             a2[factor], b2[factor]))
    return pd.DataFrame(rows, columns=["model", "subject_0", "subject_1",
                                       "cond_0", "cond_1"])


def paired_counts(responses: pd.DataFrame, pairs: pd.DataFrame,
                  min_items: int = 30) -> pd.DataFrame:
    """Per-item success counts for each pair, on shared binary items.

    `pairs` is the output of `condition_pairs`. NaN responses are dropped; an item
    is kept only if every observed response is 0/1 for both subjects and both
    observed it. Cells (id "model|cond_0|cond_1") with fewer than `min_items`
    such items are dropped. Returns [model, cell, item_id, k0, n0, k1, n1].
    """
    cols = ["model", "cell", "item_id", "k0", "n0", "k1", "n1"]
    r = responses.dropna(subset=["response"])
    r = r.assign(nb=~r["response"].isin([0.0, 1.0]))
    agg = (r.groupby(["subject_id", "item_id"])
             .agg(k=("response", "sum"), n=("response", "size"), nb=("nb", "sum"))
             .reset_index())
    by_subject = {s: g for s, g in agg.groupby("subject_id")}
    empty = agg.iloc[:0]
    out = []
    for p in pairs.itertuples(index=False):
        a = by_subject.get(p.subject_0, empty)
        b = by_subject.get(p.subject_1, empty)
        m = a.merge(b, on="item_id", suffixes=("0", "1"))
        m = m[(m["nb0"] == 0) & (m["nb1"] == 0)]
        if len(m) < min_items:
            continue
        m = m.assign(model=p.model, cell=f"{p.model}|{p.cond_0}|{p.cond_1}")
        out.append(m[cols].astype(
            {"k0": int, "n0": int, "k1": int, "n1": int}))
    if not out:
        return pd.DataFrame(columns=cols)
    return pd.concat(out, ignore_index=True)


def eligible_cells(responses: pd.DataFrame, subjects: pd.DataFrame, factor: str,
                   min_items: int = 30,
                   model_key: str = "normalized_name") -> pd.DataFrame:
    """One row per analysable (model, condition pair) cell with its basic stats.

    Returns [model, cell, cond_0, cond_1, subject_0, subject_1, n_items,
    trials_0, trials_1, mean_0, mean_1]; trials = median trials per item,
    mean = success rate, both over the shared binary items.
    """
    cols = ["model", "cell", "cond_0", "cond_1", "subject_0", "subject_1",
            "n_items", "trials_0", "trials_1", "mean_0", "mean_1"]
    pairs = condition_pairs(subjects, factor, model_key)
    counts = paired_counts(responses, pairs, min_items)
    if counts.empty:
        return pd.DataFrame(columns=cols)
    pairs = pairs.assign(cell=pairs["model"] + "|" + pairs["cond_0"] + "|" + pairs["cond_1"])
    g = counts.groupby("cell", sort=False)
    st = pd.DataFrame({
        "n_items": g.size(),
        "trials_0": g["n0"].median(), "trials_1": g["n1"].median(),
        "mean_0": g["k0"].sum() / g["n0"].sum(),
        "mean_1": g["k1"].sum() / g["n1"].sum(),
    }).reset_index()
    return st.merge(pairs, on="cell")[cols].reset_index(drop=True)


def runs_matrix(responses: pd.DataFrame, subject_ids, items=None):
    """Stack each (subject, trial) run as a row of a 0/1 matrix for person-level DIF.

    Columns are `items` (default: binary items observed in every run); runs not
    complete on those items are dropped. Returns (X, persons) with persons a
    DataFrame [subject_id, trial] in row order.
    """
    r = responses[responses["subject_id"].isin(list(subject_ids))]
    r = r.dropna(subset=["response"])
    if items is None:
        nonbin = r.loc[~r["response"].isin([0.0, 1.0]), "item_id"].unique()
        runs = r[["subject_id", "trial"]].drop_duplicates()
        seen = r.groupby("item_id").apply(
            lambda d: len(d[["subject_id", "trial"]].drop_duplicates()),
            include_groups=False)
        items = sorted(i for i in seen.index[seen == len(runs)] if i not in set(nonbin))
    items = list(items)
    wide = r[r["item_id"].isin(items)].pivot_table(
        index=["subject_id", "trial"], columns="item_id", values="response",
        aggfunc="first").reindex(columns=items)
    wide = wide.dropna()
    persons = wide.index.to_frame(index=False)
    return wide.to_numpy().astype(int), persons
