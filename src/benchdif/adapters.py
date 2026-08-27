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
