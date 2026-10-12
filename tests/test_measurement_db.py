"""Tests for the measurement-db condition-pair adapters."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchdif import condition_pairs, paired_counts, eligible_cells, runs_matrix

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from synthetic_mdb import make_synthetic  # noqa: E402


def _subjects():
    rows = [("a", "m0", "A", "low", None), ("b", "m0", "A", "high", None),
            ("c", "m0", "B", "high", None), ("d", "m1", "A", None, None),
            ("e", "m1", "A", "high", None), ("f", None, "A", "low", None)]
    return pd.DataFrame(rows, columns=["subject_id", "normalized_name", "harness",
                                       "reasoning_effort", "harness_version"])


def test_condition_pairs_holds_other_factors_fixed():
    s = _subjects()
    p = condition_pairs(s, "reasoning_effort")
    pairs = set(zip(p.subject_0, p.subject_1))
    assert {frozenset(x) for x in pairs} == {frozenset("ab"), frozenset("de")}
    assert all("c" not in x for x in pairs)          # agentB/high vs agentA/low
    h = condition_pairs(s, "harness")
    assert list(zip(h.subject_0, h.subject_1)) == [("b", "c")]


def test_condition_pairs_null_is_default_and_input_unchanged():
    s = _subjects()
    before = s.copy()
    p = condition_pairs(s, "reasoning_effort")
    row = p[p.model == "m1"].iloc[0]
    assert (row.cond_0, row.cond_1) == ("high", "default")   # known levels first
    pd.testing.assert_frame_equal(s, before)
    assert "f" not in set(p.subject_0) | set(p.subject_1)   # null model dropped
    with pytest.raises(ValueError):
        condition_pairs(s, "provider")


def _tiny():
    pairs = pd.DataFrame([("m", "s0", "s1", "high", "low")],
                         columns=["model", "subject_0", "subject_1", "cond_0", "cond_1"])
    rows = []
    for item, r0, r1 in [("i1", [1, 1, 0], [0]), ("i2", [1], [1, 1]),
                         ("i3", [0.5], [1]),            # fractional: excluded
                         ("i4", [np.nan], [1]),         # NaN-only for s0: excluded
                         ("i5", [1], [])]:              # unobserved by s1: excluded
        for t, v in enumerate(r0, 1):
            rows.append(("s0", item, t, v))
        for t, v in enumerate(r1, 1):
            rows.append(("s1", item, t, v))
    return pd.DataFrame(rows, columns=["subject_id", "item_id", "trial", "response"]), pairs


def test_paired_counts_values_and_filters():
    r, pairs = _tiny()
    out = paired_counts(r, pairs, min_items=1)
    assert out.item_id.tolist() == ["i1", "i2"]
    assert out.cell.iloc[0] == "m|high|low"
    assert out[["k0", "n0", "k1", "n1"]].values.tolist() == [[2, 3, 0, 1], [1, 1, 2, 2]]
    assert out.k0.dtype.kind == "i"
    assert paired_counts(r, pairs, min_items=3).empty
    assert list(paired_counts(r, pairs, min_items=3).columns) == \
        ["model", "cell", "item_id", "k0", "n0", "k1", "n1"]


def test_eligible_cells_on_synthetic():
    d = make_synthetic()
    e = eligible_cells(d["responses"], d["subjects"], "reasoning_effort")
    h = eligible_cells(d["responses"], d["subjects"], "harness")
    assert len(e) == 6 and len(h) == 3
    assert set(h.model) == {"m0", "m1", "m2"}
    assert (e.n_items <= 115).all() and (e.n_items > 90).all()
    assert (e.trials_0 <= 4).all()
    assert (e.cond_0 == "low").all() and (e.cond_1 == "high").all()   # effort order
    assert (e.mean_0 < e.mean_1).all()    # high effort solves more
    assert len(d["planted_dif_items"]) == 12


def test_runs_matrix_shape_and_completeness():
    d = make_synthetic()
    ids = ["m0__low__agentA", "m0__high__agentA"]
    X, persons = runs_matrix(d["responses"], ids)
    assert X.shape[0] == len(persons) <= 8 and X.shape[1] <= 115
    assert set(np.unique(X)) <= {0, 1}
    assert list(persons.columns) == ["subject_id", "trial"]
    items = [f"item{i:03d}" for i in range(10)]
    X2, p2 = runs_matrix(d["responses"], ids, items=items)
    assert X2.shape[1] == 10 and X2.shape[0] <= 8
    r = d["responses"]
    full = r[r.subject_id.isin(ids) & r.item_id.isin(items)].dropna(subset=["response"])
    n_complete = (full.groupby(["subject_id", "trial"]).size() == 10).sum()
    assert X2.shape[0] == n_complete   # only complete runs are kept
