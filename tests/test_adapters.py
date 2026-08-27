"""Tests for the reshaping adapters."""
import json

import numpy as np
import pandas as pd
import pytest

from benchdif import from_long, from_matrix, from_lm_eval, mantel_haenszel


def _long():
    rows = []
    for m in range(20):
        fam = "A" if m % 2 == 0 else "B"
        for q in range(5):
            rows.append((f"m{m}", f"q{q}", (m + q) % 2, fam))
    return pd.DataFrame(rows, columns=["model", "q", "correct", "family"])


def test_from_long_shapes_and_group_coding():
    df = _long()
    X, g, persons, items = from_long(df, person="model", item="q",
                                     score="correct", group="family")
    assert X.shape == (20, 5)
    assert set(np.unique(X)) <= {0, 1}
    assert g.sum() == 10 and g[0] == 0        # family A (sorted first) = reference
    # feeds MH without error
    assert len(mantel_haenszel(X, g)) == 5


def test_from_long_rejects_duplicates_and_missing():
    df = _long()
    with pytest.raises(ValueError):
        from_long(pd.concat([df, df.iloc[:1]]), person="model", item="q",
                  score="correct", group="family")
    with pytest.raises(ValueError):
        from_long(df.iloc[:-1], person="model", item="q",
                  score="correct", group="family")  # a missing (person,item) cell


def test_from_matrix_roundtrip():
    wide = pd.DataFrame(np.eye(4, dtype=int), index=[f"m{i}" for i in range(4)])
    grp = pd.Series(["x", "x", "y", "y"], index=wide.index)
    X, g, persons, items = from_matrix(wide, grp)
    assert X.shape == (4, 4)
    assert list(g) == [0, 0, 1, 1]


def test_from_lm_eval_parses_samples(tmp_path):
    # write two synthetic lm-eval --log_samples JSONL files
    def write(name, offset):
        p = tmp_path / name
        with open(p, "w") as f:
            for did in range(6):
                f.write(json.dumps({"doc_id": did, "target": 0,
                                    "acc": float((did + offset) % 2)}) + "\n")
        return str(p)
    files = {"llama": write("llama.jsonl", 0), "qwen": write("qwen.jsonl", 1)}
    long = from_lm_eval(files, metric="acc")
    assert set(long.columns) == {"model", "doc_id", "correct"}
    assert len(long) == 12
    long["family"] = long["model"]
    X, g, persons, items = from_long(long, person="model", item="doc_id",
                                     score="correct", group="family")
    assert X.shape == (2, 6)


def test_from_lm_eval_missing_metric(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({"doc_id": 0, "exact_match": 1.0}) + "\n")
    with pytest.raises(ValueError):
        from_lm_eval({"m": str(p)}, metric="acc")
