"""Multiplicity correction + data-adequacy checks."""
import warnings

import numpy as np
import pytest

from benchdif import adjust, check_adequacy


def test_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(0)
    p = np.clip(rng.beta(0.4, 4, 200), 1e-6, 1)
    for mine, theirs in [("bh", "fdr_bh"), ("holm", "holm")]:
        exp = sm.multipletests(p, method=theirs)[1]
        assert np.allclose(adjust(p, mine), exp, atol=1e-10)


def test_preserves_nan_and_order():
    p = np.array([0.01, np.nan, 0.5, 0.001])
    out = adjust(p, "bh")
    assert np.isnan(out[1])
    assert out[3] <= out[0] <= out[2]      # ordering preserved


def test_bad_method():
    with pytest.raises(ValueError):
        adjust([0.1, 0.2], "bogus")


def test_adequacy_flags_thin_strata_and_degenerate_items():
    rng = np.random.default_rng(1)
    # 40 persons, 300 items -> ~1 person per score stratum (the real-data failure)
    X = (rng.random((40, 300)) < 0.5).astype(int)
    X[:, 0] = 1          # degenerate: all correct
    X[:, 1] = 0          # degenerate: all wrong
    g = (np.arange(40) % 2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        info = check_adequacy(X, g)
    assert info["degenerate_items"] == 2
    assert info["persons_per_stratum"] < 3
    msgs = " ".join(str(x.message) for x in w)
    assert "per score stratum" in msgs and "zero variance" in msgs


def test_adequacy_quiet_on_healthy_data():
    rng = np.random.default_rng(2)
    X = (rng.random((2000, 20)) < 0.5).astype(int)
    g = (np.arange(2000) % 2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        check_adequacy(X, g)
    assert not w
