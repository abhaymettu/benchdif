"""IRT-LR DIF: flags injected DIF, controls false positives, separates impact."""
import numpy as np
import pytest

from benchdif.dif import irt_lr


def _sim(seed, n=1200, J=10, dif=(2, 6), impact=-0.5):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.8, 1.7, J); b = rng.normal(0, 1, J)
    a_f, b_f = a.copy(), b.copy()
    for j in dif:
        b_f[j] = b[j] + 0.9; a_f[j] = a[j] + 0.4
    th0 = rng.normal(0, 1, n); th1 = rng.normal(impact, 1, n)
    X0 = (rng.random((n, J)) < 1 / (1 + np.exp(-a * (th0[:, None] - b)))).astype(int)
    X1 = (rng.random((n, J)) < 1 / (1 + np.exp(-a_f * (th1[:, None] - b_f)))).astype(int)
    return np.vstack([X0, X1]), np.array([0] * n + [1] * n)


def test_flags_dif_and_controls_fp():
    X, g = _sim(0)
    res = irt_lr(X, g)
    tested = ~res["is_anchor"].to_numpy()
    flagged = set(np.where(res["flag"].to_numpy())[0])
    # both DIF items detected (they won't be chosen as anchors)
    assert {2, 6} <= flagged
    # few false positives among clean tested items
    clean_flagged = [j for j in flagged if j not in (2, 6)]
    assert len(clean_flagged) <= 1
    # anchors reported and excluded from testing
    assert res["is_anchor"].sum() >= 2
    assert np.isnan(res.loc[res["is_anchor"], "stat"]).all()


def test_input_validation():
    with pytest.raises(ValueError):
        irt_lr(np.zeros((4, 3)), np.array([0, 1, 2, 0]))   # bad group codes
    with pytest.raises(ValueError):
        irt_lr(np.zeros((4, 3)), np.array([0, 1]))          # length mismatch
