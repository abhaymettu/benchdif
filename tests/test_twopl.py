"""2PL estimator correctness via parameter recovery."""
import numpy as np

from benchdif.irt import fit_2pl


def test_parameter_recovery():
    rng = np.random.default_rng(1)
    n, J = 3000, 15
    a_true = rng.uniform(0.7, 1.8, J)
    b_true = rng.normal(0, 1.0, J)
    theta = rng.normal(0, 1, n)
    P = 1 / (1 + np.exp(-a_true[None, :] * (theta[:, None] - b_true[None, :])))
    X = (rng.random((n, J)) < P).astype(int)
    fit = fit_2pl(X)
    assert np.corrcoef(fit.a, a_true)[0, 1] > 0.95
    assert np.corrcoef(fit.b, b_true)[0, 1] > 0.97
    assert np.sqrt(np.mean((fit.b - b_true) ** 2)) < 0.15
    # b = -d/a identity holds
    assert np.allclose(fit.b, -fit.d / fit.a)


def test_rejects_missing():
    import pytest
    with pytest.raises(ValueError):
        fit_2pl(np.array([[1.0, np.nan], [0.0, 1.0]]))
