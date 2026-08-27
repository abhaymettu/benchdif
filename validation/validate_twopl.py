"""Validate the 2PL EM by parameter recovery across seeds."""
import numpy as np
from benchdif.irt import fit_2pl

def one(seed, n=4000, J=20):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.6, 2.0, J); b = rng.normal(0, 1, J)
    th = rng.normal(0, 1, n)
    X = (rng.random((n, J)) < 1/(1+np.exp(-a*(th[:,None]-b)))).astype(int)
    f = fit_2pl(X)
    return (np.corrcoef(f.a,a)[0,1], np.corrcoef(f.b,b)[0,1],
            np.sqrt(np.mean((f.a-a)**2)), np.sqrt(np.mean((f.b-b)**2)))

print(f"{'seed':>4} {'corr_a':>8} {'corr_b':>8} {'rmse_a':>8} {'rmse_b':>8}")
worst_ca=worst_cb=1.0
for s in range(5):
    ca,cb,ra,rb=one(s)
    worst_ca=min(worst_ca,ca); worst_cb=min(worst_cb,cb)
    print(f"{s:>4} {ca:>8.4f} {cb:>8.4f} {ra:>8.3f} {rb:>8.3f}")
assert worst_ca>0.95 and worst_cb>0.97, "2PL recovery below tolerance"
print(f"\nVALIDATED: 2PL recovers params (worst corr_a={worst_ca:.3f}, corr_b={worst_cb:.3f})")
