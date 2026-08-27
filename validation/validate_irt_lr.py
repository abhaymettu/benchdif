"""Validate IRT-LR DIF: power under injected DIF + size under pure impact."""
import numpy as np
from benchdif.dif import irt_lr


def sim(seed, n=2000, J=15, dif=(3, 9), impact=-0.5, null=False):
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.7, 1.8, J); b = rng.normal(0, 1, J)
    a_f, b_f = a.copy(), b.copy()
    if not null:
        for j in dif:
            b_f[j] = b[j] + 0.8; a_f[j] = a[j] + 0.4
    th0 = rng.normal(0, 1, n); th1 = rng.normal(impact, 1, n)
    X0 = (rng.random((n, J)) < 1 / (1 + np.exp(-a * (th0[:, None] - b)))).astype(int)
    X1 = (rng.random((n, J)) < 1 / (1 + np.exp(-a_f * (th1[:, None] - b_f)))).astype(int)
    return np.vstack([X0, X1]), np.array([0] * n + [1] * n)


dif = {3, 9}
tp = fn = fp = tn = 0
for s in range(4):
    X, g = sim(s); res = irt_lr(X, g)
    fl = set(np.where(res["flag"].to_numpy())[0])
    an = set(np.where(res["is_anchor"].to_numpy())[0])
    for j in range(X.shape[1]):
        if j in an:
            continue
        d, f = j in dif, j in fl
        tp += d and f; fn += d and not f; fp += (not d) and f; tn += (not d) and not f
print(f"DIF present: power = {tp}/{tp+fn}, FPR = {fp}/{fp+tn} = {fp/(fp+tn):.3f}")

nf = nt = 0
for s in range(6):
    X, g = sim(s, null=True); res = irt_lr(X, g)
    m = ~res["is_anchor"].to_numpy()
    nf += int(res["flag"].to_numpy()[m].sum()); nt += int(m.sum())
print(f"NULL (impact only): size = {nf}/{nt} = {nf/nt:.3f} (nominal 0.05)")
assert tp == tp + fn, "missed a DIF item"
assert fp / (fp + tn) < 0.10, "too many false positives"
assert nf / nt < 0.10, "size too high under the null"
print("VALIDATED: IRT-LR has full power, controlled FPR, correct size under impact")
