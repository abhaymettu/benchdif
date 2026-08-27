"""benchdif: differential item functioning and measurement invariance for AI evals.

Detects when a benchmark item is not score-comparable across groups (model
families, languages, prompt conditions) using validated psychometric methods,
cross-checked numerically against independent reference implementations.
"""
from benchdif.dif import mantel_haenszel, logistic, irt_lr
from benchdif.adapters import from_long, from_matrix, from_lm_eval
from benchdif.irt import fit_2pl

__all__ = ["mantel_haenszel", "logistic", "irt_lr", "from_long", "from_matrix", "from_lm_eval", "fit_2pl"]
__version__ = "0.1.0"
