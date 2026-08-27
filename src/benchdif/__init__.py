"""benchdif: differential item functioning and measurement invariance for AI evals.

Detects when a benchmark item is not score-comparable across groups (model
families, languages, prompt conditions) using validated psychometric methods,
cross-checked numerically against independent reference implementations.
"""
from benchdif.dif import mantel_haenszel, logistic
from benchdif.adapters import from_long, from_matrix

__all__ = ["mantel_haenszel", "logistic", "from_long", "from_matrix"]
__version__ = "0.1.0"
