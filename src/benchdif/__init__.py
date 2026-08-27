"""benchdif: differential item functioning and measurement invariance for AI evals.

Detects when a benchmark item is not score-comparable across groups (model
families, languages, prompt conditions) using validated psychometric methods,
cross-checked numerically against independent reference implementations.
"""
from benchdif.dif import (mantel_haenszel, logistic, irt_lr,
                          generalized_mantel_haenszel)
from benchdif.adapters import from_long, from_matrix, from_lm_eval
from benchdif.irt import fit_2pl, invariance
from benchdif.detect import detect
from benchdif.multiplicity import adjust, check_adequacy

__all__ = ["mantel_haenszel", "logistic", "irt_lr", "generalized_mantel_haenszel", "from_long", "from_matrix", "from_lm_eval", "fit_2pl", "invariance", "detect", "adjust", "check_adequacy"]
__version__ = "0.1.0"
