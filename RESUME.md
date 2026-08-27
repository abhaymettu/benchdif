# benchdif — one-paragraph framing

Built an open-source Python library that brings psychometric measurement-invariance
and differential-item-functioning (DIF) methods to AI/LLM benchmark evaluation —
detecting when a benchmark question is not score-comparable across model families,
languages, or prompt conditions. Implemented Mantel-Haenszel and logistic-regression
DIF from the Holland-Thayer and Zumbo-Thomas formulations and validated every
statistic to numerical agreement against R's reference `difR` package, then built
adapters that reshape raw benchmark result files into the person-by-item response
matrices the methods require. Fills a gap the field named but had no tooling for:
IRT/psychometric estimators for benchmarks exist, but none test invariance or DIF.
