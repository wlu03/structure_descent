"""
Baseline suite for Structure Descent.

Each baseline implements the Baseline / FittedBaseline protocols defined
in base.py. The shared evaluation harness in evaluate.py converts a
FittedBaseline into a BaselineReport with top-1/top-5/MRR/NLL/AIC/BIC
plus per-category and repeat-vs-novel breakdowns.

Available baselines (filled in as they are implemented):
  - LASSO-MNL         src/baselines/lasso_mnl.py
  - RUMBoost          src/baselines/rumboost_baseline.py
  - Paz VNS           src/baselines/paz_vns.py
  - DUET GA           src/baselines/duet_ga.py
  - Bayesian ARD      src/baselines/bayesian_ard.py
  - Delphos (stretch) src/baselines/delphos.py

Shared modules:
  - base.py           Protocols, dataclasses, shared input/output types
  - feature_pool.py   Expanded feature pool for regression-style baselines
  - evaluate.py       Shared evaluation harness
  - _synthetic.py     Synthetic BaselineEventBatch for unit tests / smoke tests
"""

from .base import (
    BaselineEventBatch,
    BaselineReport,
    FittedBaseline,
    Baseline,
)
from .evaluate import evaluate_baseline
from .feature_pool import build_expanded_pool, expand_batch

__all__ = [
    "BaselineEventBatch",
    "BaselineReport",
    "FittedBaseline",
    "Baseline",
    "evaluate_baseline",
    "build_expanded_pool",
    "expand_batch",
]
