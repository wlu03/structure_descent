"""Baseline suite for PO-LEU head-to-head comparison.

Each baseline implements the :class:`Baseline` / :class:`FittedBaseline`
protocols in :mod:`src.baselines.base`. The shared evaluation harness
:func:`src.baselines.evaluate.evaluate_baseline` converts a
:class:`FittedBaseline` into a :class:`BaselineReport` with
top-1 / top-5 / MRR / NLL / AIC / BIC plus per-category and
repeat-vs-novel breakdowns — using the *same* formulas and tie-breaking
as :mod:`src.eval.metrics` so PO-LEU and baseline numbers are directly
comparable.

Phase-1 baselines (migrated from ``old_pipeline/``):

* ``LASSO-MNL``        — :mod:`src.baselines.lasso_mnl`
* ``RandomForest``     — :mod:`src.baselines.classical_ml`
* ``GradientBoosting`` — :mod:`src.baselines.classical_ml`
* ``MLP``              — :mod:`src.baselines.classical_ml`
* ``Bayesian-ARD``     — :mod:`src.baselines.bayesian_ard`
* ``DUET``             — :mod:`src.baselines.duet_ga` (parametric ANN;
  module name is legacy — the GA variant was superseded)

Phase-3 frozen-LLM baselines:

* ``ZeroShot-Claude``      — :mod:`src.baselines.zero_shot_claude_ranker`
* ``FewShot-ICL-Claude``   — :mod:`src.baselines.few_shot_icl_ranker`
* ``ST-MLP``               — :mod:`src.baselines.st_mlp_ablation`
  (LLM-free ablation — embeds alt-metadata with the same sentence
  encoder PO-LEU uses and trains a small MLP)

Deferred (Phase 2, require porting ``old_pipeline/src/dsl.py`` +
``old_pipeline/src/inner_loop.py``):

* ``Delphos`` — DQN over DSL structures
* ``Paz-VNS`` — variable-neighborhood search over DSL structures

Dropped:

* ``RUMBoost`` — unresolved ``numpy<2`` / ``cythonbiogeme`` ABI
  conflict; see audit notes.

Shared modules:

* :mod:`src.baselines.base`          — Protocols + dataclasses
* :mod:`src.baselines.feature_pool`  — Expanded pool for LASSO / ARD
* :mod:`src.baselines.evaluate`      — Unified metric harness
* :mod:`src.baselines.data_adapter`  — Convert PO-LEU records to
  :class:`BaselineEventBatch`
* :mod:`src.baselines._synthetic`    — Unit-test fixture generator
"""

from .base import (
    Baseline,
    BaselineEventBatch,
    BaselineReport,
    FittedBaseline,
)
from .delphos import Delphos, DelphosFitted
from .evaluate import evaluate_baseline
from .feature_pool import build_expanded_pool, expand_batch
from .few_shot_icl_ranker import FewShotICLRanker, FewShotICLRankerFitted
from .lasr import Concept, ConceptLibrary, LaSR, LaSRFitted
from .llm_sr import LLMSR, LLMSRFitted
from .popularity import PopularityBaseline, PopularityFitted
from .st_mlp_ablation import STMLPChoice, STMLPFitted
from .zero_shot_claude_ranker import (
    ZeroShotClaudeRanker,
    ZeroShotClaudeRankerFitted,
)

__all__ = [
    "Baseline",
    "BaselineEventBatch",
    "BaselineReport",
    "Concept",
    "ConceptLibrary",
    "Delphos",
    "DelphosFitted",
    "FewShotICLRanker",
    "FewShotICLRankerFitted",
    "FittedBaseline",
    "LaSR",
    "LaSRFitted",
    "LLMSR",
    "LLMSRFitted",
    "PopularityBaseline",
    "PopularityFitted",
    "STMLPChoice",
    "STMLPFitted",
    "ZeroShotClaudeRanker",
    "ZeroShotClaudeRankerFitted",
    "build_expanded_pool",
    "evaluate_baseline",
    "expand_batch",
]
