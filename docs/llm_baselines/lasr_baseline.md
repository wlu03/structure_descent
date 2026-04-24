# LaSR-for-DCM: Library-Augmented Symbolic Regression for Discrete Choice

**Status**: design — implementation deferred to a follow-up agent.
**Target paths**: `src/baselines/lasr.py`,
`src/baselines/_symbolic_regression_common.py` (shared with LLM-SR; see §9),
`tests/baselines/test_lasr.py`.

Adapts Grayeli et al. 2024 (NeurIPS, "Symbolic Regression with a Learned
Concept Library", arXiv:2409.09359, hereafter **LaSR**) from continuous
symbolic regression to discrete choice. Its sibling baseline is LLM-SR
(Shojaee et al., ICLR 2025 oral, arXiv:2404.18400); the two share a
grammar + fitter + sandbox (`_symbolic_regression_common.py`) and differ
only in the outer proposal loop. See §5 for why we include both.

## 1. Recasting symbolic regression to DCM

LaSR's original objective searches programmatic hypotheses `h: R^n → R`
minimizing MSE (paper §3). We recast each hypothesis as a callable
`U_j = f(alt_features_j, coefficients)`, softmax across `J`, and fit
by conditional-logit NLL — the objective `_nll_only` already implements
at `src/baselines/lasso_mnl.py:77-87`. Features are the 4-column pool
(`price`, `popularity_rank`, `log1p_price`, `price_rank`) from
`src/baselines/data_adapter.py:86-91`.

## 2. Concept library: representation and promotion

### 2.1 Concept grammar (research question 1)

A **concept** is a named Python function over alt features plus free
scalar coefficients. Rationale: (a) the fitter already uses Python
callables, (b) promotion/unification is string-level renaming, (c) the
LaSR reference implementation stores concepts as natural-language
summaries re-expanded in-prompt (paper §3.3 "Concept Abstraction";
`trishullab/LibraryAugmentedSymbolicRegression.jl` README) — a Python
`def` with a docstring is a machine-executable specialization.

```python
@dataclass(frozen=True)
class Concept:
    name: str                   # e.g. "price_sensitivity"
    signature: str              # full "def <name>(...): return ..." source
    nl_summary: str             # one-line natural-language gloss for prompt
    arg_names: tuple[str, ...]  # feature args the concept reads
    n_coeffs: int               # free scalars injected into global coeff vec
    usage_count: int            # refcount across top-K surviving equations
    discovered_at: int          # outer-loop iteration when added
```

Example seeded concepts (bootstrapped, iteration 0) informed by the DCM
literature:

```python
def price_sensitivity(price, c0):      return -c0 * price
def log_popularity(popularity, c0):    return c0 * np.log1p(popularity)
def price_rank_centered(price_rank, c0):
    return -c0 * np.abs(price_rank - 0.5)
def linear(x, c0):                     return c0 * x
```

### 2.2 Promotion rule (research question 2)

**Hybrid**: a candidate sub-expression is promoted when **either**
(a) its canonicalized pattern appears in ≥ `concept_promotion_threshold`
(default 3) top-K survivors **or** (b) the LLM explicitly nominates it
post-fit. This mirrors LaSR's three-phase loop (§3.1 Algorithm 1):
hypothesis evolution, concept abstraction via LLM summary, concept
evolution. Frequency alone misses novel abstractions; nomination alone
is noisy — we accept frequency ≥ 3 *or* an LLM nomination backed by ≥ 1
occurrence.

### 2.3 Library size cap and pruning (research question 3)

LaSR (§4.1) caps the library at `L` concepts (paper uses 10–20). Default
`concept_library_max_size = 20`. On overflow, evict the lowest
`usage_count` concept, breaking ties by `discovered_at` (oldest first —
LRU-of-usefulness). A concept whose `usage_count` stays 0 for
`concept_ttl = 5` consecutive iterations is also evicted. The paper
discusses no explicit TTL; we add one to prevent long-run staleness.

## 3. Outer loop

```
Input: train, val, n_iters=10, proposals_per_iter=10, top_k=10
Library L <- [seeded concepts from §2.1]
Equation memory M <- []          # list of (equation_source, val_nll)

for t in 1..n_iters:
    # (i) Hypothesis evolution: propose via LLM conditioned on L and M
    proposals = []
    for _ in 1..proposals_per_iter:
        src = llm_propose_equation(library=L, memory=M[-top_k:])
        proposals.append(src)

    # Fit each proposal's free coefficients against train
    fitted = []
    for src in proposals:
        fn, n_coef = compile_equation(src)               # shared util
        theta = fit_coefficients_softmax_ce(fn, n_coef, train)  # L-BFGS-B
        val_nll = eval_nll(fn, theta, val)
        fitted.append((src, fn, theta, val_nll))

    # Survivor selection
    fitted.sort(key=lambda r: r.val_nll)
    survivors = fitted[:top_k]
    M.extend([(s.src, s.val_nll) for s in survivors])

    # (ii) Concept abstraction: frequency scan + LLM nomination
    new_concepts = extract_concepts(survivors, library=L, threshold=3)
    nominated   = llm_nominate_concepts(survivors, library=L)
    promote(L, new_concepts | nominated,
            max_size=concept_library_max_size,
            threshold=concept_promotion_threshold)

    # (iii) Concept evolution (paper §3.4): ask LLM to generalise/merge L
    L = llm_evolve_concepts(L) if t % concept_evolution_every == 0 else L

return best equation in M by val_nll, final L
```

`fit_coefficients_softmax_ce` lives in the shared module (§7); it reuses
the `p - e_chosen` gradient contraction from `lasso_mnl.py:51-74` but
drops the L1 prox — plain L-BFGS-B suffices since free-coefficient
counts are small (≤ 8 per equation).

### 3.1 Concept extraction pseudocode

```python
def extract_concepts(
    survivors: list[Survivor],
    library: list[Concept],
    threshold: int,
) -> list[Concept]:
    """
    Scan survivor ASTs for sub-expressions that (a) are not already in
    `library`, (b) have between 1 and MAX_COEFFS free numeric leaves, and
    (c) appear in at least `threshold` survivor sources after normalizing
    variable names.
    """
    candidates = Counter()
    for s in survivors:
        tree = ast.parse(s.src)
        for subexpr in subtrees_of_depth(tree, min_d=2, max_d=4):
            key = canonicalize(subexpr)   # rename vars to x0,x1,...
                                          # rename consts to c0,c1,...
            candidates[key] += 1
    new = []
    for key, freq in candidates.items():
        if freq < threshold:      continue
        if key in {c.signature for c in library}: continue
        nl = llm_summarize(key)   # one-line gloss
        new.append(Concept(
            name=auto_name(nl),
            signature=wrap_as_def(key),
            nl_summary=nl,
            arg_names=vars_in(key),
            n_coeffs=consts_in(key),
            usage_count=freq,
            discovered_at=current_iter,
        ))
    return new
```

## 4. LLM prompt template (research question 4)

The proposal-phase prompt. The library and equation memory are
rendered inline. Compare to LLM-SR's prompt (which shows only
`memory`): the delta is the `CONCEPT LIBRARY` block (~150 tokens at
library size 20).

```text
System:
You are designing a utility function for a discrete-choice model.
U_j of alternative j must be a Python expression over four features
(price, popularity_rank, log1p_price, price_rank) and free coefficients
c0, c1, .... After scoring all J alternatives, a softmax maximises the
log-probability of the observed choice. Prefer composing the provided
concepts over bare features. Return ONE new utility function as valid
Python. Do not explain.

User:
CONCEPT LIBRARY (reuse and compose these):
- price_sensitivity(price, c):  "linear disutility of price"
    def price_sensitivity(price, c): return -c * price
- log_popularity(popularity, c):  "saturating preference for popular items"
    def log_popularity(popularity, c): return c * np.log1p(popularity)
- price_rank_centered(price_rank, c):  "dislikes extreme ranks"
    def price_rank_centered(price_rank, c): return -c * np.abs(price_rank - 0.5)
  ... up to concept_library_max_size entries ...

BEST EQUATIONS SO FAR (val NLL; lower is better):
1. [val_nll=0.912]
   def U(price, popularity_rank, log1p_price, price_rank, c0, c1):
       return price_sensitivity(price, c0) + log_popularity(popularity_rank, c1)
2. [val_nll=0.928]
   def U(...): return -c0*log1p_price + c1*popularity_rank
  ... up to top_k_memory entries ...

PROPOSE A NEW UTILITY FUNCTION:
def U(price, popularity_rank, log1p_price, price_rank, c0, c1, ...):
    return
```

**Cost vs LLM-SR**: library (~200 tok) + memory (~400 tok) = ~700
input tokens per LaSR proposal vs. ~400 for LLM-SR. Over 100
proposals, LaSR spends ~30k more input tokens per customer —
negligible under prompt caching (§10).

## 5. Why include LaSR alongside LLM-SR (research question 5)

LLM-SR surfaces a best-so-far equation stack; LaSR adds **named
reusable abstractions**. Four testable claims:

1. **Accuracy**: LaSR ≥ LLM-SR on test NLL when the DCM target
   decomposes cleanly (e.g. `price_sensitivity + log_popularity`).
2. **Sample efficiency**: LaSR reaches a given val-NLL threshold in
   fewer proposals (metric: `proposals_to_threshold`).
3. **Interpretability** (load-bearing): **equation token length at
   matched test NLL**. LaSR's final equation should be shorter and use
   DCM-canonical names — reportable even when accuracy ties.
4. **Library transfer**: optional ablation — seed customer `n+1`'s
   library with customer `n`'s final library. LLM-SR has no analogue.

Without this narrative the two baselines collapse. With it they carve
out distinct ablation cells.

## 6. Module skeleton

```python
# src/baselines/lasr.py
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from .base import BaselineEventBatch, FittedBaseline
from ._symbolic_regression_common import (
    compile_equation, fit_coefficients_softmax_ce, eval_nll_val,
    SEED_CONCEPTS, extract_subexpression_candidates, canonicalize,
)
from src.outcomes.generate import LLMClient, StubLLMClient


@dataclass(frozen=True)
class Concept:
    name: str
    signature: str
    nl_summary: str
    arg_names: tuple[str, ...]
    n_coeffs: int
    usage_count: int
    discovered_at: int


@dataclass
class LaSRFitted:
    name: str
    best_equation: str
    best_coefficients: np.ndarray
    final_concept_library: List[Concept]
    equation_memory: List[tuple[str, float]]
    compiled_fn: object
    feature_names: List[str]
    train_nll: float

    def score_events(self, batch: BaselineEventBatch) -> List[np.ndarray]:
        return [
            np.array([self.compiled_fn(*feats[j], *self.best_coefficients)
                      for j in range(feats.shape[0])], dtype=np.float64)
            for feats in batch.base_features_list
        ]

    @property
    def n_params(self) -> int:
        return int(self.best_coefficients.shape[0])

    @property
    def description(self) -> str:
        return (f"LaSR |L|={len(self.final_concept_library)} "
                f"eq='{self.best_equation[:60]}...' n_coef={self.n_params}")


class LaSR:
    name = "LaSR"
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        n_iters: int = 10,
        proposals_per_iter: int = 10,
        top_k_memory: int = 10,
        concept_library_max_size: int = 20,
        concept_promotion_threshold: int = 3,
        concept_evolution_every: int = 3,
        concept_ttl: int = 5,
        seed: int = 0,
    ) -> None:
        self.llm_client = llm_client or StubLLMClient()
        # ... store all hyperparams ...

    def fit(self, train, val) -> LaSRFitted:
        ...  # outer loop from §3
```

## 7. Shared infrastructure with LLM-SR (coordination note)

Both baselines share `src/baselines/_symbolic_regression_common.py`:

| Exported symbol                          | Used by        |
| ---------------------------------------- | -------------- |
| `EquationGrammar` (allowed ops, features)| LaSR, LLM-SR   |
| `compile_equation(src)`                  | LaSR, LLM-SR   |
| `SafeSandbox` (AST allowlist)            | LaSR, LLM-SR   |
| `fit_coefficients_softmax_ce(...)`       | LaSR, LLM-SR   |
| `eval_nll_val(fn, theta, batch)`         | LaSR, LLM-SR   |
| `SEED_CONCEPTS`                          | LaSR (only)    |
| `extract_subexpression_candidates(...)`  | LaSR (only)    |
| `canonicalize(ast_node)`                 | LaSR (only)    |

LLM-SR imports only the first five; LaSR imports all eight. The
incremental LaSR code is:

- `Concept` dataclass + `final_concept_library` field (~30 LOC)
- `extract_concepts` + `promote` + `evict_lru_usage` (~90 LOC)
- LaSR prompt builder (library + memory) (~50 LOC)
- Concept-evolution phase call (~30 LOC)
- Outer-loop wiring (~40 LOC)

**~240 LOC on top of LLM-SR** (itself ~300 LOC). Estimate assumes LLM-SR
lands first; if LaSR lands first, it builds the shared module and
LLM-SR's delta is ~60 LOC (just strip the library logic).

## 8. Test strategy

1. `test_concept_dataclass_is_frozen` — mutation raises `FrozenInstanceError`.
2. `test_canonicalize_renames_variables` — `price + c0*popularity` and
   `x0 + c0*x1` canonicalize to the same key.
3. `test_extract_concepts_respects_threshold` — 3 survivors containing
   `-c*price` promote; 2 do not (threshold=3).
4. `test_extract_concepts_deduplicates_against_library` — seeded
   `price_sensitivity` blocks re-promotion.
5. `test_library_cap_evicts_lowest_usage` — fill library to 20 with
   varied `usage_count`; adding one more evicts the min-usage entry
   (tie broken by oldest `discovered_at`).
6. `test_library_ttl_evicts_zero_usage` — concept with 0 usage for
   `concept_ttl` iterations is evicted even under cap.
7. `test_prompt_contains_library_and_memory` — rendered prompt
   contains every `nl_summary` and each memory `val_nll`.
8. `test_stub_client_produces_compilable_equation` — `StubLLMClient`
   path produces a callable of expected arity with finite utilities on
   `_synthetic.make_synthetic_batch`.
9. `test_lasr_fit_on_synthetic_improves_vs_popularity` — on synthetic
   data with true utility `-0.8*price + 0.4*log1p(popularity)`, LaSR
   test NLL < Popularity test NLL.
10. `test_score_events_shape_and_finiteness` — returns `n_events`
    arrays of shape `(J,)`, all finite.
11. `test_concept_reuse_in_new_proposals` — monkeypatched proposal
    calling `price_sensitivity(price, c0)` resolves against the
    library; evaluated utility equals direct computation.
12. `test_fitter_divergence_recovered` — NaN-utility equation logs a
    warning and is skipped (doesn't fail the fit).
13. `test_concept_collapse_guard` — LLM nominating only trivial
    concepts (`def const(c): return c`) is filtered out; library
    cannot collapse to constants.
14. `test_concept_evolution_merges_duplicates` — evolution phase
    dropping duplicate canonicalized signatures.
15. `test_run_all_registry_contains_lasr` — `BASELINE_REGISTRY` in
    `run_all.py` contains `"LaSR-<model>"` rows and
    `LLM_CLIENT_FACTORIES` maps each to a factory.

## 9. Registry entry for `run_all.py`

LaSR is an LLM baseline, so it plugs into the sweep expansion at
`run_all.py:115-118`:

```python
_LLM_BASELINE_BASES: List[Tuple[str, str, str]] = [
    ("ZeroShot",    "src.baselines.zero_shot_claude_ranker", "ZeroShotClaudeRanker"),
    ("FewShot-ICL", "src.baselines.few_shot_icl_ranker",     "FewShotICLRanker"),
    ("LLM-SR",      "src.baselines.llm_sr",                  "LLMSR"),
    ("LaSR",        "src.baselines.lasr",                    "LaSR"),
]
```

The LaSR constructor already accepts `llm_client=...`, so
`LLM_CLIENT_FACTORIES` wires in every model from `LLM_MODEL_SWEEP`
without further changes. The production-safety tripwire at
`run_all.py:258-280` fires automatically for any LaSR row that
resolves to a `StubLLMClient`.

## 10. Cost estimate

Per-customer cost drivers:

- Proposal calls: `n_iters × proposals_per_iter = 100`.
- Concept abstraction calls: `n_iters ≈ 10` (one per iter, summary +
  nomination batched).
- Concept evolution calls: `n_iters / concept_evolution_every ≈ 3`.

Total ≈ **113 LLM calls per customer**, each ~700 input + 50 output
tokens before caching. Claude Sonnet 4.6 at $3/1M input, $15/1M output,
with prompt caching (system + library blocks are stable within a
customer; see `_llm_ranker_common.py:610-640` for the cache pattern
already in use):

- Cache-write (first call): 700 * 1.25 * $3/1M ≈ $0.0026
- Cache-read (calls 2..113): 112 * 700 * 0.1 * $3/1M ≈ $0.024
- Output: 113 * 50 * $15/1M ≈ $0.085
- **Per customer: ~$0.11**
- **20 customers: ~$2.20**
- **50 customers: ~$5.50**

Fit time per customer ≈ 113 calls × 1.5 s/call ≈ 3 min without
parallelism; the `_score_one_event` pattern in
`_llm_ranker_common.py:751-797` is inherently serial within a
customer but embarrassingly parallel across customers. LLM-SR costs
roughly 60% of LaSR (400-token prompts, no concept-evolution phase):
**~$1.30 / 20 customers, ~$3.30 / 50 customers**. So LaSR adds
~$0.90 (20) or ~$2.20 (50) over LLM-SR.

## 11. Known failure modes

1. **Concept-library collapse** — LLM nominates only trivial concepts
   (e.g. `def identity(x, c): return c*x`). Guard: promotion filter
   rejects concepts canonicalizing to `SEED_CONCEPTS['linear']` or with
   `n_features == 0`. Tested in `test_concept_collapse_guard`.
2. **Un-composable concepts** — LLM returns a concept using `dict` or
   `np.random.randn`. Guard: the `SafeSandbox` AST allowlist (shared
   with LLM-SR) permits only `np.log, np.log1p, np.abs, np.exp, np.sqrt,
   +, -, *, /, **` and argument refs; anything else raises at compile.
3. **Fitter divergence** — pathological equations (e.g. `exp(c0*price)`)
   overflow. Guard: wrap each proposal's fit in `try/except` and set
   `val_nll = +inf` on overflow so it falls out of top-K naturally.
4. **Prompt length blow-up** — library + memory exceeds context window.
   Guard: `concept_library_max_size = 20` and `top_k_memory = 10` keep
   prompt ≤ ~2k tokens; `build_proposal_prompt` asserts rendered length
   ≤ 4096 tokens.

## 12. References

- Grayeli, Sehgal, Costilla-Reyes, Cranmer, Chaudhuri (NeurIPS 2024),
  [Symbolic Regression with a Learned Concept Library](https://arxiv.org/abs/2409.09359).
- Shojaee et al. (ICLR 2025 oral),
  [LLM-SR: Scientific Equation Discovery via Programming with Large Language Models](https://arxiv.org/abs/2404.18400).
- Reference implementation: [trishullab/LibraryAugmentedSymbolicRegression.jl](https://github.com/trishullab/LibraryAugmentedSymbolicRegression.jl).
- Project website: [LaSR — Symbolic Regression with a Learned Concept Library](https://trishullab.github.io/lasr-web/).
