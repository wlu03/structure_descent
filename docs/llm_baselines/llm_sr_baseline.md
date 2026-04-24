# LLM-SR for Discrete Choice Modelling (LLM-SR-DCM)

**Status**: design — implementation deferred.
**Target paths**: `src/baselines/llm_sr.py`, `tests/baselines/test_llm_sr.py`.
**Companion docs**: `docs/llm_baselines/zero_shot_claude_ranker.md`,
`docs/llm_baselines/few_shot_icl_ranker.md`.

## 1. Method

We adapt LLM-SR (Shojaee et al., ICLR 2025, arXiv:2404.18400) from
continuous symbolic regression to discrete choice. In the paper's
pipeline (§3.1) an LLM proposes candidate *equation skeletons* — Python
expressions over input features with placeholder coefficients
`c1..ck` — a numerical optimizer fits the coefficients by least-squares,
and the next proposal is conditioned on an "experience buffer" of top-K
skeletons and their losses (paper §3.2).

Four adaptations for DCM:

1. **Shared utility per alternative**. Each skeleton `f(x; c)` is
   evaluated per alternative with that alternative's 4-column vector
   (`data_adapter.py:86-91`); `f` and `c` are shared across all J alts
   of an event (conditional-logit form).
2. **Softmax → NLL**. `P(j) = softmax(U_1..U_J)`; fitness is
   `train_nll + val_nll`, same loss as `lasso_mnl.py:52-74`.
3. **MNL inner loop**, not least-squares. `scipy.optimize.minimize(
   softmax_nll, x0, method="BFGS")`; no regularization (grammar caps
   k ≤ 8).
4. **Proposer context**. The LLM sees the top `top_k_memory` skeletons
   with their NLL and parameter count — never raw `(X, chosen_idx)`
   pairs.

Distinguishing axis vs. Delphos / Structure Descent: **expressivity**.
Delphos (DQN) and Structure Descent (LLM) both edit a fixed ≤ 12-atom
DSL; LLM-SR-DCM proposes arbitrary bounded numeric expressions, so its
hypothesis class strictly contains the DSL. Cost: a flat MNL inner loop
(no hierarchy). The baseline isolates whether function-class
expressivity alone closes the gap to PO-LEU.

## 2. Equation skeleton grammar

Pinned BNF. The grammar targets a utility function of one alternative's
4-column vector `x = (price, popularity_rank, log1p_price, price_rank)`
from `BUILTIN_FEATURE_NAMES` in `src/baselines/data_adapter.py:86-91`.
The adapter deliberately excludes `is_repeat` and `brand_known` for
label-leakage reasons documented there (`data_adapter.py:48-65`); the
skeleton space inherits that restriction.

```
skeleton  ::= "def utility(x, c): return " expr
expr      ::= term (addop term)*
term      ::= factor (mulop factor)*
factor    ::= atom | unary factor | atom "**" INT01_OR_2
atom      ::= coef | feat | "(" expr ")" | func "(" expr ")"
unary     ::= "+" | "-"
addop     ::= "+" | "-"
mulop     ::= "*" | "/"            ; "/" wrapped by _safe_div
func      ::= "log1p" | "exp_c" | "sqrt_abs" | "tanh"
                                  ; "exp_c" clips the argument to [-10, 10]
                                  ; "sqrt_abs(u)" = sqrt(abs(u))
coef      ::= "c[0]" | "c[1]" | ... | "c[7]"
feat      ::= "x[0]" | "x[1]" | "x[2]" | "x[3]"
INT01_OR_2::= "2"                 ; integer power, degree-2 only
```

Design constraints:

- **≤ 8 coefficients** (LLM-SR §4.1 uses ≤ 10; tighter because J=4
  events give fewer effective samples per coefficient).
- **`_safe_div(a,b)`**: `a / (b + eps*sign(b))` with eps floor — no raw
  `/` escapes the grammar.
- **`exp_c(u) = exp(clip(u, -10, 10))`**: no raw `exp`; keeps softmax
  stable.
- **Polynomial cap at degree 2**; higher degrees give zero benefit per
  LLM-SR Appendix C and ill-condition BFGS.
- **No conditionals / loops / attribute access**: enforced by the
  sandbox (§4); out-of-grammar skeletons count against the proposal's
  retry budget.

Example well-formed skeletons:

```python
def utility(x, c): return c[0]*x[0] + c[1]*x[1] + c[2]*x[2]
def utility(x, c): return -c[0]*log1p(x[0]) + c[1]*x[3] - c[2]*x[1]**2
def utility(x, c): return c[0]*x[2] + c[1]*tanh(c[2]*x[3] - c[3])
```

## 3. LLM prompt template

Exact template, rendered each iteration. Placeholders in `{braces}`.

### 3.1 System prompt (cache-eligible)

```text
You are an equation-discovery agent for discrete choice modelling. Your job
is to propose Python functions that predict which of J=4 product alternatives
a shopper will buy. You will be shown the best-so-far equations and their
negative log-likelihood (NLL) on train and validation data. Propose a NEW
equation that might fit better.

Rules:
1. Output a single Python function with signature exactly
      def utility(x, c): return <expression>
   where `x` is a length-4 per-alternative feature vector
   (x[0]=price, x[1]=popularity_rank, x[2]=log1p_price, x[3]=price_rank)
   and `c` is a coefficient array of length at most 8.
2. Use only these operators: +, -, *, /, ** (integer exponent ≤ 2).
3. Use only these functions: log1p, exp_c (clipped exp), sqrt_abs, tanh.
4. No conditionals (if/else), no loops, no attribute access, no imports,
   no global state. A single return statement only.
5. Lower NLL is better. Prefer equations that are DIFFERENT from the
   best-so-far (not minor coefficient tweaks — structural variation).
6. Return only the function definition in a ```python``` code block. No
   prose, no comments outside the block.

Feature semantics (for reasoning, not code):
- price: raw dollar price of the alternative, range roughly $5 to $500.
- popularity_rank: larger = more popular; logged or raw values possible.
- log1p_price: log1p(price). A monotone transform of price.
- price_rank: within-event dense rank of price, in [0, 1]; 0 = cheapest.

Useful intuition from consumer theory:
- Shoppers are typically price-sensitive: the coefficient on price or
  log1p_price is usually negative.
- Popular items draw share; popularity coefficient is usually positive.
- price_rank captures relative-price effects that raw price cannot.
- Nonlinear blends of price and popularity (e.g. price × popularity, or
  tanh of a linear combination) can capture diminishing-returns effects.
```

### 3.2 User prompt (changes every iteration)

```text
Iteration {t} of {T_max}.

Best {K} equations so far, ranked by combined NLL = train_nll + val_nll
(lower is better):

{memory_block}

Your turn. Output ONE new candidate utility function, different in STRUCTURE
from all of the above (not merely a coefficient change). Output only the
```python``` code block.
```

`memory_block` is rendered from the experience buffer as:

```text
[1] NLL_train=1.2431  NLL_val=1.2690  k=3
def utility(x, c): return c[0]*x[0] + c[1]*x[1] + c[2]*x[3]

[2] NLL_train=1.2502  NLL_val=1.2604  k=4
def utility(x, c): return c[0]*log1p(x[0]) + c[1]*x[1] + c[2]*x[3] + c[3]

...
```

### 3.3 Prompt caching

System prompt is static across all T proposals → send under Anthropic
`cache_control: {"type": "ephemeral"}` (pattern:
`_llm_ranker_common.py:614-632`). Cache-write once, cache-read the rest;
cuts input tokens ~90%.

## 4. Sandboxed evaluation

The LLM emits Python source. We MUST NOT `eval()` or `exec()` untrusted
strings. Contract:

### 4.1 Parse-then-compile

1. `tree = ast.parse(source, mode="exec")`.
2. Walk `tree` and reject any node whose type is not in the allowlist
   below. Any rejection is fatal for that proposal; the proposer is
   asked to retry (up to `max_retries_per_proposal = 2`).
3. If the walk passes, `compile(tree, "<llm-sr>", "exec")` and execute
   the code in a **fresh dict** whose `__builtins__` is literally the
   empty dict `{}`. No `import`, no `open`, no `__class__` lookup.
4. Inject a shadow globals dict containing ONLY
   `{"log1p", "exp_c", "sqrt_abs", "tanh", "_safe_div"}` as named
   Python callables. The produced `utility` is extracted by
   `locals["utility"]`.

### 4.2 AST allowlist

Allowed node types (everything else rejected):

```
Module, FunctionDef, arguments, arg,
Return,
Expression, Expr, Constant, Num,
Name, Load,
BinOp, UnaryOp,
Add, Sub, Mult, Div, Pow, USub, UAdd,
Call,                               ; but see §4.3 for per-call policy
Subscript, Index, Slice,            ; only for x[i], c[i]
Tuple,                              ; allowed inside Subscript only
```

Explicit denylist (rejected on sight): `Import`, `ImportFrom`, `Global`,
`Nonlocal`, `ClassDef`, `AsyncFunctionDef`, `Lambda`, `If`, `For`,
`While`, `Try`, `With`, `Assign`, `AugAssign`, `AnnAssign`, `Attribute`,
`Starred`, `ListComp`, `SetComp`, `DictComp`, `GeneratorExp`,
`FormattedValue`, `JoinedStr`, `Await`, `Yield`, `YieldFrom`, `Raise`.

### 4.3 Call-node policy

`Call` is allowed only when `func` is a `Name` whose `id` is in the
allowlist `{"log1p", "exp_c", "sqrt_abs", "tanh", "_safe_div"}`. Any
other callee (`print`, `open`, `eval`, attribute chains like
`__import__`) is rejected at the AST walk.

### 4.4 Subscript policy

`Subscript` is allowed only when:
- `value` is `Name` with `id` ∈ `{"x", "c"}`, AND
- the index is a `Constant` integer in `[0, 3]` for `x` or
  `[0, max_coefficients - 1]` for `c`.

This blocks `x[y]` (dynamic index), `x[-1]`, slice shenanigans, and
attribute dereferences via `x[0].__class__`.

### 4.5 Runtime guards

Even after the AST check passes, evaluation is wrapped in:

```python
try:
    u = utility(x_j, c)                     # produces a float
    if not np.isfinite(u):
        raise ValueError("non-finite utility")
except Exception:
    # Skeleton rejected for this fit; outer loop records NLL=+inf
    return np.inf
```

`np.seterr(all="raise")` is set inside the inner loop so divide-by-zero,
overflow, etc. surface as `FloatingPointError` rather than silent `nan`.

## 5. Coefficient fitting

### 5.1 Setup

For a fixed skeleton `utility(x, c)` with `k` coefficients:

```python
def softmax_nll(c, events):
    nll = 0.0
    for (X_e, chosen_e) in events:             # X_e: (J, 4), chosen_e: int
        U = np.array([utility(X_e[j], c) for j in range(J)])
        if not np.all(np.isfinite(U)):
            return np.inf
        U -= U.max()                           # log-sum-exp shift
        lp = U - np.log(np.exp(U).sum())
        nll -= lp[chosen_e]
    return nll
```

### 5.2 Optimisation

```python
from scipy.optimize import minimize

def fit_coefficients(skeleton_fn, k, events, seed=0):
    rng = np.random.default_rng(seed)
    best_c, best_nll = None, np.inf
    for _ in range(N_RESTARTS):                # N_RESTARTS = 4
        x0 = rng.normal(scale=0.3, size=k)     # small random init
        try:
            res = minimize(
                softmax_nll, x0, args=(events,),
                method="BFGS",
                jac="2-point",                 # numerical gradient
                options={"maxiter": 200, "gtol": 1e-5},
            )
        except (FloatingPointError, ValueError):
            continue
        if np.isfinite(res.fun) and res.fun < best_nll:
            best_nll, best_c = float(res.fun), res.x
    return best_c, best_nll                    # (None, inf) on total failure
```

### 5.3 Failure handling

- **All restarts diverged** (`res.fun = inf` or every restart raised):
  skeleton gets `(None, inf)`; outer loop drops it from the buffer.
- **One restart converged**: accept its `(c*, nll)`.
- **Non-finite / `FloatingPointError`**: treated as divergence.
- **Flat gradient at init**: scipy returns the init point; accepted if
  `res.fun < inf` (a constant utility is a valid — if poor — model).

### 5.4 Why BFGS, not L-BFGS-B

LLM-SR §3.3 uses BFGS because k is small (≤ 10). L-BFGS-B's memoryless
Hessian is overkill and its line search is less robust on the
softmax-NLL landscape. We follow the paper.

## 6. Outer loop

```python
def fit(train, val):
    memory: list[SkeletonRecord] = []
    for t in range(n_proposals):
        prompt = render_user_prompt(t, memory[:top_k_memory])
        source = call_llm(system_prompt, prompt)      # string
        skeleton_fn, k = compile_skeleton(source)     # AST walk + compile
        if skeleton_fn is None:                       # rejected
            continue
        c_star, train_nll = fit_coefficients(skeleton_fn, k, train_events)
        if c_star is None:
            continue
        val_nll = softmax_nll(c_star, val_events)
        memory.append(SkeletonRecord(
            source=source, fn=skeleton_fn,
            coeffs=c_star, k=k,
            train_nll=train_nll, val_nll=val_nll,
        ))
        memory.sort(key=lambda r: r.train_nll + r.val_nll)
        memory = memory[:max_memory]                  # cap at 50
    best = memory[0]
    return LLMSRFitted(best_skeleton=best.source, ...)
```

## 7. Class signatures

```python
# src/baselines/llm_sr.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from src.baselines.base import Baseline, BaselineEventBatch, FittedBaseline
from src.outcomes.generate import LLMClient, StubLLMClient


class LLMSR:
    """LLM-SR-for-DCM baseline. See docs/llm_baselines/llm_sr_baseline.md."""

    name: str = "LLM-SR"

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        *,
        n_proposals: int = 100,
        top_k_memory: int = 10,
        max_memory: int = 50,
        max_coefficients: int = 8,
        n_restarts: int = 4,
        max_retries_per_proposal: int = 2,
        seed: int = 0,
        prompt_version: str = "llm-sr-v1",
    ) -> None: ...

    def fit(
        self,
        train: BaselineEventBatch,
        val: BaselineEventBatch,
    ) -> "LLMSRFitted": ...


@dataclass
class LLMSRFitted:
    name: str = "LLM-SR"
    best_skeleton: str = ""             # Python source of the chosen equation
    best_coefficients: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_coefficients: int = 0
    train_nll: float = float("inf")
    val_nll: float = float("inf")
    model_id: str = "unknown"
    n_proposals_accepted: int = 0
    n_proposals_total: int = 0
    prompt_version: str = "llm-sr-v1"
    # A recompiled callable backing score_events; not serialised.
    _utility_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None

    def score_events(self, batch: BaselineEventBatch) -> List[np.ndarray]:
        """Return [U ∈ R^J] per event; evaluate_baseline log-softmaxes these."""
        ...

    @property
    def n_params(self) -> int:
        return int(self.n_coefficients)

    @property
    def description(self) -> str:
        return (
            f"LLM-SR model={self.model_id} k={self.n_coefficients} "
            f"train_nll={self.train_nll:.4f} val_nll={self.val_nll:.4f} "
            f"accepted={self.n_proposals_accepted}/{self.n_proposals_total}"
        )
```

## 8. Registry integration

Append one row to `_LLM_BASELINE_BASES`
(`src/baselines/run_all.py:115-118`):

```python
_LLM_BASELINE_BASES: List[Tuple[str, str, str]] = [
    ("ZeroShot",    "src.baselines.zero_shot_claude_ranker", "ZeroShotClaudeRanker"),
    ("FewShot-ICL", "src.baselines.few_shot_icl_ranker",     "FewShotICLRanker"),
    ("LLM-SR",      "src.baselines.llm_sr",                   "LLMSR"),
]
```

`_build_registry` (`run_all.py:126-156`) auto-expands LLM-SR across
every `LLM_MODEL_SWEEP` entry and wires each factory into
`LLM_CLIENT_FACTORIES`. The stub-safety tripwire at
`run_all.py:258-280` fires for LLM-SR identically to ZeroShot /
FewShot — a registered LLM-SR row can never silently run on the stub.

## 9. Test strategy

Target `tests/baselines/test_llm_sr.py`. Twelve tests:

1. `test_grammar_accepts_well_formed` — the three §2 examples parse,
   compile, and return finite scalars.
2. `test_grammar_rejects_attribute_access` —
   `"return x.__class__"` raises `SandboxError`.
3. `test_grammar_rejects_import` — `import os` rejected at AST walk
   (no `exec`).
4. `test_grammar_rejects_unsafe_call` — `open(...)` and
   `__import__('os')` rejected (callee not in allowlist).
5. `test_grammar_rejects_bad_subscript` — `x[10]` (out of range) and
   `x[c[0]]` (dynamic index) both rejected.
6. `test_safe_div_handles_zero` — `_safe_div(1.0, 0.0)` finite;
   `softmax_nll` never sees `inf` from the divide.
7. `test_exp_c_clips_large` — `exp_c(50.0) == exp_c(10.0)`.
8. `test_fit_coefficients_on_synthetic` — 500 synthetic events with
   `U_j = 2·x[0] - 0.5·x[1]`; recovered coeffs within ±0.1.
9. `test_fit_coefficients_handles_divergence` — blow-up skeleton
   returns `(None, inf)` without raising.
10. `test_outer_loop_with_stub_client` — `LLMSR(StubLLMClient(),
    n_proposals=5).fit(train, val)` runs offline; every proposal
    fails to parse so fitted falls back to `FALLBACK_SKELETON`
    (§10). Assert `score_events(test)` returns `n_events` arrays of
    shape `(J,)` AND `best_skeleton == FALLBACK_SKELETON`.
11. `test_outer_loop_retries_on_rejection` — client replies
    ill-formed then valid; exactly two calls per proposal 0 and the
    valid skeleton is accepted.
12. `test_registry_factory_wires_llm_sr` — `("LLM-SR-Claude-Sonnet-
    4.6", ..., "LLMSR")` in `BASELINE_REGISTRY` and
    `LLM_CLIENT_FACTORIES["LLM-SR-Claude-Sonnet-4.6"]` callable.

## 10. Stub / cold-start fallback

Per the ZeroShot stub contract (`_llm_ranker_common.py:265-287`),
`StubLLMClient` emits narrative outcome sentences — never valid
Python — so the AST walk rejects 100% of stub proposals. We seed
`memory` with a fallback skeleton BEFORE the first LLM call:

```python
FALLBACK_SKELETON = (
    "def utility(x, c): return c[0]*x[0] + c[1]*x[1] + c[2]*x[2] + c[3]*x[3]"
)
```

Fit the fallback, push onto `memory`, then enter the LLM loop. If every
LLM proposal fails (the stub case), `memory[0]` is the fallback and
scoring reduces to a plain MNL over the 4-column pool. This mirrors
the "cold-start → zero-shot" fallback in `few_shot_icl_ranker.md` §8.

## 11. Cost estimates

Per proposal: cached system ~550 tok (cache-read after first call);
user ~50 + 10·80 = 850 tok (`top_k_memory=10`); output ~60 tok.

100-proposal LLM cost at 2026-04 list prices:

| Provider            | Input $/MTok | Cached $/MTok | Output $/MTok | 100-prop |
|---------------------|--------------|---------------|---------------|----------|
| Claude Sonnet 4.6   | $3           | $0.30         | $15           | ~$0.40   |
| Claude Opus 4.6     | $15          | $1.50         | $75           | ~$2.00   |
| Gemini 2.5 Flash    | $0.30        | $0.08         | $2.50         | ~$0.05   |
| Gemini 2.5 Pro      | $1.25        | $0.31         | $10           | ~$0.20   |
| GPT-5               | $1.25        | $0.13         | $10           | ~$0.20   |

Run costs are event-independent (proposer sees memory, not data) so
20 customers (~1.8k events) and 50 customers (~4.5k events) both pay
the table above. Without caching, costs are 5–10× higher — the
spec's "~$10–20 per run" figure is the no-cache upper bound.

**CPU fitting cost**: 9k events × 100 proposals × 4 restarts × 200
BFGS iters × ~50 µs/event ≈ 36 min single-core; parallelises across
restarts / proposals.

## 12. Known failure modes

1. **Proposal collapse to trivial skeleton** (LLM repeats
   `c[0]*x[0]+...+c[3]*x[3]`). Mitigations: system-prompt rule 5 asks
   for structural variation; after t proposals of near-duplicates,
   inject `"AVOID equations structurally equivalent to memory entries
   1..K"`; expose `n_unique_skeletons` on the fitted object.
2. **Coefficient fit divergence** (e.g. `x[0]**2*x[1]**2` overflows at
   init). Caught by `N_RESTARTS=4` + `np.seterr(all="raise")`;
   skeleton dropped, not fatal.
3. **Unsafe code emission**. §4 AST walk runs before compile/exec;
   `__import__('os').system('rm -rf /')` never executes. Add a
   regression test per unsafe pattern seen in production logs.
4. **Prompt-cache drift**. Caching requires byte-identical system
   prompt across proposals; any f-string interpolation busts the
   cache. Keep system prompt static; only user prompt varies.
5. **Runaway memory growth**. `memory` capped at `max_memory=50` with
   strict NLL ordering; proposer sees only top `top_k_memory=10`.
6. **Hermetic-CI silent fallback**. `StubLLMClient` emits narrative
   text that fails parsing → fitted object is `FALLBACK_SKELETON`
   (§10). Test #10 asserts this equality so a real-LLM regression
   can't masquerade as a passing stub test.

## 13. Sibling-baseline comparison

| Baseline          | Proposer   | Hypothesis class       | Inner loop       | k typical |
|-------------------|------------|------------------------|------------------|-----------|
| Delphos           | DQN        | DSL edits (≤12 atoms)  | Hierarchical MNL | 10–30     |
| Structure Descent | LLM (DSL)  | DSL edits (≤12 atoms)  | Hierarchical MNL | 10–30     |
| **LLM-SR-DCM**    | LLM (free) | Bounded numeric exprs  | Flat MNL (BFGS)  | ≤8        |
| LASSO-MNL         | —          | Linear + interactions  | FISTA            | 10–33     |
| ZeroShot-Claude   | —          | Frozen LLM             | —                | 0         |

LLM-SR-DCM is the only row with an **LLM proposer** whose hypothesis
class is **not** a DSL atom set. It isolates the expressivity question:
does arbitrary numeric expression over the 4 per-alt features beat a
hierarchical MNL constrained to the DSL? A "no" at 100 proposals says
expressivity isn't the bottleneck; a "yes" motivates extending
Structure Descent's DSL.

## 14. References

- Shojaee, Meidani, Gupta, Farimani, Reddy. *LLM-SR: Scientific
  Equation Discovery via Programming with Large Language Models*.
  ICLR 2025. arXiv:2404.18400. §3.1 (pipeline), §3.2 (experience
  management), §3.3 (coefficient fit), §4.1 (hyperparameters).
- Beck & Teboulle (2009) FISTA — contrasted with our BFGS choice
  in §5.4.
- Hou et al. 2023 (arXiv:2305.08845), Zheng et al. 2023
  (arXiv:2309.03882) — frozen-LLM ranker context (sibling baselines).
- `src/baselines/base.py:143-176` — `FittedBaseline` protocol.
- `src/baselines/_llm_ranker_common.py:612-658` — prompt caching
  pattern reused for §3.3.
- `src/baselines/lasso_mnl.py:52-74` — MNL NLL + gradient reference.
- `src/baselines/data_adapter.py:86-91` — per-alt feature schema.
- `src/baselines/run_all.py:103-156` — LLM registry + factory pattern.
