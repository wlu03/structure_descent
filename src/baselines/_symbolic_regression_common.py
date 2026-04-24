"""Shared helpers for LLM-driven symbolic-regression baselines on DCM data.

This module is consumed by :mod:`src.baselines.llm_sr` and
:mod:`src.baselines.lasr`. It intentionally carries no baseline-specific
state (prompt strings, memory buffers, model identifiers). Anything
specific to one baseline lives in that baseline's module; everything
shared — the equation grammar, AST sandbox, coefficient fitter,
softmax-CE evaluator, numerically safe primitives, and the LaSR concept
helpers — lives here.

Public surface
--------------
- :class:`SandboxError` — raised by :func:`compile_equation` on any AST
  or subscript-policy violation.
- :class:`EquationGrammar` — the BNF allowlist / denylist constants per
  the design doc §2/§4.
- :class:`SafeSandbox` — stateless validator that walks a parsed AST
  against :class:`EquationGrammar`.
- :func:`compile_equation` — parse-walk-compile-exec pipeline that
  returns a callable ``utility(x, c) -> float`` alongside the
  coefficient count ``k``.
- :func:`fit_coefficients_softmax_ce` — BFGS multi-start fitter against
  the discrete-choice softmax-CE loss.
- :func:`eval_nll_val` — softmax-CE on a held-out batch.
- Safe primitives: :func:`_safe_div`, :func:`exp_c`, :func:`sqrt_abs`,
  :func:`tanh`. These are injected into the sandbox's globals.
- :data:`FALLBACK_SKELETON` — the linear-in-4-features skeleton used by
  the stub / cold-start fallback.
- :data:`SEED_CONCEPTS` — LaSR-only bootstrap concept list per the
  LaSR design doc §2.1.
- :class:`SubExpr` and :func:`extract_subexpression_candidates` — LaSR
  survivor-AST scan for promotion candidates.
- :func:`canonicalize` — stable canonical form over candidate subtrees
  used for frequency counting and dedup.

References
----------
- ``docs/llm_baselines/llm_sr_baseline.md`` — LLM-SR design doc. §2 pins
  the BNF; §4 pins the sandbox allowlist; §5 pins BFGS + restarts; §10
  pins the fallback skeleton.
- ``docs/llm_baselines/lasr_baseline.md`` — LaSR design doc. §2.1 pins
  the seed concepts; §3.1 pins the extraction/canonicalize contract.
- Shojaee et al. 2025 (arXiv:2404.18400) — LLM-SR.
- Grayeli et al. 2024 (arXiv:2409.09359) — LaSR.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe numeric primitives (injected as globals when compiling equations)
# ---------------------------------------------------------------------------

_EXP_CLIP: float = 10.0
"""Symmetric clip applied inside :func:`exp_c` to keep softmax stable."""

_DIV_EPS: float = 1e-8
"""Epsilon floor applied inside :func:`_safe_div` to avoid divide-by-zero."""


def _safe_div(a: float, b: float) -> float:
    """Return ``a / (b + eps*sign(b))`` with an eps floor when ``b == 0``.

    The grammar forbids raw ``/``; this helper is the only path by which a
    division reaches the NumPy arithmetic layer. Zero denominators are
    pushed off by ``_DIV_EPS``; finite denominators pass through with a
    sign-preserving floor.
    """
    b = float(b)
    if b == 0.0:
        denom = _DIV_EPS
    else:
        denom = b + _DIV_EPS * (1.0 if b > 0 else -1.0)
    return float(a) / denom


def exp_c(u: float) -> float:
    """Return ``exp(clip(u, -_EXP_CLIP, _EXP_CLIP))``.

    The grammar forbids raw ``exp``; this clipped version keeps the
    softmax numerically well behaved even when BFGS wanders into a region
    of large utilities.
    """
    v = float(u)
    if v > _EXP_CLIP:
        v = _EXP_CLIP
    elif v < -_EXP_CLIP:
        v = -_EXP_CLIP
    return float(np.exp(v))


def sqrt_abs(u: float) -> float:
    """Return ``sqrt(abs(u))``. Real-valued replacement for ``sqrt``."""
    return float(np.sqrt(abs(float(u))))


def tanh(u: float) -> float:
    """Return ``tanh(u)`` via NumPy (bounded in ``[-1, 1]``, always safe)."""
    return float(np.tanh(float(u)))


# ---------------------------------------------------------------------------
# Grammar / sandbox specification
# ---------------------------------------------------------------------------

#: Names of callables injected into the sandbox's globals. Any ``Call``
#: whose callee is a ``Name`` not in this set is rejected by the AST
#: walker (design doc §4.3).
ALLOWED_FUNCTIONS: Tuple[str, ...] = (
    "log1p",
    "exp_c",
    "sqrt_abs",
    "tanh",
    "_safe_div",
)

#: The two subscriptable names. Any ``Subscript`` whose ``value`` is a
#: ``Name`` with an id outside this set is rejected (design doc §4.4).
_SUBSCRIPTABLE_NAMES: Tuple[str, ...] = ("x", "c")

#: Max index allowed for ``x[...]`` — the per-alt feature vector has
#: exactly 4 slots (``BUILTIN_FEATURE_NAMES`` in data_adapter.py).
X_MAX_INDEX: int = 3

#: Integer exponents allowed in ``atom ** INT``. The grammar pins this to
#: ``{2}`` to bound the polynomial degree; constants 0, 1 are technically
#: allowed too since they're degenerate (``x**0 == 1``, ``x**1 == x``).
_ALLOWED_POW_INTS: frozenset[int] = frozenset({0, 1, 2})


_ALLOWED_NODE_TYPES: Tuple[type, ...] = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Return,
    ast.Expression,
    ast.Expr,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Call,
    ast.Subscript,
)

# ``ast.Index`` was removed in Python 3.9 (Subscript.slice is now the
# expression directly) and ``ast.Num`` / ``ast.Str`` were subsumed by
# ``ast.Constant`` in 3.8. We still append the legacy types when they
# exist so the allowlist is compatible with older CPython without
# special-casing, but we suppress the DeprecationWarning raised by the
# getattr probe on newer CPython where ``ast.Num`` is a stub.
import warnings as _warnings

with _warnings.catch_warnings():
    _warnings.simplefilter("ignore", DeprecationWarning)
    for _legacy in ("Index", "Slice", "Tuple"):
        _legacy_cls = getattr(ast, _legacy, None)
        if _legacy_cls is not None:
            _ALLOWED_NODE_TYPES = _ALLOWED_NODE_TYPES + (_legacy_cls,)


_DENY_NODE_NAMES: Tuple[str, ...] = (
    "Import",
    "ImportFrom",
    "Global",
    "Nonlocal",
    "ClassDef",
    "AsyncFunctionDef",
    "Lambda",
    "If",
    "IfExp",
    "For",
    "AsyncFor",
    "While",
    "Try",
    "TryStar",
    "With",
    "AsyncWith",
    "Assign",
    "AugAssign",
    "AnnAssign",
    "Attribute",
    "Starred",
    "ListComp",
    "SetComp",
    "DictComp",
    "GeneratorExp",
    "FormattedValue",
    "JoinedStr",
    "Await",
    "Yield",
    "YieldFrom",
    "Raise",
    "Assert",
    "Delete",
    "Break",
    "Continue",
    "Match",
    "NamedExpr",
)
_DENY_NODE_TYPES: Tuple[type, ...] = tuple(
    getattr(ast, name) for name in _DENY_NODE_NAMES if hasattr(ast, name)
)


class SandboxError(ValueError):
    """Raised when a candidate equation violates the grammar / sandbox policy.

    The outer LLM-SR loop catches this and records the proposal as
    rejected; it never bubbles out of ``fit``.
    """


@dataclass(frozen=True)
class EquationGrammar:
    """Allowlist / denylist constants for the LLM-SR equation grammar.

    Exposed as a dataclass so callers (LaSR, tests) can introspect the
    exact node lists rather than reading the module globals directly. No
    state beyond the class-level defaults.
    """

    allowed_nodes: Tuple[type, ...] = _ALLOWED_NODE_TYPES
    denied_nodes: Tuple[type, ...] = _DENY_NODE_TYPES
    allowed_functions: Tuple[str, ...] = ALLOWED_FUNCTIONS
    subscriptable_names: Tuple[str, ...] = _SUBSCRIPTABLE_NAMES
    x_max_index: int = X_MAX_INDEX
    allowed_pow_ints: frozenset = _ALLOWED_POW_INTS


DEFAULT_GRAMMAR: EquationGrammar = EquationGrammar()


# ---------------------------------------------------------------------------
# AST walker / sandbox
# ---------------------------------------------------------------------------


class SafeSandbox:
    """Validate a parsed equation AST against :class:`EquationGrammar`.

    The walker is stateless (beyond the injected grammar + coefficient
    cap) and produces no output; it either returns the set of coefficient
    indices referenced by the tree or raises :class:`SandboxError`. The
    caller owns compilation / execution once validation passes.
    """

    def __init__(
        self,
        *,
        grammar: EquationGrammar = DEFAULT_GRAMMAR,
        max_coefficients: int = 8,
    ) -> None:
        if max_coefficients <= 0:
            raise ValueError(
                f"max_coefficients must be positive, got {max_coefficients}"
            )
        self.grammar = grammar
        self.max_coefficients = int(max_coefficients)

    # ------------------------------------------------------------------
    def validate(self, tree: ast.AST) -> set[int]:
        """Walk ``tree`` and return the set of coefficient indices used.

        Raises :class:`SandboxError` on any violation. The returned set
        is used by the caller to deduce the number of free coefficients
        ``k`` via ``max(indices) + 1`` (so gaps are treated as zero-
        weight coefficients and BFGS still fits them; callers typically
        tighten this to ``len(indices)`` when the grammar is followed).
        """
        coef_indices: set[int] = set()
        saw_function_def = False
        for node in ast.walk(tree):
            if isinstance(node, self.grammar.denied_nodes):
                raise SandboxError(
                    f"disallowed node type: {type(node).__name__}"
                )
            if not isinstance(node, self.grammar.allowed_nodes):
                raise SandboxError(
                    f"node type not in allowlist: {type(node).__name__}"
                )
            if isinstance(node, ast.FunctionDef):
                self._check_function_def(node)
                saw_function_def = True
            elif isinstance(node, ast.Call):
                self._check_call(node)
            elif isinstance(node, ast.Subscript):
                self._check_subscript(node, coef_indices)
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                self._check_pow(node)
            elif isinstance(node, ast.Name):
                self._check_name(node)
        if not saw_function_def:
            raise SandboxError("no top-level function definition found")
        return coef_indices

    # ------------------------------------------------------------------
    def _check_function_def(self, node: ast.FunctionDef) -> None:
        if node.name != "utility":
            raise SandboxError(
                f"function must be named 'utility', got {node.name!r}"
            )
        args = node.args
        if args.vararg is not None or args.kwarg is not None:
            raise SandboxError("utility must not use *args / **kwargs")
        if args.kwonlyargs or args.posonlyargs:
            raise SandboxError(
                "utility must use two positional args (x, c) only"
            )
        arg_names = [a.arg for a in args.args]
        if arg_names != ["x", "c"]:
            raise SandboxError(
                f"utility signature must be (x, c); got {arg_names}"
            )
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Return):
            raise SandboxError(
                "utility body must be a single return statement"
            )

    # ------------------------------------------------------------------
    def _check_call(self, node: ast.Call) -> None:
        if node.keywords:
            raise SandboxError("function calls must not use keyword args")
        if not isinstance(node.func, ast.Name):
            raise SandboxError(
                "callee must be a bare Name (no attribute chains)"
            )
        if node.func.id not in self.grammar.allowed_functions:
            raise SandboxError(
                f"call to disallowed function: {node.func.id!r}"
            )

    # ------------------------------------------------------------------
    def _check_subscript(
        self, node: ast.Subscript, coef_indices: set[int]
    ) -> None:
        if not isinstance(node.value, ast.Name):
            raise SandboxError(
                "subscript target must be a Name (x or c)"
            )
        name = node.value.id
        if name not in self.grammar.subscriptable_names:
            raise SandboxError(
                f"subscript target {name!r} not in allowlist"
            )
        index_node = self._extract_index(node)
        if not isinstance(index_node, ast.Constant):
            raise SandboxError(
                "subscript index must be a literal integer constant"
            )
        value = index_node.value
        if not isinstance(value, int) or isinstance(value, bool):
            raise SandboxError(
                f"subscript index must be an int, got {type(value).__name__}"
            )
        if value < 0:
            raise SandboxError(f"subscript index {value} must be non-negative")
        if name == "x":
            if value > self.grammar.x_max_index:
                raise SandboxError(
                    f"x[{value}] out of range (max {self.grammar.x_max_index})"
                )
        else:  # name == "c"
            if value >= self.max_coefficients:
                raise SandboxError(
                    f"c[{value}] exceeds max_coefficients={self.max_coefficients}"
                )
            coef_indices.add(value)

    @staticmethod
    def _extract_index(node: ast.Subscript) -> ast.AST:
        # Python 3.9+: ``node.slice`` is the index expression directly.
        # Python 3.8 wraps it in ``ast.Index``; unwrap if needed.
        slice_node = node.slice
        legacy_index = getattr(ast, "Index", None)
        if legacy_index is not None and isinstance(slice_node, legacy_index):
            return slice_node.value  # type: ignore[attr-defined]
        return slice_node

    # ------------------------------------------------------------------
    def _check_pow(self, node: ast.BinOp) -> None:
        right = node.right
        if not isinstance(right, ast.Constant):
            raise SandboxError(
                "** exponent must be a literal integer constant"
            )
        value = right.value
        if not isinstance(value, int) or isinstance(value, bool):
            raise SandboxError(
                f"** exponent must be an int, got {type(value).__name__}"
            )
        if value not in self.grammar.allowed_pow_ints:
            raise SandboxError(
                f"** exponent {value} not in allowlist "
                f"{sorted(self.grammar.allowed_pow_ints)}"
            )

    # ------------------------------------------------------------------
    def _check_name(self, node: ast.Name) -> None:
        # Reject obvious dunder lookups that would otherwise sneak in via
        # a bare ``__builtins__`` / ``__import__`` reference. Attribute
        # access is already banned; this closes the bare-name hole.
        if node.id.startswith("__") and node.id.endswith("__"):
            raise SandboxError(f"disallowed dunder name: {node.id!r}")


# ---------------------------------------------------------------------------
# Compile pipeline
# ---------------------------------------------------------------------------


def compile_equation(
    source: str,
    *,
    max_coefficients: int = 8,
    grammar: EquationGrammar = DEFAULT_GRAMMAR,
) -> Tuple[Callable[[np.ndarray, np.ndarray], float], int]:
    """Parse, validate, and compile ``source`` into a utility callable.

    Parameters
    ----------
    source
        Python source containing exactly one top-level ``def utility(x,
        c): return <expr>``.
    max_coefficients
        Hard cap on coefficient-vector length ``k``. Any ``c[i]`` with
        ``i >= max_coefficients`` is rejected.
    grammar
        Allowlist / denylist bundle. Defaults to :data:`DEFAULT_GRAMMAR`.

    Returns
    -------
    (utility_fn, k)
        ``utility_fn(x: np.ndarray, c: np.ndarray) -> float`` and the
        inferred coefficient-vector length. ``k`` is ``max(indices) + 1``
        so gaps (e.g. ``c[0] + c[2]`` with no ``c[1]``) still produce a
        valid BFGS problem; an unused ``c[1]`` simply gets a flat
        gradient and BFGS drifts.

    Raises
    ------
    SandboxError
        On any parse error, grammar violation, or unresolved global at
        exec time.
    """
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise SandboxError(f"syntax error in equation source: {exc}") from exc

    sandbox = SafeSandbox(grammar=grammar, max_coefficients=max_coefficients)
    coef_indices = sandbox.validate(tree)
    k = (max(coef_indices) + 1) if coef_indices else 0

    try:
        code = compile(tree, "<llm-sr>", "exec")
    except (SyntaxError, ValueError) as exc:  # pragma: no cover - AST ok, compile rarely fails
        raise SandboxError(f"compile failed after AST walk: {exc}") from exc

    # Fresh namespace. __builtins__ = {} blocks every built-in lookup
    # (print, open, __import__, ...). We then inject the five allowed
    # callables as the only resolvable globals the body can see.
    sandbox_globals: dict = {
        "__builtins__": {},
        "log1p": np.log1p,
        "exp_c": exp_c,
        "sqrt_abs": sqrt_abs,
        "tanh": tanh,
        "_safe_div": _safe_div,
    }
    sandbox_locals: dict = {}
    try:
        exec(code, sandbox_globals, sandbox_locals)  # noqa: S102 - sandbox'd
    except Exception as exc:  # noqa: BLE001 - any exec failure is fatal here
        raise SandboxError(f"exec of sandboxed code failed: {exc}") from exc

    utility = sandbox_locals.get("utility")
    if utility is None:
        raise SandboxError("no 'utility' callable produced by source")
    if not callable(utility):
        raise SandboxError("'utility' is not callable")
    return utility, int(k)


# ---------------------------------------------------------------------------
# Softmax-CE loss + BFGS fit
# ---------------------------------------------------------------------------


def _per_event_utilities(
    utility_fn: Callable[[np.ndarray, np.ndarray], float],
    feats: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """Call ``utility_fn`` over each alternative in one event.

    ``feats`` has shape ``(J, F)``; we return a length-J float64 vector.
    Runtime failures (divide-by-zero, overflow under ``np.seterr(all=
    "raise")``) propagate as ``FloatingPointError`` to the caller, which
    treats them as an ``inf`` NLL for that proposal.
    """
    J = int(feats.shape[0])
    out = np.empty(J, dtype=np.float64)
    for j in range(J):
        out[j] = float(utility_fn(feats[j], c))
    return out


def _softmax_nll_one(
    utility_fn: Callable[[np.ndarray, np.ndarray], float],
    c: np.ndarray,
    feats: np.ndarray,
    chosen: int,
) -> float:
    U = _per_event_utilities(utility_fn, feats, c)
    if not np.all(np.isfinite(U)):
        return np.inf
    U = U - U.max()
    logsumexp = np.log(np.exp(U).sum())
    return float(-(U[chosen] - logsumexp))


def _softmax_nll_batch(
    c: np.ndarray,
    utility_fn: Callable[[np.ndarray, np.ndarray], float],
    feats_list: Sequence[np.ndarray],
    chosen_list: Sequence[int],
) -> float:
    """Sum softmax-CE over a list of events.

    Returns ``np.inf`` on any non-finite utility and on
    ``FloatingPointError`` from the inner calls. The BFGS caller treats
    these as "restart failed" and moves on to the next random init.
    """
    try:
        total = 0.0
        for feats, chosen in zip(feats_list, chosen_list):
            nll = _softmax_nll_one(utility_fn, c, feats, int(chosen))
            if not np.isfinite(nll):
                return np.inf
            total += nll
        return float(total)
    except (FloatingPointError, ValueError, OverflowError):
        return np.inf


def fit_coefficients_softmax_ce(
    utility_fn: Callable[[np.ndarray, np.ndarray], float],
    feats_list: Sequence[np.ndarray],
    chosen_list: Sequence[int],
    k: int,
    *,
    max_coeffs: int = 8,
    n_restarts: int = 4,
    seed: int = 0,
    maxiter: int = 200,
    gtol: float = 1e-5,
    init_scale: float = 0.3,
) -> Tuple[Optional[np.ndarray], float]:
    """Fit coefficients via ``scipy.optimize.minimize(BFGS)`` with restarts.

    Parameters
    ----------
    utility_fn
        Callable ``(x: np.ndarray, c: np.ndarray) -> float`` from
        :func:`compile_equation`.
    feats_list
        List of ``(J, F)`` per-event feature matrices.
    chosen_list
        Parallel list of chosen-alternative indices.
    k
        Coefficient-vector length for ``utility_fn``. If ``k == 0`` we
        evaluate the constant expression once and return ``(empty, nll)``
        without running BFGS.
    max_coeffs
        Hard cap (design doc §2). Raises ValueError if ``k > max_coeffs``.
    n_restarts
        Number of random ``x0`` initialisations. ``seed`` seeds the rng
        that draws them.
    maxiter, gtol, init_scale
        Forwarded to ``scipy.optimize.minimize`` / the init draw.

    Returns
    -------
    (c_star, nll)
        Best coefficient vector across all restarts and the corresponding
        training NLL. On total failure (every restart diverged) returns
        ``(None, inf)``.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if k > max_coeffs:
        raise ValueError(f"k={k} exceeds max_coeffs={max_coeffs}")
    if n_restarts <= 0:
        raise ValueError(f"n_restarts must be positive, got {n_restarts}")

    # Preserve any ambient err state; flip to "raise" inside the fit so
    # transient overflow / div-by-zero surface as FloatingPointError and
    # get caught by the wrapper instead of returning silent NaN.
    prior = np.geterr()
    np.seterr(all="raise")
    try:
        if k == 0:
            nll = _softmax_nll_batch(
                np.zeros(0, dtype=np.float64),
                utility_fn,
                feats_list,
                chosen_list,
            )
            if np.isfinite(nll):
                return np.zeros(0, dtype=np.float64), float(nll)
            return None, float("inf")

        rng = np.random.default_rng(int(seed))
        best_c: Optional[np.ndarray] = None
        best_nll: float = float("inf")
        for restart in range(int(n_restarts)):
            x0 = rng.normal(scale=float(init_scale), size=k)
            try:
                result = minimize(
                    _softmax_nll_batch,
                    x0,
                    args=(utility_fn, feats_list, chosen_list),
                    method="BFGS",
                    jac="2-point",
                    options={"maxiter": int(maxiter), "gtol": float(gtol)},
                )
            except (FloatingPointError, ValueError, OverflowError) as exc:
                logger.debug(
                    "fit_coefficients restart %d raised %s; skipping",
                    restart,
                    type(exc).__name__,
                )
                continue
            fun = float(result.fun) if result.fun is not None else float("inf")
            if np.isfinite(fun) and fun < best_nll:
                best_nll = fun
                best_c = np.asarray(result.x, dtype=np.float64).copy()
        return best_c, best_nll
    finally:
        np.seterr(**prior)


def eval_nll_val(
    utility_fn: Callable[[np.ndarray, np.ndarray], float],
    coeffs: np.ndarray,
    feats_list: Sequence[np.ndarray],
    chosen_list: Sequence[int],
) -> float:
    """Softmax-CE on a held-out batch, or ``inf`` on any numerical fault."""
    prior = np.geterr()
    np.seterr(all="raise")
    try:
        return _softmax_nll_batch(
            np.asarray(coeffs, dtype=np.float64),
            utility_fn,
            feats_list,
            chosen_list,
        )
    finally:
        np.seterr(**prior)


# ---------------------------------------------------------------------------
# Fallback skeleton (design doc §10)
# ---------------------------------------------------------------------------

FALLBACK_SKELETON: str = (
    "def utility(x, c): return c[0]*x[0] + c[1]*x[1] + c[2]*x[2] + c[3]*x[3]"
)
"""Linear-in-4-features skeleton used by the stub / cold-start fallback.

This skeleton is grammar-legal, always finite, and recoverable by BFGS
on any real-valued feature matrix. When every LLM proposal is rejected
(the hermetic-CI stub path), the LLM-SR outer loop seeds ``memory`` with
the fit of this skeleton and returns the resulting fitted object.
"""


# ---------------------------------------------------------------------------
# LaSR additions (concept library bootstrap, AST canonicalisation,
# subexpression extraction). These symbols are imported by
# :mod:`src.baselines.lasr` and are intentionally inert from LLM-SR's
# perspective — defining them here keeps the two baselines on identical
# grammar / fit / sandbox semantics without creating an import cycle.
# ---------------------------------------------------------------------------


#: LaSR-only seed concepts (design doc §2.1). Each entry is a valid
#: ``utility(x, c)`` body expression (NOT a full function source) so the
#: baseline can splice it directly into a compiled ``def utility(x, c):
#: return <expr>`` skeleton. The ``nl_summary`` field is what the LaSR
#: proposal prompt renders under the CONCEPT LIBRARY block.
#:
#: The expressions below refer only to ``x[0..3]`` (the 4 built-in
#: feature columns) and ``c[0]`` (one free coefficient per concept).
#: They all parse cleanly through :class:`SafeSandbox` and evaluate to
#: finite floats under the safe primitives above.
SEED_CONCEPTS: Tuple[dict, ...] = (
    {
        "name": "price_sensitivity",
        "body": "-c[0] * x[0]",
        "nl_summary": "linear disutility of price",
    },
    {
        "name": "log_price_utility",
        "body": "-c[0] * log1p(x[0])",
        "nl_summary": "diminishing-returns disutility of log-price",
    },
    {
        "name": "log_popularity",
        "body": "c[0] * log1p(x[1])",
        "nl_summary": "saturating preference for popular items",
    },
    {
        "name": "price_rank_centered",
        "body": "-c[0] * (x[3] - c[0]) ** 2",
        "nl_summary": "dislikes extreme within-event price ranks",
    },
    {
        "name": "linear_log1p_price",
        "body": "c[0] * x[2]",
        "nl_summary": "linear affinity for log1p_price",
    },
    {
        "name": "linear_price_rank",
        "body": "c[0] * x[3]",
        "nl_summary": "linear affinity for within-event price rank",
    },
)
"""Starter concepts bootstrapped into the LaSR library at iter 0.

Each dict has three fields:
- ``name``: stable Python identifier used for dedup + prompt rendering.
- ``body``: the ``utility(x, c)`` return-expression for this concept.
- ``nl_summary``: one-line natural-language gloss surfaced in-prompt.

All six bodies reference exactly one free coefficient ``c[0]`` so the
LaSR outer loop can rename them into a flat coefficient vector before
BFGS — e.g. a candidate that composes two concepts ``A`` and ``B``
fits with ``k=2`` after renaming ``c[0] -> c[0]`` in ``A`` and
``c[0] -> c[1]`` in ``B``.
"""


@dataclass(frozen=True)
class SubExpr:
    """One subexpression candidate returned by :func:`extract_subexpression_candidates`.

    Attributes
    ----------
    canonical
        The canonical-form string from :func:`canonicalize` — stable
        under variable renaming and constant-index renumbering.
    depth
        AST depth of the subtree; the extractor filters to ``[2, 4]``.
    n_coeffs
        Number of distinct ``c[...]`` indices appearing in the tree.
        LaSR uses this to reject degenerate 0-coef "concepts".
    n_features
        Number of distinct ``x[...]`` indices. ``n_features == 0`` marks
        the candidate as a pure-coefficient expression (filtered out).
    source
        A round-trippable Python fragment of the candidate (via
        :func:`ast.unparse`). Useful for debug logging.
    """

    canonical: str
    depth: int
    n_coeffs: int
    n_features: int
    source: str


def canonicalize(node: ast.AST) -> str:
    """Return a stable canonical string for a candidate subtree.

    The canonical form renames ``x[i]`` → ``x[0], x[1], ...`` and
    ``c[j]`` → ``c[0], c[1], ...`` in first-seen order so two surviving
    equations that differ only by which feature / coefficient index they
    reference collapse onto the same key. Numeric literals are kept
    verbatim (they're rare in LLM-SR / LaSR proposals; the grammar only
    allows integer exponents in ``**``).

    The returned string is the :func:`ast.unparse` of the renamed tree,
    which is deterministic within a single Python version and stable
    enough for frequency counting across survivors in one run. We do
    NOT normalise commutative ops (``a+b`` vs ``b+a`` remain distinct) —
    that would require a full AC-rewriter and isn't worth the ambiguity
    cost for a threshold-of-three promotion rule.
    """
    x_remap: dict = {}
    c_remap: dict = {}
    cloned = _clone_ast(node)
    for sub in ast.walk(cloned):
        if not isinstance(sub, ast.Subscript):
            continue
        if not isinstance(sub.value, ast.Name):
            continue
        name = sub.value.id
        if name not in ("x", "c"):
            continue
        index_node = SafeSandbox._extract_index(sub)
        if not isinstance(index_node, ast.Constant):
            continue
        raw = index_node.value
        if not isinstance(raw, int) or isinstance(raw, bool):
            continue
        table = x_remap if name == "x" else c_remap
        if raw not in table:
            table[raw] = len(table)
        new_index = ast.Constant(value=table[raw])
        ast.copy_location(new_index, index_node)
        legacy_index = getattr(ast, "Index", None)
        if legacy_index is not None and isinstance(sub.slice, legacy_index):
            sub.slice.value = new_index  # type: ignore[attr-defined]
        else:
            sub.slice = new_index
    try:
        return ast.unparse(cloned)
    except AttributeError:  # pragma: no cover - Python < 3.9 fallback
        return _fallback_unparse(cloned)


def _clone_ast(node: ast.AST) -> ast.AST:
    """Return a deep copy of ``node`` safe to mutate in place.

    ``ast`` doesn't expose a shallow-clone helper; ``copy.deepcopy``
    walks fine on AST nodes and preserves the ``lineno``/``col_offset``
    attributes that :func:`ast.unparse` inspects.
    """
    import copy as _copy
    return _copy.deepcopy(node)


def _fallback_unparse(node: ast.AST) -> str:  # pragma: no cover - legacy path
    """Minimal recursive unparse for Python < 3.9 (no :func:`ast.unparse`).

    Kept as a safety net; the repo pins Python 3.10+ so this never
    fires in practice. The form is not guaranteed to round-trip through
    :func:`ast.parse` but is deterministic and sufficient as a hash key.
    """
    return ast.dump(node, annotate_fields=False)


def extract_subexpression_candidates(
    source: str,
    *,
    min_depth: int = 2,
    max_depth: int = 4,
    max_coefficients: int = 8,
) -> List[SubExpr]:
    """Walk ``source`` and return candidate sub-expressions worth promoting.

    Parameters
    ----------
    source
        Full ``def utility(x, c): return <expr>`` source of a survivor
        equation. Must already have passed :func:`compile_equation`;
        this function still re-parses for independence and defensive
        validation.
    min_depth, max_depth
        Inclusive AST depth bounds (design doc §3.1 uses ``[2, 4]``).
        Depth 1 (bare ``x[i]`` / ``c[j]``) is too trivial to promote;
        depth > 4 is too specific to recur across survivors.
    max_coefficients
        Forwarded to the sandbox pre-check so malformed sources raise
        :class:`SandboxError` before the walk.

    Returns
    -------
    list[SubExpr]
        One entry per candidate subtree that (a) sits within the depth
        window, (b) references at least one ``x[...]`` (pure-coef
        expressions are useless as concepts), and (c) references ≤
        ``max_coefficients`` distinct ``c[...]`` indices. Duplicates
        under :func:`canonicalize` are NOT collapsed here — the caller
        (promotion rule) counts frequency across a list of survivors
        and collapses at that level.
    """
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise SandboxError(f"syntax error in concept source: {exc}") from exc
    sandbox = SafeSandbox(max_coefficients=max_coefficients)
    sandbox.validate(tree)

    # Pull the single ``return`` expression out of the utility def.
    body_expr: Optional[ast.AST] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "utility":
            if node.body and isinstance(node.body[0], ast.Return):
                body_expr = node.body[0].value
            break
    if body_expr is None:
        return []

    out: List[SubExpr] = []
    for sub, depth in _walk_with_depth(body_expr):
        if depth < min_depth or depth > max_depth:
            continue
        x_idx: set = set()
        c_idx: set = set()
        for inner in ast.walk(sub):
            if (
                isinstance(inner, ast.Subscript)
                and isinstance(inner.value, ast.Name)
                and inner.value.id in ("x", "c")
            ):
                index_node = SafeSandbox._extract_index(inner)
                if isinstance(index_node, ast.Constant) and isinstance(
                    index_node.value, int
                ) and not isinstance(index_node.value, bool):
                    if inner.value.id == "x":
                        x_idx.add(index_node.value)
                    else:
                        c_idx.add(index_node.value)
        if not x_idx:
            # Pure-coefficient expression — not a useful concept.
            continue
        if len(c_idx) > max_coefficients:
            continue
        canonical = canonicalize(sub)
        try:
            raw_source = ast.unparse(sub)
        except AttributeError:  # pragma: no cover - legacy
            raw_source = _fallback_unparse(sub)
        out.append(
            SubExpr(
                canonical=canonical,
                depth=depth,
                n_coeffs=len(c_idx),
                n_features=len(x_idx),
                source=raw_source,
            )
        )
    return out


def _walk_with_depth(root: ast.AST) -> List[Tuple[ast.AST, int]]:
    """Depth-annotated DFS over ``root`` (leaves depth 1).

    Only returns compound nodes (``BinOp``, ``UnaryOp``, ``Call``) —
    leaves (bare ``Subscript`` / ``Constant`` / ``Name``) are not
    candidates for promotion since a depth-1 "subtree" can't recur
    usefully across survivor equations.

    Depth here counts *structural* nesting only: operator nodes
    (``Add``, ``Mult``, ``USub``, ...) and ``Load`` context nodes are
    skipped. So a bare ``x[0]`` has depth 1, ``-c[0]*x[0]`` has depth
    3 (mult of coef and feat under a unary minus), and
    ``-c[0]*x[0] + c[1]*log1p(x[1])`` has depth 4.
    """
    out: List[Tuple[ast.AST, int]] = []
    _STRUCTURAL = (
        ast.BinOp,
        ast.UnaryOp,
        ast.Call,
        ast.Subscript,
        ast.Constant,
        ast.Name,
    )

    def _recurse(node: ast.AST) -> int:
        child_depths = [0]
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, _STRUCTURAL):
                # Skip operator / ctx nodes so depth counts logical nesting.
                continue
            child_depths.append(_recurse(child))
        my_depth = 1 + max(child_depths)
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Call)):
            out.append((node, my_depth))
        return my_depth

    _recurse(root)
    return out


__all__ = [
    "ALLOWED_FUNCTIONS",
    "DEFAULT_GRAMMAR",
    "EquationGrammar",
    "FALLBACK_SKELETON",
    "SEED_CONCEPTS",
    "SafeSandbox",
    "SandboxError",
    "SubExpr",
    "X_MAX_INDEX",
    "_safe_div",
    "canonicalize",
    "compile_equation",
    "eval_nll_val",
    "exp_c",
    "extract_subexpression_candidates",
    "fit_coefficients_softmax_ce",
    "sqrt_abs",
    "tanh",
]
