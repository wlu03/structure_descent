"""LaSR (Library-augmented Symbolic Regression) baseline for DCM.

Full design: ``docs/llm_baselines/lasr_baseline.md``. LaSR is a sibling
of :mod:`src.baselines.llm_sr` — both share the equation grammar,
sandbox, coefficient fitter, and softmax-CE evaluator in
:mod:`src.baselines._symbolic_regression_common`. The difference is
that LaSR maintains a **concept library** (reusable named
sub-expressions) that is re-expanded into every proposal prompt.

Summary of the outer loop (design doc §3, §6):

1.  Bootstrap a concept library ``L`` from :data:`SEED_CONCEPTS`.
2.  For each iteration ``t``:
    a. Render a proposal prompt showing ``L`` (name + NL gloss + full
       ``def``) and the top-K equation memory. Call the LLM for
       ``proposals_per_iter`` proposals.
    b. For each proposal: parse, compile via the shared sandbox, BFGS-
       fit coefficients on train, score val-NLL.
    c. Pick the top-K survivors by val-NLL.
    d. Scan survivor ASTs for candidate sub-expressions, canonicalise,
       and promote any pattern seen in ≥ ``concept_promotion_threshold``
       survivors (or explicitly LLM-nominated with ≥ 1 occurrence).
    e. Evict library entries under the LRU+TTL rule.
3.  Return the best-val equation + the final library in
    :class:`LaSRFitted`.

If every LLM call is rejected (hermetic-CI stub path), the loop falls
back to :data:`FALLBACK_SKELETON` so the fitted object still holds a
valid skeleton. Registry wiring is identical to LLM-SR — the
``("LaSR", ...)`` entry in :data:`src.baselines.run_all._LLM_BASELINE_BASES`
auto-expands across :data:`LLM_MODEL_SWEEP`.
"""

from __future__ import annotations

import ast
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.outcomes.generate import LLMClient, StubLLMClient

from ._symbolic_regression_common import (
    FALLBACK_SKELETON,
    SEED_CONCEPTS,
    SandboxError,
    canonicalize,
    compile_equation,
    eval_nll_val,
    extract_subexpression_candidates,
    fit_coefficients_softmax_ce,
)
from .base import BaselineEventBatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Concept dataclass and library
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Concept:
    """One entry in the LaSR concept library (design doc §2.1).

    Attributes
    ----------
    name
        Python identifier used for dedup and prompt rendering. Also
        used as the library key — two concepts with the same name are
        collapsed on promotion.
    source
        The ``utility(x, c)`` body expression (NOT a full function def)
        that this concept introduces. Parses cleanly through
        :class:`SafeSandbox`; the sandbox's check on the full proposal
        enforces the same property transitively.
    nl_summary
        One-line natural-language gloss surfaced under the CONCEPT
        LIBRARY header of the proposal prompt.
    usage_count
        Number of top-K survivors referencing this concept's canonical
        form in the most recent iteration. Drives LRU eviction.
    discovered_at
        Iteration index when the concept was promoted. Tie-breaker for
        eviction when two concepts share ``usage_count``.
    n_coeffs
        Count of distinct ``c[...]`` indices in the body. Used to
        flag trivial ``c[0]``-only "identity" concepts for the
        collapse-guard.
    """

    name: str
    source: str
    nl_summary: str
    usage_count: int = 0
    discovered_at: int = 0
    n_coeffs: int = 0


class ConceptLibrary:
    """LaSR concept store with cap, LRU-of-usefulness, and TTL eviction.

    The library is keyed by ``Concept.name``. Two promotion paths feed
    into :meth:`add`:

    * **Frequency promotion**: a canonicalised sub-expression seen in
      ≥ ``promotion_threshold`` top-K survivors this iteration.
    * **LLM nomination**: the proposer explicitly names a concept
      (prefixed ``NOMINATE:`` in the raw reply, parsed by the outer
      loop) that appears in ≥ 1 survivor.

    Eviction is applied on every :meth:`tick`:

    * **Cap**: when ``len(self) > max_size`` evict the lowest-
      ``usage_count`` entry, tie-broken by oldest ``discovered_at``.
    * **TTL**: any entry with ``usage_count == 0`` for
      ``ttl`` consecutive ticks is evicted regardless of cap.
    """

    def __init__(
        self,
        *,
        max_size: int = 20,
        ttl: int = 5,
        promotion_threshold: int = 3,
    ) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        if ttl <= 0:
            raise ValueError(f"ttl must be positive, got {ttl}")
        if promotion_threshold <= 0:
            raise ValueError(
                f"promotion_threshold must be positive, got {promotion_threshold}"
            )
        self.max_size = int(max_size)
        self.ttl = int(ttl)
        self.promotion_threshold = int(promotion_threshold)
        self._by_name: Dict[str, Concept] = {}
        # Per-concept zero-usage streak counter for TTL eviction.
        self._zero_streak: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __iter__(self):
        return iter(self._by_name.values())

    def as_list(self) -> List[Concept]:
        """Return a stable-ordered snapshot of current concepts."""
        return list(self._by_name.values())

    def get(self, name: str) -> Optional[Concept]:
        return self._by_name.get(name)

    # ------------------------------------------------------------------
    def seed(self, concepts: Sequence[Concept]) -> None:
        """Bootstrap the library with ``concepts`` (iteration-0 seed).

        Overwrites any existing entries with matching names. Does NOT
        enforce the cap — the caller is expected to pass ≤ ``max_size``
        seed concepts (``SEED_CONCEPTS`` has 6, well under the default
        20 cap).
        """
        for c in concepts:
            self._by_name[c.name] = c
            self._zero_streak[c.name] = 0

    # ------------------------------------------------------------------
    def add(self, concept: Concept) -> None:
        """Insert or upsert ``concept`` by name.

        No capacity check here — :meth:`tick` applies the cap + TTL
        rules in one pass so an iteration's full promotion batch is
        visible to the eviction logic.
        """
        self._by_name[concept.name] = concept
        self._zero_streak.setdefault(concept.name, 0)

    # ------------------------------------------------------------------
    def remove(self, name: str) -> None:
        """Explicitly evict ``name``. No-op if absent."""
        self._by_name.pop(name, None)
        self._zero_streak.pop(name, None)

    # ------------------------------------------------------------------
    def update_usage(self, usage: Dict[str, int]) -> None:
        """Write through fresh usage counts for this iteration.

        Concepts not present in ``usage`` get a 0. The zero-streak
        counter is incremented for every concept with 0 usage and
        reset otherwise.
        """
        for name, concept in list(self._by_name.items()):
            new_count = int(usage.get(name, 0))
            self._by_name[name] = Concept(
                name=concept.name,
                source=concept.source,
                nl_summary=concept.nl_summary,
                usage_count=new_count,
                discovered_at=concept.discovered_at,
                n_coeffs=concept.n_coeffs,
            )
            if new_count == 0:
                self._zero_streak[name] = self._zero_streak.get(name, 0) + 1
            else:
                self._zero_streak[name] = 0

    # ------------------------------------------------------------------
    def tick(self, *, protect: Sequence[str] = ()) -> List[str]:
        """Apply TTL and cap eviction, returning the list of evicted names.

        ``protect`` pins a set of concept names that must not be evicted
        this tick (used to retain the initial seeds for at least one
        iteration so the LLM always has something to compose with).
        """
        protect_set = set(protect)
        evicted: List[str] = []

        # Pass 1: TTL — drop concepts with zero usage for ``ttl`` ticks.
        for name, streak in list(self._zero_streak.items()):
            if name in protect_set:
                continue
            if streak >= self.ttl and name in self._by_name:
                evicted.append(name)
                self.remove(name)

        # Pass 2: cap — while over capacity, evict lowest usage (oldest first).
        while len(self._by_name) > self.max_size:
            candidates = [
                (c.usage_count, c.discovered_at, c.name)
                for c in self._by_name.values()
                if c.name not in protect_set
            ]
            if not candidates:
                break
            candidates.sort()
            name = candidates[0][2]
            evicted.append(name)
            self.remove(name)

        return evicted


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


SYSTEM_PROMPT: str = (
    "You are a symbolic-regression agent for discrete choice modelling. Your\n"
    "job is to propose Python utility functions that predict which of J=4\n"
    "product alternatives a shopper will buy. A softmax over your utility\n"
    "gives the choice probability; lower negative log-likelihood is better.\n\n"
    "You are given a library of named CONCEPTS (sub-expressions with free\n"
    "coefficients, each accompanied by a natural-language gloss) plus a\n"
    "memory of the best-so-far utility functions. Prefer COMPOSING library\n"
    "concepts over writing raw arithmetic — that is how you build DCM-\n"
    "canonical structure. You may still introduce new sub-expressions if\n"
    "they might fit better than anything in the library.\n\n"
    "Rules:\n"
    "1. Output exactly one Python function with signature\n"
    "      def utility(x, c): return <expression>\n"
    "   where `x` is a length-4 per-alternative feature vector\n"
    "   (x[0]=price, x[1]=popularity_rank, x[2]=log1p_price, x[3]=price_rank)\n"
    "   and `c` is a coefficient array of length at most 8.\n"
    "2. Use only these operators: +, -, *, /, ** (integer exponent <= 2).\n"
    "3. Use only these functions: log1p, exp_c (clipped exp), sqrt_abs, tanh.\n"
    "4. You may inline any concept body from the library verbatim; rename\n"
    "   its coefficients to fresh ``c[k]`` slots as needed.\n"
    "5. No conditionals, no loops, no attribute access, no imports. A\n"
    "   single return statement only.\n"
    "6. Return ONLY the function in a ```python``` code block. To\n"
    "   explicitly nominate a concept for promotion, prefix a line before\n"
    "   the code block with ``NOMINATE: <concept_name>``.\n\n"
    "Feature semantics (for reasoning, not code):\n"
    "- price: raw dollar price, roughly $5-$500. Shoppers are price-sensitive.\n"
    "- popularity_rank: higher = more popular; popularity coefficient is\n"
    "  usually positive.\n"
    "- log1p_price: log1p of price, monotone in price.\n"
    "- price_rank: within-event dense rank of price in [0, 1]; 0 = cheapest."
)

USER_TEMPLATE: str = (
    "Iteration {t} of {T_max}.\n\n"
    "CONCEPT LIBRARY (reuse and compose these):\n"
    "{library_block}\n\n"
    "BEST EQUATIONS SO FAR (val NLL; lower is better):\n"
    "{memory_block}\n\n"
    "Propose ONE new utility function, structurally different from those "
    "above. Prefer composing library concepts. Output only the\n"
    "```python``` code block (optionally preceded by a single "
    "``NOMINATE: <name>`` line).\n"
)


def _render_library_block(library: ConceptLibrary) -> str:
    """Render the library into the CONCEPT LIBRARY prompt block."""
    if len(library) == 0:
        return "(library empty — propose raw-feature equations)"
    lines: List[str] = []
    for concept in sorted(
        library.as_list(), key=lambda c: (-c.usage_count, c.name)
    ):
        lines.append(
            f"- {concept.name}: \"{concept.nl_summary}\"  "
            f"[usage={concept.usage_count}]"
        )
        lines.append(f"    body: {concept.source}")
    return "\n".join(lines)


def _render_memory_block(memory: Sequence["_SurvivorRecord"]) -> str:
    """Render the top-K survivors for the BEST EQUATIONS SO FAR block."""
    if not memory:
        return "(no fitted equations yet)"
    lines: List[str] = []
    for i, rec in enumerate(memory):
        lines.append(
            f"[{i + 1}] val_nll={rec.val_nll:.4f}  "
            f"train_nll={rec.train_nll:.4f}  k={rec.k}"
        )
        lines.append(rec.source.strip())
        if i < len(memory) - 1:
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Survivor record + LLM plumbing
# ---------------------------------------------------------------------------


@dataclass
class _SurvivorRecord:
    """One entry in the equation memory (top-K survivor stack)."""

    source: str
    fn: Callable[[np.ndarray, np.ndarray], float] = field(repr=False)
    coeffs: np.ndarray
    k: int
    train_nll: float
    val_nll: float

    @property
    def combined_nll(self) -> float:
        return float(self.train_nll) + float(self.val_nll)


_CODE_FENCE_RE = re.compile(
    r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE
)
_DEF_LINE_RE = re.compile(r"^\s*def\s+utility\s*\(", re.MULTILINE)
_NOMINATE_RE = re.compile(
    r"^\s*NOMINATE\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*$", re.MULTILINE
)


def _extract_equation_source(text: str) -> Optional[str]:
    """Pull a ``def utility(...)`` block out of a model response."""
    if not text:
        return None
    for m in _CODE_FENCE_RE.finditer(text):
        body = m.group(1)
        if _DEF_LINE_RE.search(body):
            return body.strip()
    m = _DEF_LINE_RE.search(text)
    if m is None:
        return None
    return text[m.start():].strip()


def _extract_nominations(text: str) -> List[str]:
    """Pull explicit ``NOMINATE: <name>`` tokens from the response."""
    if not text:
        return []
    return [m.group(1) for m in _NOMINATE_RE.finditer(text)]


def _response_as_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(response, "content", None)
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", "") or "")
        if parts:
            return "".join(parts)
    return str(response)


def _is_anthropic_client(client: LLMClient) -> bool:
    if getattr(client, "_is_stub", False):
        return False
    inner = getattr(client, "_client", None)
    if inner is None:
        return False
    messages = getattr(inner, "messages", None)
    if messages is None:
        return False
    return hasattr(messages, "create")


def _call_llm_for_proposal(
    client: LLMClient,
    system: str,
    user: str,
    *,
    temperature: float,
    max_tokens: int,
    seed: int,
    model_id: Optional[str],
) -> str:
    """Issue one proposal request, honouring the Anthropic cache pattern.

    Mirrors :func:`src.baselines.llm_sr._call_llm_for_proposal` but with
    LaSR's SYSTEM_PROMPT. We mark the system block ``cache_control:
    ephemeral`` so the 200-plus tokens of instructions + intuition land
    in Anthropic's server-side cache; every subsequent call in a single
    customer's fit cache-reads instead of cache-writing.
    """
    if getattr(client, "_is_stub", False):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        result = client.generate(
            messages,
            temperature=float(temperature),
            top_p=1.0,
            max_tokens=int(max_tokens),
            seed=int(seed),
        )
        return _response_as_text(result)
    if _is_anthropic_client(client):
        inner = getattr(client, "_client", None)
        resolved_model = (
            model_id or getattr(client, "model_id", None) or "claude-sonnet-4-6"
        )
        try:
            response = inner.messages.create(
                model=resolved_model,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user}],
                max_tokens=int(max_tokens),
                temperature=float(temperature),
            )
        except Exception as exc:  # noqa: BLE001 - SDK raises varied errors
            logger.warning(
                "LaSR Anthropic proposal call failed (%s); returning empty text.",
                exc,
            )
            return ""
        return _response_as_text(response)
    # Generic client: plain ``generate``.
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    result = client.generate(
        messages,
        temperature=float(temperature),
        top_p=1.0,
        max_tokens=int(max_tokens),
        seed=int(seed),
    )
    return _response_as_text(result)


# ---------------------------------------------------------------------------
# Fitted wrapper
# ---------------------------------------------------------------------------


@dataclass
class LaSRFitted:
    """Fitted LaSR baseline exposing the :class:`FittedBaseline` protocol.

    Attributes
    ----------
    best_equation
        Source of the survivor with the lowest val-NLL across all
        iterations.
    best_coefficients
        Fitted coefficient vector for ``best_equation``.
    final_concept_library
        Snapshot of the library at the end of the last iteration —
        exported so a subsequent customer's fit can prime
        :meth:`LaSR.fit` with the transferred library.
    equation_memory
        Top-K survivor ``(source, val_nll)`` pairs at the final
        iteration. Useful for audit / interpretability plots.
    """

    name: str = "LaSR"
    best_equation: str = ""
    best_coefficients: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_coefficients: int = 0
    train_nll: float = float("inf")
    val_nll: float = float("inf")
    model_id: str = "unknown"
    n_proposals_accepted: int = 0
    n_proposals_total: int = 0
    prompt_version: str = "lasr-v1"
    final_concept_library: List[Concept] = field(default_factory=list)
    equation_memory: List[Tuple[str, float]] = field(default_factory=list)
    _utility_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = field(
        default=None, repr=False
    )

    # ------------------------------------------------------------------
    def score_events(self, batch: BaselineEventBatch) -> List[np.ndarray]:
        """Return per-event utility vectors of shape ``(J,)``.

        Identical contract to :meth:`LLMSRFitted.score_events`: we read
        only the first 4 columns of each event's feature matrix (the
        built-in pool from ``BUILTIN_FEATURE_NAMES``).
        """
        if self._utility_fn is None:
            fn, _ = compile_equation(
                self.best_equation,
                max_coefficients=max(1, self.n_coefficients or 1),
            )
            self._utility_fn = fn
        fn = self._utility_fn
        coeffs = np.asarray(self.best_coefficients, dtype=np.float64)
        out: List[np.ndarray] = []
        for feats in batch.base_features_list:
            feats_f = np.asarray(feats, dtype=np.float64)
            feats_slice = feats_f[:, :4]
            J = int(feats_slice.shape[0])
            U = np.empty(J, dtype=np.float64)
            for j in range(J):
                try:
                    U[j] = float(fn(feats_slice[j], coeffs))
                except Exception:  # noqa: BLE001
                    U[j] = 0.0
                if not np.isfinite(U[j]):
                    U[j] = 0.0
            out.append(U)
        return out

    @property
    def n_params(self) -> int:
        return int(self.n_coefficients)

    @property
    def description(self) -> str:
        return (
            f"LaSR model={self.model_id} |L|={len(self.final_concept_library)} "
            f"k={self.n_coefficients} val_nll={self.val_nll:.4f} "
            f"accepted={self.n_proposals_accepted}/{self.n_proposals_total}"
        )


# ---------------------------------------------------------------------------
# Baseline (fit-time) class
# ---------------------------------------------------------------------------


def _seed_concepts_to_library() -> List[Concept]:
    """Materialise :data:`SEED_CONCEPTS` into :class:`Concept` instances."""
    return [
        Concept(
            name=entry["name"],
            source=entry["body"],
            nl_summary=entry["nl_summary"],
            usage_count=0,
            discovered_at=0,
            n_coeffs=entry["body"].count("c["),
        )
        for entry in SEED_CONCEPTS
    ]


def _canonicalise_concept(concept: Concept) -> Optional[str]:
    """Best-effort canonical form of a concept's body. ``None`` on failure."""
    try:
        tree = ast.parse(concept.source, mode="eval")
    except SyntaxError:
        return None
    return canonicalize(tree.body)


def _is_trivial_concept(concept: Concept) -> bool:
    """Filter for the collapse-guard (design doc §11).

    A concept with no feature reference or one that canonicalises to a
    bare ``c[0] * x[0]`` (the "linear" identity) is useless as an
    abstraction and should never be promoted.
    """
    canonical = _canonicalise_concept(concept)
    if canonical is None:
        return True
    if "x[" not in canonical:
        return True
    stripped = canonical.replace(" ", "")
    if stripped in ("c[0]*x[0]", "c[0]*x[0]+0", "0+c[0]*x[0]"):
        return True
    return False


class LaSR:
    """LaSR-for-DCM baseline. Implements the :class:`Baseline` protocol.

    Parameters
    ----------
    llm_client
        Any ``LLMClient`` from :mod:`src.outcomes.generate`. Defaults
        to :class:`StubLLMClient` for hermetic CI.
    n_iters
        Number of outer iterations (design doc §3 defaults to 10).
    proposals_per_iter
        Number of LLM calls per iteration (design doc §3 defaults to
        10; total calls ≈ ``n_iters * proposals_per_iter``).
    top_k_memory
        Number of top-val-NLL survivors held in the rolling equation
        memory and shown to the next proposal prompt.
    concept_library_max_size
        Hard cap on the concept library (design doc §2.3 defaults to
        20).
    concept_promotion_threshold
        Minimum frequency (over the top-K survivors this iteration)
        required to auto-promote a canonicalised sub-expression. LLM
        nominations bypass this threshold but still need ≥ 1
        occurrence.
    concept_ttl
        Number of consecutive zero-usage iterations before a concept
        is evicted regardless of cap (design doc §2.3 defaults to 5).
    max_coefficients
        Grammar cap on ``k`` — matches LLM-SR.
    n_restarts, max_retries_per_proposal, seed, temperature, max_tokens,
        prompt_version
        Standard LLM-SR knobs; see :class:`LLMSR`.
    transferred_library
        Optional list of :class:`Concept` entries to splice into the
        library at the start of the fit — the mechanism behind the
        "library transfer" ablation in design doc §5. Concepts added
        here are exempt from TTL eviction for the first iteration.
    """

    name: str = "LaSR"

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        *,
        n_iters: int = 10,
        proposals_per_iter: int = 10,
        top_k_memory: int = 10,
        concept_library_max_size: int = 20,
        concept_promotion_threshold: int = 3,
        concept_ttl: int = 5,
        max_coefficients: int = 8,
        n_restarts: int = 4,
        max_retries_per_proposal: int = 2,
        seed: int = 0,
        prompt_version: str = "lasr-v1",
        temperature: float = 0.7,
        max_tokens: int = 512,
        transferred_library: Optional[Sequence[Concept]] = None,
    ) -> None:
        if n_iters < 0:
            raise ValueError(f"n_iters must be >= 0, got {n_iters}")
        if proposals_per_iter <= 0:
            raise ValueError(
                f"proposals_per_iter must be positive, got {proposals_per_iter}"
            )
        if top_k_memory <= 0:
            raise ValueError(f"top_k_memory must be positive, got {top_k_memory}")
        if concept_library_max_size <= 0:
            raise ValueError(
                "concept_library_max_size must be positive, "
                f"got {concept_library_max_size}"
            )
        if concept_promotion_threshold <= 0:
            raise ValueError(
                "concept_promotion_threshold must be positive, "
                f"got {concept_promotion_threshold}"
            )
        if concept_ttl <= 0:
            raise ValueError(f"concept_ttl must be positive, got {concept_ttl}")
        if max_coefficients <= 0:
            raise ValueError(
                f"max_coefficients must be positive, got {max_coefficients}"
            )
        if n_restarts <= 0:
            raise ValueError(f"n_restarts must be positive, got {n_restarts}")
        if max_retries_per_proposal < 0:
            raise ValueError(
                "max_retries_per_proposal must be >= 0, "
                f"got {max_retries_per_proposal}"
            )

        self.llm_client: LLMClient = (
            llm_client if llm_client is not None else StubLLMClient()
        )
        self.n_iters = int(n_iters)
        self.proposals_per_iter = int(proposals_per_iter)
        self.top_k_memory = int(top_k_memory)
        self.concept_library_max_size = int(concept_library_max_size)
        self.concept_promotion_threshold = int(concept_promotion_threshold)
        self.concept_ttl = int(concept_ttl)
        self.max_coefficients = int(max_coefficients)
        self.n_restarts = int(n_restarts)
        self.max_retries_per_proposal = int(max_retries_per_proposal)
        self.seed = int(seed)
        self.prompt_version = str(prompt_version)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.transferred_library: List[Concept] = list(transferred_library or [])

    # ------------------------------------------------------------------
    def fit(
        self,
        train: BaselineEventBatch,
        val: BaselineEventBatch,
    ) -> LaSRFitted:
        """Outer loop per design doc §3/§6.

        The fallback-seeded memory path mirrors LLM-SR's §10 contract:
        if every LLM proposal is rejected (or no proposals are made),
        the fitted object still holds :data:`FALLBACK_SKELETON`.
        """
        self._validate_batches(train, val)
        train_feats = self._slice_first4(train)
        train_chosen = list(train.chosen_indices)
        val_feats = self._slice_first4(val)
        val_chosen = list(val.chosen_indices)

        # Bootstrap the library: seed concepts + any transferred ones.
        library = ConceptLibrary(
            max_size=self.concept_library_max_size,
            ttl=self.concept_ttl,
            promotion_threshold=self.concept_promotion_threshold,
        )
        library.seed(_seed_concepts_to_library())
        for c in self.transferred_library:
            library.add(c)

        # Seed equation memory with the fallback skeleton so the loop
        # has a non-empty best-so-far for the first proposal prompt.
        memory: List[_SurvivorRecord] = []
        total_proposals = 0
        accepted_proposals = 0
        fallback = self._fit_one(
            FALLBACK_SKELETON,
            train_feats,
            train_chosen,
            val_feats,
            val_chosen,
            seed=self.seed,
        )
        if fallback is not None:
            memory.append(fallback)

        all_survivors: List[_SurvivorRecord] = list(memory)

        for t in range(self.n_iters):
            top_k = sorted(memory, key=lambda r: r.val_nll)[: self.top_k_memory]
            user_prompt = USER_TEMPLATE.format(
                t=t + 1,
                T_max=self.n_iters,
                library_block=_render_library_block(library),
                memory_block=_render_memory_block(top_k),
            )
            iter_survivors: List[_SurvivorRecord] = []
            iter_nominations: List[str] = []

            for p in range(self.proposals_per_iter):
                record: Optional[_SurvivorRecord] = None
                attempts = self.max_retries_per_proposal + 1
                for attempt in range(attempts):
                    total_proposals += 1
                    call_seed = (
                        self.seed
                        + t * self.proposals_per_iter * attempts
                        + p * attempts
                        + attempt
                    )
                    try:
                        raw = _call_llm_for_proposal(
                            self.llm_client,
                            SYSTEM_PROMPT,
                            user_prompt,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            seed=call_seed,
                            model_id=getattr(self.llm_client, "model_id", None),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "LaSR proposal t=%d p=%d attempt=%d call failed: %s",
                            t, p, attempt, exc,
                        )
                        continue
                    iter_nominations.extend(_extract_nominations(raw))
                    source = _extract_equation_source(raw)
                    if source is None:
                        continue
                    record = self._fit_one(
                        source,
                        train_feats,
                        train_chosen,
                        val_feats,
                        val_chosen,
                        seed=call_seed,
                    )
                    if record is not None:
                        accepted_proposals += 1
                        break
                if record is not None:
                    iter_survivors.append(record)

            memory.extend(iter_survivors)
            memory.sort(key=lambda r: r.val_nll)
            memory = memory[: max(self.top_k_memory * 3, self.top_k_memory)]
            all_survivors.extend(iter_survivors)

            # Top-K survivors drive frequency-promotion and usage counts.
            top_k_survivors = sorted(
                iter_survivors + (memory[: self.top_k_memory]),
                key=lambda r: r.val_nll,
            )[: self.top_k_memory]

            self._update_library(
                library,
                top_k_survivors,
                nominations=iter_nominations,
                iter_index=t + 1,
                protect=(
                    set(c.name for c in self.transferred_library) if t == 0 else set()
                ),
            )

        # Pick the best across all fitted survivors.
        best = min(all_survivors, key=lambda r: r.val_nll) if all_survivors else None
        model_id = str(getattr(self.llm_client, "model_id", "unknown") or "unknown")

        if best is None:
            # Even the fallback failed -- return a zero-weighted object.
            logger.error(
                "LaSR.fit produced no viable equation; returning zero fitted object."
            )
            return LaSRFitted(
                name=self.name,
                best_equation=FALLBACK_SKELETON,
                best_coefficients=np.zeros(4, dtype=np.float64),
                n_coefficients=4,
                train_nll=float("inf"),
                val_nll=float("inf"),
                model_id=model_id,
                n_proposals_accepted=accepted_proposals,
                n_proposals_total=total_proposals,
                prompt_version=self.prompt_version,
                final_concept_library=library.as_list(),
                equation_memory=[],
            )
        return LaSRFitted(
            name=self.name,
            best_equation=best.source,
            best_coefficients=np.asarray(best.coeffs, dtype=np.float64),
            n_coefficients=int(best.k),
            train_nll=float(best.train_nll),
            val_nll=float(best.val_nll),
            model_id=model_id,
            n_proposals_accepted=accepted_proposals,
            n_proposals_total=total_proposals,
            prompt_version=self.prompt_version,
            final_concept_library=library.as_list(),
            equation_memory=[
                (r.source, float(r.val_nll))
                for r in sorted(memory, key=lambda r: r.val_nll)[: self.top_k_memory]
            ],
            _utility_fn=best.fn,
        )

    # ------------------------------------------------------------------
    # Library-update internals
    # ------------------------------------------------------------------
    def _update_library(
        self,
        library: ConceptLibrary,
        survivors: Sequence[_SurvivorRecord],
        *,
        nominations: Sequence[str],
        iter_index: int,
        protect: set,
    ) -> None:
        """Promote new concepts and apply eviction for one iteration.

        Steps:
        1. Scan survivor ASTs via :func:`extract_subexpression_candidates`
           and count canonical forms.
        2. Compute per-concept usage (existing library ∩ survivor
           canonical forms) and update the library's usage counts.
        3. Promote any canonical form with frequency ≥ threshold as a
           new concept (auto-named ``concept_{iter}_{i}``).
        4. Honour explicit LLM nominations that resolve to an existing
           library name and are backed by ≥ 1 survivor occurrence:
           refresh the usage count so TTL doesn't evict them.
        5. Apply :meth:`ConceptLibrary.tick` to enforce cap + TTL.
        """
        canonical_counts: Counter = Counter()
        per_survivor_canonicals: List[set] = []
        for rec in survivors:
            try:
                subs = extract_subexpression_candidates(
                    rec.source, max_coefficients=self.max_coefficients
                )
            except SandboxError:
                per_survivor_canonicals.append(set())
                continue
            canonicals = {s.canonical for s in subs}
            per_survivor_canonicals.append(canonicals)
            for c in canonicals:
                canonical_counts[c] += 1

        # Usage refresh for existing concepts: a concept "counts as used"
        # by a survivor if its canonical body appears in that survivor's
        # canonical set.
        existing_canonicals: Dict[str, str] = {}
        for concept in library.as_list():
            canonical = _canonicalise_concept(concept)
            if canonical is not None:
                existing_canonicals[concept.name] = canonical
        usage: Dict[str, int] = {name: 0 for name in existing_canonicals}
        for canonicals in per_survivor_canonicals:
            for name, canon in existing_canonicals.items():
                if canon in canonicals:
                    usage[name] += 1
        # LLM nominations that reference an extant concept refresh usage.
        for nom in nominations:
            if nom in usage and canonical_counts.get(
                existing_canonicals.get(nom, ""), 0
            ) >= 1:
                usage[nom] = max(usage[nom], 1)
        library.update_usage(usage)

        # Frequency promotion: any canonical form with ≥ threshold hits
        # that is NOT already represented by an existing concept.
        already_known = set(existing_canonicals.values())
        promoted = 0
        for canonical, freq in sorted(
            canonical_counts.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            if freq < self.concept_promotion_threshold:
                continue
            if canonical in already_known:
                continue
            new_name = f"concept_{iter_index}_{promoted}"
            n_coef = canonical.count("c[")
            new_concept = Concept(
                name=new_name,
                source=canonical,
                nl_summary=(
                    f"auto-promoted from {freq} survivors at iter {iter_index}"
                ),
                usage_count=int(freq),
                discovered_at=int(iter_index),
                n_coeffs=int(n_coef),
            )
            if _is_trivial_concept(new_concept):
                continue
            library.add(new_concept)
            already_known.add(canonical)
            promoted += 1

        # LLM nomination promotion: a NOMINATE that does NOT already
        # resolve to a library name *and* whose canonical form matches
        # at least one survivor sub-expression is admitted at freq ≥ 1.
        # The model named the concept but not its body; we walk the
        # canonical candidates in descending-frequency order and take
        # the first one that passes the collapse-guard and isn't
        # already represented in the library.
        ranked_candidates = sorted(
            canonical_counts.items(), key=lambda kv: (-kv[1], kv[0])
        )
        for nom in nominations:
            if nom in library:
                continue
            chosen: Optional[Tuple[str, int]] = None
            for canonical, freq in ranked_candidates:
                if freq < 1 or canonical in already_known:
                    continue
                tentative = Concept(
                    name=nom,
                    source=canonical,
                    nl_summary=f"LLM-nominated at iter {iter_index}",
                    usage_count=int(freq),
                    discovered_at=int(iter_index),
                    n_coeffs=int(canonical.count("c[")),
                )
                if _is_trivial_concept(tentative):
                    continue
                chosen = (canonical, freq)
                break
            if chosen is None:
                continue
            canonical, freq = chosen
            library.add(
                Concept(
                    name=nom,
                    source=canonical,
                    nl_summary=f"LLM-nominated at iter {iter_index}",
                    usage_count=int(freq),
                    discovered_at=int(iter_index),
                    n_coeffs=int(canonical.count("c[")),
                )
            )
            already_known.add(canonical)

        library.tick(protect=protect)

    # ------------------------------------------------------------------
    def _fit_one(
        self,
        source: str,
        train_feats: Sequence[np.ndarray],
        train_chosen: Sequence[int],
        val_feats: Sequence[np.ndarray],
        val_chosen: Sequence[int],
        *,
        seed: int,
    ) -> Optional[_SurvivorRecord]:
        """Compile, fit, score one proposal. Returns ``None`` on rejection."""
        try:
            fn, k = compile_equation(source, max_coefficients=self.max_coefficients)
        except SandboxError as exc:
            logger.debug("LaSR proposal rejected by sandbox: %s", exc)
            return None
        if k > self.max_coefficients:
            return None
        coeffs, train_nll = fit_coefficients_softmax_ce(
            fn,
            train_feats,
            train_chosen,
            k=k,
            max_coeffs=self.max_coefficients,
            n_restarts=self.n_restarts,
            seed=int(seed),
        )
        if coeffs is None or not np.isfinite(train_nll):
            logger.debug("LaSR proposal diverged on train")
            return None
        val_nll = eval_nll_val(fn, coeffs, val_feats, val_chosen)
        if not np.isfinite(val_nll):
            logger.debug("LaSR proposal diverged on val")
            return None
        return _SurvivorRecord(
            source=source.strip(),
            fn=fn,
            coeffs=coeffs,
            k=int(k),
            train_nll=float(train_nll),
            val_nll=float(val_nll),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _slice_first4(batch: BaselineEventBatch) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for feats in batch.base_features_list:
            arr = np.asarray(feats, dtype=np.float64)
            if arr.shape[1] < 4:
                raise ValueError(
                    f"LaSR requires >=4 feature columns; got {arr.shape[1]}"
                )
            out.append(arr[:, :4].copy())
        return out

    @staticmethod
    def _validate_batches(
        train: BaselineEventBatch, val: BaselineEventBatch
    ) -> None:
        if train.n_events == 0:
            raise ValueError("LaSR.fit received an empty train batch")
        if val.n_events == 0:
            raise ValueError("LaSR.fit received an empty val batch")


__all__ = [
    "Concept",
    "ConceptLibrary",
    "LaSR",
    "LaSRFitted",
    "SYSTEM_PROMPT",
    "USER_TEMPLATE",
]
