"""LLM-SR-for-DCM baseline.

Full design: ``docs/llm_baselines/llm_sr_baseline.md``.

Summary
-------
An LLM proposes candidate *equation skeletons* — Python expressions over
the 4 per-alternative features ``x = (price, popularity_rank,
log1p_price, price_rank)`` with placeholder coefficients ``c[0..k]`` — a
BFGS inner loop fits the coefficients against the softmax-CE discrete-
choice loss, and subsequent proposals are conditioned on an "experience
buffer" of the top-K fitted skeletons and their NLL. See the design doc
for the BNF grammar (§2), the AST sandbox allowlist (§4), and the outer
loop (§6).

The heavy lifting (grammar, sandbox, BFGS fitter, safe primitives) lives
in :mod:`src.baselines._symbolic_regression_common` so the forthcoming
LaSR baseline can share it verbatim — the two baselines MUST use
identical equation semantics so cross-method NLL deltas are attributable
to proposer design alone.

Registry wiring (automatic): the entry
``("LLM-SR", "src.baselines.llm_sr", "LLMSR")`` lives in
``run_all.py:_LLM_BASELINE_BASES`` and is auto-expanded across
``LLM_MODEL_SWEEP`` into ``LLM-SR-Claude-Sonnet-4.6``,
``LLM-SR-Opus-4.6``, etc.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.outcomes.generate import LLMClient, StubLLMClient

from ._symbolic_regression_common import (
    FALLBACK_SKELETON,
    SandboxError,
    compile_equation,
    eval_nll_val,
    fit_coefficients_softmax_ce,
)
from .base import BaselineEventBatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt strings (design doc §3.1, §3.2)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are an equation-discovery agent for discrete choice modelling. Your job\n"
    "is to propose Python functions that predict which of J=4 product alternatives\n"
    "a shopper will buy. You will be shown the best-so-far equations and their\n"
    "negative log-likelihood (NLL) on train and validation data. Propose a NEW\n"
    "equation that might fit better.\n\n"
    "Rules:\n"
    "1. Output a single Python function with signature exactly\n"
    "      def utility(x, c): return <expression>\n"
    "   where `x` is a length-4 per-alternative feature vector\n"
    "   (x[0]=price, x[1]=popularity_rank, x[2]=log1p_price, x[3]=price_rank)\n"
    "   and `c` is a coefficient array of length at most 8.\n"
    "2. Use only these operators: +, -, *, /, ** (integer exponent <= 2).\n"
    "3. Use only these functions: log1p, exp_c (clipped exp), sqrt_abs, tanh.\n"
    "4. No conditionals (if/else), no loops, no attribute access, no imports,\n"
    "   no global state. A single return statement only.\n"
    "5. Lower NLL is better. Prefer equations that are DIFFERENT from the\n"
    "   best-so-far (not minor coefficient tweaks -- structural variation).\n"
    "6. Return only the function definition in a ```python``` code block. No\n"
    "   prose, no comments outside the block.\n\n"
    "Feature semantics (for reasoning, not code):\n"
    "- price: raw dollar price of the alternative, range roughly $5 to $500.\n"
    "- popularity_rank: larger = more popular; logged or raw values possible.\n"
    "- log1p_price: log1p(price). A monotone transform of price.\n"
    "- price_rank: within-event dense rank of price, in [0, 1]; 0 = cheapest.\n\n"
    "Useful intuition from consumer theory:\n"
    "- Shoppers are typically price-sensitive: the coefficient on price or\n"
    "  log1p_price is usually negative.\n"
    "- Popular items draw share; popularity coefficient is usually positive.\n"
    "- price_rank captures relative-price effects that raw price cannot.\n"
    "- Nonlinear blends of price and popularity (e.g. price * popularity, or\n"
    "  tanh of a linear combination) can capture diminishing-returns effects."
)

USER_TEMPLATE: str = (
    "Iteration {t} of {T_max}.\n\n"
    "Best {K} equations so far, ranked by combined NLL = train_nll + val_nll\n"
    "(lower is better):\n\n"
    "{memory_block}\n\n"
    "Your turn. Output ONE new candidate utility function, different in STRUCTURE\n"
    "from all of the above (not merely a coefficient change). Output only the\n"
    "```python``` code block."
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class _SkeletonRecord:
    """One entry in the experience buffer.

    ``fn`` is the compiled callable; the outer loop re-uses it when
    scoring test events so we never re-compile the chosen skeleton.
    """

    source: str
    fn: Callable[[np.ndarray, np.ndarray], float] = field(repr=False)
    coeffs: np.ndarray
    k: int
    train_nll: float
    val_nll: float

    @property
    def combined_nll(self) -> float:
        return float(self.train_nll) + float(self.val_nll)


# ---------------------------------------------------------------------------
# Memory rendering helpers
# ---------------------------------------------------------------------------


def _render_memory_block(memory: Sequence[_SkeletonRecord]) -> str:
    """Render the top-K experience-buffer entries into the user prompt.

    Layout matches design doc §3.2::

        [1] NLL_train=1.2431  NLL_val=1.2690  k=3
        def utility(x, c): return c[0]*x[0] + c[1]*x[1] + c[2]*x[3]

        ...
    """
    if not memory:
        return "(no fitted skeletons yet)"
    lines: List[str] = []
    for i, rec in enumerate(memory):
        lines.append(
            f"[{i + 1}] NLL_train={rec.train_nll:.4f}  "
            f"NLL_val={rec.val_nll:.4f}  k={rec.k}"
        )
        lines.append(rec.source.strip())
        if i < len(memory) - 1:
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(
    r"```(?:python)?\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE
)
_DEF_LINE_RE = re.compile(r"^\s*def\s+utility\s*\(", re.MULTILINE)


def _extract_skeleton_source(response_text: str) -> Optional[str]:
    """Pull a ``def utility(...)`` block out of a model response.

    Preference order (design doc §3): fenced ``python`` block first, then
    any fenced block, then a raw ``def utility(`` scan over the whole
    text. Returns ``None`` if no plausible source is found so the outer
    loop can record the proposal as rejected.
    """
    if not response_text:
        return None
    # Fenced block first.
    for match in _CODE_FENCE_RE.finditer(response_text):
        body = match.group(1)
        if _DEF_LINE_RE.search(body):
            return body.strip()
    # Raw scan: take from the first ``def utility`` to the end of the
    # balanced function body (we naively take to EOF; the AST walker
    # rejects anything after the single return statement anyway).
    m = _DEF_LINE_RE.search(response_text)
    if m is None:
        return None
    return response_text[m.start():].strip()


# ---------------------------------------------------------------------------
# LLM client dispatch
# ---------------------------------------------------------------------------


def _is_anthropic_client(client: LLMClient) -> bool:
    """Duck-type check mirroring :func:`_llm_ranker_common._is_anthropic_client`."""
    if getattr(client, "_is_stub", False):
        return False
    inner = getattr(client, "_client", None)
    if inner is None:
        return False
    messages = getattr(inner, "messages", None)
    if messages is None:
        return False
    return hasattr(messages, "create")


def _response_as_text(response: Any) -> str:
    """Flatten an LLM response (GenerationResult / string / SDK object) to text."""
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
    """Issue one proposal request and return the raw text response.

    Uses the same ``cache_control: ephemeral`` pattern as
    ``_llm_ranker_common.call_llm_for_ranking`` so the static system
    prompt lands in Anthropic's server-side cache; every subsequent
    proposal cache-reads instead of cache-writing (design doc §3.3).
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
        return _call_anthropic_for_proposal(
            client,
            system=system,
            user=user,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            model_id=model_id,
        )

    # Generic LLMClient: forward through ``generate``. No caching hook.
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


def _call_anthropic_for_proposal(
    client: LLMClient,
    *,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    model_id: Optional[str],
) -> str:
    inner = getattr(client, "_client", None)
    if inner is None:  # pragma: no cover - guarded upstream
        raise RuntimeError(
            "LLMSR: expected client._client to be set for Anthropic clients."
        )
    resolved_model = model_id or getattr(client, "model_id", None) or "claude-sonnet-4-6"
    messages = [{"role": "user", "content": user}]
    kwargs: Dict[str, Any] = {
        "model": resolved_model,
        "system": [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    try:
        response = inner.messages.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 - SDK raises varied errors
        logger.warning(
            "LLMSR Anthropic proposal call failed (%s); returning empty text.",
            exc,
        )
        return ""
    return _response_as_text(response)


# ---------------------------------------------------------------------------
# Fitted wrapper
# ---------------------------------------------------------------------------


@dataclass
class LLMSRFitted:
    """Fitted LLM-SR wrapper exposing the :class:`FittedBaseline` protocol.

    ``_utility_fn`` holds the compiled callable for ``best_skeleton`` so
    ``score_events`` can evaluate without re-compiling. The field is
    ``repr=False`` / excluded from dict serialisation because it is not
    picklable in general.

    ``memory`` carries the full top-K experience buffer at the end of
    :meth:`LLMSR.fit`; used by :meth:`extra_artifacts_for_json` to
    export the top-10 equations into the leaderboard row per the
    paper-grade evaluation doc (addition 4).
    """

    name: str = "LLM-SR"
    best_skeleton: str = ""
    best_coefficients: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_coefficients: int = 0
    train_nll: float = float("inf")
    val_nll: float = float("inf")
    model_id: str = "unknown"
    n_proposals_accepted: int = 0
    n_proposals_total: int = 0
    prompt_version: str = "llm-sr-v1"
    memory: List["_SkeletonRecord"] = field(default_factory=list)
    _utility_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = field(
        default=None, repr=False
    )

    # ------------------------------------------------------------------
    # FittedBaseline protocol
    # ------------------------------------------------------------------
    def score_events(self, batch: BaselineEventBatch) -> List[np.ndarray]:
        """Return per-event utility vectors of shape ``(J,)``.

        Evaluates the fitted skeleton on the 4 built-in per-alt columns
        (``price, popularity_rank, log1p_price, price_rank``). The harness
        log-softmaxes these into probabilities.
        """
        if self._utility_fn is None:
            # Re-compile from source if the fitted object was deserialised.
            fn, _ = compile_equation(
                self.best_skeleton, max_coefficients=max(1, self.n_coefficients or 1)
            )
            self._utility_fn = fn
        fn = self._utility_fn
        coeffs = np.asarray(self.best_coefficients, dtype=np.float64)
        out: List[np.ndarray] = []
        for feats in batch.base_features_list:
            feats_f = np.asarray(feats, dtype=np.float64)
            # LLM-SR operates on the first 4 columns only
            # (BUILTIN_FEATURE_NAMES, data_adapter.py). If the caller
            # passes extra columns (extra_feature_names), we ignore them.
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
            f"LLM-SR model={self.model_id} k={self.n_coefficients} "
            f"train_nll={self.train_nll:.4f} val_nll={self.val_nll:.4f} "
            f"accepted={self.n_proposals_accepted}/{self.n_proposals_total}"
        )

    # ------------------------------------------------------------------
    # Paper-grade evaluation hook (addition 4)
    # ------------------------------------------------------------------
    def extra_artifacts_for_json(self) -> Optional[Dict[str, Any]]:
        """Export the top-10 fitted skeletons for the leaderboard row.

        Contract (``docs/paper_evaluation_additions.md`` §4):

        * Key ``llm_sr_top_equations`` maps to a list of dicts with
          fields ``{"source", "nll_train", "nll_val", "coefficients"}``.
        * Ordered ascending by ``nll_train + nll_val`` (lower is better).
        * Capped at 10 entries.
        * Returns ``None`` when :attr:`memory` is empty so the row
          surfaces ``extra_artifacts: null`` rather than an empty list.
        """
        if not self.memory:
            return None
        sorted_mem = sorted(self.memory, key=lambda r: r.combined_nll)[:10]
        return {
            "llm_sr_top_equations": [
                {
                    "source": str(r.source),
                    "nll_train": float(r.train_nll),
                    "nll_val": float(r.val_nll),
                    "coefficients": [float(c) for c in np.asarray(r.coeffs).ravel()],
                }
                for r in sorted_mem
            ]
        }


# ---------------------------------------------------------------------------
# Baseline (fit-time) class
# ---------------------------------------------------------------------------


class LLMSR:
    """LLM-SR for DCM. Implements the :class:`Baseline` protocol.

    Parameters
    ----------
    llm_client
        Any ``LLMClient`` from :mod:`src.outcomes.generate`. Defaults to
        :class:`StubLLMClient` so hermetic CI runs work without network.
        Under the stub path every proposal is rejected (stub emits
        narrative text, not Python) and :attr:`LLMSRFitted.best_skeleton`
        falls back to :data:`FALLBACK_SKELETON` (§10).
    n_proposals
        Number of LLM calls (design doc §4.1 uses 100).
    top_k_memory
        Number of top-NLL skeletons shown to the proposer in the user
        prompt (design doc §3.2 uses 10).
    max_memory
        Hard cap on the experience-buffer size (§6).
    max_coefficients
        Grammar hard cap on ``k`` (§2 pins this to 8).
    n_restarts
        Multi-start count for the BFGS inner loop (§5 uses 4).
    max_retries_per_proposal
        If a proposal is rejected (parse error, grammar violation, BFGS
        divergence), the proposer is re-queried up to this many extra
        times for iteration ``t`` before moving on (design doc §9 test
        11).
    seed
        Base seed for the BFGS restarts and per-proposal LLM call seeds.
    prompt_version
        Identifier stamped on the fitted object for audit.
    """

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
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> None:
        if n_proposals < 0:
            raise ValueError(f"n_proposals must be >= 0, got {n_proposals}")
        if top_k_memory <= 0:
            raise ValueError(f"top_k_memory must be positive, got {top_k_memory}")
        if max_memory < top_k_memory:
            raise ValueError(
                f"max_memory={max_memory} must be >= top_k_memory={top_k_memory}"
            )
        if max_coefficients <= 0:
            raise ValueError(
                f"max_coefficients must be positive, got {max_coefficients}"
            )
        if n_restarts <= 0:
            raise ValueError(f"n_restarts must be positive, got {n_restarts}")
        if max_retries_per_proposal < 0:
            raise ValueError(
                f"max_retries_per_proposal must be >= 0, got {max_retries_per_proposal}"
            )

        self.llm_client: LLMClient = (
            llm_client if llm_client is not None else StubLLMClient()
        )
        self.n_proposals = int(n_proposals)
        self.top_k_memory = int(top_k_memory)
        self.max_memory = int(max_memory)
        self.max_coefficients = int(max_coefficients)
        self.n_restarts = int(n_restarts)
        self.max_retries_per_proposal = int(max_retries_per_proposal)
        self.seed = int(seed)
        self.prompt_version = str(prompt_version)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        train: BaselineEventBatch,
        val: BaselineEventBatch,
    ) -> LLMSRFitted:
        """Outer loop per design doc §6 + §10 fallback seeding.

        1. Pre-seed memory with :data:`FALLBACK_SKELETON` so a total
           LLM-failure run still returns a valid fitted object.
        2. For each iteration ``t``:
           a. Render the user prompt from the top-K memory entries.
           b. Call the LLM (with up to ``max_retries_per_proposal`` extra
              retries on rejection).
           c. Parse the returned code block; sandbox-compile.
           d. BFGS-fit coefficients on ``train``; score ``val``.
           e. Push the skeleton record onto memory; sort + cap.
        3. Return the min-combined-NLL record wrapped in
           :class:`LLMSRFitted`.
        """
        self._validate_batches(train, val)

        train_feats_list = self._slice_first4(train)
        train_chosen = list(train.chosen_indices)
        val_feats_list = self._slice_first4(val)
        val_chosen = list(val.chosen_indices)

        memory: List[_SkeletonRecord] = []
        total_proposals = 0
        accepted_proposals = 0

        # Step 1: fallback seeding.
        fallback = self._fit_one_skeleton(
            FALLBACK_SKELETON,
            train_feats_list,
            train_chosen,
            val_feats_list,
            val_chosen,
            seed=self.seed,
        )
        if fallback is not None:
            memory.append(fallback)

        # Step 2: outer LLM loop.
        for t in range(self.n_proposals):
            user_prompt = USER_TEMPLATE.format(
                t=t + 1,
                T_max=self.n_proposals,
                K=self.top_k_memory,
                memory_block=_render_memory_block(memory[: self.top_k_memory]),
            )
            record: Optional[_SkeletonRecord] = None
            # ``max_retries_per_proposal`` extra attempts on top of the
            # first call -- so up to (retries + 1) total calls per t.
            attempts_budget = self.max_retries_per_proposal + 1
            for attempt in range(attempts_budget):
                total_proposals += 1
                call_seed = (
                    self.seed + t * (self.max_retries_per_proposal + 1) + attempt
                )
                try:
                    raw_text = _call_llm_for_proposal(
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
                        "LLMSR proposal t=%d attempt=%d call failed: %s",
                        t, attempt, exc,
                    )
                    continue
                source = _extract_skeleton_source(raw_text)
                if source is None:
                    logger.debug(
                        "LLMSR proposal t=%d attempt=%d: no parseable block",
                        t, attempt,
                    )
                    continue
                record = self._fit_one_skeleton(
                    source,
                    train_feats_list,
                    train_chosen,
                    val_feats_list,
                    val_chosen,
                    seed=call_seed,
                )
                if record is not None:
                    accepted_proposals += 1
                    break
            if record is not None:
                memory.append(record)
                memory.sort(key=lambda r: r.combined_nll)
                memory = memory[: self.max_memory]

        # Step 3: pick the best.
        model_id = str(getattr(self.llm_client, "model_id", "unknown") or "unknown")
        if not memory:
            # Even the fallback failed -- emit a zero-weighted fitted
            # object so the harness reports rather than crashing.
            logger.error(
                "LLMSR.fit produced no viable skeleton; returning a zero fitted object."
            )
            return LLMSRFitted(
                name=self.name,
                best_skeleton=FALLBACK_SKELETON,
                best_coefficients=np.zeros(4, dtype=np.float64),
                n_coefficients=4,
                train_nll=float("inf"),
                val_nll=float("inf"),
                model_id=model_id,
                n_proposals_accepted=accepted_proposals,
                n_proposals_total=total_proposals,
                prompt_version=self.prompt_version,
                memory=[],
                _utility_fn=None,
            )
        best = memory[0]
        return LLMSRFitted(
            name=self.name,
            best_skeleton=best.source,
            best_coefficients=np.asarray(best.coeffs, dtype=np.float64),
            n_coefficients=int(best.k),
            train_nll=float(best.train_nll),
            val_nll=float(best.val_nll),
            model_id=model_id,
            n_proposals_accepted=accepted_proposals,
            n_proposals_total=total_proposals,
            prompt_version=self.prompt_version,
            memory=list(memory),
            _utility_fn=best.fn,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _fit_one_skeleton(
        self,
        source: str,
        train_feats_list: Sequence[np.ndarray],
        train_chosen: Sequence[int],
        val_feats_list: Sequence[np.ndarray],
        val_chosen: Sequence[int],
        *,
        seed: int,
    ) -> Optional[_SkeletonRecord]:
        """Compile, fit, and score one skeleton. ``None`` on any rejection."""
        try:
            fn, k = compile_equation(source, max_coefficients=self.max_coefficients)
        except SandboxError as exc:
            logger.debug("LLMSR skeleton rejected: %s", exc)
            return None
        if k > self.max_coefficients:
            logger.debug("LLMSR skeleton k=%d exceeds cap", k)
            return None
        coeffs, train_nll = fit_coefficients_softmax_ce(
            fn,
            train_feats_list,
            train_chosen,
            k=k,
            max_coeffs=self.max_coefficients,
            n_restarts=self.n_restarts,
            seed=int(seed),
        )
        if coeffs is None or not np.isfinite(train_nll):
            logger.debug("LLMSR skeleton diverged on train")
            return None
        val_nll = eval_nll_val(fn, coeffs, val_feats_list, val_chosen)
        if not np.isfinite(val_nll):
            logger.debug("LLMSR skeleton diverged on val")
            return None
        return _SkeletonRecord(
            source=source.strip(),
            fn=fn,
            coeffs=coeffs,
            k=int(k),
            train_nll=float(train_nll),
            val_nll=float(val_nll),
        )

    @staticmethod
    def _slice_first4(batch: BaselineEventBatch) -> List[np.ndarray]:
        """Extract the 4 built-in per-alt columns per event.

        The adapter's :data:`BUILTIN_FEATURE_NAMES` pins the first 4
        columns as ``(price, popularity_rank, log1p_price, price_rank)``.
        Any ``extra_feature_names`` passed to
        :func:`records_to_baseline_batch` produce columns 4..F-1 which
        LLM-SR ignores.
        """
        out: List[np.ndarray] = []
        for feats in batch.base_features_list:
            arr = np.asarray(feats, dtype=np.float64)
            if arr.shape[1] < 4:
                raise ValueError(
                    f"LLM-SR requires at least 4 feature columns; got {arr.shape[1]}"
                )
            out.append(arr[:, :4].copy())
        return out

    @staticmethod
    def _validate_batches(
        train: BaselineEventBatch, val: BaselineEventBatch
    ) -> None:
        if train.n_events == 0:
            raise ValueError("LLMSR.fit received an empty train batch")
        if val.n_events == 0:
            raise ValueError("LLMSR.fit received an empty val batch")


__all__ = [
    "LLMSR",
    "LLMSRFitted",
    "SYSTEM_PROMPT",
    "USER_TEMPLATE",
]
