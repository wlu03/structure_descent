"""Few-shot in-context-learning frozen-LLM ranker baseline for PO-LEU.

Full design: ``docs/llm_baselines/few_shot_icl_ranker.md``.

Summary
-------
Companion to :class:`ZeroShotClaudeRanker`. The test-event portion of the
prompt is byte-identical to the zero-shot baseline; what changes is a
prefix of ``n_shots`` in-context demonstrations drawn from the same
customer's earlier training events (``(c_d, alternatives, CHOSEN: letter)``
tuples, chronological order). The model is asked to generalise that
customer's revealed preferences from the demonstrations to the test
event. Like zero-shot, we run K letter-permutation rotations and average
the per-call distributions; crucially, **the ``CHOSEN:`` letter inside
each ICL example is rotated in lockstep with the query's permutation**
so the ICL labels stay aligned with the query's letter scheme (design
doc §6).

Cold-start semantics
--------------------
A test event cold-starts when :func:`select_icl_examples` returns ``[]``
(customer unseen in train, or every train event is dated at-or-after the
test, or ``train.raw_events is None``). In that case the prompt falls
back to the zero-shot template for that single call and
``_cold_start_count`` is incremented; the ratio is surfaced in
``description`` for audit.

Registry entry (to be added manually to src/baselines/run_all.py):
  ("FewShot-ICL-Claude", "src.baselines.few_shot_icl_ranker", "FewShotICLRanker"),
__init__.py export (to be added manually):
  from .few_shot_icl_ranker import FewShotICLRanker, FewShotICLRankerFitted
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.outcomes.generate import LLMClient, StubLLMClient

from .base import BaselineEventBatch
from ._llm_ranker_common import (
    DEFAULT_LETTERS,
    ICLExample,
    LLMRankerBase,
    _LOGPROB_FLOOR,
    build_customer_timeline,
    call_llm_for_ranking,
    letter_permutations,
    render_alternatives,
)
from .zero_shot_claude_ranker import SYSTEM_PROMPT, USER_TEMPLATE

# ---------------------------------------------------------------------------
# Prompt constants (design doc §5)
# ---------------------------------------------------------------------------

#: Split marker between the cacheable ICL prefix and the per-event suffix.
#: Mirrors the ``_USER_SPLIT_MARKER`` pattern from
#: ``src/outcomes/generate.py``; everything up to and including this line
#: goes in a cacheable content block.
ICL_SPLIT_MARKER: str = "<<<END OF PRIOR CHOICES>>>"

ICL_HEADER: str = "PRIOR CHOICES BY THIS CUSTOMER (chronological):\n"

#: Rough heuristic used by the context-overflow guard (design doc §13).
#: We avoid importing tiktoken / anthropic tokenizers here so the module
#: stays dependency-free under the stub path.
_CHARS_PER_TOKEN: float = 4.0


def _estimate_tokens(text: str) -> int:
    """Cheap token-count estimate: ``ceil(len(text) / 4)``."""
    if not text:
        return 0
    return (len(text) + int(_CHARS_PER_TOKEN) - 1) // int(_CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# ICL example selection (design doc §4)
# ---------------------------------------------------------------------------


def select_icl_examples(
    customer_id: str,
    test_order_date: pd.Timestamp,
    timeline: Mapping[str, Sequence[ICLExample]],
    n_shots: int,
) -> List[ICLExample]:
    """Return the ``n_shots`` most-recent strictly-earlier ICL examples.

    The strict ``<`` comparison defends against tied timestamps from the
    same checkout (design doc §4): a training event sharing the test
    event's ``order_date`` is NOT a valid demonstration, and any such
    event would encode a label leak if returned.

    Parameters
    ----------
    customer_id
        Customer identifier for the test event.
    test_order_date
        ``order_date`` of the test event. Examples with
        ``e.order_date >= test_order_date`` are filtered out.
    timeline
        Mapping from customer id to an ascending-sorted list of
        :class:`ICLExample`.
    n_shots
        Maximum number of examples to return; the N most-recent strictly
        earlier events are kept.

    Returns
    -------
    list of ICLExample
        Chronological (ascending) order. Empty when the customer is
        unseen or all training events are dated at-or-after the test.
    """
    if n_shots <= 0:
        return []
    history = timeline.get(str(customer_id), [])
    if not history:
        return []
    prior = [e for e in history if e.order_date < test_order_date]
    if not prior:
        return []
    return list(prior[-n_shots:])


# ---------------------------------------------------------------------------
# Prompt rendering (design doc §5, §6)
# ---------------------------------------------------------------------------


def _render_icl_example(
    idx: int,
    example: ICLExample,
    pi: Sequence[int],
    letters: Sequence[str] = DEFAULT_LETTERS,
) -> str:
    """Render one ``=== Example K ===`` block under a specific permutation.

    ``pi`` is the letter-slot → canonical-index mapping currently in
    force for the query; we apply the same mapping to each ICL example's
    alternatives AND to its ``CHOSEN:`` letter. That means: if the
    customer's true ``chosen_idx`` is c, we first find the slot s such
    that ``pi[s] == c`` and then emit ``CHOSEN: {letters[s]}``. This
    keeps the ICL labels aligned with the query's letter scheme (design
    doc §6 — the critical failsafe).
    """
    if len(pi) != len(letters):
        raise ValueError(
            f"permutation length {len(pi)} != letter count {len(letters)}"
        )
    if len(example.alt_texts) != len(letters):
        raise ValueError(
            f"ICL example {idx} has {len(example.alt_texts)} alt_texts; "
            f"expected {len(letters)}"
        )
    if not (0 <= example.chosen_idx < len(letters)):
        raise ValueError(
            f"ICL example {idx} chosen_idx={example.chosen_idx} out of range "
            f"for {len(letters)} alternatives"
        )
    # Permute alt_texts in lockstep with the query.
    permuted_alt_texts = [example.alt_texts[pi[s]] for s in range(len(letters))]
    alternatives = render_alternatives(permuted_alt_texts, letters)
    # Find the slot under the current permutation where the true chosen
    # alternative lives; emit the corresponding letter.
    chosen_slot: Optional[int] = None
    for s, canonical in enumerate(pi):
        if canonical == example.chosen_idx:
            chosen_slot = s
            break
    if chosen_slot is None:  # pragma: no cover - validated above
        raise ValueError(
            f"ICL example {idx} chosen_idx={example.chosen_idx} not found "
            f"in permutation {tuple(pi)}"
        )
    chosen_letter = letters[chosen_slot]
    return (
        f"=== Example {idx + 1} ===\n"
        f"CONTEXT:\n{example.c_d}\n\n"
        f"ALTERNATIVES:\n{alternatives}\n\n"
        f"CHOSEN: {chosen_letter}"
    )


def _render_icl_prefix(
    examples: Sequence[ICLExample],
    pi: Sequence[int],
    letters: Sequence[str] = DEFAULT_LETTERS,
) -> str:
    """Render the header + ``N`` example blocks + split-marker sentinel.

    Returns the empty string when ``examples`` is empty so the caller
    falls back to the byte-identical zero-shot prompt.
    """
    if not examples:
        return ""
    blocks = [_render_icl_example(i, ex, pi, letters) for i, ex in enumerate(examples)]
    return (
        ICL_HEADER
        + "\n"
        + "\n\n".join(blocks)
        + "\n\n"
        + ICL_SPLIT_MARKER
        + "\n\n"
    )


def _truncate_to_token_budget(
    examples: Sequence[ICLExample],
    pi: Sequence[int],
    max_prefix_tokens: int,
    letters: Sequence[str] = DEFAULT_LETTERS,
) -> List[ICLExample]:
    """Drop oldest ICL examples until the prefix fits ``max_prefix_tokens``.

    Estimates the ICL-prefix token count with :func:`_estimate_tokens`;
    if it already fits, returns ``examples`` unchanged. Otherwise trims
    from the front (oldest) until either the budget is met or no
    examples are left. Truncation order is deterministic (oldest first).
    """
    kept = list(examples)
    while kept:
        prefix = _render_icl_prefix(kept, pi, letters)
        if _estimate_tokens(prefix) <= max_prefix_tokens:
            return kept
        kept = kept[1:]  # drop oldest
    return []


def build_user_prompt_with_icl(
    c_d: str,
    alt_texts_permuted: Sequence[Mapping[str, Any]],
    icl_examples: Sequence[ICLExample],
    pi: Sequence[int],
    letters: Sequence[str] = DEFAULT_LETTERS,
    max_prefix_tokens: int = 12_000,
) -> str:
    """Compose the full user prompt: ICL prefix (if any) + test event.

    The test-event suffix is produced via the same :data:`USER_TEMPLATE`
    used by :class:`ZeroShotClaudeRanker`; cold-start falls through to
    the byte-identical zero-shot string.
    """
    alternatives = render_alternatives(alt_texts_permuted, letters)
    base_prompt = USER_TEMPLATE.format(c_d=c_d, alternatives=alternatives)
    if not icl_examples:
        return base_prompt
    kept = _truncate_to_token_budget(
        icl_examples, pi, max_prefix_tokens, letters
    )
    if not kept:
        return base_prompt
    prefix = _render_icl_prefix(kept, pi, letters)
    return prefix + base_prompt


# ---------------------------------------------------------------------------
# Concrete LLMRankerBase subclass with the few-shot prompt hooks
# ---------------------------------------------------------------------------


class _FewShotRankerImpl(LLMRankerBase):
    """Handles the K-permutation scoring loop with per-permutation ICL."""

    def __init__(
        self,
        llm_client: LLMClient,
        n_permutations: int,
        seed: int,
        n_shots: int,
        timeline: Dict[str, List[ICLExample]],
        letters: Sequence[str],
        max_prefix_tokens: int,
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            n_permutations=n_permutations,
            seed=seed,
        )
        self._letters = tuple(letters)
        self._n_shots = int(n_shots)
        self._timeline = timeline
        self._max_prefix_tokens = int(max_prefix_tokens)
        self.cold_start_count = 0

    def build_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build_user_prompt(
        self,
        event: Mapping[str, Any],
        alt_texts_permuted: Sequence[Mapping[str, Any]],
        letters: Sequence[str],
    ) -> str:
        # Not used — ``_score_one_event`` is overridden below so it can
        # weave the per-permutation ICL prefix in with the same ``pi``
        # already applied to ``alt_texts_permuted``.
        raise NotImplementedError(
            "_FewShotRankerImpl uses _score_one_event directly."
        )

    # ------------------------------------------------------------------
    # Override _score_one_event so we can keep the ICL ``pi`` in scope
    # ------------------------------------------------------------------
    def _score_one_event(  # type: ignore[override]
        self,
        event: Mapping[str, Any],
        letters: Sequence[str] = DEFAULT_LETTERS,
        *,
        event_idx: int = 0,
        temperature: float = 0.0,
        max_tokens: int = 2,
    ) -> np.ndarray:
        alt_texts = list(event["alt_texts"])
        n_alts = len(alt_texts)
        if n_alts != len(letters):
            raise ValueError(
                f"event has {n_alts} alternatives but {len(letters)} letters "
                "were provided; ranker baselines require n_alts == len(letters)."
            )

        perms = letter_permutations(n_alts, self.n_permutations, seed=self.seed)

        # Resolve ICL examples once per event (permutation-independent).
        icl_examples = self._select_icl_for_event(event)
        if not icl_examples:
            self.cold_start_count += 1

        system = self.build_system_prompt()
        c_d = str(event.get("c_d", ""))

        un_permuted = np.zeros((len(perms), n_alts), dtype=np.float64)
        for k, pi in enumerate(perms):
            permuted_alt_texts = [alt_texts[pi[s]] for s in range(n_alts)]
            user = build_user_prompt_with_icl(
                c_d=c_d,
                alt_texts_permuted=permuted_alt_texts,
                icl_examples=icl_examples,
                pi=pi,
                letters=letters,
                max_prefix_tokens=self._max_prefix_tokens,
            )
            letter_probs = call_llm_for_ranking(
                self.llm_client,
                system,
                user,
                letters,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=self.seed + event_idx * self.n_permutations + k,
            )
            for s in range(n_alts):
                canonical_j = pi[s]
                un_permuted[k, canonical_j] = letter_probs[s]

        p_hat = un_permuted.mean(axis=0)
        # Reuse the same floor as the base class so log(p_hat) stays finite.
        p_hat = np.maximum(p_hat, _LOGPROB_FLOOR)
        return np.log(p_hat)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _select_icl_for_event(
        self, event: Mapping[str, Any]
    ) -> List[ICLExample]:
        cid = str(event.get("customer_id", ""))
        order_date_val = event.get("order_date")
        if order_date_val is None:
            # Without an order_date we cannot safely filter; treat as
            # cold-start (design doc §8 cases (a)/(b)/(c) cover this).
            return []
        try:
            ts = pd.Timestamp(order_date_val)
        except (TypeError, ValueError):
            return []
        return select_icl_examples(
            customer_id=cid,
            test_order_date=ts,
            timeline=self._timeline,
            n_shots=self._n_shots,
        )


# ---------------------------------------------------------------------------
# Fitted wrapper
# ---------------------------------------------------------------------------


@dataclass
class FewShotICLRankerFitted:
    """Fitted wrapper: holds the customer timeline + LLM client.

    Satisfies the :class:`FittedBaseline` protocol. Like the zero-shot
    ranker, ``n_params`` is zero — the timeline is a lookup table, not
    trained coefficients — which keeps AIC/BIC comparable across
    frozen-LLM baselines (design doc §7).
    """

    name: str = "FewShot-ICL-Claude"
    client: LLMClient = field(default_factory=StubLLMClient, repr=False)
    timeline: Dict[str, List[ICLExample]] = field(default_factory=dict)
    n_shots: int = 3
    n_permutations: int = 4
    seed: int = 0
    temperature: float = 0.0
    max_tokens: int = 2
    prompt_version: str = "few-shot-icl-rank-v1"
    letters: tuple[str, ...] = DEFAULT_LETTERS
    model_id: str = "unknown"
    max_prefix_tokens: int = 12_000
    _cold_start_count: int = 0
    _total_events: int = 0

    # ------------------------------------------------------------------
    # FittedBaseline protocol
    # ------------------------------------------------------------------
    def score_events(self, batch: BaselineEventBatch) -> List[np.ndarray]:
        if batch.n_alternatives != len(self.letters):
            raise ValueError(
                f"FewShot-ICL-Claude scores batches with "
                f"n_alternatives={len(self.letters)} only; got "
                f"{batch.n_alternatives}."
            )
        if batch.raw_events is None:
            raise ValueError(
                "FewShot-ICL-Claude requires raw_events with per-event "
                "'c_d', 'alt_texts', 'customer_id', and 'order_date' keys; "
                "BaselineEventBatch.raw_events is None."
            )

        ranker = _FewShotRankerImpl(
            llm_client=self.client,
            n_permutations=self.n_permutations,
            seed=self.seed,
            n_shots=self.n_shots,
            timeline=self.timeline,
            letters=self.letters,
            max_prefix_tokens=self.max_prefix_tokens,
        )

        out: List[np.ndarray] = []
        for idx, rec in enumerate(batch.raw_events):
            if not isinstance(rec, Mapping):
                raise ValueError(
                    f"raw_events[{idx}] is not a Mapping; "
                    "FewShot-ICL-Claude expects dict records."
                )
            if "alt_texts" not in rec or "c_d" not in rec:
                raise ValueError(
                    f"raw_events[{idx}] missing 'c_d' / 'alt_texts'; "
                    "FewShot-ICL-Claude requires both keys."
                )
            logp = ranker._score_one_event(
                rec,
                letters=self.letters,
                event_idx=idx,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            out.append(logp)

        self._cold_start_count = int(ranker.cold_start_count)
        self._total_events = int(len(batch.raw_events))
        return out

    @property
    def n_params(self) -> int:
        """Zero trainable parameters (frozen LLM).

        The timeline is a lookup table, not fit coefficients, so
        reporting 0 keeps AIC/BIC comparable with the zero-shot ranker
        (design doc §7).
        """
        return 0

    @property
    def description(self) -> str:
        return (
            f"FewShot-ICL-Claude model={self.model_id} "
            f"n_shots={self.n_shots} K={self.n_permutations} "
            f"T={self.temperature:g} prompt={self.prompt_version} "
            f"cold_start={self._cold_start_count}/{self._total_events}"
        )


# ---------------------------------------------------------------------------
# Fit-time class
# ---------------------------------------------------------------------------


class FewShotICLRanker:
    """Few-shot ICL frozen-LLM ranker (implements the :class:`Baseline` protocol).

    Parameters
    ----------
    n_shots
        Number of prior demonstrations per test event. Default 3 matches
        the design doc §7 recommendation — keeps the prefix ≈1k tokens.
    llm_client
        Any :class:`LLMClient`. Defaults to :class:`StubLLMClient`.
    n_permutations
        K letter-permutation rotations; same K=4 Latin square as the
        zero-shot baseline (design doc §6 — letter bias is independent
        of the ICL prefix, so the rotation must still happen).
    seed
        Base seed forwarded to each per-permutation LLM call as
        ``seed + event_idx * K + k``.
    max_prefix_tokens
        Truncation guard (design doc §13). If the rendered prefix plus
        the test-event suffix exceeds this, oldest ICL examples are
        dropped first.
    """

    name: str = "FewShot-ICL-Claude"

    def __init__(
        self,
        n_shots: int = 3,
        llm_client: Optional[LLMClient] = None,
        n_permutations: int = 4,
        seed: int = 0,
        max_prefix_tokens: int = 12_000,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2,
        prompt_version: str = "few-shot-icl-rank-v1",
    ) -> None:
        if n_shots < 0:
            raise ValueError(f"n_shots must be non-negative, got {n_shots}")
        if n_permutations <= 0:
            raise ValueError(f"n_permutations must be positive, got {n_permutations}")
        if max_prefix_tokens <= 0:
            raise ValueError(
                f"max_prefix_tokens must be positive, got {max_prefix_tokens}"
            )
        self.n_shots = int(n_shots)
        self.llm_client: LLMClient = (
            llm_client if llm_client is not None else StubLLMClient()
        )
        self.n_permutations = int(n_permutations)
        self.seed = int(seed)
        self.max_prefix_tokens = int(max_prefix_tokens)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.prompt_version = str(prompt_version)

    def fit(
        self,
        train: BaselineEventBatch,
        val: Optional[BaselineEventBatch] = None,
    ) -> FewShotICLRankerFitted:
        """Materialise the per-customer timeline; no LLM calls at fit time.

        Validates batch shape and builds
        ``customer_id -> [ICLExample, ...]`` via
        :func:`build_customer_timeline` (which returns ``{}`` gracefully
        when ``train.raw_events is None``).
        """
        if train.n_events == 0:
            raise ValueError(
                "FewShotICLRanker.fit received an empty train batch"
            )
        if train.n_alternatives != len(DEFAULT_LETTERS):
            raise ValueError(
                f"FewShot-ICL-Claude requires n_alternatives={len(DEFAULT_LETTERS)} "
                f"(got {train.n_alternatives})."
            )

        timeline = build_customer_timeline(train)
        model_id = str(getattr(self.llm_client, "model_id", "unknown") or "unknown")
        return FewShotICLRankerFitted(
            name=self.name,
            client=self.llm_client,
            timeline=timeline,
            n_shots=self.n_shots,
            n_permutations=self.n_permutations,
            seed=self.seed,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            prompt_version=self.prompt_version,
            letters=DEFAULT_LETTERS,
            model_id=model_id,
            max_prefix_tokens=self.max_prefix_tokens,
        )


__all__ = [
    "ICL_HEADER",
    "ICL_SPLIT_MARKER",
    "FewShotICLRanker",
    "FewShotICLRankerFitted",
    "build_user_prompt_with_icl",
    "select_icl_examples",
]
