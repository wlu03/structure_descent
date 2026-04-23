"""Zero-shot frozen-LLM ranker baseline for PO-LEU.

Full design: ``docs/llm_baselines/zero_shot_claude_ranker.md``.

Summary
-------
Given a PO-LEU choice event with J=4 alternatives, we ask a frozen Claude
model "Which alternative is this person most likely to buy?" once per
permutation (K=4 Latin-square rotations), read a length-4 probability
vector over the letter tokens ``A/B/C/D``, un-permute each call back to
canonical alternative order, arithmetic-average, and return
``log p_hat(j)`` as the utility score. The baseline is "fit" trivially:
``fit()`` just stores the client and returns a fitted wrapper.

The heavy lifting (Latin-square schedule, prompt rendering, logprob
extraction, stub fallback, Anthropic SDK dispatch) lives in
``_llm_ranker_common.py`` so the forthcoming Few-Shot ICL ranker can
share it verbatim — the two baselines MUST use identical letters,
permutation seeds, and extraction paths so ΔNLL between them is
attributable to the ICL prefix alone.

Registry integration
--------------------
This module deliberately does not edit ``src/baselines/__init__.py`` or
``src/baselines/run_all.py``. To wire the baseline into the standard
harness, add (manually):

    # src/baselines/__init__.py
    from .zero_shot_claude_ranker import (
        ZeroShotClaudeRanker,
        ZeroShotClaudeRankerFitted,
    )

    # src/baselines/run_all.py — append to BASELINE_REGISTRY
    ("ZeroShot-Claude", "src.baselines.zero_shot_claude_ranker", "ZeroShotClaudeRanker"),
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np

from src.outcomes.generate import LLMClient, StubLLMClient

from .base import Baseline, BaselineEventBatch, FittedBaseline
from ._llm_ranker_common import (
    DEFAULT_LETTERS,
    LLMRankerBase,
    render_alternatives,
)

# ---------------------------------------------------------------------------
# Prompt strings (design doc §2)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are a careful shopping-decision analyst. Given a person's "
    "profile and four product alternatives labeled A, B, C, D, identify "
    "which one the person is most likely to purchase next. Respond with a "
    "single capital letter: A, B, C, or D. Do not explain."
)

USER_TEMPLATE: str = (
    "PERSON:\n"
    "{c_d}\n\n"
    "ALTERNATIVES:\n"
    "{alternatives}\n\n"
    "Which alternative is this person most likely to purchase? Answer "
    "with a single capital letter (A, B, C, or D)."
)


# ---------------------------------------------------------------------------
# Fitted wrapper
# ---------------------------------------------------------------------------


@dataclass
class ZeroShotClaudeRankerFitted:
    """Fitted wrapper around a ``ZeroShotClaudeRanker`` (see module docstring).

    ``fit()`` is trivial (no training), so the fitted object is just the
    original ranker state: the client, the K-permutation count, the base
    seed, and decoding knobs. We still expose ``n_params``, ``description``,
    and ``score_events`` so this type satisfies the
    :class:`FittedBaseline` protocol and plugs into
    ``src.baselines.evaluate.evaluate_baseline`` without special casing.
    """

    name: str = "ZeroShot-Claude"
    llm_client: LLMClient = field(default_factory=StubLLMClient, repr=False)
    n_permutations: int = 4
    seed: int = 0
    temperature: float = 0.0
    max_tokens: int = 2
    prompt_version: str = "zero-shot-rank-v1"
    letters: tuple[str, ...] = DEFAULT_LETTERS
    model_id: str = "unknown"

    # ------------------------------------------------------------------
    # FittedBaseline protocol
    # ------------------------------------------------------------------
    def score_events(self, batch: BaselineEventBatch) -> List[np.ndarray]:
        """Return ``[log p_hat(j) of shape (J,)]`` for each event in ``batch``.

        ``evaluate_baseline`` then runs ``log_softmax`` on these scores so
        downstream metrics see a well-defined distribution even when one
        of the log-probabilities is near the floor.
        """
        if batch.n_alternatives != len(self.letters):
            raise ValueError(
                f"ZeroShot-Claude scores batches with "
                f"n_alternatives={len(self.letters)} only; got "
                f"{batch.n_alternatives}."
            )
        if batch.raw_events is None:
            raise ValueError(
                "ZeroShot-Claude requires raw_events with per-event 'c_d' and "
                "'alt_texts' keys. BaselineEventBatch.raw_events is None."
            )

        ranker = _build_ranker(self)

        out: List[np.ndarray] = []
        for idx, rec in enumerate(batch.raw_events):
            if not isinstance(rec, Mapping):
                raise ValueError(
                    f"raw_events[{idx}] is not a Mapping; "
                    "ZeroShot-Claude expects dict records."
                )
            if "alt_texts" not in rec or "c_d" not in rec:
                raise ValueError(
                    f"raw_events[{idx}] missing 'c_d' / 'alt_texts'; "
                    "ZeroShot-Claude requires both keys."
                )
            logp = ranker._score_one_event(
                rec,
                letters=self.letters,
                event_idx=idx,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            out.append(logp)
        return out

    @property
    def n_params(self) -> int:
        """Zero trainable parameters (frozen LLM).

        The harness's AIC/BIC code floors this at 1, so reporting 0 here
        is safe and truthful — the only "parameter" is the prompt
        template itself, which is not fitted.
        """
        return 0

    @property
    def description(self) -> str:
        return (
            f"ZeroShot-Claude model={self.model_id} "
            f"K={self.n_permutations} T={self.temperature:g} "
            f"prompt={self.prompt_version}"
        )


# ---------------------------------------------------------------------------
# Private wiring
# ---------------------------------------------------------------------------


class _ZeroShotRankerImpl(LLMRankerBase):
    """Concrete ``LLMRankerBase`` subclass with the zero-shot prompt hooks."""

    def __init__(
        self,
        llm_client: LLMClient,
        n_permutations: int,
        seed: int,
        letters: Sequence[str],
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            n_permutations=n_permutations,
            seed=seed,
        )
        self._letters = tuple(letters)

    def build_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build_user_prompt(
        self,
        event: Mapping[str, Any],
        alt_texts_permuted: Sequence[Mapping[str, Any]],
        letters: Sequence[str],
    ) -> str:
        c_d = str(event.get("c_d", ""))
        alternatives = render_alternatives(alt_texts_permuted, letters)
        return USER_TEMPLATE.format(c_d=c_d, alternatives=alternatives)


def _build_ranker(fitted: ZeroShotClaudeRankerFitted) -> _ZeroShotRankerImpl:
    return _ZeroShotRankerImpl(
        llm_client=fitted.llm_client,
        n_permutations=fitted.n_permutations,
        seed=fitted.seed,
        letters=fitted.letters,
    )


# ---------------------------------------------------------------------------
# Fit-time class
# ---------------------------------------------------------------------------


class ZeroShotClaudeRanker:
    """Frozen-LLM zero-shot ranker (implements the :class:`Baseline` protocol).

    Parameters
    ----------
    client
        Any ``LLMClient`` from ``src.outcomes.generate``. Defaults to
        :class:`StubLLMClient` so tests and hermetic CI runs work without
        network access.
    K
        Number of Latin-square permutations (design doc §4). K=4 is the
        minimum for first-order positional debiasing on J=4.
    temperature
        LLM decoding temperature forwarded to every per-permutation call.
    max_tokens
        Max output tokens per call. Two is enough for ``"A)"``.
    seed
        Base seed forwarded to per-call ``seed`` arguments as
        ``seed + event_idx * K + k``.
    prompt_version
        Identifier baked into the fitted object's ``description``; bump
        this when the prompt template changes so cached results never
        silently mix versions.
    """

    name: str = "ZeroShot-Claude"

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        *,
        K: int = 4,
        temperature: float = 0.0,
        max_tokens: int = 2,
        seed: int = 0,
        prompt_version: str = "zero-shot-rank-v1",
    ) -> None:
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")
        self.llm_client: LLMClient = llm_client if llm_client is not None else StubLLMClient()
        self.K = int(K)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.seed = int(seed)
        self.prompt_version = str(prompt_version)

    def fit(
        self,
        train: BaselineEventBatch,
        val: Optional[BaselineEventBatch] = None,
    ) -> ZeroShotClaudeRankerFitted:
        """Trivial fit: no training. Validates batch shape + returns wrapper.

        Only validation:

        - asserts ``train.n_alternatives == 4`` (baseline is hard-coded to J=4).
        - warns-via-raise if ``raw_events`` is missing; the fitted object
          cannot score without per-event ``c_d`` / ``alt_texts``.
        """
        if train.n_events == 0:
            raise ValueError("ZeroShotClaudeRanker.fit received an empty train batch")
        if train.n_alternatives != len(DEFAULT_LETTERS):
            raise ValueError(
                f"ZeroShot-Claude requires n_alternatives={len(DEFAULT_LETTERS)} "
                f"(got {train.n_alternatives}). Adjust the letter set or use a "
                "J=4 batch."
            )
        if train.raw_events is None:
            raise ValueError(
                "ZeroShot-Claude needs raw_events with 'c_d' and 'alt_texts'; "
                "BaselineEventBatch.raw_events is None. Rebuild the batch from "
                "records_to_baseline_batch so these fields survive."
            )

        model_id = str(getattr(self.llm_client, "model_id", "unknown") or "unknown")
        return ZeroShotClaudeRankerFitted(
            name=self.name,
            llm_client=self.llm_client,
            n_permutations=self.K,
            seed=self.seed,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            prompt_version=self.prompt_version,
            letters=DEFAULT_LETTERS,
            model_id=model_id,
        )


__all__ = [
    "SYSTEM_PROMPT",
    "USER_TEMPLATE",
    "ZeroShotClaudeRanker",
    "ZeroShotClaudeRankerFitted",
]
