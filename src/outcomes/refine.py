"""Critique-and-revise loop for PO-LEU outcome refinement.

Pure orchestration around two existing :class:`LLMClient`-protocol clients —
a *writer* (same model family that produced the original v1 outcomes, e.g.
Anthropic) and a *critic* (a different family, e.g. OpenAI or Gemini, so the
critic isn't echo-chambering the writer). This module stays free of
provider-specific code; both clients conform to the same protocol used by
:mod:`src.outcomes.generate`.

Flow per (customer, alternative):

1. Build critic prompt over the K v1 outcomes →
   :func:`src.outcomes.prompts.build_critic_messages`.
2. Critic LLM returns strict JSON: ``{plausibility, diversity, notes}``.
3. If both scores ≥ ``accept_threshold``, return the originals unchanged
   (no API spend, no churn — refining a passable case adds noise).
4. Otherwise build reviser prompt with the critique → writer LLM emits K
   revised outcomes (same parser/diversity-filter contract as v1).
5. The driver (:mod:`scripts.refine_outcomes`) writes the revised outcomes
   to :class:`OutcomesCache` under ``REFINED_PROMPT_VERSION``.

The cascade lookup in :func:`src.data.batching.assemble_batch` then prefers
refined entries on the third PO-LEU training round and falls back to v1 for
non-failure events.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from src.outcomes.generate import LLMClient, parse_completion
from src.outcomes.prompts import (
    REFINED_PROMPT_VERSION,
    build_critic_messages,
    build_reviser_messages,
    build_reviser_per_outcome_messages,
)

logger = logging.getLogger(__name__)


# Default decoding params for both clients. ``max_tokens`` is sized for
# reasoning-model criticism: GPT-5 / o1 / o3 spend output tokens on chain-
# of-thought before emitting the visible answer, so anything below ~1500
# truncates the JSON or returns empty. Non-reasoning models (Claude,
# Gemini Flash) only need ~150 tokens and are unaffected by the larger
# ceiling. Bumped from 320 → 2048 after observing GPT-5 hitting the
# "model output limit was reached" 400 + empty-completion fallback in the
# Apr-27 sweep. Mirrors v1 generator defaults from
# configs/default.yaml § outcomes.generator otherwise.
_DEFAULT_GEN_KWARGS: dict[str, Any] = {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 2048,
}


@dataclass
class CritiqueResult:
    """Structured critic output. ``raw`` is the unparsed completion text.

    ``weak_outcome_indices`` is the per-outcome flag list (0-indexed)
    introduced for the per-outcome reviser path: when non-empty, only
    those positions are rewritten and the strong outcomes are preserved
    byte-identical via splice. Empty list means either every outcome is
    acceptable, or the critic didn't supply position-level info — in
    the latter case the reviser falls back to the monolithic rewrite.
    """

    plausibility: int
    diversity: int
    notes: str
    raw: str
    model_id: str = ""
    weak_outcome_indices: list[int] = field(default_factory=list)


@dataclass
class RefinementResult:
    """End-to-end output of one (customer, alt) refinement call."""

    revised_outcomes: list[str]
    critique: CritiqueResult
    skipped: bool = False  # True when the critique passed accept_threshold
    metadata: dict[str, Any] = field(default_factory=dict)


def _parse_critique_json(raw: str) -> dict[str, Any]:
    """Tolerant JSON parse: strip code fences, find first ``{...}`` block."""
    text = raw.strip()
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # Find the first {...} block.
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"critic returned no JSON object: {raw[:200]!r}")
    return json.loads(m.group(0))


def critique(
    *,
    c_d: str,
    alt: Mapping[str, Any],
    outcomes: list[str],
    K: int,
    critic_client: LLMClient,
    seed: int,
    generation_kwargs: Mapping[str, Any] | None = None,
) -> CritiqueResult:
    """Score K candidate outcomes on plausibility + diversity.

    Returns ``CritiqueResult`` with both scores in 1-5 and a free-text notes
    field. Scores are clamped to 1-5 even if the critic emits out-of-range
    values; an unparseable critique falls back to ``plausibility=3,
    diversity=3, notes="<parse error>"`` so the pipeline still makes
    progress (the reviser will treat a 3/3 as "modest improvement needed").
    """
    if len(outcomes) != int(K):
        raise ValueError(
            f"critique: got {len(outcomes)} outcomes, expected K={K}"
        )
    messages = build_critic_messages(c_d=c_d, alt=alt, outcomes=outcomes, K=int(K))
    gen_kwargs = dict(_DEFAULT_GEN_KWARGS)
    if generation_kwargs:
        gen_kwargs.update(dict(generation_kwargs))
    # Lower temperature for the critic — JSON adherence matters more than
    # creativity here. Caller can override via ``generation_kwargs``.
    gen_kwargs.setdefault("temperature", 0.2)

    result = critic_client.generate(
        messages,
        temperature=float(gen_kwargs["temperature"]),
        top_p=float(gen_kwargs["top_p"]),
        max_tokens=int(gen_kwargs["max_tokens"]),
        seed=int(seed),
    )
    raw = getattr(result, "text", "") or ""
    try:
        parsed = _parse_critique_json(raw)
        p = int(parsed.get("plausibility", 3))
        d = int(parsed.get("diversity", 3))
        notes = str(parsed.get("notes", "")).strip()
        p = max(1, min(5, p))
        d = max(1, min(5, d))
        # Per-outcome flag list. Tolerant: missing field, non-list, or
        # mixed types all degrade gracefully to an empty list (which
        # routes the reviser to the monolithic path so behavior matches
        # the prior contract on critics that don't yet return the field).
        raw_indices = parsed.get("weak_outcome_indices", []) or []
        weak: list[int] = []
        if isinstance(raw_indices, list):
            seen: set[int] = set()
            for v in raw_indices:
                if isinstance(v, bool):
                    continue
                try:
                    iv = int(v)
                except (TypeError, ValueError):
                    continue
                if 0 <= iv < int(K) and iv not in seen:
                    weak.append(iv)
                    seen.add(iv)
            weak.sort()
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        logger.warning(
            "critique JSON parse failed (%s); defaulting to 3/3", exc
        )
        p, d, notes, weak = 3, 3, "<parse error>", []
    return CritiqueResult(
        plausibility=p,
        diversity=d,
        notes=notes,
        raw=raw,
        model_id=str(getattr(result, "model_id", "")),
        weak_outcome_indices=weak,
    )


def revise(
    *,
    c_d: str,
    alt: Mapping[str, Any],
    outcomes: list[str],
    K: int,
    critique_result: CritiqueResult,
    writer_client: LLMClient,
    seed: int,
    diversity_filter: Any | None = None,
    generation_kwargs: Mapping[str, Any] | None = None,
    per_outcome: bool = True,
) -> list[str]:
    """Rewrite outcomes given the critic's feedback.

    Two paths, selected by ``per_outcome`` × the critic output:

    * **Per-outcome (preferred)** — when ``per_outcome=True`` AND
      ``critique_result.weak_outcome_indices`` is non-empty, the writer
      regenerates ONLY those positions. Strong outcomes are preserved
      byte-identical via splice in this function. Less variance, less
      thematic drift on the strong outcomes.

    * **Monolithic (fallback)** — when ``per_outcome=False`` OR the
      critic returned no per-position info, the writer rewrites all K
      outcomes (the original v2_refined contract).

    Reuses :func:`src.outcomes.generate.parse_completion` (same parser
    the v1 generator uses) so the revised list goes through the same
    validation contract: N sentences (per-outcome) or K sentences
    (monolithic), each within length bounds. Optional
    ``diversity_filter`` is applied to the spliced final K-sentence
    result so diversity coherence holds across the strong + revised
    mixed set.
    """
    if len(outcomes) != int(K):
        raise ValueError(
            f"revise: got {len(outcomes)} outcomes, expected K={K}"
        )

    weak_indices = list(critique_result.weak_outcome_indices or [])
    use_per_outcome = bool(per_outcome) and len(weak_indices) > 0

    gen_kwargs = dict(_DEFAULT_GEN_KWARGS)
    if generation_kwargs:
        gen_kwargs.update(dict(generation_kwargs))

    if use_per_outcome:
        n = len(weak_indices)
        messages = build_reviser_per_outcome_messages(
            c_d=c_d,
            alt=alt,
            outcomes=outcomes,
            K=int(K),
            weak_indices=weak_indices,
            plausibility=int(critique_result.plausibility),
            diversity=int(critique_result.diversity),
            notes=critique_result.notes,
        )
        result = writer_client.generate(
            messages,
            temperature=float(gen_kwargs["temperature"]),
            top_p=float(gen_kwargs["top_p"]),
            max_tokens=int(gen_kwargs["max_tokens"]),
            seed=int(seed),
        )
        raw = getattr(result, "text", "") or ""
        parsed_n = parse_completion(raw, K=int(n))
        if len(parsed_n) != n:
            raise AssertionError(
                f"revise (per-outcome): parser returned {len(parsed_n)} "
                f"outcomes, expected N={n}"
            )
        # Splice: keep strong outcomes verbatim, replace weak positions.
        merged = list(outcomes)
        for slot_idx, new_text in zip(weak_indices, parsed_n):
            merged[slot_idx] = new_text
        parsed = merged
    else:
        messages = build_reviser_messages(
            c_d=c_d,
            alt=alt,
            outcomes=outcomes,
            K=int(K),
            plausibility=int(critique_result.plausibility),
            diversity=int(critique_result.diversity),
            notes=critique_result.notes,
        )
        result = writer_client.generate(
            messages,
            temperature=float(gen_kwargs["temperature"]),
            top_p=float(gen_kwargs["top_p"]),
            max_tokens=int(gen_kwargs["max_tokens"]),
            seed=int(seed),
        )
        raw = getattr(result, "text", "") or ""
        parsed = parse_completion(raw, K=int(K))

    if diversity_filter is not None:
        try:
            rewritten, ok = diversity_filter(parsed)
            if ok:
                parsed = rewritten
            else:
                logger.warning(
                    "revise: diversity_filter rejected; keeping unrewritten"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("revise: diversity_filter raised %s; ignoring", exc)

    if len(parsed) != int(K):
        raise AssertionError(
            f"revise: parser returned {len(parsed)} outcomes, expected K={K}"
        )
    return parsed


def refine_outcomes(
    *,
    c_d: str,
    alt: Mapping[str, Any],
    outcomes: list[str],
    K: int,
    writer_client: LLMClient,
    critic_client: LLMClient,
    seed: int,
    accept_threshold: int = 4,
    diversity_filter: Any | None = None,
    generation_kwargs: Mapping[str, Any] | None = None,
    per_outcome: bool = True,
) -> RefinementResult:
    """End-to-end critique + (optionally) revise for one (customer, alt).

    If both critic scores are >= ``accept_threshold`` AND the critic
    flagged no specific weak positions, returns the originals unchanged
    with ``skipped=True`` (no revise call).

    Otherwise runs :func:`revise`. With ``per_outcome=True`` (default),
    the reviser only rewrites positions in
    ``critique_result.weak_outcome_indices``; strong outcomes are
    preserved byte-identical via splice. When the critic doesn't supply
    weak indices, the reviser falls back to the monolithic K-sentence
    rewrite (matches the prior v2_refined contract).
    """
    crit = critique(
        c_d=c_d,
        alt=alt,
        outcomes=outcomes,
        K=int(K),
        critic_client=critic_client,
        seed=int(seed),
        generation_kwargs=generation_kwargs,
    )
    has_weak = len(crit.weak_outcome_indices) > 0
    scores_pass = (
        crit.plausibility >= accept_threshold
        and crit.diversity >= accept_threshold
    )
    # Skip the revise call when scores pass AND the critic didn't single
    # out any specific weak positions. If scores pass but the critic
    # nonetheless flagged a position as weak, trust the per-outcome
    # signal and rewrite that one slot — this is the cheap, high-
    # surgical-precision path the new contract is designed for.
    if scores_pass and not has_weak:
        return RefinementResult(
            revised_outcomes=list(outcomes),
            critique=crit,
            skipped=True,
            metadata={
                "accept_threshold": int(accept_threshold),
                "critic_model_id": crit.model_id,
                "writer_model_id": "",
                "prompt_version": REFINED_PROMPT_VERSION,
                "weak_outcome_indices": list(crit.weak_outcome_indices),
                "per_outcome": False,
            },
        )
    revised = revise(
        c_d=c_d,
        alt=alt,
        outcomes=outcomes,
        K=int(K),
        critique_result=crit,
        writer_client=writer_client,
        seed=int(seed),
        diversity_filter=diversity_filter,
        generation_kwargs=generation_kwargs,
        per_outcome=per_outcome,
    )
    revise_path = (
        "per_outcome" if (per_outcome and has_weak) else "monolithic"
    )
    return RefinementResult(
        revised_outcomes=revised,
        critique=crit,
        skipped=False,
        metadata={
            "accept_threshold": int(accept_threshold),
            "critic_model_id": crit.model_id,
            "writer_model_id": str(getattr(writer_client, "model_id", "")),
            "prompt_version": REFINED_PROMPT_VERSION,
            "weak_outcome_indices": list(crit.weak_outcome_indices),
            "revise_path": revise_path,
        },
    )
