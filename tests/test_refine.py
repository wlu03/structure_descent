"""Tests for the curriculum-refinement loop (src/outcomes/refine.py).

Covers:
  - JSON-tolerant parse for the critic completion (raw, fenced, junk).
  - critique() with a canned client returns the structured scores.
  - revise() with a canned client respects K and reuses parse_completion.
  - refine_outcomes() skips revise when both scores >= accept_threshold.
  - refine_outcomes() runs revise when either score < accept_threshold.

LLM clients are stubbed by a tiny ``_CannedClient`` class — both real
clients (Anthropic, OpenAI, Gemini) conform to the same protocol used
here, so the test surface stays provider-free.
"""

from __future__ import annotations

import pytest

from src.outcomes.generate import GenerationResult
from src.outcomes.refine import (
    CritiqueResult,
    _parse_critique_json,
    critique,
    refine_outcomes,
    revise,
)


# ---------------------------------------------------------------------------
# Tiny canned LLM stub. Real clients implement exactly the same generate()
# signature, so tests don't pin to any provider SDK.
# ---------------------------------------------------------------------------


class _CannedClient:
    def __init__(self, model_id: str, scripted_replies: list[str]) -> None:
        self.model_id = model_id
        self._replies = list(scripted_replies)
        self.calls: list[dict] = []

    def generate(
        self,
        messages,
        *,
        temperature,
        top_p,
        max_tokens,
        seed,
    ) -> GenerationResult:
        self.calls.append({
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        })
        if not self._replies:
            raise AssertionError("_CannedClient ran out of scripted replies")
        text = self._replies.pop(0)
        return GenerationResult(
            text=text, finish_reason="end_turn", model_id=self.model_id,
        )


_ALT = {
    "title": "Generic widget",
    "category": "widgets",
    "price": "12.99",
    "popularity_rank": "high",
}
_C_D = "Customer is a 32-year-old in a mid-size city with a routine of "
_C_D += "weekly online orders."


# ---------------------------------------------------------------------------
# JSON parse
# ---------------------------------------------------------------------------


def test_parse_critique_json_plain() -> None:
    text = '{"plausibility": 4, "diversity": 5, "notes": "ok"}'
    out = _parse_critique_json(text)
    assert out == {"plausibility": 4, "diversity": 5, "notes": "ok"}


def test_parse_critique_json_fenced() -> None:
    text = "```json\n{\"plausibility\": 2, \"diversity\": 3, \"notes\": \"vague\"}\n```"
    out = _parse_critique_json(text)
    assert out["plausibility"] == 2
    assert out["diversity"] == 3
    assert out["notes"] == "vague"


def test_parse_critique_json_with_preamble() -> None:
    text = "Here is the analysis.\n{\"plausibility\":1,\"diversity\":2,\"notes\":\"too generic\"}"
    out = _parse_critique_json(text)
    assert out["plausibility"] == 1
    assert out["diversity"] == 2


def test_parse_critique_json_no_json_raises() -> None:
    with pytest.raises(ValueError):
        _parse_critique_json("Sorry, I can't comply.")


# ---------------------------------------------------------------------------
# critique()
# ---------------------------------------------------------------------------


def test_critique_returns_clamped_scores() -> None:
    """Out-of-range JSON scores are clamped into 1-5."""
    client = _CannedClient(
        "stub-critic",
        ['{"plausibility": 9, "diversity": -2, "notes": "boundary test"}'],
    )
    outs = ["o1.", "o2.", "o3."]
    cr = critique(
        c_d=_C_D, alt=_ALT, outcomes=outs, K=3,
        critic_client=client, seed=7,
    )
    assert cr.plausibility == 5  # clamped from 9
    assert cr.diversity == 1     # clamped from -2
    assert "boundary" in cr.notes
    assert cr.model_id == "stub-critic"


def test_critique_falls_back_on_parse_error() -> None:
    """Unparseable critique → 3/3 fallback so the pipeline keeps moving."""
    client = _CannedClient("stub-critic", ["I refuse to score."])
    cr = critique(
        c_d=_C_D, alt=_ALT, outcomes=["a.", "b.", "c."], K=3,
        critic_client=client, seed=0,
    )
    assert cr.plausibility == 3
    assert cr.diversity == 3
    assert "<parse error>" in cr.notes


# ---------------------------------------------------------------------------
# revise()
# ---------------------------------------------------------------------------


def test_revise_returns_K_outcomes() -> None:
    """Writer returns K newline-separated sentences; parser yields a list[K]."""
    revised_text = (
        "I save twelve dollars this month on my grocery routine.\n"
        "I move through checkout in under ninety seconds, which fits my schedule.\n"
        "I feel relieved that the brand matches what my partner uses."
    )
    writer = _CannedClient("stub-writer", [revised_text])
    crit = CritiqueResult(
        plausibility=2, diversity=2, notes="too generic", raw="", model_id=""
    )
    out = revise(
        c_d=_C_D, alt=_ALT, outcomes=["a.", "b.", "c."], K=3,
        critique_result=crit, writer_client=writer, seed=11,
    )
    assert isinstance(out, list)
    assert len(out) == 3
    assert all(isinstance(s, str) and len(s) > 0 for s in out)


# ---------------------------------------------------------------------------
# refine_outcomes() (full critique + maybe revise)
# ---------------------------------------------------------------------------


def test_refine_skips_when_scores_above_threshold() -> None:
    """Both scores >= accept_threshold → originals returned, writer not called."""
    critic = _CannedClient(
        "stub-critic",
        ['{"plausibility": 5, "diversity": 4, "notes": "fine"}'],
    )
    writer = _CannedClient("stub-writer", [])  # zero replies — must not be called
    originals = ["o1.", "o2.", "o3."]
    result = refine_outcomes(
        c_d=_C_D, alt=_ALT, outcomes=originals, K=3,
        writer_client=writer, critic_client=critic, seed=1,
        accept_threshold=4,
    )
    assert result.skipped is True
    assert result.revised_outcomes == originals
    assert result.critique.plausibility == 5
    assert result.critique.diversity == 4
    assert len(writer.calls) == 0


def test_refine_runs_revise_when_below_threshold() -> None:
    """Either score < accept_threshold → revise() runs, new outcomes returned."""
    critic = _CannedClient(
        "stub-critic",
        ['{"plausibility": 2, "diversity": 5, "notes": "more specific"}'],
    )
    new_text = (
        "I save fourteen dollars on the grocery bill this week.\n"
        "I cut my cooking time by twenty minutes on weeknights.\n"
        "I feel more in control of my routine and post about it once."
    )
    writer = _CannedClient("stub-writer", [new_text])
    originals = ["a.", "b.", "c."]
    result = refine_outcomes(
        c_d=_C_D, alt=_ALT, outcomes=originals, K=3,
        writer_client=writer, critic_client=critic, seed=2,
        accept_threshold=4,
    )
    assert result.skipped is False
    assert result.revised_outcomes != originals
    assert len(result.revised_outcomes) == 3
    assert len(writer.calls) == 1
    # Critic was called exactly once.
    assert len(critic.calls) == 1
