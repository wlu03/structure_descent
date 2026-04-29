"""Tests for the anchored K=5 prompt path (Group-3 fix).

The anchored path forces the LLM to produce exactly one outcome per M=5
attribute-head axis in canonical order, eliminating the starved-head failure
mode that K=3 produced (m0 near-zero variance, m3 runaway scale).
"""

from __future__ import annotations

import pytest

from src.outcomes.prompts import (
    ANCHORED_AXES,
    PROMPT_VERSION_ANCHORED,
    SYSTEM_PROMPT_ANCHORED,
    build_messages_anchored,
)


def _alt() -> dict:
    return {
        "title": "SuperWidget 3000",
        "category": "Home & Kitchen",
        "price": 29.99,
        "popularity_rank": 1234,
    }


def test_prompt_version_constant():
    assert PROMPT_VERSION_ANCHORED == "v3_anchored"


def test_anchored_axes_count():
    assert len(ANCHORED_AXES) == 5


def test_anchored_axes_order():
    assert ANCHORED_AXES == (
        "financial",
        "health",
        "convenience",
        "emotional",
        "social",
    )


def test_build_messages_anchored_renders_axes():
    messages = build_messages_anchored(c_d="I am a person.", alt=_alt(), K=5)
    assert messages[0]["role"] == "system"
    sys_content = messages[0]["content"]
    for axis in ANCHORED_AXES:
        assert axis in sys_content, f"axis {axis!r} missing from system prompt"
    # Sanity: SYSTEM_PROMPT_ANCHORED was used, not the legacy SYSTEM_PROMPT.
    assert sys_content == SYSTEM_PROMPT_ANCHORED


@pytest.mark.parametrize("bad_K", [1, 3, 4, 6, 10])
def test_build_messages_anchored_rejects_K_mismatch(bad_K: int):
    with pytest.raises(ValueError) as exc:
        build_messages_anchored(c_d="ctx", alt=_alt(), K=bad_K)
    msg = str(exc.value)
    assert "K=5" in msg, f"error message must name expected K=5; got {msg!r}"


def test_anchored_distinct_cache_key():
    """Regression for the cache cascade: same (cust, asin, seed, K, model_id,
    c_d) tuple under prompt_version="v3_anchored" must NOT collide with the
    same tuple under "v2"."""
    from src.outcomes.generate import build_cache_prompt_version

    common = dict(K=5, model_id="stub", c_d="I am a person.")
    v2_key = build_cache_prompt_version(prompt_version="v2", **common)
    v3_key = build_cache_prompt_version(
        prompt_version="v3_anchored", **common
    )
    assert v2_key != v3_key
    assert v3_key.startswith("v3_anchored-")
    assert v2_key.startswith("v2-")
