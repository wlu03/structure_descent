"""Tests for ``src.outcomes.prompts`` (redesign.md §3.2)."""

from __future__ import annotations

import pytest

from src.outcomes.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    USER_BLOCK_TEMPLATE,
    build_messages,
    build_system_prompt,
    build_user_block,
)


# Golden reference for the §3.2 system prompt. Any edit here MUST also be a
# deliberate edit to `SYSTEM_PROMPT` and a `PROMPT_VERSION` bump.
SYSTEM_PROMPT_GOLDEN: str = (
    "You generate short first-person outcome narratives for a decision-maker "
    "considering an alternative. Produce exactly {K} sentences, each 10–25 "
    "words, each describing a *different type* of consequence: financial, "
    "health/physical, convenience/time, emotional/identity, social/relational. "
    "Sentences are in the first person, present or near-future tense, and "
    "grounded in the person's specific context. Do not describe the product. "
    "Do not hedge (\"might\", \"could\" are fine; \"I think\" is not). Do not "
    "number the sentences; separate them with newlines."
)


def _alt() -> dict:
    return {
        "title": "SuperWidget 3000",
        "category": "Home & Kitchen",
        "price": 29.99,
        "popularity_rank": 1234,
    }


def test_system_prompt_contains_key_phrases():
    prompt = build_system_prompt(K=3)
    assert "first-person" in prompt
    assert "financial" in prompt
    assert "different type" in prompt
    # K is substituted literally as the sentence count.
    assert "exactly 3 sentences" in prompt


def test_system_prompt_frozen():
    # The module constant (still containing the {K} placeholder) must match the
    # golden reference exactly, modulo surrounding whitespace.
    assert SYSTEM_PROMPT.strip() == SYSTEM_PROMPT_GOLDEN.strip()


def test_prompt_version_is_v1():
    assert PROMPT_VERSION == "v1"


def test_user_block_substitutes():
    block = build_user_block(c_d="I am a person.", alt=_alt(), K=3)
    assert "SuperWidget 3000" in block
    assert "Home & Kitchen" in block
    assert "$29.99" in block
    assert "1234" in block
    assert "K=3" in block
    assert "I am a person." in block


@pytest.mark.parametrize("missing", ["title", "category", "price", "popularity_rank"])
def test_user_block_missing_field_raises(missing: str):
    alt = _alt()
    del alt[missing]
    with pytest.raises(ValueError):
        build_user_block(c_d="ctx", alt=alt, K=3)


def test_optional_fields_render():
    block = build_user_block(
        c_d="ctx",
        alt=_alt(),
        K=3,
        optional_fields={"brand": "Acme"},
    )
    assert "- brand: Acme" in block
    # The optional line must appear after the fixed fields and before the
    # "Generate K=" closing line.
    pop_idx = block.index("Popularity:")
    brand_idx = block.index("- brand: Acme")
    gen_idx = block.index("Generate K=")
    assert pop_idx < brand_idx < gen_idx


def test_build_messages_shape():
    c_d = "Person profile\n- Age: mid-30s"
    messages = build_messages(c_d=c_d, alt=_alt(), K=3)
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert c_d in messages[1]["content"]
    # System content is the K-substituted prompt, not the raw template.
    assert "{K}" not in messages[0]["content"]
    assert "exactly 3 sentences" in messages[0]["content"]


def test_no_trailing_k_leak():
    # A prior K=3 call must not contaminate a later K=7 call.
    _ = build_system_prompt(K=3)
    _ = build_user_block(c_d="c", alt=_alt(), K=3)

    system_7 = build_system_prompt(K=7)
    block_7 = build_user_block(c_d="c", alt=_alt(), K=7)

    assert "exactly 7 sentences" in system_7
    assert "exactly 3 sentences" not in system_7
    assert "K=7" in block_7
    assert "K=3" not in block_7


def test_template_contains_placeholders():
    # Sanity check: the raw template still carries its format placeholders so a
    # regression (e.g. someone pre-substituting the template) fails loudly.
    for token in ("{c_d}", "{title}", "{category}", "{price}", "{popularity_rank}", "{K}", "{optional_fields}"):
        assert token in USER_BLOCK_TEMPLATE


def test_no_optional_fields_has_no_stray_blank_line():
    block = build_user_block(c_d="ctx", alt=_alt(), K=3)
    # When there are no optional fields the Popularity line is followed by the
    # single mandated blank line and then "Generate K=...", with no double
    # blank line in between.
    assert "\n\n\nGenerate K=" not in block
    assert "Popularity: 1234\n\nGenerate K=3" in block


def test_invalid_K_raises():
    with pytest.raises(ValueError):
        build_system_prompt(K=0)
    with pytest.raises(ValueError):
        build_system_prompt(K=-1)
    with pytest.raises(ValueError):
        build_user_block(c_d="c", alt=_alt(), K=0)
