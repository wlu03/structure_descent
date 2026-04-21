"""Tests for ``src.outcomes.generate`` (redesign.md §3.3, §3.4, §3.5 hooks).

All tests use the hermetic :class:`StubLLMClient`; the one case that exercises
:class:`AnthropicLLMClient` does so via ``sys.modules`` monkeypatching so the
real SDK is never contacted.
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any

import pytest

from src.outcomes import generate as gen_mod
from src.outcomes.cache import OutcomesCache
from src.outcomes.generate import (
    SENTINEL_OUTCOME,
    AnthropicLLMClient,
    GenerationResult,
    OutcomesPayload,
    StubLLMClient,
    generate_outcomes,
    parse_completion,
)
from src.outcomes.prompts import PROMPT_VERSION, build_messages


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

_ALT = {
    "title": "Noise-cancelling Headphones",
    "category": "Electronics",
    "price": 129.99,
    "popularity_rank": 42,
}
_C_D = "Person profile\n- Age: mid-30s; household of 4.\n- Income: about $55k/year."


def _make_cache(tmp_cache_dir: Path) -> OutcomesCache:
    return OutcomesCache(tmp_cache_dir / "outcomes.sqlite")


def _count_words(sentence: str) -> int:
    return len(sentence.split())


# ---------------------------------------------------------------------------
# StubLLMClient
# ---------------------------------------------------------------------------

def test_stub_deterministic_same_seed() -> None:
    """Same (messages, seed) → identical completion text and identical
    outcomes list after parsing."""
    client = StubLLMClient()
    messages = build_messages(c_d=_C_D, alt=_ALT, K=3)

    r1 = client.generate(messages, temperature=0.8, top_p=0.95, max_tokens=180, seed=7)
    r2 = client.generate(messages, temperature=0.8, top_p=0.95, max_tokens=180, seed=7)

    assert r1.text == r2.text
    assert r1.model_id == r2.model_id == "stub-v1"
    assert r1.finish_reason == "stop"

    # Also round-trip via generate_outcomes without caching to confirm
    # stability at the higher level.
    p1 = generate_outcomes(
        "cust", "asin-1", _C_D, _ALT,
        K=3, seed=7, prompt_version=PROMPT_VERSION, client=client,
    )
    p2 = generate_outcomes(
        "cust", "asin-1", _C_D, _ALT,
        K=3, seed=7, prompt_version=PROMPT_VERSION, client=client,
    )
    assert p1.outcomes == p2.outcomes


def test_stub_changes_with_seed() -> None:
    """Different seeds on the same messages must change at least one line."""
    client = StubLLMClient()
    messages = build_messages(c_d=_C_D, alt=_ALT, K=3)

    out_a = client.generate(messages, temperature=0.8, top_p=0.95, max_tokens=180, seed=1).text.split("\n")

    # Scan a handful of seeds to guarantee divergence somewhere — even if
    # one byte of the digest happens to collide, others won't.
    diverged = False
    for alt_seed in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16):
        out_b = client.generate(
            messages, temperature=0.8, top_p=0.95, max_tokens=180, seed=alt_seed
        ).text.split("\n")
        if any(a != b for a, b in zip(out_a, out_b)):
            diverged = True
            break
    assert diverged, "StubLLMClient returned identical outputs for many distinct seeds"


def test_stub_output_shape() -> None:
    """Exactly K outputs, first person, 10-25 words each — for a range of K."""
    client = StubLLMClient()
    for K in (1, 3, 5):
        messages = build_messages(c_d=_C_D, alt=_ALT, K=K)
        res = client.generate(messages, temperature=0.8, top_p=0.95, max_tokens=180, seed=42)
        lines = res.text.split("\n")
        assert len(lines) == K
        for line in lines:
            assert line.startswith("I "), f"not first person: {line!r}"
            wc = _count_words(line)
            assert 10 <= wc <= 25, f"word count {wc} out of [10, 25]: {line!r}"


# ---------------------------------------------------------------------------
# parse_completion
# ---------------------------------------------------------------------------

def test_parse_completion_truncates_to_K() -> None:
    """Extra lines beyond K are dropped; blank lines are ignored."""
    text = "\n".join(
        [
            "I feel better.",
            "",
            "I save money.",
            "I sleep well.",
            "I am happy.",
            "I am relaxed.",
        ]
    )
    out = parse_completion(text, K=3)
    assert out == ["I feel better.", "I save money.", "I sleep well."]


def test_parse_completion_pads_with_sentinel(caplog: pytest.LogCaptureFixture) -> None:
    """Short completions get padded with SENTINEL_OUTCOME and a warning fires."""
    short_text = "I save money."
    with caplog.at_level(logging.WARNING, logger="src.outcomes.generate"):
        out = parse_completion(
            short_text,
            K=3,
            context={"customer_id": "cust-1", "asin": "B0001"},
        )

    assert out == ["I save money.", SENTINEL_OUTCOME, SENTINEL_OUTCOME]
    warnings = [rec for rec in caplog.records if rec.levelno == logging.WARNING]
    assert warnings, "expected a warning when padding with sentinel"
    joined = " ".join(rec.getMessage() for rec in warnings)
    assert "cust-1" in joined and "B0001" in joined


# ---------------------------------------------------------------------------
# generate_outcomes — cache behaviour
# ---------------------------------------------------------------------------

class _CountingClient:
    """Wraps a real client, counting the number of generate() invocations."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.model_id = getattr(inner, "model_id", "counting")
        self.calls = 0

    def generate(self, messages, *, temperature, top_p, max_tokens, seed):
        self.calls += 1
        return self.inner.generate(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )


def test_cache_miss_then_hit(tmp_cache_dir: Path) -> None:
    """First call hits the client; second call with the same key doesn't."""
    client = _CountingClient(StubLLMClient())
    cache = _make_cache(tmp_cache_dir)
    try:
        p1 = generate_outcomes(
            "cust", "B01", _C_D, _ALT,
            K=3, seed=11, prompt_version=PROMPT_VERSION,
            client=client, cache=cache,
        )
        assert client.calls == 1

        p2 = generate_outcomes(
            "cust", "B01", _C_D, _ALT,
            K=3, seed=11, prompt_version=PROMPT_VERSION,
            client=client, cache=cache,
        )
        # Cache hit: client not invoked a second time.
        assert client.calls == 1
        assert p1.outcomes == p2.outcomes
        assert p1.metadata["seed"] == p2.metadata["seed"]
    finally:
        cache.close()


def test_cache_value_shape(tmp_cache_dir: Path) -> None:
    """Cached dict has outcomes + metadata; metadata holds every required field."""
    client = StubLLMClient()
    cache = _make_cache(tmp_cache_dir)
    try:
        generate_outcomes(
            "cust-A", "B-A", _C_D, _ALT,
            K=3, seed=1, prompt_version=PROMPT_VERSION,
            client=client, cache=cache,
        )
        raw = cache.get_outcomes("cust-A", "B-A", 1, PROMPT_VERSION)
    finally:
        cache.close()

    assert raw is not None
    assert set(raw.keys()) == {"outcomes", "metadata"}
    assert len(raw["outcomes"]) == 3

    metadata = raw["metadata"]
    for required in (
        "temperature",
        "top_p",
        "max_tokens",
        "model_id",
        "finish_reason",
        "seed",
        "prompt_version",
        "timestamp",
    ):
        assert required in metadata, f"missing metadata field: {required}"
    assert metadata["prompt_version"] == PROMPT_VERSION
    assert metadata["model_id"] == "stub-v1"


def test_generation_kwargs_respected(tmp_cache_dir: Path) -> None:
    """Custom generation kwargs propagate into metadata."""
    client = StubLLMClient()
    payload = generate_outcomes(
        "cust", "B02", _C_D, _ALT,
        K=3, seed=0, prompt_version=PROMPT_VERSION,
        client=client,
        generation_kwargs={"temperature": 0.5, "top_p": 0.9, "max_tokens": 140},
    )
    assert isinstance(payload, OutcomesPayload)
    assert payload.metadata["temperature"] == 0.5
    assert payload.metadata["top_p"] == 0.9
    assert payload.metadata["max_tokens"] == 140


# ---------------------------------------------------------------------------
# Diversity filter wiring
# ---------------------------------------------------------------------------

class _AlwaysFailFilter:
    """Diversity filter that never passes; counts the invocations."""

    def __init__(self) -> None:
        self.calls = 0
        self.last_seed_seen_in_text: list[str] = []

    def __call__(self, outcomes: list[str]) -> tuple[list[str], bool]:
        self.calls += 1
        return outcomes, False


class _FirstOkFilter:
    """Diversity filter that returns ok=True on its very first call."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, outcomes: list[str]) -> tuple[list[str], bool]:
        self.calls += 1
        return outcomes, True


def test_diversity_filter_retry_then_accept() -> None:
    """An always-fail filter still terminates after max_retries+1 client calls."""
    client = _CountingClient(StubLLMClient())
    diversity_filter = _AlwaysFailFilter()
    max_retries = 2

    payload = generate_outcomes(
        "cust", "B03", _C_D, _ALT,
        K=3, seed=5, prompt_version=PROMPT_VERSION,
        client=client, diversity_filter=diversity_filter, max_retries=max_retries,
    )

    assert client.calls == max_retries + 1  # 3 attempts total
    assert diversity_filter.calls == max_retries + 1
    # Final seed recorded should be seed + (max_retries)  -> 5 + 2 = 7.
    assert payload.metadata["seed"] == 5 + max_retries
    assert len(payload.outcomes) == 3


def test_diversity_filter_ok_breaks_loop() -> None:
    """Filter returning ok=True on the first try short-circuits the retry loop."""
    client = _CountingClient(StubLLMClient())
    diversity_filter = _FirstOkFilter()

    payload = generate_outcomes(
        "cust", "B04", _C_D, _ALT,
        K=3, seed=9, prompt_version=PROMPT_VERSION,
        client=client, diversity_filter=diversity_filter, max_retries=2,
    )

    assert client.calls == 1
    assert diversity_filter.calls == 1
    assert payload.metadata["seed"] == 9  # no seed bump


# ---------------------------------------------------------------------------
# AnthropicLLMClient lazy-import contract
# ---------------------------------------------------------------------------

def test_anthropic_client_lazy_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the module must not require `anthropic`; only constructing
    :class:`AnthropicLLMClient` should raise :class:`ImportError` when the
    package is missing."""

    # Simulate "anthropic not installed" by blocking imports of it.
    real_anthropic = sys.modules.get("anthropic")
    monkeypatch.setitem(sys.modules, "anthropic", None)

    # Drop any cached copy of the generate module so the reload really
    # re-executes the top-level code in a pristine environment.
    monkeypatch.delitem(sys.modules, "src.outcomes.generate", raising=False)

    reloaded = importlib.import_module("src.outcomes.generate")
    # Module import succeeds even though `anthropic` is unimportable — this is
    # the hermetic guarantee we need for CI workers without the llm extra.
    assert hasattr(reloaded, "AnthropicLLMClient")

    # Constructing it must now raise ImportError (lazy import fails).
    with pytest.raises(ImportError):
        reloaded.AnthropicLLMClient("claude-3-5-sonnet-20241022", api_key="dummy")

    # Cleanup: restore the real module so other tests in this session
    # (and any module state) remain sane.
    if real_anthropic is not None:
        monkeypatch.setitem(sys.modules, "anthropic", real_anthropic)
    # Re-import the canonical module to leave sys.modules in the normal state.
    monkeypatch.delitem(sys.modules, "src.outcomes.generate", raising=False)
    importlib.import_module("src.outcomes.generate")
