"""Tests for ``src.outcomes.generate`` (redesign.md §3.3, §3.4, §3.5 hooks).

All tests use the hermetic :class:`StubLLMClient`; the one case that exercises
:class:`AnthropicLLMClient` does so via ``sys.modules`` monkeypatching so the
real SDK is never contacted.
"""

from __future__ import annotations

import hashlib
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
# Cache-key suffix component for the canonical test c_d (V5-B1 fix:
# generate.py folds a sha256(c_d)[:16] prefix into cache_prompt_version
# so per-event c_d variations don't collide under the same (cust, asin,
# seed) tuple). Tests that read cache entries by composite key must use
# this fragment.
_CD_HASH = hashlib.sha256(_C_D.encode("utf-8")).hexdigest()[:16]


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
        # generate_outcomes folds K, the producing client's model_id,
        # AND a sha256 prefix of c_d into the cache-key's prompt_version
        # field (e.g. "v2-K3-stub-v1-cd<hash>"); the raw key-level
        # read-back must use that full composite value to find the entry.
        raw = cache.get_outcomes(
            "cust-A", "B-A", 1,
            f"{PROMPT_VERSION}-K3-stub-v1-cd{_CD_HASH}",
        )
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
        "cache_prompt_version",
        "timestamp",
    ):
        assert required in metadata, f"missing metadata field: {required}"
    # User-supplied prompt_version is preserved verbatim; the composite
    # cache_prompt_version folds in K and the sanitised model_id so a
    # stub-run entry can't collide with a real-run entry.
    assert metadata["prompt_version"] == PROMPT_VERSION
    assert metadata["cache_prompt_version"] == (
        f"{PROMPT_VERSION}-K3-stub-v1-cd{_CD_HASH}"
    )
    assert metadata["model_id"] == "stub-v1"


class _FakeRealClient:
    """Non-stub pseudo-client used to verify cache-key isolation.

    Does NOT set ``_is_stub`` and does NOT inherit a "Stub"-prefixed
    class name, so :func:`src.data.batching._is_stub_client` will
    classify it as real. ``model_id`` is a plausible real value so the
    cache composite ``{prompt_version}-K{K}-{model_id}`` diverges from
    the stub's ``stub-v1`` fragment.
    """

    def __init__(self, model_id: str = "claude-test-7") -> None:
        self.model_id = model_id
        self.calls = 0

    def generate(self, messages, *, temperature, top_p, max_tokens, seed):
        self.calls += 1
        # Deterministic, recognisably-non-stub completion so the test
        # can tell at a glance that the "real" path produced it.
        lines = [f"I am a real-client outcome number {i + 1}." for i in range(3)]
        # Pad / expand so word counts don't trip any downstream validator.
        lines = [
            f"{ln} It runs for the sake of the cache-isolation test."
            for ln in lines
        ]
        return GenerationResult(
            text="\n".join(lines),
            finish_reason="stop",
            model_id=self.model_id,
        )


def test_cache_key_separates_stub_and_real(tmp_cache_dir: Path) -> None:
    """A cache populated by a stub must NOT short-circuit a real-client
    call on the same ``(customer_id, asin, seed, prompt_version)`` tuple.

    This is the regression the model_id fold is for: before the fix, a
    stub run would happily hand its outcomes back to a subsequent real
    run sharing the base key. Post-fix the composite
    ``{prompt_version}-K{K}-{model_id}`` is different, so the real
    client must actually be invoked (cache miss on the real-keyed
    lookup) and its outcomes must be what we get back.
    """
    stub = StubLLMClient()
    real = _FakeRealClient(model_id="claude-test-7")
    cache = _make_cache(tmp_cache_dir)
    try:
        # Populate the cache with a stub outcome.
        stub_payload = generate_outcomes(
            "cust", "B01", _C_D, _ALT,
            K=3, seed=11, prompt_version=PROMPT_VERSION,
            client=stub, cache=cache,
        )
        assert stub_payload.metadata["model_id"] == "stub-v1"

        # Now call with a "real" client using the SAME base tuple.
        # Pre-fix behaviour: this would have returned the stub outcomes.
        # Post-fix: cache miss on the real-keyed composite -> real client
        # is invoked exactly once and its outcomes come back.
        real_payload = generate_outcomes(
            "cust", "B01", _C_D, _ALT,
            K=3, seed=11, prompt_version=PROMPT_VERSION,
            client=real, cache=cache,
        )

        assert real.calls == 1, "real client must be invoked on cache miss"
        assert real_payload.metadata["model_id"] == "claude-test-7"
        # Outcomes must come from the real client, NOT the stub.
        assert real_payload.outcomes != stub_payload.outcomes
        # Sanity: both cache entries coexist under distinct composite keys.
        stub_raw = cache.get_outcomes(
            "cust", "B01", 11,
            f"{PROMPT_VERSION}-K3-stub-v1-cd{_CD_HASH}",
        )
        real_raw = cache.get_outcomes(
            "cust", "B01", 11,
            f"{PROMPT_VERSION}-K3-claude-test-7-cd{_CD_HASH}",
        )
        assert stub_raw is not None
        assert real_raw is not None
        assert stub_raw["outcomes"] == stub_payload.outcomes
        assert real_raw["outcomes"] == real_payload.outcomes
    finally:
        cache.close()


def test_stub_llm_client_carries_is_stub_flag() -> None:
    """``StubLLMClient`` must expose ``_is_stub=True`` so downstream
    guards can reliably distinguish it from real clients without
    resorting to class-name sniffing."""
    client = StubLLMClient()
    assert getattr(client, "_is_stub", False) is True
    # Class-level presence: subclasses or test wrappers that expose the
    # attribute still test True.
    assert getattr(StubLLMClient, "_is_stub", False) is True


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
# AnthropicLLMClient cache_control wiring
# ---------------------------------------------------------------------------

class _FakeUsage:
    """Mimic the ``response.usage`` struct emitted by the real SDK."""

    def __init__(
        self,
        *,
        cache_creation_input_tokens: int | None = 0,
        cache_read_input_tokens: int | None = 0,
    ) -> None:
        self.cache_creation_input_tokens = cache_creation_input_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(
        self,
        text: str = "I save money.\nI feel better.\nI sleep well.",
        *,
        model: str = "claude-test",
        stop_reason: str = "end_turn",
        usage: _FakeUsage | None = None,
    ) -> None:
        self.content = [_FakeTextBlock(text)]
        self.stop_reason = stop_reason
        self.model = model
        self.usage = usage if usage is not None else _FakeUsage()


class _CapturingMessages:
    """Captures the kwargs passed to ``messages.create`` for assertions."""

    def __init__(self, response_factory, raise_first: Exception | None = None):
        self._response_factory = response_factory
        self._raise_first = raise_first
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise_first is not None and len(self.calls) == 1:
            exc = self._raise_first
            self._raise_first = None
            raise exc
        return self._response_factory()


class _FakeAnthropicSDK:
    """Namespace that mimics ``anthropic.Anthropic(...)`` construction."""

    def __init__(self, messages_stub: _CapturingMessages) -> None:
        self._messages_stub = messages_stub
        # Expose ``BadRequestError`` so generate.py's ``except`` clause
        # binds to the real class — the test has already imported it.
        import anthropic as _real
        self.BadRequestError = _real.BadRequestError

    def Anthropic(self, **_kwargs):  # noqa: N802  (match SDK attr name)
        outer = self

        class _Client:
            messages = outer._messages_stub

        return _Client()


def _build_fake_anthropic(
    monkeypatch: pytest.MonkeyPatch,
    messages_stub: _CapturingMessages,
) -> None:
    """Install the fake anthropic module under the real one's name.

    We monkeypatch only the attributes the client touches
    (``Anthropic`` constructor + ``BadRequestError``) so that the
    ``except anthropic.BadRequestError`` branch in generate.py still
    catches instances of the real class.
    """
    fake = _FakeAnthropicSDK(messages_stub)
    import anthropic as real_mod

    monkeypatch.setattr(real_mod, "Anthropic", fake.Anthropic)


def _make_anthropic_client(monkeypatch: pytest.MonkeyPatch, stub: _CapturingMessages):
    """Construct an ``AnthropicLLMClient`` wired to ``stub.create``."""
    _build_fake_anthropic(monkeypatch, stub)
    return AnthropicLLMClient("claude-test-model", api_key="dummy-key")


def _anthropic_messages(K: int = 3) -> list[dict]:
    return build_messages(c_d=_C_D, alt=_ALT, K=K)


def test_anthropic_client_includes_cache_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cache_control must land on the system prompt and the c_d prefix of the
    user message; the ``ALTERNATIVE`` suffix must stay uncached."""
    stub = _CapturingMessages(lambda: _FakeResponse())
    client = _make_anthropic_client(monkeypatch, stub)

    client.generate(
        _anthropic_messages(),
        temperature=0.8,
        top_p=0.95,
        max_tokens=180,
        seed=7,
    )

    assert len(stub.calls) == 1
    kw = stub.calls[0]

    # System block: list of one typed dict with ephemeral cache_control.
    assert isinstance(kw["system"], list)
    assert len(kw["system"]) == 1
    sys_block = kw["system"][0]
    assert sys_block["type"] == "text"
    assert sys_block["cache_control"] == {"type": "ephemeral"}
    assert "generate" in sys_block["text"].lower()

    # User message: content is a list of 2 typed dicts; first has
    # cache_control and contains CONTEXT:, second is uncached and starts
    # with the ALTERNATIVE: marker.
    assert isinstance(kw["messages"], list) and len(kw["messages"]) == 1
    user_msg = kw["messages"][0]
    assert user_msg["role"] == "user"
    blocks = user_msg["content"]
    assert isinstance(blocks, list) and len(blocks) == 2
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert blocks[0]["text"].startswith("CONTEXT:")
    assert "ALTERNATIVE:" not in blocks[0]["text"]
    assert "cache_control" not in blocks[1]
    assert blocks[1]["text"].startswith("\n\nALTERNATIVE:")
    assert "Generate K=3" in blocks[1]["text"]


def test_anthropic_client_falls_back_on_bad_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the server rejects cache_control, the client retries with the
    plain-string shape and returns a normal completion."""
    import anthropic
    import httpx

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(400, request=req)
    err = anthropic.BadRequestError(
        message="cache_control not supported",
        response=resp,
        body={"error": "cache_control not supported"},
    )

    stub = _CapturingMessages(lambda: _FakeResponse(), raise_first=err)
    client = _make_anthropic_client(monkeypatch, stub)

    result = client.generate(
        _anthropic_messages(),
        temperature=0.8,
        top_p=0.95,
        max_tokens=180,
        seed=3,
    )

    # Two calls: the first (cached) raised, the second (plain) succeeded.
    assert len(stub.calls) == 2

    first, second = stub.calls
    # The first call carried the cached shape.
    assert isinstance(first["system"], list)
    assert isinstance(first["messages"][0]["content"], list)

    # The retry falls back to plain strings.
    assert isinstance(second["system"], str)
    assert isinstance(second["messages"], list)
    assert isinstance(second["messages"][0]["content"], str)

    assert isinstance(result, GenerationResult)
    assert result.text.startswith("I ")


def test_anthropic_client_no_split_marker_single_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User messages lacking ``\\n\\nALTERNATIVE:`` fall back to a single
    uncached block."""
    stub = _CapturingMessages(lambda: _FakeResponse())
    client = _make_anthropic_client(monkeypatch, stub)

    # Minimal hand-rolled messages that skip the USER_BLOCK_TEMPLATE
    # entirely — simulates a future prompt schema change.
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Hello with no split marker here."},
    ]

    client.generate(
        messages,
        temperature=0.8,
        top_p=0.95,
        max_tokens=180,
        seed=1,
    )

    kw = stub.calls[0]
    blocks = kw["messages"][0]["content"]
    assert isinstance(blocks, list) and len(blocks) == 1
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"] == "Hello with no split marker here."
    assert "cache_control" not in blocks[0]


def test_cache_usage_logged_at_debug(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``cache_creation_input_tokens`` + ``cache_read_input_tokens`` from
    ``response.usage`` must be emitted at DEBUG level once per call."""
    usage = _FakeUsage(
        cache_creation_input_tokens=512,
        cache_read_input_tokens=128,
    )
    stub = _CapturingMessages(lambda: _FakeResponse(usage=usage))
    client = _make_anthropic_client(monkeypatch, stub)

    with caplog.at_level(logging.DEBUG, logger="src.outcomes.generate"):
        client.generate(
            _anthropic_messages(),
            temperature=0.8,
            top_p=0.95,
            max_tokens=180,
            seed=0,
        )

    debug_msgs = [
        rec.getMessage()
        for rec in caplog.records
        if rec.levelno == logging.DEBUG
        and rec.name == "src.outcomes.generate"
    ]
    joined = "\n".join(debug_msgs)
    assert "cache_creation_input_tokens=512" in joined
    assert "cache_read_input_tokens=128" in joined


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
