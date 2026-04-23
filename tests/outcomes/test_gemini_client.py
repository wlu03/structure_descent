"""Tests for :class:`src.outcomes._gemini_client.GeminiLLMClient`.

All tests install a fake ``google.genai`` SDK into :data:`sys.modules`
before importing / instantiating the client, so real network calls to
Vertex AI are never issued during ``pytest``. Each fake captures the
construction kwargs and the ``generate_content`` call kwargs so we can
assert on how the client maps Protocol parameters into the SDK.
"""

from __future__ import annotations

import sys
import types
from typing import Any, List

import pytest

from src.outcomes.generate import GenerationResult


# ---------------------------------------------------------------------------
# Fake google-genai SDK
# ---------------------------------------------------------------------------


class _FakePart:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeContent:
    def __init__(self, parts: List[_FakePart]) -> None:
        self.parts = parts


class _FakeCandidate:
    def __init__(self, text: str = "hello", finish_reason: Any = "STOP") -> None:
        self.content = _FakeContent([_FakePart(text)])
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(
        self,
        text: str = "hello",
        *,
        finish_reason: Any = "STOP",
        model_version: str | None = None,
    ) -> None:
        self.text = text
        self.candidates = [_FakeCandidate(text=text, finish_reason=finish_reason)]
        if model_version is not None:
            self.model_version = model_version


class _FakeModels:
    def __init__(self) -> None:
        self.calls: List[dict[str, Any]] = []
        self.response_factory = lambda: _FakeResponse()
        self.raise_on_call: Exception | None = None

    def generate_content(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if self.raise_on_call is not None:
            raise self.raise_on_call
        return self.response_factory()


class _FakeClient:
    """Captures constructor kwargs and exposes a stub ``.models``."""

    instances: List["_FakeClient"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.models = _FakeModels()
        _FakeClient.instances.append(self)


class _FakeHttpOptions:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeGenerateContentConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeAPIError(Exception):
    """Stand-in for ``google.genai.errors.APIError``."""


class _FakeClientError(_FakeAPIError):
    """Stand-in for ``google.genai.errors.ClientError``."""


def _install_fake_genai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a minimal fake ``google.genai`` SDK into ``sys.modules``.

    The fake mirrors only the surface the client touches:

    * ``google.genai.Client`` — constructor capturing kwargs, exposing
      ``.models.generate_content``.
    * ``google.genai.types.GenerateContentConfig`` / ``HttpOptions`` —
      kwarg-capturing shells.
    * ``google.genai.errors.APIError`` / ``ClientError`` — exception
      types the client catches.
    """
    _FakeClient.instances.clear()

    # google package (may exist as a real namespace package; we
    # install only what we need and do not clobber siblings).
    google_mod = sys.modules.get("google")
    if google_mod is None:
        google_mod = types.ModuleType("google")
        monkeypatch.setitem(sys.modules, "google", google_mod)

    # google.genai
    genai_mod = types.ModuleType("google.genai")
    genai_mod.Client = _FakeClient  # type: ignore[attr-defined]

    # google.genai.types
    types_mod = types.ModuleType("google.genai.types")
    types_mod.GenerateContentConfig = _FakeGenerateContentConfig  # type: ignore[attr-defined]
    types_mod.HttpOptions = _FakeHttpOptions  # type: ignore[attr-defined]

    # google.genai.errors
    errors_mod = types.ModuleType("google.genai.errors")
    errors_mod.APIError = _FakeAPIError  # type: ignore[attr-defined]
    errors_mod.ClientError = _FakeClientError  # type: ignore[attr-defined]
    errors_mod.ServerError = _FakeAPIError  # type: ignore[attr-defined]

    genai_mod.types = types_mod  # type: ignore[attr-defined]
    genai_mod.errors = errors_mod  # type: ignore[attr-defined]

    monkeypatch.setattr(google_mod, "genai", genai_mod, raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)
    monkeypatch.setitem(sys.modules, "google.genai.errors", errors_mod)


# ---------------------------------------------------------------------------
# Construction / env-var resolution
# ---------------------------------------------------------------------------


def test_init_resolves_project_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing kwargs must fall back to the Vertex env vars."""
    _install_fake_genai(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project-id")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    from src.outcomes._gemini_client import GeminiLLMClient

    client = GeminiLLMClient()

    assert client.model_id == "gemini-2.5-flash"
    assert _FakeClient.instances, "fake genai.Client was not constructed"
    init_kwargs = _FakeClient.instances[-1].init_kwargs
    assert init_kwargs["project"] == "env-project-id"
    assert init_kwargs["location"] == "us-central1"
    # HttpOptions(api_version="v1") should have been passed.
    assert isinstance(init_kwargs["http_options"], _FakeHttpOptions)
    assert init_kwargs["http_options"].kwargs == {"api_version": "v1"}


def test_init_respects_explicit_project_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit kwarg must win over the env var."""
    _install_fake_genai(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project-id")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "env-location")

    from src.outcomes._gemini_client import GeminiLLMClient

    GeminiLLMClient(project="kwarg-project", location="kwarg-location")

    init_kwargs = _FakeClient.instances[-1].init_kwargs
    assert init_kwargs["project"] == "kwarg-project"
    assert init_kwargs["location"] == "kwarg-location"


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def test_generate_converts_messages_to_gemini_contents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """System -> system_instruction; assistant -> model; same-role runs merge."""
    _install_fake_genai(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "l")

    from src.outcomes._gemini_client import GeminiLLMClient

    client = GeminiLLMClient(model_id="gemini-2.5-flash")
    fake_models = _FakeClient.instances[-1].models

    messages = [
        {"role": "system", "content": "SYS-A"},
        {"role": "system", "content": "SYS-B"},
        {"role": "user", "content": "U1"},
        {"role": "user", "content": "U2"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U3"},
    ]
    client.generate(messages, temperature=0.4, top_p=0.9, max_tokens=50, seed=1)

    assert len(fake_models.calls) == 1
    call = fake_models.calls[0]

    # The config received the joined system instruction.
    config = call["config"]
    assert isinstance(config, _FakeGenerateContentConfig)
    assert config.kwargs["system_instruction"] == "SYS-A\n\nSYS-B"

    # Contents must carry no system turn, and assistant must be "model".
    contents = call["contents"]
    assert isinstance(contents, list)
    assert all(turn["role"] in ("user", "model") for turn in contents)

    # Same-role runs must be merged into a single turn with multiple parts.
    assert contents[0]["role"] == "user"
    assert [p["text"] for p in contents[0]["parts"]] == ["U1", "U2"]
    assert contents[1]["role"] == "model"
    assert [p["text"] for p in contents[1]["parts"]] == ["A1"]
    assert contents[2]["role"] == "user"
    assert [p["text"] for p in contents[2]["parts"]] == ["U3"]


# ---------------------------------------------------------------------------
# GenerationResult shape
# ---------------------------------------------------------------------------


def test_generate_returns_generation_result_with_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "l")

    from src.outcomes._gemini_client import GeminiLLMClient

    client = GeminiLLMClient(model_id="gemini-2.5-flash")
    fake_models = _FakeClient.instances[-1].models
    fake_models.response_factory = lambda: _FakeResponse(
        text="outcome-text", finish_reason="STOP", model_version="gemini-2.5-flash-001"
    )

    result = client.generate(
        [{"role": "user", "content": "hi"}],
        temperature=0.1,
        top_p=0.9,
        max_tokens=10,
        seed=42,
    )

    assert isinstance(result, GenerationResult)
    assert result.text == "outcome-text"
    # Server-reported model id wins when present.
    assert result.model_id == "gemini-2.5-flash-001"


def test_generate_maps_finish_reason_strings(monkeypatch: pytest.MonkeyPatch) -> None:
    """``STOP`` -> ``"stop"``; ``MAX_TOKENS`` -> ``"max_tokens"``; safety variants."""
    _install_fake_genai(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "l")

    from src.outcomes._gemini_client import GeminiLLMClient

    client = GeminiLLMClient()
    fake_models = _FakeClient.instances[-1].models

    fake_models.response_factory = lambda: _FakeResponse(finish_reason="STOP")
    r = client.generate(
        [{"role": "user", "content": "x"}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        seed=0,
    )
    assert r.finish_reason == "stop"

    fake_models.response_factory = lambda: _FakeResponse(finish_reason="MAX_TOKENS")
    r = client.generate(
        [{"role": "user", "content": "x"}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        seed=0,
    )
    assert r.finish_reason == "max_tokens"

    fake_models.response_factory = lambda: _FakeResponse(finish_reason="SAFETY")
    r = client.generate(
        [{"role": "user", "content": "x"}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        seed=0,
    )
    assert r.finish_reason == "safety"


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------


def test_generate_passes_temperature_top_p_max_tokens_seed_to_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``max_tokens`` must be forwarded as ``max_output_tokens``; other params pass through."""
    _install_fake_genai(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "l")

    from src.outcomes._gemini_client import GeminiLLMClient

    client = GeminiLLMClient(model_id="gemini-2.5-pro")
    fake_models = _FakeClient.instances[-1].models

    client.generate(
        [{"role": "user", "content": "ping"}],
        temperature=0.37,
        top_p=0.81,
        max_tokens=256,
        seed=2024,
    )

    call = fake_models.calls[0]
    assert call["model"] == "gemini-2.5-pro"
    cfg = call["config"].kwargs
    assert cfg["temperature"] == pytest.approx(0.37)
    assert cfg["top_p"] == pytest.approx(0.81)
    assert cfg["max_output_tokens"] == 256
    assert cfg["seed"] == 2024


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_generate_raises_on_sdk_error_with_prefixed_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_genai(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "l")

    from src.outcomes._gemini_client import GeminiLLMClient

    client = GeminiLLMClient()
    fake_models = _FakeClient.instances[-1].models
    fake_models.raise_on_call = _FakeClientError("quota exceeded")

    with pytest.raises(RuntimeError) as excinfo:
        client.generate(
            [{"role": "user", "content": "x"}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            seed=0,
        )

    msg = str(excinfo.value)
    assert msg.startswith("GeminiLLMClient:")
    assert "quota exceeded" in msg


# ---------------------------------------------------------------------------
# Stub contamination guard
# ---------------------------------------------------------------------------


def test_not_flagged_as_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_is_stub`` must not be set so the batching guard treats Gemini as real."""
    _install_fake_genai(monkeypatch)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "l")

    from src.outcomes._gemini_client import GeminiLLMClient

    client = GeminiLLMClient()
    assert getattr(client, "_is_stub", False) is False
