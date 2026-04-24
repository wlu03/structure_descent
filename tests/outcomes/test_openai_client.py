"""Tests for :class:`src.outcomes._openai_client.OpenAILLMClient`.

All tests mock the ``openai`` SDK — **no network calls**. We install a
fake ``openai`` module under :mod:`sys.modules` so the lazy import inside
:class:`OpenAILLMClient.__init__` / :meth:`generate` binds to our stub
regardless of whether the real SDK is installed in the test environment.

The fake exposes:

* ``openai.OpenAI`` — constructor capturing ``api_key`` / ``organization``
  kwargs and wiring ``chat.completions.create`` to a capturing stub.
* ``openai.OpenAIError`` and the subclasses the client's ``except`` clause
  needs to catch (``APIConnectionError``, ``RateLimitError``).

The capturing stub records every kwarg payload so the assertions can
inspect exactly what the client sent on the wire.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import pytest


# ---------------------------------------------------------------------------
# Fake ``openai`` SDK
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str, finish_reason: str) -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(
        self,
        content: str = "hello from openai",
        *,
        finish_reason: str = "stop",
        model: str = "gpt-5-2026-04-01",
    ) -> None:
        self.choices = [_FakeChoice(content, finish_reason)]
        self.model = model


class _CapturingCompletions:
    """Records the kwargs passed to ``chat.completions.create``."""

    def __init__(
        self,
        response_factory: Callable[[], Any] | None = None,
        raise_exc: Exception | None = None,
    ) -> None:
        self._response_factory = response_factory or (lambda: _FakeResponse())
        self._raise_exc = raise_exc
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._response_factory()


class _CapturingOpenAI:
    """Drop-in for ``openai.OpenAI``.

    Instances carry a ``chat.completions.create`` chain and record the
    constructor kwargs (``api_key`` / ``organization`` / ...) on
    :attr:`init_kwargs` so tests can assert the client forwarded them
    correctly.
    """

    last_instance: "_CapturingOpenAI | None" = None

    def __init__(
        self,
        *,
        completions: _CapturingCompletions | None = None,
        **init_kwargs: Any,
    ) -> None:
        self.init_kwargs = dict(init_kwargs)
        self.completions = completions if completions is not None else _CapturingCompletions()
        self.chat = SimpleNamespace(completions=self.completions)
        type(self).last_instance = self


class _FakeOpenAIError(Exception):
    """Base error class used by the fake SDK (matches ``openai.OpenAIError``)."""


class _FakeAPIConnectionError(_FakeOpenAIError):
    pass


class _FakeRateLimitError(_FakeOpenAIError):
    pass


def _install_fake_openai(
    monkeypatch: pytest.MonkeyPatch,
    completions: _CapturingCompletions | None = None,
) -> _CapturingCompletions:
    """Install a fake ``openai`` module exposing the surface we use.

    Returns the shared ``_CapturingCompletions`` stub so the caller can
    mutate its factory / raise_exc after construction and still have
    subsequent ``create`` calls hit the updated behaviour.
    """
    shared_completions = completions if completions is not None else _CapturingCompletions()

    fake_module = ModuleType("openai")
    fake_module.OpenAIError = _FakeOpenAIError  # type: ignore[attr-defined]
    fake_module.APIConnectionError = _FakeAPIConnectionError  # type: ignore[attr-defined]
    fake_module.RateLimitError = _FakeRateLimitError  # type: ignore[attr-defined]

    def _factory(**kwargs: Any) -> _CapturingOpenAI:
        return _CapturingOpenAI(completions=shared_completions, **kwargs)

    fake_module.OpenAI = _factory  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "openai", fake_module)

    # Force a fresh import of the client module so the lazy ``from openai
    # import OpenAI`` inside __init__ binds to OUR fake the first time the
    # test constructs a client. Not strictly required (__init__ re-imports
    # each time) but makes the intent explicit.
    monkeypatch.delitem(sys.modules, "src.outcomes._openai_client", raising=False)

    return shared_completions


def _messages() -> list[dict]:
    """Minimal system+user+assistant chat payload used across tests."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Pick one of A/B/C/D."},
        {"role": "assistant", "content": "The answer is ("},
    ]


# ---------------------------------------------------------------------------
# __init__ credential resolution
# ---------------------------------------------------------------------------


def test_init_resolves_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no explicit kwarg, the SDK must be constructed with
    ``OPENAI_API_KEY`` from the environment."""
    _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    monkeypatch.delenv("OPENAI_ORGANIZATION", raising=False)

    from src.outcomes._openai_client import OpenAILLMClient

    client = OpenAILLMClient("gpt-5")

    assert _CapturingOpenAI.last_instance is not None
    assert _CapturingOpenAI.last_instance.init_kwargs["api_key"] == "sk-from-env"
    # No explicit org env -> organization kwarg not forwarded to SDK.
    assert "organization" not in _CapturingOpenAI.last_instance.init_kwargs
    assert client.model_id == "gpt-5"


def test_init_respects_explicit_api_key_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``api_key`` kwarg wins over the environment."""
    _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    from src.outcomes._openai_client import OpenAILLMClient

    OpenAILLMClient("gpt-5", api_key="sk-explicit")

    assert _CapturingOpenAI.last_instance is not None
    assert _CapturingOpenAI.last_instance.init_kwargs["api_key"] == "sk-explicit"


def test_init_respects_organization_env_and_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``organization`` resolves from kwarg first, then
    ``OPENAI_ORGANIZATION``; both paths forward to the SDK."""
    _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("OPENAI_ORGANIZATION", "org-from-env")

    from src.outcomes._openai_client import OpenAILLMClient

    # Env fallback path.
    OpenAILLMClient("gpt-5")
    inst_env = _CapturingOpenAI.last_instance
    assert inst_env is not None
    assert inst_env.init_kwargs["organization"] == "org-from-env"

    # Explicit kwarg path.
    OpenAILLMClient("gpt-5", organization="org-explicit")
    inst_kw = _CapturingOpenAI.last_instance
    assert inst_kw is not None
    assert inst_kw.init_kwargs["organization"] == "org-explicit"


# ---------------------------------------------------------------------------
# generate() payload shape
# ---------------------------------------------------------------------------


def test_generate_passes_messages_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI's Chat format matches ours — messages must go through
    verbatim with no shape rewriting (system / user / assistant all kept)."""
    completions = _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

    from src.outcomes._openai_client import OpenAILLMClient

    client = OpenAILLMClient("gpt-4o-mini")
    msgs = _messages()
    client.generate(msgs, temperature=0.5, top_p=0.9, max_tokens=64, seed=11)

    assert len(completions.calls) == 1
    sent = completions.calls[0]["messages"]
    assert sent == msgs  # verbatim
    # Ensure we did NOT accidentally mutate role names or collapse any block.
    assert [m["role"] for m in sent] == ["system", "user", "assistant"]


def test_generate_returns_generation_result_with_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The returned object must be a ``GenerationResult`` carrying the
    content, normalised finish_reason, and the model id the server
    reported (falling back to ``self.model_id`` if absent)."""
    completions = _install_fake_openai(monkeypatch)
    completions._response_factory = lambda: _FakeResponse(
        content="I save money.\nI feel better.",
        finish_reason="stop",
        model="gpt-4.1-2025-01-01",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

    from src.outcomes._openai_client import OpenAILLMClient
    from src.outcomes.generate import GenerationResult

    client = OpenAILLMClient("gpt-4.1")
    result = client.generate(
        _messages(), temperature=0.0, top_p=1.0, max_tokens=32, seed=0
    )

    assert isinstance(result, GenerationResult)
    assert result.text == "I save money.\nI feel better."
    assert result.finish_reason == "stop"
    assert result.model_id == "gpt-4.1-2025-01-01"


def test_generate_passes_temperature_top_p_max_tokens_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-reasoning models: decoding kwargs must land on the SDK call
    verbatim, with ``max_tokens`` under its legacy name."""
    completions = _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

    from src.outcomes._openai_client import OpenAILLMClient

    client = OpenAILLMClient("gpt-4o")
    client.generate(
        _messages(), temperature=0.73, top_p=0.91, max_tokens=123, seed=42
    )

    kw = completions.calls[0]
    assert kw["model"] == "gpt-4o"
    assert kw["temperature"] == 0.73
    assert kw["top_p"] == 0.91
    assert kw["max_tokens"] == 123
    assert kw["seed"] == 42
    # Reasoning-only kwarg must NOT leak into the non-reasoning path.
    assert "max_completion_tokens" not in kw


def test_generate_normalizes_finish_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """``"stop"`` passes through; ``"length"`` must become ``"max_tokens"``
    so callers can detect token-cap hits without special-casing OpenAI."""
    completions = _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

    from src.outcomes._openai_client import OpenAILLMClient

    client = OpenAILLMClient("gpt-4o")

    completions._response_factory = lambda: _FakeResponse(finish_reason="stop")
    r1 = client.generate(_messages(), temperature=0.0, top_p=1.0, max_tokens=8, seed=0)
    assert r1.finish_reason == "stop"

    completions._response_factory = lambda: _FakeResponse(finish_reason="length")
    r2 = client.generate(_messages(), temperature=0.0, top_p=1.0, max_tokens=8, seed=0)
    assert r2.finish_reason in ("length", "max_tokens")
    # We specifically chose "max_tokens" as the normalised form.
    assert r2.finish_reason == "max_tokens"

    completions._response_factory = lambda: _FakeResponse(finish_reason="content_filter")
    r3 = client.generate(_messages(), temperature=0.0, top_p=1.0, max_tokens=8, seed=0)
    assert r3.finish_reason == "filter"


def test_generate_raises_on_openai_error_with_prefixed_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Any SDK error must surface as :class:`RuntimeError` with the
    ``OpenAILLMClient:`` prefix; the original exception chains via
    ``__cause__`` so callers keep full diagnostic context."""
    completions = _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

    import openai  # our fake

    completions._raise_exc = openai.RateLimitError("rate limited")

    from src.outcomes._openai_client import OpenAILLMClient

    client = OpenAILLMClient("gpt-4o")
    with pytest.raises(RuntimeError) as excinfo:
        client.generate(
            _messages(), temperature=0.0, top_p=1.0, max_tokens=8, seed=0
        )
    msg = str(excinfo.value)
    assert msg.startswith("OpenAILLMClient:")
    assert "RateLimitError" in msg
    assert "rate limited" in msg
    # Original SDK exception must be chained, not swallowed.
    assert isinstance(excinfo.value.__cause__, openai.OpenAIError)


# ---------------------------------------------------------------------------
# Stub-contamination guard
# ---------------------------------------------------------------------------


def test_not_flagged_as_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """The batching-layer guard keys on ``_is_stub``; OpenAI is a real
    LLM and must not be flagged."""
    _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

    from src.outcomes._openai_client import OpenAILLMClient

    client = OpenAILLMClient("gpt-5")
    assert getattr(client, "_is_stub", False) is False


# ---------------------------------------------------------------------------
# Reasoning-model max_completion_tokens routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id",
    [
        "o1-preview",
        "o3-mini",
        "gpt-5",
        "gpt-5-thinking",
        "GPT-5-Thinking-2026-04",
        "gpt-5-mini",
    ],
)
def test_model_id_routes_max_completion_tokens_for_reasoning_models(
    monkeypatch: pytest.MonkeyPatch, model_id: str
) -> None:
    """Reasoning-family prefixes (``o1``, ``o3``, ``gpt-5*``; case-insensitive)
    must send ``max_completion_tokens`` instead of the legacy ``max_tokens``
    kwarg, reserve ``_REASONING_BUDGET_TOKENS`` extra budget for the
    reasoning trace, and strip the ``temperature``/``top_p`` sampler knobs
    (these models only accept the default value=1 and reject any forwarded
    override)."""
    completions = _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

    from src.outcomes._openai_client import (
        OpenAILLMClient,
        _REASONING_BUDGET_TOKENS,
    )

    client = OpenAILLMClient(model_id)
    client.generate(
        _messages(), temperature=0.0, top_p=1.0, max_tokens=256, seed=7
    )

    kw = completions.calls[0]
    assert kw["model"] == model_id
    assert kw["max_completion_tokens"] == 256 + _REASONING_BUDGET_TOKENS
    assert "max_tokens" not in kw
    # Reasoning models reject non-default temperature/top_p at the request
    # layer — the client must strip them from the payload so the caller's
    # deterministic temperature=0.0 (used by the rankers) doesn't hard-fail.
    assert "temperature" not in kw
    assert "top_p" not in kw


def test_non_reasoning_model_uses_plain_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control for the routing branch: vanilla GPT models keep
    the legacy ``max_tokens`` parameter."""
    completions = _install_fake_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")

    from src.outcomes._openai_client import OpenAILLMClient

    client = OpenAILLMClient("gpt-4.1")
    client.generate(
        _messages(), temperature=0.0, top_p=1.0, max_tokens=64, seed=0
    )

    kw = completions.calls[0]
    assert kw["max_tokens"] == 64
    assert "max_completion_tokens" not in kw


# ---------------------------------------------------------------------------
# Lazy-import contract
# ---------------------------------------------------------------------------


def test_module_imports_without_openai_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the client module must succeed even when ``openai`` is
    not installed; only instantiating the class surfaces ImportError."""
    # Pretend the SDK is not installed.
    monkeypatch.setitem(sys.modules, "openai", None)
    monkeypatch.delitem(sys.modules, "src.outcomes._openai_client", raising=False)

    reloaded = importlib.import_module("src.outcomes._openai_client")
    assert hasattr(reloaded, "OpenAILLMClient")

    with pytest.raises(ImportError):
        reloaded.OpenAILLMClient("gpt-5", api_key="dummy")
