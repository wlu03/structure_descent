"""OpenAI-backed :class:`LLMClient` for the Phase-3 LLM baselines.

This module provides :class:`OpenAILLMClient`, a drop-in implementation of
the :class:`src.outcomes.generate.LLMClient` Protocol that targets the
OpenAI Chat Completions API via the v1.x Python SDK (``openai>=1.0``).

Authentication
--------------
Standard OpenAI API key flow. The client resolves credentials from the
constructor kwargs first, then falls back to the environment:

* ``api_key`` / ``OPENAI_API_KEY``
* ``organization`` / ``OPENAI_ORGANIZATION``

No Application Default Credentials path, no custom auth headers.

Supported models
----------------
Any Chat-Completions-compatible model. Validated targets at time of
writing:

* ``gpt-5``
* ``gpt-4.1``
* ``gpt-4o``
* ``gpt-4o-mini``

Reasoning-family models (``o1-*``, ``o3-*``, ``gpt-5-*``) accept
``max_completion_tokens`` rather than the legacy ``max_tokens`` parameter;
the client detects these by model-id prefix and forwards the caller's
``max_tokens`` argument under the correct kwarg name. Plain ``gpt-5`` is
also covered — as of the 2026-Q1 API, all ``gpt-5`` variants (thinking,
mini, etc.) reject ``max_tokens`` at the request layer. No other parameter
routing differs.

Logprob support
---------------
OpenAI exposes per-token logprobs via the ``logprobs=True`` and
``top_logprobs=N`` request flags; the response surface is
``response.choices[0].logprobs.content[0].top_logprobs`` (a list of
``{token, logprob, bytes}`` records for the first generated position).

This client does **not** wire up native logprob extraction yet. Phase-3's
:mod:`src.baselines._llm_ranker_common` currently hard-codes the Anthropic
``content[i].logprobs.top_logprobs`` shape; until a parallel OpenAI branch
lands there, OpenAI calls route through the verbalised-JSON fallback in
``call_llm_for_ranking``. When the OpenAI logprob path is added, the
relevant fields on a ``ChatCompletion`` object are:

* ``response.choices[0].logprobs.content[i].token`` — the generated token
* ``response.choices[0].logprobs.content[i].logprob`` — its logprob
* ``response.choices[0].logprobs.content[i].top_logprobs`` — list of
  ``{token, logprob, bytes}`` records

Stub-contamination guard
------------------------
This client deliberately does **not** set ``_is_stub``; the batching
layer's stub-contamination guard
(``src/data/batching.py::_is_stub_client``) treats a missing / falsy
``_is_stub`` attribute as "real LLM" and will therefore correctly allow
OpenAI-backed pipelines to write to the outcomes cache.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.outcomes.generate import GenerationResult

logger = logging.getLogger(__name__)


# Prefixes (case-insensitive) that identify reasoning-family models whose
# Chat Completions payloads must use ``max_completion_tokens`` instead of
# the legacy ``max_tokens`` parameter. Kept here as a module-level constant
# so tests can inspect / monkeypatch it if OpenAI ever adds new families.
_REASONING_MODEL_PREFIXES: tuple[str, ...] = (
    "o1",
    "o3",
    "gpt-5",
)

# Reasoning-family models burn internal tokens on a reasoning trace BEFORE
# producing visible output, and both budgets share the single
# ``max_completion_tokens`` allowance. Empirically GPT-5 needs ~500 tokens
# before emitting a single word; 1000 gives headroom without dominating the
# caller's answer budget. Exposed at module scope so tests can reference it
# cleanly and so a future per-model override table can live next to it.
_REASONING_BUDGET_TOKENS: int = 1000


# Error-message prefix used for every re-raised OpenAI SDK failure.
# Consumed by callers (and tests) that want to distinguish OpenAI-backed
# failures from Anthropic / stub failures without catching SDK-specific
# exception types.
_ERROR_PREFIX: str = "OpenAILLMClient:"


def _uses_max_completion_tokens(model_id: str) -> bool:
    """Return True when ``model_id`` requires ``max_completion_tokens``.

    Matches the prefix list in :data:`_REASONING_MODEL_PREFIXES`
    case-insensitively. Exposed at module scope (not as a method) so the
    prefix check can be unit-tested without instantiating the client.
    """
    mid = str(model_id).strip().lower()
    return any(mid.startswith(prefix) for prefix in _REASONING_MODEL_PREFIXES)


def _normalize_finish_reason(raw: str | None) -> str:
    """Map OpenAI ``finish_reason`` strings onto a small canonical set.

    OpenAI Chat Completions returns one of ``"stop"``, ``"length"``,
    ``"content_filter"``, ``"tool_calls"``, or ``"function_call"``. Callers
    (cache metadata, ranker fallback logic) mainly care whether the model
    hit the token cap, so we normalize:

    * ``"stop"``           -> ``"stop"``
    * ``"length"``         -> ``"max_tokens"``  (hit the token cap)
    * ``"content_filter"`` -> ``"filter"``
    * anything else        -> pass through verbatim (e.g. ``"tool_calls"``)

    Unknown / ``None`` reasons return an empty string so the downstream
    metadata schema (``finish_reason: str``) stays satisfied.
    """
    if raw is None:
        return ""
    r = str(raw)
    if r == "length":
        return "max_tokens"
    if r == "content_filter":
        return "filter"
    return r


class OpenAILLMClient:
    """Pluggable real client backed by the ``openai`` v1.x Python SDK.

    Mirrors :class:`src.outcomes.generate.AnthropicLLMClient` in shape:

    * The ``openai`` package is imported **lazily** inside ``__init__``
      so ``import src.outcomes._openai_client`` succeeds on machines
      without the SDK installed. Only instantiating the client surfaces
      :class:`ImportError`.
    * ``self.model_id`` is set on the instance for the cache-key
      machinery in :func:`src.outcomes.generate.generate_outcomes` to
      read (via ``getattr(client, "model_id", ...)``).
    * No ``_is_stub`` attribute — the stub-contamination guard must treat
      this client as a real LLM.

    Parameters
    ----------
    model_id:
        OpenAI model identifier (e.g. ``"gpt-5"``, ``"gpt-4.1"``,
        ``"gpt-4o"``, ``"gpt-4o-mini"``). Reasoning-family prefixes
        (``o1``, ``o3``, ``gpt-5-thinking``) trigger the
        ``max_completion_tokens`` routing automatically.
    api_key:
        Optional API key. ``None`` falls back to ``OPENAI_API_KEY`` in
        the environment — matching the SDK's own default. Resolved here
        rather than at module import so importing this file does not
        read the environment.
    organization:
        Optional OpenAI organization ID. ``None`` falls back to
        ``OPENAI_ORGANIZATION`` in the environment. Passed straight
        through to the SDK constructor.
    """

    def __init__(
        self,
        model_id: str = "gpt-5",
        *,
        api_key: str | None = None,
        organization: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(
                "OpenAILLMClient requires the `openai` package (v1.x). "
                "Install it with `pip install openai>=1.0`."
            ) from exc

        self.model_id = model_id
        resolved_key = (
            api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        )
        resolved_org = (
            organization
            if organization is not None
            else os.environ.get("OPENAI_ORGANIZATION")
        )
        self._api_key = resolved_key
        self._organization = resolved_org

        # The SDK tolerates ``organization=None`` but chokes on an empty
        # string in some versions; only forward the kwarg when we have a
        # non-empty value so defaults continue to flow through the SDK's
        # own env-var lookup.
        sdk_kwargs: dict[str, Any] = {"api_key": resolved_key}
        if resolved_org:
            sdk_kwargs["organization"] = resolved_org
        self._client = OpenAI(**sdk_kwargs)

    # ------------------------------------------------------------------
    # LLMClient protocol
    # ------------------------------------------------------------------
    def generate(
        self,
        messages: list[dict],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
    ) -> GenerationResult:
        """Issue a single Chat Completions request and map it to
        :class:`GenerationResult`.

        OpenAI's Chat Completions ``messages`` format is identical to the
        shape used throughout this codebase (``[{"role": ..., "content":
        ...}]``), so messages pass through verbatim. ``temperature``,
        ``top_p``, ``seed`` are forwarded directly. ``max_tokens`` is
        routed to either ``max_tokens`` or ``max_completion_tokens``
        based on :func:`_uses_max_completion_tokens`.

        Any ``openai.OpenAIError`` (including ``APIConnectionError`` and
        ``RateLimitError``) is re-raised with a consistent
        ``OpenAILLMClient:`` prefix so callers that log / surface the
        error see a recognisable origin.
        """
        try:
            import openai  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenAILLMClient requires the `openai` package (v1.x). "
                "Install it with `pip install openai>=1.0`."
            ) from exc

        # Messages pass through verbatim — OpenAI Chat format == our format.
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": list(messages),
            "seed": seed,
        }
        if _uses_max_completion_tokens(self.model_id):
            # Reasoning-family models (o1/o3/gpt-5) reject any non-default
            # temperature / top_p at the request layer ("Only the default
            # (1) value is supported"). Skip forwarding them so the caller's
            # temperature=0.0 (used by the deterministic rankers) doesn't
            # hard-fail the request. Decoding is driven by the reasoning
            # trace, not the classical sampler knobs.
            #
            # These models also burn tokens on an internal reasoning pass
            # BEFORE producing visible output, and both budgets come out of
            # max_completion_tokens. With small caller budgets (e.g.
            # max_tokens=2 from ZeroShot), reasoning consumes the entire
            # allowance and the response comes back empty with
            # finish_reason=length. The +_REASONING_BUDGET_TOKENS reserve
            # (module-level constant) ensures the caller's answer budget
            # survives with headroom.
            kwargs["max_completion_tokens"] = int(max_tokens) + _REASONING_BUDGET_TOKENS
        else:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
            kwargs["max_tokens"] = int(max_tokens)

        try:
            response = self._client.chat.completions.create(**kwargs)
        except openai.OpenAIError as exc:
            # Covers APIConnectionError, RateLimitError, APIStatusError,
            # AuthenticationError, etc. — all derive from OpenAIError in
            # the v1.x SDK. Chain the original via ``from exc`` so
            # tracebacks retain the SDK context.
            raise RuntimeError(f"{_ERROR_PREFIX} {type(exc).__name__}: {exc}") from exc

        choices = getattr(response, "choices", None) or []
        if not choices:  # pragma: no cover - defensive; OpenAI always returns >=1
            raise RuntimeError(
                f"{_ERROR_PREFIX} response contained no choices: {response!r}"
            )
        first = choices[0]
        message = getattr(first, "message", None)
        text = getattr(message, "content", "") if message is not None else ""
        if text is None:
            text = ""
        finish_reason = _normalize_finish_reason(getattr(first, "finish_reason", None))
        model_id = getattr(response, "model", self.model_id) or self.model_id

        return GenerationResult(
            text=str(text),
            finish_reason=finish_reason,
            model_id=str(model_id),
        )


__all__ = ["OpenAILLMClient"]
