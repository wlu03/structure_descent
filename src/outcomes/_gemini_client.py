"""Gemini (Vertex AI) ``LLMClient`` implementation for the PO-LEU LLM baselines.

This module plugs Google's Gemini family of models into the same
:class:`src.outcomes.generate.LLMClient` Protocol that the frozen-LLM
baselines (``ZeroShotClaudeRanker``, few-shot ICL ranker) already call
through. The client is provider-agnostic at the call site — any ranker
that accepts an ``LLMClient`` will also accept a :class:`GeminiLLMClient`.

Authentication
--------------
The client targets **Vertex AI** via Application Default Credentials
(ADC). Callers are expected to have run ``gcloud auth
application-default login`` locally; no API key is required. The
project ID and region are read from the standard Vertex environment
variables when not provided explicitly to the constructor:

* ``GOOGLE_CLOUD_PROJECT`` — GCP project ID (e.g.
  ``"en-decision-modeling-bffc"``).
* ``GOOGLE_CLOUD_LOCATION`` — Vertex region (e.g. ``"global"``,
  ``"us-central1"``).
* ``GOOGLE_GENAI_USE_VERTEXAI`` — set to ``"True"`` so the SDK routes
  through Vertex AI rather than the public Generative Language API.

Model choices
-------------
The default ``model_id`` is ``"gemini-2.5-flash"``; callers can pass
``"gemini-2.5-pro"`` or any other Vertex-hosted Gemini model. The
model_id is surfaced on every :class:`GenerationResult` so the
outcomes cache keys on it (``generate.py`` folds ``model_id`` into
``cache_prompt_version``), preventing cross-provider cache
contamination.

Logprob path
------------
Gemini's ``generate_content`` API exposes only limited logprob support
(``response_logprobs`` returns a small fixed number of candidates and
is not universally available across models). The frozen-LLM ranker
baselines therefore route Gemini calls through the **verbalised-JSON
fallback path** in ``src/baselines/_llm_ranker_common.py``: the ranker
asks Gemini for ``{"A": p, "B": p, "C": p, "D": p}`` and parses the
returned JSON fragment. This client intentionally does not implement
``messages_create_logprobs`` — the generic ``extract_letter_logprobs``
helper handles the string response.

Lazy import
-----------
``google.genai`` is imported inside the constructor so
``import src.outcomes._gemini_client`` stays side-effect-free and the
project remains importable on machines that have not installed
``google-genai``. Only instantiating :class:`GeminiLLMClient` raises
:class:`ImportError` when the SDK is missing.

Stub-contamination guard
------------------------
:class:`GeminiLLMClient` deliberately does **not** set
``_is_stub = True``. :func:`src.data.batching._is_stub_client` treats
the attribute as authoritative, and real Gemini outputs must never be
classified as stub-origin cache entries.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from src.outcomes.generate import GenerationResult

logger = logging.getLogger(__name__)


__all__ = ["GeminiLLMClient"]


# ---------------------------------------------------------------------------
# Finish-reason normalisation
# ---------------------------------------------------------------------------

# Stable string labels we expose to callers. Mirrors the set of labels
# the outcomes-cache metadata schema already understands.
_FINISH_REASON_STOP = "stop"
_FINISH_REASON_MAX_TOKENS = "max_tokens"
_FINISH_REASON_SAFETY = "safety"
_FINISH_REASON_OTHER = "other"


def _normalise_finish_reason(raw: Any) -> str:
    """Map a Gemini ``FinishReason`` enum / string / ``None`` to a stable label.

    The SDK has returned finish reasons as both an enum (``FinishReason.STOP``)
    and a bare string (``"STOP"``) across versions, so we stringify and
    match on the suffix rather than importing the enum.
    """
    if raw is None:
        return _FINISH_REASON_STOP
    # Enum -> "FinishReason.STOP"; bare str -> "STOP"; some SDKs expose
    # a ``.name`` attribute.
    name = getattr(raw, "name", None)
    token = (name if isinstance(name, str) else str(raw)).upper()
    # Strip enum prefix like "FINISHREASON.STOP" -> "STOP".
    if "." in token:
        token = token.rsplit(".", 1)[-1]
    if token in ("STOP", "FINISH_REASON_STOP", "END_OF_TURN"):
        return _FINISH_REASON_STOP
    if token in ("MAX_TOKENS", "LENGTH"):
        return _FINISH_REASON_MAX_TOKENS
    if token in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "RECITATION"):
        return _FINISH_REASON_SAFETY
    return _FINISH_REASON_OTHER


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def _convert_messages(
    messages: list[dict],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Split OpenAI-style messages into ``(system_instruction, contents)``.

    The Gemini SDK expects the system prompt via the top-level
    ``system_instruction`` config field and the conversation turns as a
    list of ``{"role": "user"|"model", "parts": [{"text": ...}]}``
    dictionaries. This helper:

    * Concatenates all ``system`` messages (joined with two newlines) into
      a single system string — Gemini only accepts one.
    * Maps ``assistant`` -> ``model`` for the ``role`` field.
    * Consolidates consecutive messages that share a role into a single
      content entry (multiple text parts) because Gemini rejects
      back-to-back ``user``/``user`` turns.
    """
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        text = msg.get("content", "")
        if role == "system":
            if text:
                system_parts.append(text)
            continue
        if role == "assistant":
            gemini_role = "model"
        else:
            # Everything else (including unknown roles) routes to "user".
            gemini_role = "user"
        if contents and contents[-1]["role"] == gemini_role:
            contents[-1]["parts"].append({"text": text})
        else:
            contents.append({"role": gemini_role, "parts": [{"text": text}]})

    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class GeminiLLMClient:
    """Pluggable real client backed by the ``google-genai`` Python SDK.

    Conforms structurally to :class:`src.outcomes.generate.LLMClient` so
    it can be passed as ``llm_client=`` to any baseline that accepts the
    Protocol. Constructor arguments select the Vertex AI project /
    region; per-call decoding settings flow through ``.generate``.

    Parameters
    ----------
    model_id:
        Vertex-hosted Gemini model identifier. Default
        ``"gemini-2.5-flash"``.
    project:
        GCP project ID. When ``None`` we fall back to the
        ``GOOGLE_CLOUD_PROJECT`` env var. An explicit kwarg always
        wins over the env var.
    location:
        Vertex region (e.g. ``"global"``, ``"us-central1"``). When
        ``None`` we fall back to ``GOOGLE_CLOUD_LOCATION``.

    Notes
    -----
    * The ``google.genai`` SDK is imported *lazily* inside ``__init__``
      so ``import src.outcomes._gemini_client`` succeeds even when the
      ``google-genai`` package is not installed. Only instantiating
      this class raises :class:`ImportError` when the dependency is
      missing.
    * ``_is_stub`` is deliberately unset — see module docstring.
    """

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        *,
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        try:
            from google import genai  # type: ignore[import-not-found]
            from google.genai.types import HttpOptions  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(
                "GeminiLLMClient requires the `google-genai` package. "
                "Install with `pip install google-genai>=1.51.0`."
            ) from exc

        resolved_project = project if project is not None else os.environ.get(
            "GOOGLE_CLOUD_PROJECT"
        )
        resolved_location = location if location is not None else os.environ.get(
            "GOOGLE_CLOUD_LOCATION"
        )

        self.model_id = model_id
        self._project = resolved_project
        self._location = resolved_location

        client_kwargs: dict[str, Any] = {
            "http_options": HttpOptions(api_version="v1"),
        }
        if resolved_project is not None:
            client_kwargs["project"] = resolved_project
        if resolved_location is not None:
            client_kwargs["location"] = resolved_location

        self._client = genai.Client(**client_kwargs)

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
        """Dispatch a single Gemini completion and map to :class:`GenerationResult`.

        The ``google-genai`` SDK takes the system prompt via a dedicated
        ``system_instruction`` field in :class:`GenerateContentConfig`
        rather than as a role-``system`` message, so we split the
        system text off before the call. Decoding parameters are
        forwarded via the config object; ``max_tokens`` is renamed to
        the SDK's ``max_output_tokens``.

        Errors raised by the SDK (``google.genai.errors.ClientError``,
        ``ServerError``, ``APIError``, and their subclasses) are caught
        and re-raised as :class:`RuntimeError` with a
        ``"GeminiLLMClient:"`` prefix so callers can distinguish Gemini
        failures from Anthropic / other provider failures when parsing
        logs.
        """
        try:
            from google import genai  # type: ignore[import-not-found]  # noqa: F401
            from google.genai import errors as genai_errors  # type: ignore[import-not-found]
            from google.genai.types import (  # type: ignore[import-not-found]
                GenerateContentConfig,
                ThinkingConfig,
            )
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "GeminiLLMClient requires the `google-genai` package. "
                "Install with `pip install google-genai>=1.51.0`."
            ) from exc

        system_instruction, contents = _convert_messages(messages)

        # Gemini 2.5 models (both Pro and Flash) consume internal "thinking"
        # tokens out of max_output_tokens. With a small budget (e.g.
        # max_tokens=2 for the letter-token rankers), thinking burns the
        # entire allowance and the response comes back empty with
        # finish_reason=max_tokens.
        #
        # Flash supports disabling thinking outright via thinking_budget=0,
        # so we take that path — the ranker needs deterministic letter-only
        # output, not chain-of-thought. Pro's minimum thinking budget is 128
        # (it cannot be 0); for Pro we allocate the 128-token minimum and
        # bump max_output_tokens accordingly so the caller's answer budget
        # survives.
        model_lower = self.model_id.lower()
        effective_max_tokens = int(max_tokens)
        thinking_config: Any = None
        if "gemini-2.5-pro" in model_lower:
            pro_thinking_budget = 128
            effective_max_tokens = int(max_tokens) + pro_thinking_budget
            thinking_config = ThinkingConfig(
                thinking_budget=pro_thinking_budget,
                include_thoughts=False,
            )
        elif "gemini-2.5-flash" in model_lower:
            thinking_config = ThinkingConfig(
                thinking_budget=0,
                include_thoughts=False,
            )

        config_kwargs: dict[str, Any] = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_output_tokens": effective_max_tokens,
            "seed": int(seed),
        }
        if system_instruction is not None:
            config_kwargs["system_instruction"] = system_instruction
        if thinking_config is not None:
            config_kwargs["thinking_config"] = thinking_config
        config = GenerateContentConfig(**config_kwargs)

        try:
            response = self._client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=config,
            )
        except genai_errors.APIError as exc:
            raise RuntimeError(
                f"GeminiLLMClient: generate_content failed ({type(exc).__name__}): {exc}"
            ) from exc

        text = self._extract_text(response)
        finish_reason = self._extract_finish_reason(response)
        model_id = self._extract_model_id(response)

        return GenerationResult(
            text=text,
            finish_reason=finish_reason,
            model_id=model_id,
        )

    # ------------------------------------------------------------------
    # Response unpackers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_text(response: Any) -> str:
        """Concatenate text parts from a Gemini response.

        Prefers the SDK's ``.text`` accessor (which joins all text parts
        from the first candidate); falls back to a manual scan over
        ``candidates[0].content.parts`` when ``.text`` is absent or
        ``None`` (e.g. safety-blocked responses).
        """
        text = getattr(response, "text", None)
        if isinstance(text, str):
            return text
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return ""
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        collected: list[str] = []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str):
                collected.append(part_text)
        return "".join(collected)

    @staticmethod
    def _extract_finish_reason(response: Any) -> str:
        """Read the first candidate's ``finish_reason`` and normalise it."""
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return _FINISH_REASON_STOP
        raw = getattr(candidates[0], "finish_reason", None)
        return _normalise_finish_reason(raw)

    def _extract_model_id(self, response: Any) -> str:
        """Return the server-reported model id if present, else the configured one."""
        reported = getattr(response, "model_version", None) or getattr(
            response, "model", None
        )
        if isinstance(reported, str) and reported:
            return reported
        return self.model_id
