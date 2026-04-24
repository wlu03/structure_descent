"""LLM outcome-narrative generator with pluggable client and cache plumbing.

Implements redesign.md §3.3 (decoding settings and parsing) and §3.4 (caching),
and wires §3.5's diversity filter as an injected callable so this module stays
decoupled from its implementation. Every call that hits the LLM is deterministic
given the explicit ``seed``; retries advance the seed so cache entries for the
same ``(customer_id, asin, seed)`` never change beneath a caller.

Public surface
--------------
- :class:`LLMClient` (Protocol) and :class:`GenerationResult`.
- :class:`StubLLMClient` — hermetic, deterministic default client.
- :class:`AnthropicLLMClient` — optional real client; imports ``anthropic``
  lazily so ``import src.outcomes.generate`` stays side-effect-free even when
  the ``llm`` extra is not installed.
- :data:`SENTINEL_OUTCOME` / :func:`parse_completion` — parsing helpers.
- :class:`OutcomesPayload` and :func:`generate_outcomes` — end-to-end entry
  point used by the data-prep pipeline.

Design notes
------------
* All randomness funnels through the explicit ``seed``. The stub hashes
  ``(serialized messages || seed)`` so two distinct seeds on the same messages
  yield different completions (required by the diversity-filter tests).
* The function-level ``generate_outcomes`` loops up to ``max_retries + 1``
  times, bumping ``seed`` by the attempt index each iteration; after the
  budget is exhausted it accepts the last list (§3.5 "accept after 2
  retries").
* The Anthropic client is imported lazily inside ``__init__`` / ``generate``
  so that ``import src.outcomes.generate`` succeeds in environments that do
  not have the ``llm`` extra installed (e.g. CI runners that only exercise
  the stub).
* ``generate_outcomes`` folds ``K`` AND the client's ``model_id`` into the
  ``prompt_version`` argument passed to the cache. The §3.4 cache layer's
  four-field key semantics stay unchanged; the caller-layer composite
  (e.g. ``"v2-K3-claude-sonnet-4-6"``) guarantees that stub outputs and
  real-LLM outputs can never be served interchangeably for the same
  ``(customer_id, asin, seed, prompt_version)`` base tuple.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from src.outcomes.prompts import build_messages

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentinel and client protocol
# ---------------------------------------------------------------------------

SENTINEL_OUTCOME: str = "no additional consequence."
"""Exact sentence used to pad short completions up to ``K`` (§3.3)."""


@dataclass
class GenerationResult:
    """Structured return value from :meth:`LLMClient.generate`.

    Attributes
    ----------
    text:
        Raw completion text. ``parse_completion`` splits this on newlines.
    finish_reason:
        Provider-supplied reason the model stopped (e.g. ``"stop"``,
        ``"length"``). Logged into the outcomes cache metadata.
    model_id:
        Model identifier (e.g. ``"claude-3-5-sonnet"`` or ``"stub-v1"``).
        Logged into metadata for reproducibility (§15 checklist).
    """

    text: str
    finish_reason: str
    model_id: str


@runtime_checkable
class LLMClient(Protocol):
    """Pluggable LLM client used by :func:`generate_outcomes`.

    Implementations must be deterministic given ``seed``. They must not
    inspect globals or env vars at call time; any configuration needed for
    routing (API key, base URL) should be captured in ``__init__``.
    """

    def generate(
        self,
        messages: list[dict],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
    ) -> GenerationResult:  # pragma: no cover - Protocol body
        ...


# ---------------------------------------------------------------------------
# Deterministic stub client
# ---------------------------------------------------------------------------

# Five template families, one per attribute axis listed in SYSTEM_PROMPT. Each
# family holds several phrasings so different seeds on the same messages pick
# different sentences. Every phrasing is first-person, starts with "I ", and
# is between 10 and 25 words (spec §3.2).
_STUB_TEMPLATES: dict[str, tuple[str, ...]] = {
    "financial": (
        "I save roughly twelve dollars this month and redirect the difference toward groceries without juggling my usual weekend spending plans.",
        "I free up about fifteen dollars from my budget and finally cover the overdue electric bill without another late fee.",
        "I spend a little less than expected and set aside the leftover cash toward the kids' upcoming school supplies.",
        "I keep forty dollars out of the credit-card cycle this week and breathe a bit easier when payday finally arrives.",
        "I trim roughly a tenth off this month's discretionary spending and quietly move the savings into the emergency cushion.",
    ),
    "health": (
        "I sleep noticeably better across the week because the evening routine gets simpler and my shoulders stop aching by bedtime.",
        "I skip two trips to the corner pharmacy and my back holds up through the usual Saturday errand run without complaint.",
        "I drink more water throughout the afternoon and finish the week without the familiar mid-day headache I usually fight through.",
        "I avoid a minor kitchen burn by switching tools and my wrist recovers from last month's strain faster than expected.",
        "I get outside a few extra times this week and my lower back stops tightening up during the evening commute home.",
    ),
    "convenience": (
        "I shave about ten minutes off the Tuesday errand loop and reach the school pickup line before it wraps around the block.",
        "I stop juggling three separate shopping apps and my weekend morning actually starts with coffee rather than checkout screens.",
        "I cut down the kitchen cleanup by fifteen minutes and still have time to read to the kids before lights out.",
        "I skip one grocery trip this week and spend that evening catching up on laundry that otherwise bleeds into Sunday night.",
        "I find everything for Monday in a single errand and the week's first morning stops feeling like a minor logistics crisis.",
    ),
    "emotional": (
        "I feel quietly proud about the decision tonight and the low hum of decision fatigue I usually carry finally eases a little.",
        "I notice my shoulders drop on the drive home and the familiar Sunday-night dread shrinks into something manageable for once.",
        "I stop second-guessing the purchase by Wednesday and the calmer mood leaks into how I talk with my partner at dinner.",
        "I feel a small spark of relief after the box arrives and the anxiety about next month's expenses softens noticeably by Friday.",
        "I feel more like myself for an hour this evening and the tension I carry into bedtime finally loosens around the edges.",
    ),
    "social": (
        "I earn a small round of thanks from my partner at dinner and the household logistics feel briefly like a shared project again.",
        "I pass a useful recommendation to a neighbor at pickup and my standing in the informal parent network edges up a notch.",
        "I surprise my kids with an unexpected weekend upgrade and the family text thread briefly lights up with approving emoji.",
        "I bring one less complaint to my partner this week and our short evening conversations feel a touch warmer than usual.",
        "I look slightly more put-together at Friday drop-off and a fellow parent actually asks me for a tip by the gate.",
    ),
}

# Fixed attribute order — mirrors the order listed in SYSTEM_PROMPT so the
# first K outputs are always drawn from the first K attribute axes.
_STUB_ATTRIBUTE_ORDER: tuple[str, ...] = (
    "financial",
    "health",
    "convenience",
    "emotional",
    "social",
)


def _stub_hash(messages: list[dict], seed: int) -> bytes:
    """Derive a deterministic 32-byte digest from ``(messages, seed)``.

    The digest feeds every downstream random choice in :class:`StubLLMClient`,
    so identical inputs yield identical completions and distinct seeds yield
    distinct completions. Messages are serialized by stable repr since the
    keys are always ``"role"``/``"content"`` from ``build_messages``.
    """
    payload = repr(messages).encode("utf-8") + b"\x00" + str(int(seed)).encode("utf-8")
    return hashlib.sha256(payload).digest()


class StubLLMClient:
    """Hermetic, deterministic :class:`LLMClient` used by tests and CI.

    Given ``(messages, seed)`` the client picks one phrasing from each of
    the first ``K`` attribute families (``financial``, ``health``,
    ``convenience``, ``emotional``, ``social``) via a SHA-256 digest, joins
    them with ``"\\n"``, and returns the resulting text.

    ``K`` is inferred from the user prompt's ``Generate K={K}`` footer; if
    absent or malformed we fall back to ``K=3`` (the spec default).

    Carries a class-level ``_is_stub = True`` marker so downstream guards
    (e.g. :func:`src.data.batching.assemble_batch`) can cheaply distinguish
    hermetic CI / test runs from production LLM runs without string-matching
    class names. Real clients (e.g. :class:`AnthropicLLMClient`) do not set
    this attribute.
    """

    # Defense-in-depth marker consumed by assemble_batch's stub guard.
    _is_stub: bool = True

    def __init__(self, model_id: str = "stub-v1") -> None:
        self.model_id = model_id

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
        """Return ``K`` deterministic canned outcome sentences.

        ``temperature``, ``top_p``, ``max_tokens`` are accepted (and must be
        honored by real clients) but intentionally ignored here so the stub
        stays byte-identical across runs.
        """
        K = self._infer_K(messages)
        digest = _stub_hash(messages, seed)

        lines: list[str] = []
        # Cycle through the attribute families in order, picking one phrasing
        # per family by indexing into the digest byte-by-byte. Each byte
        # selects a phrasing via modulo the family size; seed changes flip
        # those selections independently per family.
        for idx, attr in enumerate(_STUB_ATTRIBUTE_ORDER[:K]):
            bucket = _STUB_TEMPLATES[attr]
            choice = digest[idx] % len(bucket)
            lines.append(bucket[choice])

        return GenerationResult(
            text="\n".join(lines),
            finish_reason="stop",
            model_id=self.model_id,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_K(messages: list[dict]) -> int:
        """Best-effort ``K`` extraction from the user block's footer.

        ``prompts.build_user_block`` ends with ``"Generate K={K} outcome
        sentences."`` so a simple token scan suffices. We default to 3 on
        any parse failure; spec default (§3, Appendix B).
        """
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            marker = "Generate K="
            if marker in content:
                tail = content.split(marker, 1)[1]
                digits: list[str] = []
                for ch in tail:
                    if ch.isdigit():
                        digits.append(ch)
                    else:
                        break
                if digits:
                    try:
                        k = int("".join(digits))
                        if k > 0:
                            return k
                    except ValueError:
                        pass
        return 3


# ---------------------------------------------------------------------------
# Optional Anthropic client
# ---------------------------------------------------------------------------

class AnthropicLLMClient:
    """Pluggable real client backed by the ``anthropic`` Python SDK.

    The ``anthropic`` package is imported *lazily* inside ``__init__`` (and
    again inside :meth:`generate` if needed) so that ``import
    src.outcomes.generate`` succeeds even when the ``llm`` extra is not
    installed. Only instantiating this class raises :class:`ImportError`
    when the dependency is missing.
    """

    def __init__(self, model_id: str, *, api_key: str | None = None) -> None:
        """Construct the client.

        Parameters
        ----------
        model_id:
            Anthropic model identifier (e.g. ``"claude-3-5-sonnet-20241022"``).
        api_key:
            Optional API key. When ``None`` we fall back to the
            ``ANTHROPIC_API_KEY`` env var, matching the SDK's own default.
            Reading the env var happens inside this constructor — never at
            module import — so importing this file does not touch the
            environment.
        """
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ImportError(
                "AnthropicLLMClient requires the `anthropic` package. "
                "Install the llm extra: `pip install -e .[llm]`."
            ) from exc

        self.model_id = model_id
        resolved_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        self._api_key = resolved_key
        self._client = anthropic.Anthropic(api_key=resolved_key)

    # Split marker used to cleave the user message into a cacheable prefix
    # (CONTEXT + c_d) and a per-alternative suffix. Kept as a class-level
    # constant so tests can assert against the exact string.
    _USER_SPLIT_MARKER: str = "\n\nALTERNATIVE:"

    def generate(
        self,
        messages: list[dict],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
    ) -> GenerationResult:
        """Dispatch a single completion request and map it to
        :class:`GenerationResult`.

        The ``anthropic`` SDK expects the system prompt as a top-level
        ``system`` argument rather than a message with ``role="system"``,
        so we split it off here before issuing the call.

        ``seed`` is not forwarded because the Anthropic API does not
        currently expose a seed parameter; callers upstream still see seed
        effects because the outcomes cache keys on ``seed`` directly
        (§3.4).

        Prompt caching (Anthropic Messages API, 2025): we mark the frozen
        system prompt and the ``CONTEXT`` + ``c_d`` prefix of the user
        message with ``cache_control={"type": "ephemeral"}`` so Anthropic
        caches them server-side for the 5-minute TTL. The per-alternative
        suffix (``ALTERNATIVE:`` + trailing metadata + ``Generate K=``)
        is left uncached since it churns on every call. If a server
        rejects ``cache_control`` (older model / new error message), we
        fall back to the plain-string shape transparently — see the
        ``anthropic.BadRequestError`` branch below.
        """
        # Import defensively in case the constructor path was bypassed
        # (e.g. subclassing with a custom client factory).
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "AnthropicLLMClient requires the `anthropic` package. "
                "Install the llm extra: `pip install -e .[llm]`."
            ) from exc

        system_text: str | None = None
        user_messages_plain: list[dict] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_text = msg.get("content", "")
            else:
                user_messages_plain.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

        # Anthropic's newer models reject simultaneously-specified
        # ``temperature`` and ``top_p``; we forward only ``temperature``
        # (§3.3 sets both but the API now disallows the combo). The
        # ``top_p`` value is silently dropped for Anthropic calls;
        # stub / other clients still receive both parameters.
        base_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Build a cached-shape request: typed content blocks with
        # cache_control on the system prompt and on the c_d prefix.
        cached_kwargs: dict[str, Any] = dict(base_kwargs)
        cached_kwargs["messages"] = [
            {
                "role": msg["role"],
                "content": self._split_user_content(msg["content"]),
            }
            for msg in user_messages_plain
        ]
        if system_text is not None:
            cached_kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        try:
            response = self._client.messages.create(**cached_kwargs)
        except anthropic.BadRequestError as exc:
            # Fallback for servers / models that don't support the
            # cache_control field. Retry once with the plain-string shape
            # so the caller still gets a completion; cost is unchanged
            # versus the pre-caching behaviour.
            logger.warning(
                "AnthropicLLMClient cache_control rejected (%s); "
                "retrying without cache breakpoints.",
                exc,
            )
            plain_kwargs: dict[str, Any] = dict(base_kwargs)
            plain_kwargs["messages"] = user_messages_plain
            if system_text is not None:
                plain_kwargs["system"] = system_text
            response = self._client.messages.create(**plain_kwargs)

        # Emit cache-usage stats once per call so operators can observe
        # cache hit / miss rates without parsing the Anthropic dashboard.
        usage = getattr(response, "usage", None)
        if usage is not None:
            created = getattr(usage, "cache_creation_input_tokens", None)
            read = getattr(usage, "cache_read_input_tokens", None)
            if created is not None or read is not None:
                logger.debug(
                    "anthropic cache usage: cache_creation_input_tokens=%s "
                    "cache_read_input_tokens=%s",
                    created,
                    read,
                )

        # The SDK returns a list of content blocks; concatenate all text
        # blocks for the parser. Unknown block types are skipped.
        text_parts: list[str] = []
        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))
        text = "".join(text_parts)

        finish_reason = getattr(response, "stop_reason", "") or ""
        model_id = getattr(response, "model", self.model_id) or self.model_id

        return GenerationResult(text=text, finish_reason=finish_reason, model_id=model_id)

    @classmethod
    def _split_user_content(cls, content: str) -> list[dict]:
        """Split a user message into cached prefix + uncached suffix blocks.

        The split point is ``"\\n\\nALTERNATIVE:"`` so everything up to
        (but not including) the ``ALTERNATIVE:`` line — i.e. ``CONTEXT:``
        plus the ``c_d`` block — ends up in the cacheable prefix, while
        the per-alternative metadata and the trailing ``Generate K=``
        line stay uncached. Messages lacking the marker fall back to a
        single uncached block (defensive: keeps the client working if
        ``USER_BLOCK_TEMPLATE`` ever changes shape).
        """
        marker = cls._USER_SPLIT_MARKER
        if marker in content:
            prefix, suffix = content.split(marker, 1)
            return [
                {
                    "type": "text",
                    "text": prefix,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": marker + suffix},
            ]
        return [{"type": "text", "text": content}]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_completion(
    text: str,
    K: int,
    *,
    context: Mapping[str, Any] | None = None,
) -> list[str]:
    """Split completion ``text`` into exactly ``K`` outcome sentences.

    Following §3.3: split on newlines, strip whitespace, drop empty lines,
    keep the first ``K``. If fewer than ``K`` survive, pad with
    :data:`SENTINEL_OUTCOME` and emit a ``logging`` warning. The optional
    ``context`` mapping (``{"customer_id", "asin"}``) is interpolated into
    the warning so operators can trace which pair produced a malformed
    completion.

    Parameters
    ----------
    text:
        Raw completion string from an :class:`LLMClient`.
    K:
        Target number of outcomes (spec default ``K=3``).
    context:
        Optional ``(customer_id, asin)`` mapping for log enrichment.

    Returns
    -------
    list[str]
        Exactly ``K`` strings.
    """
    if not isinstance(K, int) or isinstance(K, bool) or K <= 0:
        raise ValueError(f"K must be a positive int, got {K!r}")

    raw_lines = text.split("\n") if text else []
    cleaned = [ln.strip() for ln in raw_lines]
    non_empty = [ln for ln in cleaned if ln]

    truncated = non_empty[:K]
    if len(truncated) < K:
        deficit = K - len(truncated)
        ctx = dict(context) if context else {}
        logger.warning(
            "parse_completion padding with sentinel: missing=%d K=%d context=%s",
            deficit,
            K,
            ctx,
        )
        truncated = truncated + [SENTINEL_OUTCOME] * deficit

    return truncated


# ---------------------------------------------------------------------------
# Outcomes payload + top-level entry point
# ---------------------------------------------------------------------------

@dataclass
class OutcomesPayload:
    """Container returned by :func:`generate_outcomes`.

    Attributes
    ----------
    outcomes:
        Exactly ``K`` outcome sentences.
    metadata:
        Decoding settings and provenance fields (see
        :func:`generate_outcomes` for the full list of keys).
    """

    outcomes: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


# Decoding defaults straight from configs/default.yaml § outcomes.generator.
_DEFAULT_GEN_KWARGS: dict[str, Any] = {
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 180,
}


def _sanitize_model_id(model_id: str) -> str:
    """Normalize ``model_id`` for safe inclusion in cache keys.

    Strips whitespace, lower-cases, and replaces ``/`` with ``_`` so that
    forward-slashed provider-namespaced ids (e.g. ``"anthropic/claude-..."``)
    stay filesystem- and cache-safe. Pure-string transform; no external
    dependencies. Consumed by :func:`generate_outcomes` when composing
    ``cache_prompt_version``.
    """
    return str(model_id).strip().replace("/", "_").lower()


def generate_outcomes(
    customer_id: str,
    asin: str,
    c_d: str,
    alt: Mapping[str, Any],
    *,
    K: int = 3,
    seed: int,
    prompt_version: str,
    client: LLMClient,
    cache: Any | None = None,
    generation_kwargs: Mapping[str, Any] | None = None,
    diversity_filter: Callable[[list[str]], tuple[list[str], bool]] | None = None,
    max_retries: int = 2,
) -> OutcomesPayload:
    """End-to-end outcome generation for a single ``(customer_id, asin)``.

    Flow (redesign.md §3.3–§3.5):

    1. Cache lookup. On hit, return the stored payload verbatim.
    2. Build chat messages via :func:`prompts.build_messages`.
    3. Loop up to ``max_retries + 1`` times:
       - call ``client.generate`` with ``seed = seed + attempt_idx``;
       - parse the completion with :func:`parse_completion`;
       - if a ``diversity_filter`` callable is provided, invoke it on the
         parsed list; accept the (possibly rewritten) list when it returns
         ``ok=True``.
    4. Record metadata (decoding settings + provenance) and persist to the
       cache.

    Cache-key composition
    ---------------------
    The cache layer (§3.4) keys on four fields:
    ``(customer_id, asin, seed, prompt_version)``. To keep that layer
    spec-compliant while still distinguishing runs that must never share
    entries, we fold two extra dimensions into the ``prompt_version``
    argument passed down to the cache:

    * ``K`` — so ``K=1`` validation runs never collide with ``K=3``
      ablation runs on the same base prompt.
    * ``client.model_id`` (sanitised via :func:`_sanitize_model_id`) —
      so a stub client's outputs can never be read back by a real-LLM
      run with the same base tuple, and vice versa. This is the
      defence against the 44k-entry stub contamination regression.

    The composite key lives in metadata as ``cache_prompt_version``
    (e.g. ``"v2-K3-claude-sonnet-4-6"``); the user-supplied
    ``prompt_version`` (e.g. ``"v2"``) is preserved verbatim in the
    ``prompt_version`` metadata field.

    Parameters
    ----------
    customer_id, asin:
        Identifiers used for cache keying (§3.4) and log enrichment.
    c_d, alt:
        Context string and alternative attributes forwarded to the prompt
        builder.
    K:
        Number of outcome sentences (§3 default: 3).
    seed:
        Base seed; retries add the attempt index so the same base seed
        always produces the same sequence of attempts.
    prompt_version:
        Tag from :data:`prompts.PROMPT_VERSION`; part of the cache key.
    client:
        Any :class:`LLMClient`. Defaults are not supplied — callers must
        choose (stub vs. real) deliberately.
    cache:
        Optional :class:`OutcomesCache`. ``None`` disables caching
        entirely (useful for tests that want to exercise the generate path
        without touching disk).
    generation_kwargs:
        Overrides for ``temperature`` / ``top_p`` / ``max_tokens``. Values
        that land on the client are also recorded in metadata.
    diversity_filter:
        Optional callable. Receives the parsed list and must return
        ``(possibly_rewritten_list, ok: bool)``. Implemented by a sibling
        module (``src/outcomes/diversity_filter.py``).
    max_retries:
        Upper bound on filter-driven retries. The loop runs at most
        ``max_retries + 1`` times total (§3.5 "At most 2 retries; after
        that, accept.").

    Returns
    -------
    OutcomesPayload
        ``outcomes`` is exactly ``K`` strings; ``metadata`` carries the
        fields listed in the module docstring.
    """
    # Cache-key ``prompt_version`` folds in ``K``, the client's
    # ``model_id``, *and* a hash of the per-event context string ``c_d``.
    # §3.4 lists four key fields (customer_id, asin, seed,
    # prompt_version); we fold the extra dimensions into
    # ``prompt_version`` rather than adding fifth/sixth key fields,
    # keeping the cache layer itself spec-compliant while every caller-
    # layer dimension that must change the key is accounted for.
    #
    # K fold: different K values (K=1 validation vs K=3 ablation) must
    # not share cache entries — the outcome *count* is a user-visible
    # output shape.
    #
    # model_id fold: fix for a silent contamination bug — a stub-LLM
    # run wrote entries under a given (cust, asin, seed, prompt_version-K)
    # composite, and a subsequent real-LLM run with the same base tuple
    # would transparently hit the stub's outputs.
    #
    # c_d fold (V5-B1 fix): per-event context strings vary by recent-
    # purchases slice (Wave-12 per-event c_d), so the same
    # (customer_id, asin, seed) tuple can legitimately render different
    # outcomes on different events. Without folding c_d into the key,
    # whichever event ran first would silently poison the cache for
    # every subsequent event sharing the base tuple. A 16-char sha256
    # prefix is collision-resistant in practice for the scales we run.
    #
    # Sanitisation (``_sanitize_model_id``): strip whitespace, lower,
    # replace ``/`` → ``_``. Keeps the composite safe as a cache/fs
    # fragment regardless of provider namespacing convention.
    model_id_tag = _sanitize_model_id(getattr(client, "model_id", "unknown"))
    cd_hash = hashlib.sha256(c_d.encode("utf-8")).hexdigest()[:16]
    cache_prompt_version = (
        f"{prompt_version}-K{int(K)}-{model_id_tag}-cd{cd_hash}"
    )
    logger.debug(
        "generate_outcomes cache composite prompt_version=%r "
        "(base=%r K=%d model_id=%r cd_hash=%s)",
        cache_prompt_version,
        prompt_version,
        int(K),
        model_id_tag,
        cd_hash,
    )

    # ---- 1. cache lookup ------------------------------------------------
    if cache is not None:
        cached = cache.get_outcomes(customer_id, asin, seed, cache_prompt_version)
        if cached is not None:
            cached_outcomes = list(cached.get("outcomes", []))
            # Defensive: if a legacy (pre-K-fold) entry somehow survived,
            # its length may not match K. Bail on the cache hit and
            # regenerate rather than tripping the batching invariant.
            if len(cached_outcomes) == int(K):
                return OutcomesPayload(
                    outcomes=cached_outcomes,
                    metadata=dict(cached.get("metadata", {})),
                )

    # ---- 2. build messages ---------------------------------------------
    messages = build_messages(c_d=c_d, alt=alt, K=K)

    # ---- 3. call + parse + diversity loop ------------------------------
    merged_kwargs: dict[str, Any] = {**_DEFAULT_GEN_KWARGS}
    if generation_kwargs:
        merged_kwargs.update(dict(generation_kwargs))

    temperature = float(merged_kwargs["temperature"])
    top_p = float(merged_kwargs["top_p"])
    max_tokens = int(merged_kwargs["max_tokens"])

    attempts_budget = max_retries + 1
    final_outcomes: list[str] = []
    result: GenerationResult | None = None
    seed_used = seed
    # Track whether the accepted completion was short (had to be padded
    # with SENTINEL_OUTCOME). We prefer to retry short completions before
    # accepting them rather than silently training on sentinels.
    accepted_was_padded = False

    def _count_nonempty_lines(txt: str) -> int:
        """Count valid outcome lines a ``parse_completion`` call would keep."""
        if not txt:
            return 0
        return sum(1 for ln in txt.split("\n") if ln.strip())

    for attempt_idx in range(attempts_budget):
        seed_used = seed + attempt_idx
        result = client.generate(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed_used,
        )
        # Short-completion retry: if the model returned fewer than K
        # non-empty lines and we still have attempts left, reissue with
        # a different seed rather than padding with SENTINEL_OUTCOME and
        # teaching downstream embeddings on a dummy sentence. Only the
        # last attempt accepts a short completion.
        nonempty = _count_nonempty_lines(result.text)
        is_short = nonempty < int(K)
        is_last_attempt = attempt_idx == attempts_budget - 1
        if is_short and not is_last_attempt:
            logger.warning(
                "generate_outcomes: short completion (%d/%d lines) on "
                "attempt %d/%d for customer_id=%r asin=%r — retrying "
                "with seed+%d",
                nonempty,
                int(K),
                attempt_idx + 1,
                attempts_budget,
                customer_id,
                asin,
                attempt_idx + 1,
            )
            continue
        parsed = parse_completion(
            result.text,
            K,
            context={"customer_id": customer_id, "asin": asin},
        )
        if diversity_filter is None:
            final_outcomes = parsed
            accepted_was_padded = is_short  # may be True on last attempt
            break

        rewritten, ok = diversity_filter(parsed)
        final_outcomes = list(rewritten)
        if ok:
            accepted_was_padded = is_short
            break
    else:  # pragma: no cover - for clarity; loop always executes once
        pass

    # Even if the loop exhausted without ``ok=True`` the spec instructs us to
    # accept the last produced list. ``final_outcomes`` already holds it.
    if result is None:  # pragma: no cover - only reachable with max_retries < 0
        raise RuntimeError("generate_outcomes: client.generate was never called")

    # ---- 4. metadata + cache write -------------------------------------
    # ``prompt_version`` is the user-supplied tag (e.g. "v2"); the
    # ``cache_prompt_version`` we *actually* keyed the cache by folds in
    # K and the sanitised model_id. We write both so cache introspection
    # tooling can tell real-LLM entries apart from stub entries without
    # re-deriving the composite.
    metadata: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "model_id": result.model_id,
        "finish_reason": result.finish_reason,
        "seed": seed_used,
        "prompt_version": prompt_version,
        "cache_prompt_version": cache_prompt_version,
        "timestamp": time.time(),
        # ``was_padded`` marks cached entries whose last attempt still
        # produced fewer than K lines and therefore got SENTINEL_OUTCOME
        # padding. Downstream audits can filter these out (they are a
        # tiny fraction of calls — ~0.1% on Amazon — but a paper-grade
        # run may want to exclude them).
        "was_padded": bool(accepted_was_padded),
    }

    if cache is not None:
        cache.put_outcomes(
            customer_id,
            asin,
            seed,
            cache_prompt_version,
            final_outcomes,
            metadata,
        )

    return OutcomesPayload(outcomes=final_outcomes, metadata=metadata)


__all__ = [
    "SENTINEL_OUTCOME",
    "GenerationResult",
    "LLMClient",
    "StubLLMClient",
    "AnthropicLLMClient",
    "OutcomesPayload",
    "parse_completion",
    "generate_outcomes",
]
