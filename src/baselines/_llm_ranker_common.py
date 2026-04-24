"""Shared helpers for the frozen-LLM ranker baselines.

Both ``ZeroShotClaudeRanker`` (see
``docs/llm_baselines/zero_shot_claude_ranker.md``) and the forthcoming
few-shot ICL variant (``docs/llm_baselines/few_shot_icl_ranker.md``) share
the same mechanical pipeline:

1. Build a 4-alternative prompt with letters ``A/B/C/D`` mapped to the
   canonical alternative indices via a deterministic permutation.
2. Call a frozen Claude model once per permutation.
3. Extract a 4-vector of per-letter log-probabilities from the response
   (primary path: Anthropic ``top_logprobs``; fallback: verbalised JSON
   elicitation; stub path: SHA-256-derived deterministic vector).
4. Un-permute each call's distribution back to canonical order.
5. Average the K distributions arithmetically and return ``log p_hat``.

This module factors every step that is identical across the two baselines
into stateless helpers so both callers can stay small and the permutation-
unwind logic is tested in one place.

The shared ``LLMRankerBase.__init__`` accepts the ``llm_client``,
``n_permutations`` (K), and ``seed`` triple; concrete baselines layer
their own state on top (e.g. the few-shot ranker owns a customer
timeline). ``_score_one_event`` implements the K-permutation average so
both subclasses inherit it unchanged.

References
----------
- Hou et al. 2023, "Large Language Models are Zero-Shot Rankers"
  (arXiv:2305.08845).
- Zheng et al. 2023, "LLMs Are Not Robust Multiple Choice Selectors"
  (arXiv:2309.03882).
- Pezeshkpour & Hruschka 2024, "LLMs Sensitivity to the Order of Options"
  (arXiv:2308.11483).
- Wang et al. 2024, "My Answer is C: First-Token Probabilities..."
  (arXiv:2402.14499).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.outcomes.generate import LLMClient, StubLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LETTERS: tuple[str, ...] = (
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
)
"""Canonical ordered letter labels for up to J=10 alternatives.

Matches the Amazon dataset's ``choice_set_size=10``. Callers with
smaller J should pass ``letters=DEFAULT_LETTERS[:J]`` at construction;
the Latin-square helper :func:`letter_permutations` reads the first
``len(letters)`` entries."""

# Floor applied before ``np.log`` so aggregated probabilities never produce
# ``-inf`` scores (see design doc §Known failure modes #1).
_LOGPROB_FLOOR: float = 1e-12


# ---------------------------------------------------------------------------
# ICLExample placeholder (populated by the few-shot agent)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ICLExample:
    """One in-context demonstration drawn from a customer's training history.

    Populated by :func:`build_customer_timeline` at fit time and consumed by
    the few-shot ranker at score time. The zero-shot ranker does not use
    this type; it is exported here so both baselines can share the import.
    Fields mirror the schema documented in
    ``docs/llm_baselines/few_shot_icl_ranker.md`` §3.

    Attributes
    ----------
    order_date
        ``pd.Timestamp`` of the training event; used to sort the timeline
        ascending and to filter strictly earlier examples at test time.
    c_d
        Context string from the training record (``rec["c_d"]``).
    alt_texts
        Length-J list of 7-key adapter dicts for that event.
    chosen_idx
        Canonical alternative index in ``[0, J)`` that the customer chose.
    """

    order_date: pd.Timestamp
    c_d: str
    alt_texts: List[Dict[str, Any]]
    chosen_idx: int


def build_customer_timeline(train: Any) -> Dict[str, List[ICLExample]]:
    """Build ``customer_id -> [ICLExample, ...]`` sorted ascending by date.

    Iterates ``train.raw_events`` once, bucketing by ``customer_id``, and
    sorts each customer's list by ``order_date`` so downstream ICL
    selection can take the N most-recent prior events in O(1) with a
    tail slice.

    Returns ``{}`` when ``train.raw_events is None`` so every test event
    cold-starts to the zero-shot prompt (design doc §3, §8). A KeyError
    on any required record key is allowed to propagate — the adapter
    should always populate them.
    """
    raw_events = getattr(train, "raw_events", None)
    if raw_events is None:
        return {}
    timeline: Dict[str, List[ICLExample]] = {}
    for rec in raw_events:
        cid = str(rec["customer_id"])
        timeline.setdefault(cid, []).append(
            ICLExample(
                order_date=pd.Timestamp(rec["order_date"]),
                c_d=str(rec["c_d"]),
                alt_texts=list(rec["alt_texts"]),
                chosen_idx=int(rec["chosen_idx"]),
            )
        )
    for cid in timeline:
        timeline[cid].sort(key=lambda e: e.order_date)
    return timeline


# ---------------------------------------------------------------------------
# Permutation schedule
# ---------------------------------------------------------------------------


def letter_permutations(
    n_alts: int, K: int, seed: int = 0
) -> List[tuple[int, ...]]:
    """Return the K-permutation schedule mapping letter slots → alt indices.

    For the canonical J=4, K=4 configuration this is the left-rotation
    Latin square prescribed in ``zero_shot_claude_ranker.md`` §4::

        k=0: (0, 1, 2, 3)   identity
        k=1: (1, 2, 3, 0)   left-rotate by 1
        k=2: (2, 3, 0, 1)   left-rotate by 2
        k=3: (3, 0, 1, 2)   left-rotate by 3

    Each returned tuple ``pi`` has length ``n_alts``: ``pi[s]`` is the
    canonical alternative index displayed in letter-slot ``s`` (so slot 0
    → letter A). A Latin-square rotation guarantees every alternative
    appears in every letter-slot exactly once across the K calls, which
    is the minimum condition for first-order positional-bias cancellation
    (Pezeshkpour & Hruschka 2024).

    Parameters
    ----------
    n_alts
        Number of alternatives per event (PO-LEU uses 4).
    K
        Number of permutations. For ``K <= n_alts`` we return the first K
        rows of the left-rotation Latin square; this hits the minimum-bias
        criterion when ``K == n_alts``. For ``K > n_alts`` we wrap around
        (still deterministic in ``seed``) — callers asking for more than
        n_alts permutations are typically doing higher-order debiasing
        experiments.
    seed
        Reserved for future random shuffles over which rotation to start
        from. Currently unused because the rotation order is fully
        specified by the Latin square; exposed so subclasses that want a
        non-canonical starting rotation can override.

    Returns
    -------
    list of tuple[int, ...]
        Length K, each tuple of length ``n_alts``.
    """
    if n_alts <= 0:
        raise ValueError(f"n_alts must be positive, got {n_alts}")
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    del seed  # currently unused — see docstring
    base = list(range(n_alts))
    perms: List[tuple[int, ...]] = []
    for k in range(K):
        shift = k % n_alts
        rotated = tuple(base[shift:] + base[:shift])
        perms.append(rotated)
    return perms


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def render_alternatives(
    alt_texts: Sequence[Mapping[str, Any]],
    letters: Sequence[str] = DEFAULT_LETTERS,
) -> str:
    """Render the ``ALTERNATIVES:`` block used by both ranker baselines.

    Uses the subset of the 7-key ``alt_texts`` adapter schema documented
    in ``docs/llm_baselines/zero_shot_claude_ranker.md`` §2 (``title``,
    ``category``, ``price``, ``popularity_rank``, ``brand``; ``state`` is
    dropped because it is a context attribute, not an alternative
    attribute, and ``is_repeat`` is dropped because it is label-leaky
    under the current adapter (see comment on the render loop).
    Missing keys render as ``"n/a"`` so a partially populated
    ``alt_texts`` dict never raises.

    Parameters
    ----------
    alt_texts
        Length-J list of mappings; typically from
        ``record["alt_texts"]`` produced by the PO-LEU adapter.
    letters
        Letter labels for each slot. Defaults to ``("A", "B", "C", "D")``.
        ``len(letters)`` must equal ``len(alt_texts)``.

    Returns
    -------
    str
        Multi-line block with each alternative prefixed by ``(LETTER)`` and
        the six fields indented two spaces. No trailing newline.
    """
    if len(alt_texts) != len(letters):
        raise ValueError(
            f"render_alternatives: n_alts={len(alt_texts)} does not match "
            f"n_letters={len(letters)}"
        )
    lines: List[str] = []
    for letter, alt in zip(letters, alt_texts):
        title = alt.get("title", "n/a") if isinstance(alt, Mapping) else "n/a"
        category = alt.get("category", "n/a") if isinstance(alt, Mapping) else "n/a"
        price = alt.get("price", "n/a") if isinstance(alt, Mapping) else "n/a"
        popularity = (
            alt.get("popularity_rank", "n/a") if isinstance(alt, Mapping) else "n/a"
        )
        brand = alt.get("brand", "n/a") if isinstance(alt, Mapping) else "n/a"

        lines.append(f"({letter}) Title: {title}")
        lines.append(f"    Category: {category}")
        lines.append(f"    Price: {price}")
        lines.append(f"    Popularity: {popularity}")
        lines.append(f"    Brand: {brand}")
        # Intentionally omitted: ``is_repeat``. The adapter sets it to True
        # only on the chosen alt (negatives from the ASIN lookup carry no
        # per-customer history) — so rendering it to the LLM is direct
        # label leakage. A paper-grade reintroduction would need a
        # per-customer train-history lookup, not the alt_texts flag.
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Logprob extraction
# ---------------------------------------------------------------------------


def _stub_letter_probs(text: str, letters: Sequence[str]) -> np.ndarray:
    """Derive deterministic letter probabilities from a SHA-256 digest.

    Design doc §Stub Behavior option (b): the stub ``LLMClient`` emits
    narrative outcome sentences, not rankings, so we cannot read a real
    distribution out of its text. Instead we hash the text and derive a
    softmax over ``len(letters)`` logits from the first few digest bytes.
    Identical stub output yields identical distributions (determinism
    preserved); distinct stub output yields distinct distributions
    (non-trivial sensitivity), which is all we need to exercise the
    permutation-unwind plumbing in tests.
    """
    n = len(letters)
    if n <= 0:
        raise ValueError("letters must be non-empty")
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Stretch the digest if caller ever passes a larger letter set.
    raw = np.frombuffer(digest[: max(n, 4)], dtype=np.uint8).astype(np.float64)
    logits = raw[:n] / 32.0
    logits = logits - logits.max()
    exp = np.exp(logits)
    probs = exp / exp.sum()
    return probs.astype(np.float64)


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_verbalized_json(text: str, letters: Sequence[str]) -> Optional[np.ndarray]:
    """Parse a ``{"A": p, "B": p, ...}`` JSON fragment from the response.

    Returns ``None`` if parsing fails or if fewer than one letter matches;
    callers fall back to the stub-hash path in that case.
    """
    match = _JSON_OBJECT_RE.search(text or "")
    if match is None:
        return None
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    probs = np.zeros(len(letters), dtype=np.float64)
    any_hit = False
    for i, letter in enumerate(letters):
        v = obj.get(letter, obj.get(letter.lower()))
        if v is None:
            continue
        try:
            probs[i] = float(v)
            any_hit = True
        except (TypeError, ValueError):
            continue
    if not any_hit:
        return None
    total = probs.sum()
    if total <= 0:
        return None
    return probs / total


def extract_letter_logprobs(
    response: Any,
    letters: Sequence[str] = DEFAULT_LETTERS,
) -> np.ndarray:
    """Extract a length-``len(letters)`` probability vector from a response.

    Strategy (matches design doc §3):

    1. If ``response`` is an Anthropic-SDK ``Message`` object carrying
       ``content[0].logprobs.top_logprobs``, scan that list for each
       letter (including the ``" A"`` / ``"A)"`` tokenizer variants) and
       softmax the resulting log-probabilities over the letter set.
    2. If ``response`` is a string, try to parse verbalised JSON first.
    3. If neither works, fall back to the SHA-256-hash stub path so the
       returned vector is well-defined and deterministic.

    Parameters
    ----------
    response
        Either an Anthropic SDK response object, a raw string, or any
        object exposing a ``.text`` / ``.content`` attribute we can
        stringify.
    letters
        Ordered list of letter labels we want probabilities for.

    Returns
    -------
    np.ndarray
        Shape ``(len(letters),)``, dtype float64, summing to 1.0.
    """
    n = len(letters)
    # Anthropic SDK path: response.content[0].logprobs.top_logprobs.
    logprobs_vec = _try_extract_anthropic_top_logprobs(response, letters)
    if logprobs_vec is not None:
        # Softmax over the four letter logprobs (ignore the rest of the
        # vocabulary); missing letters sit at ``-inf`` and softmax to 0.
        shifted = logprobs_vec - np.max(logprobs_vec)
        exp = np.exp(shifted)
        total = exp.sum()
        if total > 0 and np.isfinite(total):
            return (exp / total).astype(np.float64)
        # Fall through to text parsing.

    # String / stub path.
    text = _response_as_text(response)
    verbal = _parse_verbalized_json(text, letters)
    if verbal is not None:
        return verbal
    return _stub_letter_probs(text, letters)


def _response_as_text(response: Any) -> str:
    """Best-effort extraction of a text blob from a response object."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    # GenerationResult-like objects expose ``.text``.
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    # Anthropic SDK: response.content is a list of blocks.
    content = getattr(response, "content", None)
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                parts.append(getattr(block, "text", "") or "")
        if parts:
            return "".join(parts)
    return str(response)


def _try_extract_anthropic_top_logprobs(
    response: Any, letters: Sequence[str]
) -> Optional[np.ndarray]:
    """Return a length-``len(letters)`` logprob vector from an Anthropic response.

    The Anthropic Messages API (as of 2026-04) exposes
    ``response.content[i].logprobs.top_logprobs`` as a list of
    ``{token, logprob}`` entries for the first generated position. We
    pick the block whose ``logprobs`` attribute is populated and scan its
    top-K list for each letter (plus its ``" A"`` and ``"A)"`` tokenizer
    variants, picking the max logprob per letter).

    Returns ``None`` if no block carries ``logprobs`` — callers then use
    the text / stub-hash path.
    """
    content = getattr(response, "content", None)
    if not isinstance(content, list) or not content:
        return None
    for block in content:
        lp = getattr(block, "logprobs", None)
        if lp is None:
            continue
        top_list = getattr(lp, "top_logprobs", None)
        if not top_list:
            continue
        return _scan_top_logprobs_for_letters(top_list, letters)
    return None


def _scan_top_logprobs_for_letters(
    top_list: Any, letters: Sequence[str]
) -> np.ndarray:
    """Return ``[lp_A, lp_B, lp_C, lp_D]`` with ``-inf`` for missing letters."""
    out = np.full(len(letters), -np.inf, dtype=np.float64)
    # ``top_list`` is an iterable of ``{token, logprob}`` mappings or SDK
    # objects with ``.token`` / ``.logprob`` attributes.
    for entry in top_list:
        token = _get(entry, "token")
        logprob = _get(entry, "logprob")
        if token is None or logprob is None:
            continue
        try:
            lp_val = float(logprob)
        except (TypeError, ValueError):
            continue
        stripped = str(token).strip().rstrip(")")
        for i, letter in enumerate(letters):
            if stripped == letter and lp_val > out[i]:
                out[i] = lp_val
    return out


def _get(entry: Any, key: str) -> Any:
    """Mapping-or-attribute accessor used for Anthropic's polymorphic entries."""
    if isinstance(entry, Mapping):
        return entry.get(key)
    return getattr(entry, key, None)


# ---------------------------------------------------------------------------
# Single-call dispatch
# ---------------------------------------------------------------------------


def _is_anthropic_client(client: LLMClient) -> bool:
    """Duck-type check: does ``client`` wrap an Anthropic SDK handle?

    The design doc §6 says we detect the Anthropic path by looking for a
    ``_client.messages.create`` attribute chain rather than importing
    ``AnthropicLLMClient`` (which pulls in the optional ``anthropic``
    extra even when the caller only wants the stub path). ``StubLLMClient``
    explicitly sets ``_is_stub = True`` so we short-circuit that case
    first.
    """
    if getattr(client, "_is_stub", False):
        return False
    inner = getattr(client, "_client", None)
    if inner is None:
        return False
    messages = getattr(inner, "messages", None)
    if messages is None:
        return False
    return hasattr(messages, "create")


def call_llm_for_ranking(
    client: LLMClient,
    system: str,
    user: str,
    letters: Sequence[str] = DEFAULT_LETTERS,
    *,
    temperature: float = 0.0,
    max_tokens: int = 2,
    seed: int = 0,
    top_logprobs: int = 20,
    model_id: Optional[str] = None,
) -> np.ndarray:
    """Issue a single LLM call and return a length-J probability vector.

    Dispatches on the client type (design doc §6):

    - ``StubLLMClient`` (``_is_stub = True``) — run ``client.generate`` to
      get a narrative string, then hash it to deterministic letter
      probabilities. This keeps the plumbing testable without network.
    - Anthropic-flavoured client (``client._client.messages.create``
      available) — bypass ``client.generate`` and call the SDK directly so
      we can request ``logprobs=True`` + ``top_logprobs=N``. If the API
      rejects the logprobs flag (any ``Exception`` raised at runtime), we
      fall back to verbalised-JSON elicitation on the same client. Both
      paths produce a valid length-J vector.
    - Any other ``LLMClient`` — call ``client.generate`` and run the
      response through ``extract_letter_logprobs``, which first tries the
      verbalised-JSON path and falls back to the SHA-256 stub path on
      failure.

    Parameters
    ----------
    client
        Any ``LLMClient`` from ``src.outcomes.generate``.
    system
        System prompt text.
    user
        User prompt text (already includes the alternatives block).
    letters
        Letter labels; probabilities are returned in the same order.
    temperature, max_tokens, seed
        Forwarded to ``client.generate`` for the non-Anthropic path, and
        to ``client._client.messages.create`` for the Anthropic path.
    top_logprobs
        Number of top-logprob entries to request on the Anthropic path
        (0..20; anything outside that range is clamped by the SDK).
    model_id
        Optional override for the Anthropic model identifier. When
        ``None`` we read ``client.model_id``.

    Returns
    -------
    np.ndarray
        Length ``len(letters)``, float64, sums to 1.0.
    """
    if getattr(client, "_is_stub", False):
        # Stub path: run the client's ``generate`` to get its deterministic
        # text, then hash → letter probs. The Stub doesn't understand the
        # ranker prompt shape so we just hand it the concatenation; any
        # input yielding distinct output is fine here.
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        result = client.generate(
            messages,
            temperature=float(temperature),
            top_p=1.0,
            max_tokens=int(max_tokens),
            seed=int(seed),
        )
        return _stub_letter_probs(_response_as_text(result), letters)

    if _is_anthropic_client(client):
        return _call_anthropic_for_ranking(
            client,
            system=system,
            user=user,
            letters=letters,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            top_logprobs=int(top_logprobs),
            model_id=model_id,
        )

    # Generic LLMClient: go through ``generate`` and parse whatever comes
    # back (verbalised JSON, raw letter, or fall through to the hash).
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    result = client.generate(
        messages,
        temperature=float(temperature),
        top_p=1.0,
        max_tokens=int(max_tokens),
        seed=int(seed),
    )
    return extract_letter_logprobs(result, letters)


def _call_anthropic_for_ranking(
    client: LLMClient,
    *,
    system: str,
    user: str,
    letters: Sequence[str],
    temperature: float,
    max_tokens: int,
    top_logprobs: int,
    model_id: Optional[str],
) -> np.ndarray:
    """Anthropic-specific primary/fallback path (design doc §3)."""
    inner = getattr(client, "_client", None)
    if inner is None:  # pragma: no cover - guarded by _is_anthropic_client
        raise RuntimeError(
            "call_llm_for_ranking: expected client._client to be set for "
            "Anthropic-flavoured clients."
        )

    resolved_model = model_id or getattr(client, "model_id", None) or "claude-sonnet-4-6"

    # Primary path: request top_logprobs. Use a one-token prefill ("The
    # answer is (") to force the next generated token into letter space,
    # per the design doc §Known failure modes #2.
    primary_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "assistant", "content": "The answer is ("},
    ]
    primary_kwargs: Dict[str, Any] = {
        "model": resolved_model,
        "system": [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": primary_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }
    try:
        response = inner.messages.create(**primary_kwargs)
        vec = _try_extract_anthropic_top_logprobs(response, letters)
        if vec is not None:
            shifted = vec - np.max(vec)
            exp = np.exp(shifted)
            total = exp.sum()
            if total > 0 and np.isfinite(total):
                return (exp / total).astype(np.float64)
        # No usable logprobs → fall through to the verbalised path below.
        logger.warning(
            "Anthropic response for ranker did not carry usable "
            "top_logprobs; falling back to verbalised elicitation."
        )
    except Exception as exc:  # noqa: BLE001 - SDK raises varied errors here
        logger.warning(
            "Anthropic messages.create(logprobs=True) failed (%s); "
            "falling back to verbalised JSON elicitation.",
            exc,
        )

    # Verbalised-JSON fallback: ask for probabilities as JSON, parse.
    verbal_system = (
        system
        + "\n\nReturn JSON with keys "
        + ", ".join(f'"{l}"' for l in letters)
        + " and probabilities summing to 1. Do not explain."
    )
    verbal_messages = [
        {
            "role": "user",
            "content": user,
        }
    ]
    verbal_kwargs: Dict[str, Any] = {
        "model": resolved_model,
        "system": verbal_system,
        "messages": verbal_messages,
        "max_tokens": 128,
        "temperature": temperature,
    }
    try:
        response = inner.messages.create(**verbal_kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Anthropic verbalised fallback also failed (%s); returning "
            "hash-derived placeholder probabilities.",
            exc,
        )
        return _stub_letter_probs(f"{system}|{user}", letters)

    return extract_letter_logprobs(response, letters)


# ---------------------------------------------------------------------------
# Shared base class
# ---------------------------------------------------------------------------


class LLMRankerBase:
    """Shared state + scoring loop for frozen-LLM ranker baselines.

    Holds the ``(llm_client, n_permutations, seed)`` triple and implements
    the per-event K-permutation average (design doc §4). Subclasses layer
    their own state (e.g. the few-shot ranker stashes a customer
    timeline) and supply the system / user prompt text by overriding
    :meth:`build_system_prompt` and :meth:`build_user_prompt`.

    Parameters
    ----------
    llm_client
        Any ``LLMClient`` from ``src.outcomes.generate``. ``StubLLMClient``
        is supported for tests.
    n_permutations
        Number of letter-slot rotations (design doc §4 uses K=4).
    seed
        Base seed forwarded to each per-permutation LLM call as
        ``seed + k`` so the stub honours it exactly and any future
        caching layer can key on it.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        n_permutations: int = 10,
        seed: int = 0,
    ) -> None:
        if n_permutations <= 0:
            raise ValueError(
                f"n_permutations must be positive, got {n_permutations}"
            )
        self.llm_client: LLMClient = llm_client if llm_client is not None else StubLLMClient()
        self.n_permutations = int(n_permutations)
        self.seed = int(seed)

    # ------------------------------------------------------------------
    # Prompt hooks (subclasses override)
    # ------------------------------------------------------------------
    def build_system_prompt(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def build_user_prompt(
        self,
        event: Mapping[str, Any],
        alt_texts_permuted: Sequence[Mapping[str, Any]],
        letters: Sequence[str],
    ) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared scoring loop
    # ------------------------------------------------------------------
    def _score_one_event(
        self,
        event: Mapping[str, Any],
        letters: Sequence[str] = DEFAULT_LETTERS,
        *,
        event_idx: int = 0,
        temperature: float = 0.0,
        max_tokens: int = 2,
    ) -> np.ndarray:
        """Return ``log p_hat`` of shape ``(n_alts,)`` for one event.

        Runs ``n_permutations`` LLM calls, un-permutes each returned
        letter-distribution back to canonical alternative order, and
        averages arithmetically (design doc §4 Aggregation).
        """
        alt_texts = list(event["alt_texts"])
        n_alts = len(alt_texts)
        if n_alts != len(letters):
            raise ValueError(
                f"event has {n_alts} alternatives but {len(letters)} letters "
                "were provided; ranker baselines require n_alts == len(letters)."
            )

        perms = letter_permutations(n_alts, self.n_permutations, seed=self.seed)
        un_permuted = np.zeros((len(perms), n_alts), dtype=np.float64)
        system = self.build_system_prompt()
        for k, pi in enumerate(perms):
            permuted_alt_texts = [alt_texts[pi[s]] for s in range(n_alts)]
            user = self.build_user_prompt(event, permuted_alt_texts, letters)
            letter_probs = call_llm_for_ranking(
                self.llm_client,
                system,
                user,
                letters,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=self.seed + event_idx * self.n_permutations + k,
            )
            # letter_probs[s] is P(letter slot s). Canonical alt at slot s
            # is pi[s]; store the probability under that canonical index.
            for s in range(n_alts):
                canonical_j = pi[s]
                un_permuted[k, canonical_j] = letter_probs[s]

        p_hat = un_permuted.mean(axis=0)
        p_hat = np.maximum(p_hat, _LOGPROB_FLOOR)
        return np.log(p_hat)


__all__ = [
    "DEFAULT_LETTERS",
    "ICLExample",
    "LLMRankerBase",
    "build_customer_timeline",
    "call_llm_for_ranking",
    "extract_letter_logprobs",
    "letter_permutations",
    "render_alternatives",
]
