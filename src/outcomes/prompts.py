"""Prompt templates for the PO-LEU LLM outcome generator (redesign.md §3.2).

This module exposes the *frozen* system prompt and the dynamic user-block
template used by ``src/outcomes/generate.py``. Only template strings and pure
builder helpers live here; no file I/O, no globals beyond the template strings,
no LLM / encoder dependencies.

Public surface
--------------
- ``SYSTEM_PROMPT`` : the verbatim §3.2 system prompt with the literal sentence
  count interpolated at build time (see NOTES.md for the K-substitution
  rationale).
- ``USER_BLOCK_TEMPLATE`` : the verbatim §3.2 user-block template with Python
  ``str.format``-style placeholders.
- ``PROMPT_VERSION`` : version tag that feeds the generation cache key (§3.4).
  Bump whenever either template changes.
- ``build_system_prompt(K)`` / ``build_user_block(...)`` / ``build_messages(...)``
  : pure, deterministic helpers.
"""

from __future__ import annotations

from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Frozen text (redesign.md §3.2). The literal ``{K}`` placeholder inside
# ``SYSTEM_PROMPT`` is substituted by :func:`build_system_prompt` at call time;
# the rest of the string is verbatim from the spec and MUST NOT drift.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You generate short first-person outcome narratives for a decision-maker "
    "considering an alternative. Produce exactly {K} sentences, each 10–25 "
    "words, each describing a *different type* of consequence: financial, "
    "health/physical, convenience/time, emotional/identity, social/relational. "
    "Sentences are in the first person, present or near-future tense, and "
    "grounded in the person's specific context. Do not describe the product. "
    "Do not hedge (\"might\", \"could\" are fine; \"I think\" is not). Do not "
    "number the sentences; separate them with newlines."
)

USER_BLOCK_TEMPLATE: str = (
    "CONTEXT:\n"
    "{c_d}\n"
    "\n"
    "ALTERNATIVE:\n"
    "- Name/title: {title}\n"
    "- Category: {category}\n"
    "- Price: ${price}\n"
    "- Popularity: {popularity_rank}\n"
    "{optional_fields}"
    "\n"
    "Generate K={K} outcome sentences."
)

PROMPT_VERSION: str = "v2"

# ---------------------------------------------------------------------------
# Anchored K=5 prompt (Group 3 fix). Forces one outcome per named axis in the
# canonical order matching the M=5 attribute heads' post-hoc labels in
# src/eval/interpret.py. Addresses starved-heads diagnosis (m0 near-zero
# variance, m3 runaway scale = random-walk fingerprints) where K=3 left two
# heads with no aligned outcome signal.
# ---------------------------------------------------------------------------

PROMPT_VERSION_ANCHORED: str = "v3_anchored"

# matches the M=5 attribute heads' post-hoc axis labels in src/eval/interpret.py head naming order
ANCHORED_AXES: tuple[str, ...] = (
    "financial",
    "health",
    "convenience",
    "emotional",
    "social",
)

SYSTEM_PROMPT_ANCHORED: str = (
    "Generate exactly K=5 outcomes, one per axis, in this canonical order: "
    "financial, health, convenience, emotional, social. Each outcome describes "
    "a plausible consequence FOR THIS PERSON considering THIS alternative, "
    "scored along ITS axis. If an axis is genuinely irrelevant for this "
    "category, write a brief honest acknowledgement; the salience layer will "
    "down-weight irrelevant axes. Use first-person 'I' narration."
)

# ---------------------------------------------------------------------------
# Mobility-anchored prompt (Boston / urban movement). M=5 axes tuned to
# trip-choice rather than purchase-choice. Bumps the cache key separately
# so existing v3_anchored entries are not silently reused.
# ---------------------------------------------------------------------------

PROMPT_VERSION_MOBILITY_ANCHORED: str = "v4_mobility_anchored"

# Maps 1:1 onto the M=5 attribute heads when used with anchored generation.
# Order is canonical and must match the head-naming list passed to
# :func:`src.eval.interpret.head_naming_report`.
MOBILITY_ANCHORED_AXES: tuple[str, ...] = (
    "convenience",
    "routine",
    "purpose",
    "social",
    "leisure",
)

SYSTEM_PROMPT_MOBILITY_ANCHORED: str = (
    "You generate short first-person outcome narratives for a person "
    "deciding whether to visit a particular place. Produce exactly K=5 "
    "sentences, one per axis, in this canonical order: convenience, "
    "routine, purpose, social, leisure. Each outcome describes a plausible "
    "consequence FOR THIS PERSON if they go to THIS place, scored along "
    "ITS axis:\n"
    "  - convenience: travel time, distance, effort, friction.\n"
    "  - routine: how this fits the person's habits — familiar vs. novel.\n"
    "  - purpose: what need this trip serves (errand, work, food, etc.).\n"
    "  - social: who they are likely to be with or encounter.\n"
    "  - leisure: enjoyment, identity, comfort.\n"
    "Each sentence is 10-25 words, first-person, present or near-future "
    "tense. Do not describe the place itself. Do not number sentences; "
    "separate them with newlines. If an axis is genuinely irrelevant for "
    "this trip, write a brief honest acknowledgement; the salience layer "
    "will down-weight it."
)

# ---------------------------------------------------------------------------
# Curriculum refinement prompts (commit: critique-and-revise loop). Used by
# :mod:`src.outcomes.refine` to fix outcomes for events the model fails on.
# Bump :data:`REFINED_PROMPT_VERSION` when either template changes — it is
# folded into the cache key so prior refined entries are not silently reused.
# ---------------------------------------------------------------------------

REFINED_PROMPT_VERSION: str = "v3_refined"

CRITIC_SYSTEM_PROMPT: str = (
    "You are a careful critic of first-person outcome narratives. You receive "
    "K candidate outcomes that another model wrote for a person considering "
    "an alternative. Score them on two axes AND identify which specific "
    "outcomes are weak so the reviser can rewrite only those — strong "
    "outcomes get preserved verbatim, reducing churn.\n"
    "  - plausibility (1-5): does each outcome describe something that could "
    "realistically happen to THIS person given THIS alternative? "
    "Penalize generic or off-topic claims.\n"
    "  - diversity (1-5): do the K outcomes cover *different types* of "
    "consequences (financial, health, convenience, emotional, social) — or "
    "are they paraphrases of each other?\n"
    "  - weak_outcome_indices: 0-indexed positions of outcomes that need "
    "rewriting (generic, off-topic, redundant with another, factually "
    "incoherent). EMPTY list when every outcome is acceptable. Aim to "
    "flag only outcomes that materially degrade the set; leave the others "
    "alone so their signal carries through.\n"
    "Return STRICT JSON with this schema and nothing else:\n"
    "  {\"plausibility\": <int 1-5>, \"diversity\": <int 1-5>, "
    "\"weak_outcome_indices\": [<0-indexed ints>], "
    "\"notes\": \"<one short sentence per weak outcome, naming the issue>\"}"
)

CRITIC_USER_TEMPLATE: str = (
    "PERSON CONTEXT:\n"
    "{c_d}\n"
    "\n"
    "ALTERNATIVE:\n"
    "- Name/title: {title}\n"
    "- Category: {category}\n"
    "- Price: ${price}\n"
    "- Popularity: {popularity_rank}\n"
    "{optional_fields}"
    "\n"
    "CANDIDATE OUTCOMES (K={K}):\n"
    "{outcomes_block}\n"
    "\n"
    "Return STRICT JSON only."
)

REVISER_SYSTEM_PROMPT: str = (
    "You rewrite first-person outcome narratives based on a critic's feedback. "
    "Produce exactly {K} sentences, each 10-25 words, each describing a "
    "*different type* of consequence (financial, health/physical, "
    "convenience/time, emotional/identity, social/relational). Sentences are "
    "first-person, present or near-future tense, and grounded in the person's "
    "specific context. Do not describe the product. Do not number sentences; "
    "separate them with newlines. Address the critic's concerns directly: if "
    "the critic flagged genericness, get specific; if the critic flagged "
    "redundancy, replace duplicate consequence types."
)

REVISER_USER_TEMPLATE: str = (
    "PERSON CONTEXT:\n"
    "{c_d}\n"
    "\n"
    "ALTERNATIVE:\n"
    "- Name/title: {title}\n"
    "- Category: {category}\n"
    "- Price: ${price}\n"
    "- Popularity: {popularity_rank}\n"
    "{optional_fields}"
    "\n"
    "PRIOR OUTCOMES (to be improved):\n"
    "{outcomes_block}\n"
    "\n"
    "CRITIC FEEDBACK:\n"
    "- plausibility: {plausibility}/5\n"
    "- diversity: {diversity}/5\n"
    "- notes: {notes}\n"
    "\n"
    "Generate K={K} revised outcome sentences."
)


# Per-outcome reviser. When the critic flags only specific positions as
# weak, this prompt asks for ONLY those rewrites — strong outcomes stay
# byte-identical via splice in the orchestrator. Different system prompt
# from the monolithic reviser because the contract is a partial output
# count + position-ordered emission, not the K-sentences-total contract.
REVISER_PER_OUTCOME_SYSTEM_PROMPT: str = (
    "You rewrite a small subset of weak outcome narratives based on a "
    "critic's per-position feedback. The strong outcomes are kept verbatim "
    "by the caller — DO NOT include them in your output. Produce exactly N "
    "sentences, one for each weak position the critic flagged, in the "
    "SAME order as the position list (ascending). Each sentence is 10-25 "
    "words, first-person, present or near-future tense, and grounded in "
    "the person's specific context. Do not describe the product. Do not "
    "number sentences; separate them with newlines. Each new sentence "
    "must complement (not paraphrase) the strong outcomes that will sit "
    "alongside it in the final K-sentence set."
)

REVISER_PER_OUTCOME_USER_TEMPLATE: str = (
    "PERSON CONTEXT:\n"
    "{c_d}\n"
    "\n"
    "ALTERNATIVE:\n"
    "- Name/title: {title}\n"
    "- Category: {category}\n"
    "- Price: ${price}\n"
    "- Popularity: {popularity_rank}\n"
    "{optional_fields}"
    "\n"
    "ALL CURRENT OUTCOMES (numbered 1..K, weak ones marked):\n"
    "{outcomes_block}\n"
    "\n"
    "WEAK POSITIONS (1-indexed, in ascending order): {weak_positions_1based}\n"
    "STRONG POSITIONS (kept verbatim, do NOT regenerate these): "
    "{strong_positions_1based}\n"
    "\n"
    "CRITIC FEEDBACK:\n"
    "- plausibility: {plausibility}/5\n"
    "- diversity: {diversity}/5\n"
    "- notes: {notes}\n"
    "\n"
    "Output exactly N={N} sentences, one per weak position, in ascending "
    "order. Each new sentence must cover a *consequence type* that "
    "complements (does not duplicate) the strong outcomes' types."
)

# Required keys for the ``alt`` mapping passed to :func:`build_user_block`.
_REQUIRED_ALT_FIELDS: tuple[str, ...] = (
    "title",
    "category",
    "price",
    "popularity_rank",
)


def build_system_prompt(K: int) -> str:
    """Return the frozen §3.2 system prompt with the literal sentence count
    embedded.

    Parameters
    ----------
    K : int
        Number of outcome sentences to request. Must be a positive int.

    Returns
    -------
    str
        The system prompt with ``{K}`` substituted. No trailing newline.
    """
    if not isinstance(K, int) or isinstance(K, bool) or K <= 0:
        raise ValueError(f"K must be a positive int, got {K!r}")
    return SYSTEM_PROMPT.format(K=K)


def _render_optional_fields(optional_fields: Mapping[str, Any] | None) -> str:
    """Render an optional-fields mapping as zero or more ``- key: value`` lines.

    Each entry becomes its own line ending in ``\\n``. Returns the empty string
    when ``optional_fields`` is falsy so that the surrounding template collapses
    cleanly (no stray blank line is inserted).
    """
    if not optional_fields:
        return ""
    lines = [f"- {key}: {value}" for key, value in optional_fields.items()]
    return "\n".join(lines) + "\n"


def build_user_block(
    c_d: str,
    alt: Mapping[str, Any],
    K: int,
    optional_fields: Mapping[str, Any] | None = None,
) -> str:
    """Substitute all fields into :data:`USER_BLOCK_TEMPLATE`.

    Parameters
    ----------
    c_d : str
        The person context string (§2.2). Rendered verbatim under ``CONTEXT:``.
    alt : Mapping
        Alternative attributes. Must contain ``title``, ``category``, ``price``,
        ``popularity_rank``; a :class:`ValueError` is raised otherwise.
    K : int
        Number of outcome sentences requested; rendered as ``K={K}``.
    optional_fields : Mapping, optional
        Additional ``- key: value`` lines appended after ``Popularity``.

    Returns
    -------
    str
        The fully-substituted user block.
    """
    if not isinstance(K, int) or isinstance(K, bool) or K <= 0:
        raise ValueError(f"K must be a positive int, got {K!r}")

    missing = [field for field in _REQUIRED_ALT_FIELDS if field not in alt]
    if missing:
        raise ValueError(
            "alt is missing required field(s): "
            + ", ".join(missing)
            + f"; required keys are {list(_REQUIRED_ALT_FIELDS)}"
        )

    # V2-6 fix: reject any ``optional_fields`` entry whose key collides
    # with a canonical §3.2 field. Canonical fields render via their
    # dedicated lines in ``USER_BLOCK_TEMPLATE``; allowing them to slip
    # in through ``optional_fields`` would double-render them (once
    # canonically, once as "- key: value") and desync the cache-key
    # prompt_version from the visible prompt text.
    if optional_fields:
        collisions = [
            k for k in optional_fields.keys() if k in _REQUIRED_ALT_FIELDS
        ]
        if collisions:
            raise ValueError(
                "optional_fields contains canonical alt key(s): "
                + ", ".join(collisions)
                + f"; canonical keys {list(_REQUIRED_ALT_FIELDS)} must be "
                "set via the ``alt`` mapping, not ``optional_fields``."
            )

    # Any keys in ``alt`` beyond the required four (e.g. ``brand``,
    # ``is_repeat``, ``state`` added by the Wave-11 richer alt_text)
    # are merged into ``optional_fields`` so they render as
    # "- key: value" lines after the canonical four. An explicit
    # ``optional_fields`` kwarg wins on key collision for non-canonical
    # keys (canonical collisions are rejected above).
    extras_from_alt = {
        k: alt[k] for k in alt.keys() if k not in _REQUIRED_ALT_FIELDS
    }
    if optional_fields:
        extras_from_alt.update(dict(optional_fields))
    return USER_BLOCK_TEMPLATE.format(
        c_d=c_d,
        title=alt["title"],
        category=alt["category"],
        price=alt["price"],
        popularity_rank=alt["popularity_rank"],
        optional_fields=_render_optional_fields(extras_from_alt),
        K=K,
    )


def build_messages(
    c_d: str,
    alt: Mapping[str, Any],
    K: int,
    optional_fields: Mapping[str, Any] | None = None,
) -> list[dict]:
    """Return chat-completion-style messages for the outcome generator.

    Returns
    -------
    list of dict
        ``[{"role": "system", "content": ...}, {"role": "user", "content": ...}]``.
        Both ``content`` values are fully substituted; no further formatting is
        required before handing the list to an LLM client.
    """
    system_content = build_system_prompt(K)
    user_content = build_user_block(
        c_d=c_d,
        alt=alt,
        K=K,
        optional_fields=optional_fields,
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_messages_anchored(
    c_d: str,
    alt: Mapping[str, Any],
    K: int,
    optional_fields: Mapping[str, Any] | None = None,
) -> list[dict]:
    """Anchored variant of :func:`build_messages` (Group-3 fix).

    Forces ``K == len(ANCHORED_AXES) == 5`` and uses
    :data:`SYSTEM_PROMPT_ANCHORED` so the LLM produces exactly one outcome per
    named axis in canonical order. Each outcome aligns 1:1 with one of the
    M=5 attribute heads, eliminating the starved-head failure mode at K=3.
    """
    if K != len(ANCHORED_AXES):
        raise ValueError(
            f"expected K=5 to match ANCHORED_AXES; got K={K}"
        )
    user_content = build_user_block(
        c_d=c_d,
        alt=alt,
        K=K,
        optional_fields=optional_fields,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_ANCHORED},
        {"role": "user", "content": user_content},
    ]


def build_messages_mobility_anchored(
    c_d: str,
    alt: Mapping[str, Any],
    K: int,
    optional_fields: Mapping[str, Any] | None = None,
) -> list[dict]:
    """Mobility-anchored variant of :func:`build_messages`.

    Forces ``K == len(MOBILITY_ANCHORED_AXES) == 5`` and uses
    :data:`SYSTEM_PROMPT_MOBILITY_ANCHORED` so the LLM emits one outcome
    per mobility-tuned axis (convenience, routine, purpose, social,
    leisure) in canonical order. Each outcome aligns 1:1 with one of the
    M=5 attribute heads under the head_names override
    :data:`MOBILITY_ANCHORED_AXES`.
    """
    if K != len(MOBILITY_ANCHORED_AXES):
        raise ValueError(
            f"expected K=5 to match MOBILITY_ANCHORED_AXES; got K={K}"
        )
    user_content = build_user_block(
        c_d=c_d,
        alt=alt,
        K=K,
        optional_fields=optional_fields,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_MOBILITY_ANCHORED},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Curriculum-refinement prompt builders (parallel to build_messages)
# ---------------------------------------------------------------------------


def _render_outcomes_block(outcomes: list[str]) -> str:
    """Render K candidate outcomes as a numbered block for the critic/revise."""
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(outcomes))


def _render_outcomes_block_marked(
    outcomes: list[str], weak_indices: list[int]
) -> str:
    """Render K outcomes with a ``*** WEAK ***`` tag on flagged positions.

    Used by the per-outcome reviser so the model has both the full set
    (for diversity coherence) and an explicit flag of which positions to
    rewrite. ``weak_indices`` is 0-indexed; tags are appended to the same
    line as the outcome text.
    """
    weak_set = set(int(i) for i in weak_indices)
    lines: list[str] = []
    for i, s in enumerate(outcomes):
        marker = "  *** WEAK — REWRITE ***" if i in weak_set else ""
        lines.append(f"{i + 1}. {s}{marker}")
    return "\n".join(lines)


def build_critic_messages(
    c_d: str,
    alt: Mapping[str, Any],
    outcomes: list[str],
    K: int,
    optional_fields: Mapping[str, Any] | None = None,
) -> list[dict]:
    """Build the chat messages that score K candidate outcomes."""
    if not isinstance(K, int) or isinstance(K, bool) or K <= 0:
        raise ValueError(f"K must be a positive int, got {K!r}")
    if len(outcomes) != K:
        raise ValueError(
            f"build_critic_messages: got {len(outcomes)} outcomes, expected K={K}"
        )
    missing = [field for field in _REQUIRED_ALT_FIELDS if field not in alt]
    if missing:
        raise ValueError(
            "alt is missing required field(s): " + ", ".join(missing)
        )
    extras = {
        k: alt[k] for k in alt.keys() if k not in _REQUIRED_ALT_FIELDS
    }
    if optional_fields:
        extras.update(dict(optional_fields))
    user_content = CRITIC_USER_TEMPLATE.format(
        c_d=c_d,
        title=alt["title"],
        category=alt["category"],
        price=alt["price"],
        popularity_rank=alt["popularity_rank"],
        optional_fields=_render_optional_fields(extras),
        K=K,
        outcomes_block=_render_outcomes_block(outcomes),
    )
    return [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_reviser_messages(
    c_d: str,
    alt: Mapping[str, Any],
    outcomes: list[str],
    K: int,
    plausibility: int,
    diversity: int,
    notes: str,
    optional_fields: Mapping[str, Any] | None = None,
) -> list[dict]:
    """Build the chat messages that rewrite K outcomes given critic feedback."""
    if not isinstance(K, int) or isinstance(K, bool) or K <= 0:
        raise ValueError(f"K must be a positive int, got {K!r}")
    if len(outcomes) != K:
        raise ValueError(
            f"build_reviser_messages: got {len(outcomes)} outcomes, expected K={K}"
        )
    missing = [field for field in _REQUIRED_ALT_FIELDS if field not in alt]
    if missing:
        raise ValueError(
            "alt is missing required field(s): " + ", ".join(missing)
        )
    extras = {
        k: alt[k] for k in alt.keys() if k not in _REQUIRED_ALT_FIELDS
    }
    if optional_fields:
        extras.update(dict(optional_fields))
    system_content = REVISER_SYSTEM_PROMPT.format(K=K)
    user_content = REVISER_USER_TEMPLATE.format(
        c_d=c_d,
        title=alt["title"],
        category=alt["category"],
        price=alt["price"],
        popularity_rank=alt["popularity_rank"],
        optional_fields=_render_optional_fields(extras),
        K=K,
        outcomes_block=_render_outcomes_block(outcomes),
        plausibility=int(plausibility),
        diversity=int(diversity),
        notes=str(notes),
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_reviser_per_outcome_messages(
    c_d: str,
    alt: Mapping[str, Any],
    outcomes: list[str],
    K: int,
    weak_indices: list[int],
    plausibility: int,
    diversity: int,
    notes: str,
    optional_fields: Mapping[str, Any] | None = None,
) -> list[dict]:
    """Build chat messages for the per-outcome reviser path.

    Asks the LLM to rewrite ONLY the weak positions; strong outcomes
    are kept verbatim by the orchestrator via splice. Returns a prompt
    that requests exactly ``N = len(weak_indices)`` sentences in
    ascending position order.

    The strong outcomes are still rendered in the prompt (with a
    ``*** WEAK — REWRITE ***`` tag on the weak ones) so the reviser
    knows what's already covered and avoids generating paraphrases of
    the strong outcomes' consequence types.
    """
    if not isinstance(K, int) or isinstance(K, bool) or K <= 0:
        raise ValueError(f"K must be a positive int, got {K!r}")
    if len(outcomes) != K:
        raise ValueError(
            f"build_reviser_per_outcome_messages: got {len(outcomes)} "
            f"outcomes, expected K={K}"
        )
    missing = [field for field in _REQUIRED_ALT_FIELDS if field not in alt]
    if missing:
        raise ValueError(
            "alt is missing required field(s): " + ", ".join(missing)
        )

    weak_idx_sorted = sorted({int(i) for i in (weak_indices or [])})
    if not weak_idx_sorted:
        raise ValueError(
            "build_reviser_per_outcome_messages: weak_indices is empty; "
            "use build_reviser_messages for monolithic rewrites instead."
        )
    if any(i < 0 or i >= K for i in weak_idx_sorted):
        raise ValueError(
            f"build_reviser_per_outcome_messages: weak_indices "
            f"{weak_idx_sorted!r} out of range [0, {K})"
        )
    strong_idx_sorted = [i for i in range(K) if i not in set(weak_idx_sorted)]

    extras = {
        k: alt[k] for k in alt.keys() if k not in _REQUIRED_ALT_FIELDS
    }
    if optional_fields:
        extras.update(dict(optional_fields))

    n = len(weak_idx_sorted)
    weak_1based = ", ".join(str(i + 1) for i in weak_idx_sorted)
    strong_1based = (
        ", ".join(str(i + 1) for i in strong_idx_sorted) or "(none)"
    )

    user_content = REVISER_PER_OUTCOME_USER_TEMPLATE.format(
        c_d=c_d,
        title=alt["title"],
        category=alt["category"],
        price=alt["price"],
        popularity_rank=alt["popularity_rank"],
        optional_fields=_render_optional_fields(extras),
        outcomes_block=_render_outcomes_block_marked(outcomes, weak_idx_sorted),
        weak_positions_1based=weak_1based,
        strong_positions_1based=strong_1based,
        plausibility=int(plausibility),
        diversity=int(diversity),
        notes=str(notes),
        N=n,
    )
    return [
        {"role": "system", "content": REVISER_PER_OUTCOME_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


__all__ = [
    "SYSTEM_PROMPT",
    "USER_BLOCK_TEMPLATE",
    "PROMPT_VERSION",
    "PROMPT_VERSION_ANCHORED",
    "ANCHORED_AXES",
    "SYSTEM_PROMPT_ANCHORED",
    "PROMPT_VERSION_MOBILITY_ANCHORED",
    "MOBILITY_ANCHORED_AXES",
    "SYSTEM_PROMPT_MOBILITY_ANCHORED",
    "build_messages_mobility_anchored",
    "build_system_prompt",
    "build_user_block",
    "build_messages",
    "build_messages_anchored",
    "REFINED_PROMPT_VERSION",
    "REVISER_PER_OUTCOME_SYSTEM_PROMPT",
    "REVISER_PER_OUTCOME_USER_TEMPLATE",
    "build_critic_messages",
    "build_reviser_messages",
    "build_reviser_per_outcome_messages",
]
