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

PROMPT_VERSION: str = "v1"

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

    return USER_BLOCK_TEMPLATE.format(
        c_d=c_d,
        title=alt["title"],
        category=alt["category"],
        price=alt["price"],
        popularity_rank=alt["popularity_rank"],
        optional_fields=_render_optional_fields(optional_fields),
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


__all__ = [
    "SYSTEM_PROMPT",
    "USER_BLOCK_TEMPLATE",
    "PROMPT_VERSION",
    "build_system_prompt",
    "build_user_block",
    "build_messages",
]
