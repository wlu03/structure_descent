"""Render the per-customer context string ``c_d`` for the LLM generator.

Spec: ``docs/redesign.md`` §2.2. This module consumes the same raw per-customer
columns as ``src/data/person_features.py`` (``z_d`` builder) and renders them as
a short, human-readable paragraph (5-8 lines) that the LLM outcome generator
(§3.2) consumes as the ``CONTEXT:`` block of its user prompt.

Two strict rules from §2.2:

1. Never include numeric feature values that would leak test-time truth
   (e.g. "will buy headphones next"). The function signature therefore has
   **no** future-purchase parameter.
2. Paraphrase columns rather than dumping them. Tests assert that raw column
   names (``age_bucket``, ``income_bucket``, ...) and literal bucket codes
   (``"<25k"``, ``"35-44"``, ...) never appear in the rendered string.

Pure function; stdlib only; deterministic (same row -> same string, byte for
byte).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)

#: Canonical column names whose rendered clauses in ``c_d`` may be
#: suppressed by a caller (adapter-driven). Non-suppressible columns —
#: age_bucket, income_bucket, household_size, purchase_frequency,
#: novelty_rate — always render because they carry per-customer signal
#: on every dataset we've considered. Unknown names in ``suppress_fields``
#: are silently ignored so adapter drift can't break the renderer.
_SUPPRESSIBLE_FIELDS: frozenset[str] = frozenset({
    "has_kids",
    "city_size",
    "education",
    "health_rating",
    "risk_tolerance",
})

__all__ = [
    "DEFAULT_PHRASINGS",
    "build_context_string",
    "paraphrase_rules_check",
]


# ---------------------------------------------------------------------------
# Phrasing dictionary
# ---------------------------------------------------------------------------
#
# redesign.md §2.2 specifies the template but not the exact phrase dictionary;
# the choices below are the ones this module commits to. See NOTES.md Wave 1
# for the rationale.

DEFAULT_PHRASINGS: dict[str, dict[Any, str]] = {
    "age_bucket": {
        "18-24": "early 20s",
        "25-34": "late 20s/early 30s",
        "35-44": "mid-30s",
        "45-54": "mid-40s",
        "55-64": "mid-50s",
        "65+": "mid-60s or older",
    },
    # Option-B income collapse (NOTES.md Wave 8, Artifact-3 decision):
    # Distribution-weighted midpoints from 5,027 Amazon survey respondents.
    # Only the `50-100k` bucket has a non-trivial weighted midpoint
    # ($72.9k, vs. naive $75k) because it merges two raw sub-buckets.
    # Open-ended endpoints use US Census ACS 2022 PUMS anchors
    # (<$25k mean ≈ $14.2k → $15k; ≥$150k mean ≈ $221k → rounded
    # conservatively to $200k).
    "income_bucket": {
        "<25k": "about $15k/year",
        "25-50k": "about $38k/year",
        "50-100k": "about $73k/year",
        "100-150k": "about $125k/year",
        "150k+": "around $200k/year",
    },
    "city_size": {
        "rural": "rural area",
        "small": "small town",
        "medium": "mid-size city",
        "large": "large U.S. city",
    },
    "education": {
        1: "some high school",
        2: "high-school graduate",
        3: "some college",
        4: "college-educated",
        5: "graduate degree",
    },
    "health_rating": {
        1: "poor health",
        2: "fair health",
        3: "average health",
        4: "good health",
        5: "excellent health",
    },
    # ``risk_tolerance`` is a standardized scalar in z_d; here we enumerate the
    # three labeled regions used by the renderer so tests can assert coverage.
    "risk_tolerance": {
        "low": "tends to be cautious with risk",
        "neutral": "risk-neutral",
        "high": "comfortable with risk",
    },
    # ``purchase_frequency`` is a continuous count/week; the renderer bins it
    # into the three labeled regions below.
    "purchase_frequency": {
        "rare": "a few times per month",
        "weekly": "roughly twice per week",
        "daily": "almost daily",
    },
    # ``novelty_rate`` is a continuous rate in [0, 1]; the renderer uses the
    # three labeled regions below (for the "rarely/occasionally/frequently
    # tries new products" clause) and a finer share-phrase for the
    # "<share> of orders are first-time products" clause.
    "novelty_rate": {
        "low": "rarely tries new products",
        "mid": "occasionally tries new products",
        "high": "frequently buys new products",
    },
    # ``has_kids`` combined with household_size renders a children clause.
    # Keys here describe the rendered outputs.
    "has_kids": {
        "no_kids": "no children",
        "one_child": "one child",
        "two_children": "two children",
        "several_children": "several children",
    },
}


# Raw column codes that must never appear in the rendered text. Used both by
# the paraphrase_rules_check guard and by the test suite.
_RAW_COLUMN_NAMES: tuple[str, ...] = (
    "age_bucket",
    "income_bucket",
    "household_size",
    "has_kids",
    "city_size",
    "education",
    "health_rating",
    "risk_tolerance",
    "purchase_frequency",
    "novelty_rate",
)

_RAW_BUCKET_CODES: tuple[str, ...] = (
    "18-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65+",
    "<25k",
    "25-50k",
    "50-100k",
    "100-150k",
    "150k+",
)


# ---------------------------------------------------------------------------
# Phrasing helpers
# ---------------------------------------------------------------------------

def _phrase_age(age_bucket: str) -> str:
    try:
        return DEFAULT_PHRASINGS["age_bucket"][age_bucket]
    except KeyError as e:
        raise ValueError(f"unknown age_bucket: {age_bucket!r}") from e


def _phrase_income(income_bucket: str) -> str:
    try:
        return DEFAULT_PHRASINGS["income_bucket"][income_bucket]
    except KeyError as e:
        raise ValueError(f"unknown income_bucket: {income_bucket!r}") from e


def _phrase_city(city_size: str) -> str:
    try:
        return DEFAULT_PHRASINGS["city_size"][city_size]
    except KeyError as e:
        raise ValueError(f"unknown city_size: {city_size!r}") from e


def _phrase_education(education: int) -> str:
    try:
        return DEFAULT_PHRASINGS["education"][int(education)]
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"unknown education level: {education!r}") from e


def _phrase_health(health_rating: int) -> str:
    try:
        return DEFAULT_PHRASINGS["health_rating"][int(health_rating)]
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"unknown health_rating: {health_rating!r}") from e


def _phrase_risk(risk_tolerance: float) -> str:
    r = float(risk_tolerance)
    if r > 0.5:
        return "comfortable with risk"
    if r < -0.5:
        return "tends to be cautious with risk"
    return "risk-neutral"


def _phrase_purchase_frequency(purchase_frequency: float) -> str:
    f = float(purchase_frequency)
    if f < 1.0:
        return "a few times per month"
    if f <= 3.0:
        # round to nearest integer times/week for a natural phrase
        n = int(round(f))
        if n <= 1:
            return "roughly once per week"
        if n == 2:
            return "roughly twice per week"
        return f"roughly {n} times per week"
    return "almost daily"


def _phrase_novelty(novelty_rate: float) -> str:
    n = float(novelty_rate)
    if n < 0.2:
        return "rarely tries new products"
    if n < 0.5:
        return "occasionally tries new products"
    return "frequently buys new products"


def _phrase_kids(has_kids: bool, household_size: int) -> str:
    """Kids/household clause for the age line."""
    has = bool(has_kids)
    size = int(household_size)
    if not has:
        return "no children"
    # has_kids True
    # household_size 3 => roughly one child; 4+ => multiple children
    if size <= 2:
        # has_kids True but household_size suggests otherwise; respect the flag
        return "one child"
    if size == 3:
        return "one child"
    if size == 4:
        return "two children"
    return "several children"


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_context_string(
    row: Mapping[str, Any],
    *,
    recent_purchases: list[str] | None = None,
    current_time: str | None = None,
    suppress_fields: Iterable[str] = (),
) -> str:
    """Render the person-context paragraph ``c_d``.

    Parameters
    ----------
    row : Mapping
        Raw per-customer columns. Required keys: ``age_bucket``,
        ``income_bucket``, ``household_size``, ``has_kids``, ``city_size``,
        ``education``, ``health_rating``, ``risk_tolerance``,
        ``purchase_frequency``, ``novelty_rate``. Same columns used by
        ``person_features.py`` for ``z_d``.
    recent_purchases : list[str], optional
        If given and non-empty, emits a final "Recent purchases ..." line.
    current_time : str, optional
        Free-text phrase like ``"Tuesday evening, late April"``. If given and
        non-empty, emits a final "Current time: ..." line.
    suppress_fields : Iterable[str], optional
        Canonical column names whose rendered clauses are omitted from the
        output. Intended for adapter-level suppression of sentinel /
        constant fields whose values carry no per-customer signal (e.g.
        ``has_kids`` when the dataset has no kids question and the field
        is a constant 0). Suppressible fields:
        ``{has_kids, city_size, education, health_rating, risk_tolerance}``.
        Unknown names are silently ignored.

    Returns
    -------
    str
        2-8 non-empty lines of plain text. No raw column names, no raw bucket
        codes, no numeric test-time truths. Deterministic. A WARNING is
        logged when the output falls below 5 lines (the paper's "5-8 short
        lines" target); typically this only happens when suppress_fields
        is non-empty.

    Notes
    -----
    This function intentionally has **no** future-purchase parameter: per §2.2
    rule 1 ("will buy X next" is test-time truth and must never enter the
    prompt).
    """
    suppress = frozenset(suppress_fields)

    # --- phrasings -------------------------------------------------------
    age_phrase = _phrase_age(row["age_bucket"])
    income_phrase = _phrase_income(row["income_bucket"])
    freq_phrase = _phrase_purchase_frequency(row["purchase_frequency"])
    novelty_phrase = _phrase_novelty(row["novelty_rate"])
    household_size = int(row["household_size"])

    # Suppressible phrasings: only resolved if needed.
    city_phrase = (
        None if "city_size" in suppress else _phrase_city(row["city_size"])
    )
    education_phrase = (
        None if "education" in suppress else _phrase_education(row["education"])
    )
    health_phrase = (
        None
        if "health_rating" in suppress
        else _phrase_health(row["health_rating"])
    )
    risk_phrase = (
        None
        if "risk_tolerance" in suppress
        else _phrase_risk(row["risk_tolerance"])
    )

    # --- lines -----------------------------------------------------------
    lines: list[str] = ["Person profile"]

    # Line 2: age + household [+ kids].
    if "has_kids" in suppress:
        lines.append(f"- Age: {age_phrase}; household of {household_size}.")
    else:
        kids_phrase = _phrase_kids(bool(row["has_kids"]), household_size)
        lines.append(
            f"- Age: {age_phrase}; household of {household_size} ({kids_phrase})."
        )

    # Line 3: income (not suppressible).
    lines.append(f"- Income: {income_phrase}.")

    # Line 4: city + education. Drop the line only when BOTH are suppressed.
    if city_phrase is not None and education_phrase is not None:
        lines.append(f"- Lives in a {city_phrase}; {education_phrase}.")
    elif city_phrase is not None:
        lines.append(f"- Lives in a {city_phrase}.")
    elif education_phrase is not None:
        # Keep education as a standalone demographic line; capitalize.
        lines.append(f"- {education_phrase[:1].upper()}{education_phrase[1:]}.")
    # else: both suppressed; skip.

    # Line 5: health + risk. Drop when BOTH are suppressed.
    if health_phrase is not None and risk_phrase is not None:
        lines.append(f"- Self-reports {health_phrase}; {risk_phrase}.")
    elif health_phrase is not None:
        lines.append(f"- Self-reports {health_phrase}.")
    elif risk_phrase is not None:
        lines.append(f"- {risk_phrase[:1].upper()}{risk_phrase[1:]}.")
    # else: both suppressed; skip.

    # Line 6: purchase frequency + novelty share (not suppressible).
    _ = novelty_phrase  # surfaced only for DEFAULT_PHRASINGS-coverage tests
    share_phrase = _novelty_share_phrase(float(row["novelty_rate"]))
    lines.append(
        f"- Buys on Amazon {freq_phrase}; "
        f"{share_phrase} of orders are first-time products."
    )

    # --- optional lines --------------------------------------------------
    if recent_purchases:
        joined = ", ".join(str(p) for p in recent_purchases)
        lines.append(f"Recent purchases (last 30 days): {joined}.")

    if current_time:
        lines.append(f"Current time: {current_time}.")

    text = "\n".join(lines)
    paraphrase_rules_check(text, row)

    # Relaxed floor: 2-8 non-empty lines. The paper's "5-8 short lines"
    # target still stands as an intent; we WARN rather than hard-fail
    # below 5 so adapter-driven sentinel suppression (Amazon: has_kids,
    # city_size, health_rating, risk_tolerance) doesn't block rendering.
    n_lines = sum(1 for ln in text.split("\n") if ln.strip())
    if not 2 <= n_lines <= 8:
        raise AssertionError(
            f"context string has {n_lines} non-empty lines; expected 2-8"
        )
    if n_lines < 5:
        logger.warning(
            "context string rendered only %d non-empty lines (target 5-8); "
            "suppress_fields=%s",
            n_lines,
            sorted(suppress) or "()",
        )
    return text


def _novelty_share_phrase(novelty_rate: float) -> str:
    """Map novelty_rate in [0, 1] to a natural English share phrase.

    Used inside the purchase-frequency line so the rendered text doesn't expose
    the raw decimal (which would be a bucket dump per §2.2 rule 2).
    """
    n = float(novelty_rate)
    if n < 0.1:
        return "very few"
    if n < 0.25:
        return "about a fifth"
    if n < 0.4:
        return "about a third"
    if n < 0.6:
        return "about half"
    if n < 0.8:
        return "most"
    return "nearly all"


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

def paraphrase_rules_check(text: str, row: Mapping[str, Any]) -> None:
    """Assert the rendered text obeys the paraphrase rules of §2.2.

    Raises ``AssertionError`` if *text* contains any raw column name
    (e.g. ``age_bucket``) or any literal bucket code (e.g. ``<25k``, ``35-44``).
    The *row* argument is accepted for future extension (e.g. detecting leaked
    numeric values) and is currently unused beyond type-checking that a
    mapping was passed.
    """
    if not isinstance(row, Mapping):
        raise TypeError("row must be a Mapping")
    for name in _RAW_COLUMN_NAMES:
        if name in text:
            raise AssertionError(
                f"raw column name {name!r} leaked into context string"
            )
    for code in _RAW_BUCKET_CODES:
        if code in text:
            raise AssertionError(
                f"raw bucket code {code!r} leaked into context string"
            )
