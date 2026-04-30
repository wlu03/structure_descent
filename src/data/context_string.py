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
    "extract_extra_fields_from_row",
    "compute_customer_aggregates",
    "compute_mobility_aggregates",
    "format_event_time_phrase",
    "paraphrase_rules_check",
]


# ---------------------------------------------------------------------------
# Event-time phrase helper (used by build_choice_sets when the dataset
# wants the per-event time-of-day + weekday rendered into c_d).
# ---------------------------------------------------------------------------

# Six daypart buckets — matches what the LLM understands without
# needing exact hour numerics. Ranges are inclusive on the low end.
_DAYPART_BANDS: tuple[tuple[int, str], ...] = (
    (5,  "early morning"),
    (9,  "morning"),
    (12, "midday"),
    (17, "afternoon"),
    (21, "evening"),
    (24, "late night"),
)
_WEEKDAY_NAMES: tuple[str, ...] = (
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
)
# Month -> "<early|mid|late> <season>" using meteorological seasons
# (winter = Dec-Feb, etc.). Dec is the first month of winter so Dec =
# "early winter", Jan = "mid winter", Feb = "late winter".
_MONTH_TO_SEASON: dict[int, str] = {
    12: "early winter", 1: "mid winter", 2: "late winter",
    3:  "early spring", 4: "mid spring", 5: "late spring",
    6:  "early summer", 7: "mid summer", 8: "late summer",
    9:  "early fall",  10: "mid fall",  11: "late fall",
}


def _daypart_for_hour(hour: int) -> str:
    h = int(hour) % 24
    for cutoff, label in _DAYPART_BANDS:
        if h < cutoff:
            return label
    return "late night"


def format_event_time_phrase(timestamp) -> str:
    """Render an event time as a short natural-English phrase.

    Output looks like ``"Saturday morning, early summer"`` or
    ``"Tuesday evening, late fall (weekend)"``. Used to populate the
    ``current_time`` parameter of :func:`build_context_string` so the
    LLM outcome generator sees the when-they-decided context with
    enough resolution to ground time-conditional outcomes (e.g.
    "I'll get caught in afternoon rain" in spring vs. winter). Pure:
    no I/O, no globals.

    Returns an empty string when *timestamp* is missing or unparseable
    so the caller can pass the result straight through to
    :func:`build_context_string` (which treats empty / falsy
    ``current_time`` as "skip the line").
    """
    if timestamp is None:
        return ""
    try:
        # Lazy-import pandas only when we actually need to coerce.
        import pandas as _pd  # noqa: PLC0415

        ts = _pd.Timestamp(timestamp)
    except Exception:  # noqa: BLE001 — return empty rather than raising
        return ""
    if _pd_is_nat(ts):
        return ""
    hour = int(ts.hour)
    weekday_idx = int(ts.weekday())
    weekday = _WEEKDAY_NAMES[weekday_idx]
    daypart = _daypart_for_hour(hour)
    season = _MONTH_TO_SEASON.get(int(ts.month), "")
    is_weekend = weekday_idx >= 5
    suffix = " (weekend)" if is_weekend else ""
    if season:
        return f"{weekday} {daypart}, {season}{suffix}"
    return f"{weekday} {daypart}{suffix}"


def _pd_is_nat(ts) -> bool:
    """True for pandas NaT / NumPy NaT timestamps; lazy import."""
    try:
        import pandas as _pd  # noqa: PLC0415
        return bool(_pd.isna(ts))
    except Exception:
        return False


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
    # ``purchase_frequency`` is a continuous events-per-week rate.
    # The driver (scripts/run_dataset.py) normalizes the raw
    # derived_from_events count by the train-window duration in weeks
    # BEFORE feeding persons_canonical to build_choice_sets, so the
    # renderer sees a rate, not a total count.
    #
    # Bucket thresholds (events/week):
    #   <0.1  -> "very_rare"  (≲ once every 2-3 months)
    #   <1    -> "rare"       ("a few times per month")
    #   1-3   -> "weekly"     ("roughly N times per week")
    #   >3    -> "daily"      ("almost daily")
    "purchase_frequency": {
        "very_rare": "rarely",
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
    # Wave 11 c_d signal enrichment: gender / life event / Amazon-use-frequency
    # clauses surfaced through the ``extra_fields`` kwarg of
    # ``build_context_string``. These are NOT z_d columns (the p=23 contract
    # stays fixed); they ride alongside c_d only.
    "gender": {
        "Female": "female",
        "Male": "male",
    },
    # Raw Amazon ``Q-amazon-use-how-oft`` values → natural-English paraphrases.
    # The raw strings must NOT appear verbatim in the rendered c_d (they
    # contain "5" / "10" substrings from the survey scale); see
    # test_extra_amazon_frequency_paraphrased.
    "amazon_frequency": {
        "Less than 5 times per month": "a few times a month",
        "5 - 10 times per month": "several times a month",
        "More than 10 times per month": "more than ten times a month",
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
    """Render a natural-English phrase for events-per-week rate.

    Input is an events-per-week rate; the driver normalizes the raw
    derived_from_events count by the train-window duration before this
    function sees it. Bucket boundaries: <0.1 → "rarely shops",
    <1 → "a few times per month", 1-3 → "roughly N times per week",
    >3 → "almost daily".

    Prior bug (fixed Wave 11): this function was being called with the
    raw total event count instead of a rate, so customers with low
    absolute counts (e.g. 9 events over 5 years) rendered as
    "almost daily". The driver now divides by window_weeks before
    calling.
    """
    f = float(purchase_frequency)
    if f < 0.1:
        return DEFAULT_PHRASINGS["purchase_frequency"]["very_rare"]
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
    extra_fields: Mapping[str, Any] | None = None,
    event_origin: str | None = None,
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
    extra_fields : Mapping[str, Any], optional
        Wave-11 c_d signal-enrichment kwarg. Optional narrative signals
        that do NOT belong to z_d (which is pinned by the §2.1 / Wave-8
        p=23 contract) but are useful context for the LLM generator.
        Recognized keys:

        * ``gender`` — ``"Female"`` / ``"Male"`` emits an inline
          "as a female" / "as a male" phrase in the age/household line.
          Any other value (or absent key / ``None``) omits the clause.
        * ``life_event`` — non-empty string emits an additional line
          ``"- Recent life event: <event>."`` right after the income
          line. ``None``, empty string, and pandas NaN sentinel are
          all treated as "no clause".
        * ``amazon_frequency`` — raw Amazon ``Q-amazon-use-how-oft``
          value. Rendered as a paraphrased "- Shops on Amazon <freq>."
          line near the purchase_frequency line. Unknown values are
          dropped silently (never passed through verbatim).

        When ``None`` (the default), this function produces
        byte-identical output to the pre-fix behavior.

    Returns
    -------
    str
        2-12 non-empty lines of plain text (extras + optional lines can
        push the count above the paper's 5-8 target). No raw column
        names, no raw bucket codes, no numeric test-time truths.
        Deterministic. A WARNING is logged when the output falls below
        5 lines (the paper's "5-8 short lines" target); typically this
        only happens when suppress_fields is non-empty.

    Notes
    -----
    This function intentionally has **no** future-purchase parameter: per §2.2
    rule 1 ("will buy X next" is test-time truth and must never enter the
    prompt).
    """
    suppress = frozenset(suppress_fields)
    extras = _normalize_extra_fields(extra_fields)

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

    # Line 2: age + household [+ kids] [+ gender extra]. Gender renders as
    # an inline "as a <gender>" phrase appended to the subject description.
    gender_suffix = ""
    if extras.get("gender"):
        gender_suffix = f", as a {extras['gender']}"
    if "has_kids" in suppress:
        lines.append(
            f"- Age: {age_phrase}; household of {household_size}"
            f"{gender_suffix}."
        )
    else:
        kids_phrase = _phrase_kids(bool(row["has_kids"]), household_size)
        lines.append(
            f"- Age: {age_phrase}; household of {household_size} "
            f"({kids_phrase}){gender_suffix}."
        )

    # Line 3: income (not suppressible).
    lines.append(f"- Income: {income_phrase}.")

    # Wave-11 extra: Recent life event line, right after income.
    if extras.get("life_event"):
        lines.append(f"- Recent life event: {extras['life_event']}.")

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
    # Domain-overridable verb / noun / object phrasing — defaults are
    # Amazon-tuned and reproduce the prior literal output byte-for-byte.
    # A non-Amazon dataset can override via extra_fields:
    #   domain_verb       e.g. "Travels around Boston"   (default "Buys on Amazon")
    #   activity_noun     e.g. "trips"                   (default "orders")
    #   novelty_object    e.g. "places"                  (default "products")
    #   self_report_verb  e.g. "Visits places"           (default "Shops on Amazon")
    _ = novelty_phrase  # surfaced only for DEFAULT_PHRASINGS-coverage tests
    share_phrase = _novelty_share_phrase(float(row["novelty_rate"]))
    domain_verb = extras.get("domain_verb") or "Buys on Amazon"
    activity_noun = extras.get("activity_noun") or "orders"
    novelty_object = extras.get("novelty_object") or "products"
    lines.append(
        f"- {domain_verb} {freq_phrase}; "
        f"{share_phrase} of {activity_noun} are first-time {novelty_object}."
    )

    # Wave-11 extra: self-reported domain cadence (complements the
    # event-derived purchase_frequency line above; they measure different
    # signals and agreement / disagreement is itself informative).
    if extras.get("amazon_frequency"):
        self_report_verb = extras.get("self_report_verb") or "Shops on Amazon"
        lines.append(f"- {self_report_verb} {extras['amazon_frequency']}.")

    # Customer-aggregate enrichment (opt-in via YAML
    # ``data.enrich_customer_context``; ignored when fields are absent).
    # The encoder cannot otherwise see brand affinity / category density
    # / typical price tier — these clauses give it that signal.
    if extras.get("top_brand"):
        lines.append(
            f"- Most-purchased brand in their history: {extras['top_brand']}."
        )
    if extras.get("top_categories"):
        cats = extras["top_categories"]
        if isinstance(cats, (list, tuple)) and cats:
            joined = ", ".join(str(c) for c in cats[:3])
            lines.append(f"- Top categories they shop: {joined}.")
    if extras.get("avg_price") is not None:
        try:
            ap = float(extras["avg_price"])
            if ap > 0.0:
                lines.append(
                    f"- Typical purchase price: about ${ap:.0f}."
                )
        except (TypeError, ValueError):
            pass
    if extras.get("repeat_rate") is not None:
        try:
            rr = float(extras["repeat_rate"])
            if 0.0 <= rr <= 1.0:
                lines.append(
                    f"- About {int(round(100 * rr))}% of their purchases "
                    f"are repeats of items they have bought before."
                )
        except (TypeError, ValueError):
            pass

    # Mobility profile aggregates (computed by
    # :func:`compute_mobility_aggregates`). Adapter-agnostic — Amazon
    # callers don't populate these so the lines silently drop.
    if extras.get("typical_distance_km") is not None:
        try:
            d_km = float(extras["typical_distance_km"])
            if d_km > 0.0:
                lines.append(
                    f"- Typical trip length: about {d_km:.1f} km."
                )
        except (TypeError, ValueError):
            pass
    if extras.get("weekend_share") is not None:
        try:
            ws = float(extras["weekend_share"])
            if 0.0 <= ws <= 1.0:
                if ws >= 0.6:
                    lines.append("- Mostly travels on weekends.")
                elif ws <= 0.2:
                    lines.append("- Mostly travels on weekdays.")
                # else: roughly even split — skip the clause.
        except (TypeError, ValueError):
            pass
    if extras.get("daypart_preference"):
        lines.append(
            f"- Most active during the {extras['daypart_preference']}."
        )

    # --- optional lines --------------------------------------------------
    if event_origin:
        # Per-event origin context — for mobility this renders where the
        # agent is coming from ("home" / "their workplace" / "a Food and
        # Accommodation place"). Pure narrative; no leak (computed from
        # from_place_id of the current event, which is the previous
        # event's destination — strict prefix info).
        lines.append(f"- Just came from {event_origin}.")

    if recent_purchases:
        joined = ", ".join(str(p) for p in recent_purchases)
        lines.append(f"Recent purchases (last 30 days): {joined}.")

    if current_time:
        lines.append(f"Current time: {current_time}.")

    text = "\n".join(lines)
    paraphrase_rules_check(text, row)

    # Relaxed floor: 2-16 non-empty lines. The paper's "5-8 short lines"
    # target still stands as an intent; we WARN rather than hard-fail
    # below 5 so adapter-driven sentinel suppression (Amazon: has_kids,
    # city_size, health_rating, risk_tolerance) doesn't block rendering.
    # The ``extra_fields`` kwarg can push the count upward — gender rides
    # inline so adds 0 lines, life_event +1, amazon_frequency +1, the
    # customer-aggregate enrichment (top_brand, top_categories,
    # avg_price, repeat_rate) +4, and the two optional lines (recent /
    # current_time) +2 — hence the relaxed ceiling of 16.
    n_lines = sum(1 for ln in text.split("\n") if ln.strip())
    if not 2 <= n_lines <= 16:
        raise AssertionError(
            f"context string has {n_lines} non-empty lines; expected 2-16"
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
# Extra-field normalization (Wave 11 c_d signal enrichment)
# ---------------------------------------------------------------------------


def _is_missing(value: Any) -> bool:
    """True if *value* should be treated as "no clause": None, empty string,
    or the pandas/NumPy NaN sentinel (float('nan')).
    """
    if value is None:
        return True
    # pandas NaN is a float that != itself; this catches NumPy NaN too.
    try:
        if isinstance(value, float) and value != value:  # noqa: PLR0124
            return True
    except Exception:
        pass
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "nan":
            return True
    return False


def _normalize_extra_fields(
    extra_fields: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a dict of already-paraphrased extra-field phrases.

    Missing / unrecognized entries are silently dropped (so callers can
    pass raw survey values without a pre-flight guard). Recognized keys:

    * ``gender`` — lowercased "female" / "male"; anything else is dropped
      (so "Other" / "Prefer not to say" produce no clause).
    * ``life_event`` — passed through as a trimmed string when non-empty,
      else dropped. pandas NaN is detected via _is_missing.
    * ``amazon_frequency`` — mapped via
      ``DEFAULT_PHRASINGS["amazon_frequency"]`` when the raw value matches
      a known survey option; otherwise dropped (never passed through, to
      avoid leaking "More than 10 times per month" verbatim).
    """
    out: dict[str, Any] = {}
    if extra_fields is None:
        return out

    raw_gender = extra_fields.get("gender")
    if not _is_missing(raw_gender):
        # Accept both raw-survey ("Female"/"Male") and already-paraphrased
        # ("female"/"male") inputs — the helper upstream may have mapped
        # through the YAML already.
        gender_map = DEFAULT_PHRASINGS["gender"]
        phrased = gender_map.get(raw_gender)
        if phrased is None and isinstance(raw_gender, str):
            for key, val in gender_map.items():
                if key.lower() == raw_gender.lower():
                    phrased = val
                    break
        if phrased is not None:
            out["gender"] = phrased
        # else: "Other" / "Prefer not to say" / unknown → drop silently.

    raw_life = extra_fields.get("life_event")
    if not _is_missing(raw_life):
        text = str(raw_life).strip()
        if text:
            out["life_event"] = text

    raw_freq = extra_fields.get("amazon_frequency")
    if not _is_missing(raw_freq):
        phrased = DEFAULT_PHRASINGS["amazon_frequency"].get(raw_freq)
        if phrased is not None:
            out["amazon_frequency"] = phrased
        # else: unknown freq → drop silently (never leak raw "10 times"
        # through; paraphrase_rules_check wouldn't block it, but the
        # Wave-11 test asserts the raw form is absent).

    # Customer-aggregate enrichment keys (opt-in via
    # ``data.enrich_customer_context`` in run_dataset.py). These are
    # already-paraphrased / numeric values; pass through unchanged when
    # present and non-missing. The renderer guards against bad numerics
    # itself (e.g. negative avg_price → skipped clause), so we don't
    # need to re-validate here.
    for key in (
        "top_brand", "top_categories", "avg_price", "repeat_rate",
        "typical_distance_km", "weekend_share", "daypart_preference",
    ):
        val = extra_fields.get(key)
        if not _is_missing(val):
            out[key] = val

    # Domain-override knobs — strings substituted directly into the
    # purchase_frequency/novelty line. Pass through unchanged when present.
    for key in (
        "domain_verb",
        "activity_noun",
        "novelty_object",
        "self_report_verb",
    ):
        val = extra_fields.get(key)
        if not _is_missing(val):
            out[key] = str(val)

    return out


def extract_extra_fields_from_row(
    row: Mapping[str, Any],
    yaml_block: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Translate a raw persons-row dict into the canonical ``extra_fields``
    dict that :func:`build_context_string` consumes.

    Parameters
    ----------
    row :
        One raw persons-row mapping (e.g. a ``pandas.Series`` or dict of
        the original survey columns for a single customer).
    yaml_block :
        The parsed ``persons.c_d_extra_fields`` YAML block, as a free-form
        Python dict. Keys are canonical extra-field names (``gender``,
        ``life_event``, ``amazon_frequency``); values are per-field spec
        dicts with at least a ``source`` (raw column name) and a ``kind``
        (one of ``passthrough``, ``categorical_map`` — matching the
        schema_map vocabulary).

        ``None`` or empty → returns an empty dict.

    Returns
    -------
    dict
        Canonical extra-fields dict ready to pass as
        ``build_context_string(..., extra_fields=...)``. Missing /
        unknown raw values are dropped (they simply don't appear in the
        returned dict), matching the spec's "drop the clause entirely"
        rule for ambiguous or NaN signals.

    Notes
    -----
    This helper lives alongside the renderer rather than in
    ``schema_map.py`` because extra_fields are c_d-only — they never
    touch z_d and therefore aren't part of the frozen
    :class:`DatasetSchema` contract. Driver code (e.g.
    ``scripts/run_dataset.py`` or ``src/data/choice_sets.py``) calls
    this helper after :func:`schema_map.translate_persons` to plumb the
    extra signals through to ``build_context_string``.
    """
    if not yaml_block:
        return {}

    out: dict[str, Any] = {}
    for canonical, spec in yaml_block.items():
        if not isinstance(spec, Mapping):
            continue
        kind = spec.get("kind", "passthrough")
        # ``kind: constant`` carries no source — it's a YAML-supplied
        # broadcast value (e.g. dataset-level domain_verb / activity_noun
        # for non-Amazon adapters). Surface it before we look at the row.
        if kind == "constant":
            val = spec.get("value")
            if val is not None and not _is_missing(val):
                out[canonical] = val
            continue
        source = spec.get("source")
        if source is None or source not in row:
            continue
        raw_value = row[source]
        if _is_missing(raw_value):
            continue

        if kind == "passthrough":
            out[canonical] = raw_value
        elif kind == "categorical_map":
            values = spec.get("values") or {}
            drop = tuple(spec.get("drop_on_unknown") or ())
            if raw_value in drop:
                continue
            if raw_value in values:
                out[canonical] = values[raw_value]
            # else: unknown value; drop silently.
        else:
            # Unknown kind: be conservative and drop. (This helper
            # intentionally supports only the two kinds used by the
            # Wave-11 Amazon YAML; extending it is a small follow-up.)
            continue
    return out


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

def compute_customer_aggregates(events_df, *, train_only: bool = True) -> dict:
    """Compute per-customer history aggregates for c_d enrichment.

    For each customer, returns a dict with the keys consumed by the
    ``extra_fields`` hook of :func:`build_context_string`:

    * ``top_brand``       — most-frequent brand string (excluding empty)
    * ``top_categories``  — list[str] of top-3 categories by purchase count
    * ``avg_price``       — float, mean purchase price
    * ``repeat_rate``     — float in [0, 1], fraction of events with
                            ``routine > 0`` (a repeat purchase)

    ``train_only=True`` filters by ``events_df["split"] == "train"`` when
    that column is present, preventing val/test leakage when the
    aggregates are used to render c_d for held-out customer-events. When
    the column is absent (synthetic / pre-split inputs), the full frame
    is used and a debug log is emitted.

    Returns
    -------
    dict[str, dict[str, Any]]
        Maps ``customer_id`` -> aggregate dict. Customers absent from
        ``events_df`` (after the split filter) do not appear in the
        output. Callers should fall back gracefully on missing keys.
    """
    import pandas as pd  # local import keeps this module light-weight

    df = events_df
    if train_only and "split" in df.columns:
        df = df[df["split"] == "train"]
    if df.empty:
        return {}

    out: dict[str, dict] = {}

    has_brand = "brand" in df.columns
    has_price = "price" in df.columns
    has_routine = "routine" in df.columns

    for cid, group in df.groupby("customer_id", sort=False):
        agg: dict = {}
        if has_brand:
            brands = group["brand"].astype(str)
            brands = brands[
                brands.notna() & (brands != "") & (brands != "nan")
            ]
            if not brands.empty:
                agg["top_brand"] = brands.mode().iloc[0]
        if "category" in group.columns:
            cat_counts = group["category"].astype(str).value_counts()
            agg["top_categories"] = cat_counts.head(3).index.tolist()
        if has_price:
            prices = pd.to_numeric(group["price"], errors="coerce")
            prices = prices[prices > 0]
            if not prices.empty:
                agg["avg_price"] = float(prices.mean())
        if has_routine:
            agg["repeat_rate"] = float((group["routine"] > 0).mean())
        out[cid] = agg
    return out


def compute_mobility_aggregates(
    events_df, *, train_only: bool = True
) -> dict:
    """Per-customer mobility summary stats for c_d enrichment.

    Returns ``dict[customer_id, agg]`` with keys:

    * ``typical_distance_km`` (float) — median trip distance over the
      customer's events. Uses the ``price`` column (= haversine
      geodistance under the mobility adapter).
    * ``weekend_share`` (float in [0, 1]) — fraction of events on
      Saturday/Sunday.
    * ``daypart_preference`` (str) — the most-frequent daypart label
      across the customer's events ("late night", "early morning",
      "morning", "midday", "afternoon", "evening").

    ``train_only=True`` filters by ``events_df["split"] == "train"`` when
    present, preventing val/test leakage when these aggregates are
    rendered into c_d for held-out customer-events. Both renderings
    apply the SAME train-fit per customer, so the c_d the model sees at
    val/test time is consistent with what it saw at train time.

    Customers absent from ``events_df`` (after the optional split
    filter) do not appear in the output. Callers should fall back
    gracefully on missing keys.
    """
    import pandas as _pd  # noqa: PLC0415 — keep module light-weight

    df = events_df
    if train_only and "split" in df.columns:
        df = df[df["split"] == "train"]
    if df.empty:
        return {}

    out: dict[object, dict] = {}
    has_price = "price" in df.columns
    has_date = "order_date" in df.columns

    daypart_bins = [-1, 4, 8, 11, 16, 20, 23]
    daypart_labels = [
        "late night",
        "early morning",
        "morning",
        "midday",
        "afternoon",
        "evening",
    ]

    for cid, group in df.groupby("customer_id", sort=False):
        agg: dict = {}
        if has_price:
            prices = _pd.to_numeric(group["price"], errors="coerce")
            valid = prices[prices > 0]
            if not valid.empty:
                agg["typical_distance_km"] = float(valid.median())
        if has_date:
            dates = _pd.to_datetime(group["order_date"], errors="coerce")
            dates = dates.dropna()
            if not dates.empty:
                agg["weekend_share"] = float((dates.dt.weekday >= 5).mean())
                hours = dates.dt.hour
                dayparts = _pd.cut(
                    hours,
                    bins=daypart_bins,
                    labels=daypart_labels,
                    include_lowest=True,
                )
                counts = dayparts.value_counts()
                if not counts.empty:
                    agg["daypart_preference"] = str(counts.index[0])
        out[cid] = agg
    return out


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
