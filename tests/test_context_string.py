"""Tests for ``src/data/context_string.py`` (redesign.md §2.2)."""

from __future__ import annotations

import inspect

import pytest

from src.data.context_string import (
    DEFAULT_PHRASINGS,
    build_context_string,
    extract_extra_fields_from_row,
    paraphrase_rules_check,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _base_row(**overrides):
    """A legal synthetic row covering every raw column c_d consumes."""
    row = {
        "age_bucket": "35-44",
        "income_bucket": "50-100k",
        "household_size": 4,
        "has_kids": True,
        "city_size": "large",
        "education": 4,
        "health_rating": 4,
        "risk_tolerance": -0.8,   # cautious
        "purchase_frequency": 2.0,  # roughly twice per week
        "novelty_rate": 0.33,       # about a third
    }
    row.update(overrides)
    return row


def _nonempty_lines(s: str) -> list[str]:
    return [ln for ln in s.split("\n") if ln.strip()]


# ---------------------------------------------------------------------------
# 1. Line-count invariant
# ---------------------------------------------------------------------------

def test_line_count_mandatory_only():
    row = _base_row()
    out = build_context_string(row)
    n = len(_nonempty_lines(out))
    assert 5 <= n <= 8, f"mandatory-only render produced {n} lines:\n{out}"


def test_line_count_with_recent_and_time():
    row = _base_row()
    out = build_context_string(
        row,
        recent_purchases=["kids' backpacks", "a coffee maker", "batteries"],
        current_time="Tuesday evening, late April",
    )
    n = len(_nonempty_lines(out))
    assert 5 <= n <= 8, f"full render produced {n} lines:\n{out}"


# ---------------------------------------------------------------------------
# 2. No raw column names / bucket codes leak into the rendered text
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "leak",
    [
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
        "<25k",
        "25-50k",
        "50-100k",
        "100-150k",
        "150k+",
        "18-24",
        "25-34",
        "35-44",
        "45-54",
        "55-64",
        "65+",
    ],
)
def test_no_raw_column_names_or_codes(leak):
    row = _base_row()
    out = build_context_string(row)
    assert leak not in out, f"leaked token {leak!r} in:\n{out}"


# ---------------------------------------------------------------------------
# 3. has_kids phrasing
# ---------------------------------------------------------------------------

def test_has_kids_phrasing_family_of_four():
    row = _base_row(has_kids=True, household_size=4)
    out = build_context_string(row)
    assert "two children" in out


def test_no_kids_no_child_mention():
    row = _base_row(has_kids=False, household_size=2)
    out = build_context_string(row)
    # "no children" is the only child-adjacent phrase allowed
    assert "children" not in out or "no children" in out
    assert "one child" not in out
    assert "two children" not in out
    assert "several children" not in out


def test_has_kids_phrasing_three_person_household():
    row = _base_row(has_kids=True, household_size=3)
    out = build_context_string(row)
    assert "one child" in out


# ---------------------------------------------------------------------------
# 4. Every bucket in DEFAULT_PHRASINGS renders without error
# ---------------------------------------------------------------------------

def test_all_phrase_buckets_render():
    """For every categorical bucket in DEFAULT_PHRASINGS, each legal category
    must produce a valid 5-8 line c_d when plugged into the otherwise-default
    row. Scalar buckets (risk_tolerance, purchase_frequency, novelty_rate,
    has_kids) are covered by the targeted tests below.
    """
    # age_bucket
    for code in DEFAULT_PHRASINGS["age_bucket"]:
        out = build_context_string(_base_row(age_bucket=code))
        assert 5 <= len(_nonempty_lines(out)) <= 8

    # income_bucket
    for code in DEFAULT_PHRASINGS["income_bucket"]:
        out = build_context_string(_base_row(income_bucket=code))
        assert 5 <= len(_nonempty_lines(out)) <= 8

    # city_size
    for code in DEFAULT_PHRASINGS["city_size"]:
        out = build_context_string(_base_row(city_size=code))
        assert 5 <= len(_nonempty_lines(out)) <= 8

    # education (ordinal 1..5)
    for level in DEFAULT_PHRASINGS["education"]:
        out = build_context_string(_base_row(education=level))
        assert 5 <= len(_nonempty_lines(out)) <= 8

    # health_rating (ordinal 1..5)
    for level in DEFAULT_PHRASINGS["health_rating"]:
        out = build_context_string(_base_row(health_rating=level))
        assert 5 <= len(_nonempty_lines(out)) <= 8


def test_scalar_buckets_cover_every_region():
    # risk_tolerance: low / neutral / high regions.
    for r, expected_any in [
        (-1.0, DEFAULT_PHRASINGS["risk_tolerance"]["low"]),
        (0.0, DEFAULT_PHRASINGS["risk_tolerance"]["neutral"]),
        (1.0, DEFAULT_PHRASINGS["risk_tolerance"]["high"]),
    ]:
        out = build_context_string(_base_row(risk_tolerance=r))
        assert expected_any in out

    # purchase_frequency: rare / weekly / daily.
    for f, expected in [
        (0.2, "a few times per month"),
        (2.0, "roughly twice per week"),
        (10.0, "almost daily"),
    ]:
        out = build_context_string(_base_row(purchase_frequency=f))
        assert expected in out

    # novelty_rate: low / mid / high.
    # (The rendered line uses the share-phrase, not the tries-new phrase; the
    # DEFAULT_PHRASINGS coverage here is that every region produces a valid
    # render.)
    for n in [0.0, 0.3, 0.7]:
        out = build_context_string(_base_row(novelty_rate=n))
        assert 5 <= len(_nonempty_lines(out)) <= 8

    # has_kids: all four rendered phrases are reachable.
    assert "no children" in build_context_string(
        _base_row(has_kids=False, household_size=1)
    )
    assert "one child" in build_context_string(
        _base_row(has_kids=True, household_size=3)
    )
    assert "two children" in build_context_string(
        _base_row(has_kids=True, household_size=4)
    )
    assert "several children" in build_context_string(
        _base_row(has_kids=True, household_size=6)
    )


# ---------------------------------------------------------------------------
# 5. No test-time leakage: signature contract
# ---------------------------------------------------------------------------

def test_no_test_time_leakage():
    """The function signature must have no future-purchase parameter (§2.2
    rule 1). The docstring must document this contract.
    """
    sig = inspect.signature(build_context_string)
    param_names = set(sig.parameters.keys())

    # None of these may appear as parameters:
    forbidden = {
        "next_purchase",
        "future_purchase",
        "will_buy",
        "target",
        "chosen",
        "label",
        "y",
    }
    leaked = forbidden & param_names
    assert not leaked, f"function signature leaks test-time truth: {leaked}"

    # Docstring contract: must mention the no-leak rule.
    doc = (build_context_string.__doc__ or "").lower()
    assert "no" in doc and "future-purchase" in doc, (
        "build_context_string docstring must document the §2.2 rule that the "
        "function has no future-purchase parameter."
    )


# ---------------------------------------------------------------------------
# 6. Deterministic
# ---------------------------------------------------------------------------

def test_deterministic():
    row = _base_row()
    a = build_context_string(
        row,
        recent_purchases=["apples", "bread"],
        current_time="Friday morning, mid-May",
    )
    b = build_context_string(
        row,
        recent_purchases=["apples", "bread"],
        current_time="Friday morning, mid-May",
    )
    assert a == b


# ---------------------------------------------------------------------------
# 7. paraphrase_rules_check rejects raw bucket codes
# ---------------------------------------------------------------------------

def test_paraphrase_rules_check_rejects_raw_bucket_code():
    row = _base_row()
    bad = "Person profile\n- Age: <25k.\n- etc.\n- etc.\n- etc.\n"
    with pytest.raises(AssertionError):
        paraphrase_rules_check(bad, row)


def test_paraphrase_rules_check_rejects_raw_column_name():
    row = _base_row()
    bad = "Person profile\n- age_bucket=3.\n- etc.\n- etc.\n- etc.\n"
    with pytest.raises(AssertionError):
        paraphrase_rules_check(bad, row)


def test_paraphrase_rules_check_passes_clean_output():
    row = _base_row()
    good = build_context_string(row)
    # Must not raise.
    paraphrase_rules_check(good, row)


# ---------------------------------------------------------------------------
# 8. Optional lines
# ---------------------------------------------------------------------------

def test_optional_lines_present_when_supplied():
    row = _base_row()
    out = build_context_string(
        row,
        recent_purchases=["bananas", "notebook"],
        current_time="Monday afternoon, early March",
    )
    assert "Recent purchases" in out
    assert "Monday afternoon, early March" in out
    n = len(_nonempty_lines(out))
    assert 5 <= n <= 8


def test_optional_lines_omitted_when_absent():
    row = _base_row()
    out = build_context_string(row)
    assert "Recent purchases" not in out
    assert "Current time" not in out
    n = len(_nonempty_lines(out))
    assert 5 <= n <= 8


def test_optional_lines_partial():
    """Only recent_purchases supplied; only current_time supplied."""
    row = _base_row()

    a = build_context_string(row, recent_purchases=["apples"])
    assert "Recent purchases" in a
    assert "Current time" not in a
    assert 5 <= len(_nonempty_lines(a)) <= 8

    b = build_context_string(row, current_time="Sunday")
    assert "Current time" in b
    assert "Recent purchases" not in b
    assert 5 <= len(_nonempty_lines(b)) <= 8


# ---------------------------------------------------------------------------
# Wave 9: suppress_fields — adapter-driven sentinel suppression.
# ---------------------------------------------------------------------------

def test_suppress_has_kids_removes_kids_clause():
    """Suppressing ``has_kids`` drops the kids parenthetical from line 2."""
    row = _base_row(has_kids=True, household_size=4)
    with_kids = build_context_string(row)
    without = build_context_string(row, suppress_fields=["has_kids"])
    # The kids phrasing is present in the default render...
    assert "two children" in with_kids
    # ...and gone in the suppressed render.
    assert "two children" not in without
    # Line 2 should still exist and mention the household size.
    assert "household of 4" in without


def test_suppress_city_size_removes_city_clause():
    """Suppressing ``city_size`` drops the "Lives in a …" clause.

    Education still renders — the line collapses to a standalone
    education clause rather than disappearing.
    """
    row = _base_row(city_size="large", education=4)
    full = build_context_string(row)
    sup = build_context_string(row, suppress_fields=["city_size"])
    assert "large U.S. city" in full
    assert "large U.S. city" not in sup
    assert "Lives in" not in sup
    # The surviving education clause should still be in the output.
    assert "college-educated" in sup.lower() or "College-educated" in sup


def test_suppress_health_and_risk_drops_entire_line():
    """Suppressing both ``health_rating`` and ``risk_tolerance`` drops line 5."""
    row = _base_row()
    full = build_context_string(row)
    sup = build_context_string(
        row, suppress_fields=["health_rating", "risk_tolerance"]
    )
    # Line 5 ("Self-reports … ; …") should be present in full, absent when
    # both clauses are suppressed.
    assert "Self-reports" in full
    assert "Self-reports" not in sup


def test_suppress_only_health_keeps_risk_standalone():
    """Suppressing only health still renders the risk clause alone."""
    row = _base_row(risk_tolerance=-1.0)  # cautious
    sup = build_context_string(row, suppress_fields=["health_rating"])
    # "Self-reports" prefix should be gone; risk phrase remains.
    assert "Self-reports" not in sup
    assert "cautious" in sup.lower()


def test_suppress_only_risk_keeps_health_standalone():
    row = _base_row(health_rating=5)
    sup = build_context_string(row, suppress_fields=["risk_tolerance"])
    assert "Self-reports excellent health" in sup
    # No dangling separator.
    assert ";" not in sup.split("Self-reports", 1)[1].splitlines()[0]


def test_suppress_all_four_amazon_sentinels_produces_minimal_c_d(caplog):
    """Amazon's has_kids + city_size + health_rating + risk_tolerance
    suppression yields a minimal but coherent c_d with at least 2 lines.

    Logs a WARNING because the result falls below the paper's 5-line
    target.
    """
    import logging
    row = _base_row()
    caplog.set_level(logging.WARNING, logger="src.data.context_string")
    sup = build_context_string(
        row,
        suppress_fields=[
            "has_kids", "city_size", "health_rating", "risk_tolerance",
        ],
    )
    lines = _nonempty_lines(sup)
    # Must have: profile header, age/household, income, education
    # (standalone), purchase/novelty. That's 5 lines — still at the
    # paper's floor. A WARNING might still fire if rendering dropped
    # to 4; either way must be >= 2 and <= 8.
    assert 2 <= len(lines) <= 8
    # Amazon's sentinels must not leak any of the suppressed content.
    assert "children" not in sup
    assert "city" not in sup.lower()
    assert "Self-reports" not in sup
    assert "risk" not in sup.lower()


def test_suppress_paraphrase_rules_still_pass():
    """Suppression must not break paraphrase_rules_check."""
    row = _base_row()
    sup = build_context_string(
        row,
        suppress_fields=[
            "has_kids", "city_size", "health_rating", "risk_tolerance",
        ],
    )
    # paraphrase_rules_check runs inside build_context_string. If it
    # failed we'd have seen an AssertionError. Re-run explicitly to
    # ensure idempotence.
    paraphrase_rules_check(sup, row)


def test_suppress_unknown_field_silently_ignored():
    """Unknown names in suppress_fields do not raise."""
    row = _base_row()
    out = build_context_string(row, suppress_fields=["not_a_real_field"])
    # Should render identically to the default call.
    default = build_context_string(row)
    assert out == default


def test_suppress_below_five_lines_logs_warning(caplog):
    """Force a <5-line render via suppression; assert WARNING is logged."""
    import logging
    row = _base_row()
    caplog.set_level(logging.WARNING, logger="src.data.context_string")
    # Suppress enough to force 4 lines. Suppressing both city and
    # education drops line 4 entirely; suppressing both health and risk
    # drops line 5 entirely. That leaves: profile, age/household,
    # income, purchase/novelty = 4 lines.
    out = build_context_string(
        row,
        suppress_fields=[
            "city_size", "education", "health_rating", "risk_tolerance",
        ],
    )
    assert len(_nonempty_lines(out)) == 4
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    assert "4 non-empty lines" in warns[0].getMessage()


def test_suppress_default_empty_tuple_changes_nothing():
    """Calling with the default empty suppress_fields matches pre-kwarg behavior."""
    row = _base_row()
    a = build_context_string(row)
    b = build_context_string(row, suppress_fields=())
    c = build_context_string(row, suppress_fields=[])
    assert a == b == c
    assert 5 <= len(_nonempty_lines(a)) <= 8  # target range unaffected


# ---------------------------------------------------------------------------
# Wave 11: extra_fields kwarg — c_d signal enrichment
# ---------------------------------------------------------------------------

def test_extra_gender_inline():
    """extras={'gender':'Female'} → 'as a female' appears in the age line."""
    row = _base_row()
    out_f = build_context_string(row, extra_fields={"gender": "Female"})
    out_m = build_context_string(row, extra_fields={"gender": "Male"})
    # The gender clause is expected to be inline in the second (age) line.
    age_line_f = _nonempty_lines(out_f)[1]
    age_line_m = _nonempty_lines(out_m)[1]
    assert "as a female" in age_line_f, f"gender=Female did not render inline: {age_line_f!r}"
    assert "as a male" in age_line_m, f"gender=Male did not render inline: {age_line_m!r}"
    # And the raw "Female"/"Male" titlecase strings must not leak.
    assert "Female" not in out_f
    assert "Male" not in out_m


def test_extra_gender_other_dropped():
    """extras={'gender':'Other'} → no gender clause."""
    row = _base_row()
    base = build_context_string(row)
    out = build_context_string(row, extra_fields={"gender": "Other"})
    # 'Other' maps to no paraphrase → no clause → output identical to base.
    assert out == base
    assert "as a " not in out.split("\n")[1].lower()


def test_extra_gender_prefer_not_to_say_dropped():
    """'Prefer not to say' is treated the same as 'Other' — no clause."""
    row = _base_row()
    base = build_context_string(row)
    out = build_context_string(row, extra_fields={"gender": "Prefer not to say"})
    assert out == base


def test_extra_life_event_adds_line():
    """extras={'life_event':'Had a child'} → a new line containing that phrase."""
    row = _base_row()
    base = build_context_string(row)
    out = build_context_string(row, extra_fields={"life_event": "Had a child"})
    assert len(_nonempty_lines(out)) == len(_nonempty_lines(base)) + 1
    assert "Had a child" in out
    assert "Recent life event" in out


def test_extra_life_event_empty_no_line():
    """extras={'life_event':''} → no extra line added."""
    row = _base_row()
    base = build_context_string(row)
    out = build_context_string(row, extra_fields={"life_event": ""})
    assert out == base


def test_extra_life_event_nan_handled():
    """extras={'life_event': float('nan')} → no extra line (pandas-NaN guard)."""
    row = _base_row()
    base = build_context_string(row)
    out = build_context_string(row, extra_fields={"life_event": float("nan")})
    assert out == base
    assert "Recent life event" not in out


def test_extra_life_event_none_handled():
    """extras={'life_event': None} → no extra line."""
    row = _base_row()
    base = build_context_string(row)
    out = build_context_string(row, extra_fields={"life_event": None})
    assert out == base


def test_extra_amazon_frequency_paraphrased():
    """'More than 10 times per month' → 'more than ten times a month'.

    The exact raw string must NOT appear verbatim.
    """
    row = _base_row()
    out = build_context_string(
        row,
        extra_fields={"amazon_frequency": "More than 10 times per month"},
    )
    assert "more than ten times a month" in out
    # Raw form must be fully paraphrased:
    assert "More than 10 times per month" not in out
    # 'Shops on Amazon' prefix indicates the new extras line.
    assert "Shops on Amazon more than ten times a month" in out


def test_extra_amazon_frequency_all_three_values_paraphrased():
    """Each of the three Q-amazon-use-how-oft values has a paraphrase."""
    row = _base_row()
    raw_to_paraphrase = {
        "Less than 5 times per month": "a few times a month",
        "5 - 10 times per month": "several times a month",
        "More than 10 times per month": "more than ten times a month",
    }
    for raw, phrase in raw_to_paraphrase.items():
        out = build_context_string(row, extra_fields={"amazon_frequency": raw})
        assert phrase in out
        assert raw not in out


def test_extras_default_none_unchanged():
    """Not passing extra_fields (or passing None) must be byte-identical to
    the pre-Wave-11 call site."""
    row = _base_row()
    a = build_context_string(row)
    b = build_context_string(row, extra_fields=None)
    c = build_context_string(row, extra_fields={})
    assert a == b == c


def test_extras_compose_with_suppression():
    """Amazon's 4 sentinels suppressed + all 3 extras provided → 5-8 lines
    restored (the paper's target range)."""
    row = _base_row()
    sup = ["has_kids", "city_size", "health_rating", "risk_tolerance"]
    extras = {
        "gender": "Female",
        "life_event": "Had a child",
        "amazon_frequency": "5 - 10 times per month",
    }
    out = build_context_string(row, suppress_fields=sup, extra_fields=extras)
    n = len(_nonempty_lines(out))
    assert 5 <= n <= 8, f"expected 5-8 lines after restoring extras; got {n}:\n{out}"
    # All three signals should have shown up.
    assert "as a female" in out
    assert "Recent life event: Had a child" in out
    assert "several times a month" in out


def test_extras_paraphrase_rules_still_pass():
    """Extras must not break paraphrase_rules_check."""
    row = _base_row()
    extras = {
        "gender": "Female",
        "life_event": "Lost a job",
        "amazon_frequency": "More than 10 times per month",
    }
    out = build_context_string(row, extra_fields=extras)
    # Explicit idempotent re-check.
    paraphrase_rules_check(out, row)


def test_extra_amazon_frequency_unknown_dropped():
    """Unknown freq value → dropped silently (never leaked verbatim)."""
    row = _base_row()
    base = build_context_string(row)
    out = build_context_string(
        row, extra_fields={"amazon_frequency": "some unknown cadence"}
    )
    # The unknown raw value is dropped entirely.
    assert out == base
    assert "some unknown cadence" not in out


# ---------------------------------------------------------------------------
# extract_extra_fields_from_row helper
# ---------------------------------------------------------------------------

def test_extract_extra_fields_from_row():
    """Given raw persons row + minimal YAML-like block, return canonical dict."""
    raw_row = {
        "Survey ResponseID": "R_x",
        "Q-demos-gender": "Female",
        "Q-life-changes": "Had a child",
        "Q-amazon-use-how-oft": "5 - 10 times per month",
    }
    yaml_block = {
        "gender": {
            "source": "Q-demos-gender",
            "kind": "categorical_map",
            "values": {"Female": "Female", "Male": "Male"},
            "drop_on_unknown": ["Other", "Prefer not to say"],
        },
        "life_event": {"source": "Q-life-changes", "kind": "passthrough"},
        "amazon_frequency": {
            "source": "Q-amazon-use-how-oft",
            "kind": "passthrough",
        },
    }
    out = extract_extra_fields_from_row(raw_row, yaml_block)
    assert set(out.keys()) == {"gender", "life_event", "amazon_frequency"}
    assert out["gender"] == "Female"
    assert out["life_event"] == "Had a child"
    assert out["amazon_frequency"] == "5 - 10 times per month"

    # The returned dict plugs straight into build_context_string.
    row = _base_row()
    rendered = build_context_string(row, extra_fields=out)
    assert "as a female" in rendered
    assert "Had a child" in rendered
    assert "several times a month" in rendered


def test_extract_extra_fields_respects_drop_on_unknown():
    """Gender='Other' → helper omits the 'gender' key from its output."""
    raw_row = {
        "Q-demos-gender": "Other",
        "Q-life-changes": "Moved place of residence",
    }
    yaml_block = {
        "gender": {
            "source": "Q-demos-gender",
            "kind": "categorical_map",
            "values": {"Female": "Female", "Male": "Male"},
            "drop_on_unknown": ["Other", "Prefer not to say"],
        },
        "life_event": {"source": "Q-life-changes", "kind": "passthrough"},
    }
    out = extract_extra_fields_from_row(raw_row, yaml_block)
    assert "gender" not in out, "Other / Prefer not to say must be dropped"
    assert out["life_event"] == "Moved place of residence"


def test_extract_extra_fields_empty_inputs():
    """None / {} yaml_block → empty dict; missing source column → skip."""
    raw_row = {"Q-demos-gender": "Female"}
    assert extract_extra_fields_from_row(raw_row, None) == {}
    assert extract_extra_fields_from_row(raw_row, {}) == {}
    # Missing source column in the raw row: the canonical key should be
    # absent from the output.
    yaml_block = {
        "life_event": {"source": "Q-life-changes", "kind": "passthrough"},
    }
    out = extract_extra_fields_from_row(raw_row, yaml_block)
    assert "life_event" not in out


def test_extract_extra_fields_skips_nan_source_values():
    """Raw life_event = NaN → canonical dict has no life_event key."""
    raw_row = {"Q-life-changes": float("nan")}
    yaml_block = {
        "life_event": {"source": "Q-life-changes", "kind": "passthrough"},
    }
    out = extract_extra_fields_from_row(raw_row, yaml_block)
    assert "life_event" not in out


# ---------------------------------------------------------------------------
# Wave 11: purchase_frequency bucket boundaries on events/week rate.
# ---------------------------------------------------------------------------


def test_phrase_purchase_frequency_very_rare_bucket():
    """Very low rates (<0.1 events/week ≈ < once every 2.5 months) render
    as 'rarely shops on Amazon', not the prior miscalibrated 'a few times
    per month'. This pins the Wave-11 count-as-rate bug fix.
    """
    from src.data.context_string import _phrase_purchase_frequency
    # 9 events over 5 years → ~0.034 events/week → must be very_rare.
    # Phrase is "rarely" (short form) so it composes inside the template
    # "Buys on Amazon <phrase>" as "Buys on Amazon rarely" without
    # duplicating "shops on Amazon".
    assert _phrase_purchase_frequency(0.034) == "rarely"
    assert _phrase_purchase_frequency(0.0) == "rarely"
    assert _phrase_purchase_frequency(0.09) == "rarely"


def test_phrase_purchase_frequency_boundary_crossings():
    """Boundary transitions between all four buckets."""
    from src.data.context_string import _phrase_purchase_frequency
    # 0.1 is the boundary between very_rare and rare
    assert _phrase_purchase_frequency(0.10) == "a few times per month"
    # 0.99 is still in the monthly bucket
    assert _phrase_purchase_frequency(0.99) == "a few times per month"
    # Weekly starts at 1.0
    assert _phrase_purchase_frequency(1.0) == "roughly once per week"
    assert _phrase_purchase_frequency(2.0) == "roughly twice per week"
    assert _phrase_purchase_frequency(3.0) == "roughly 3 times per week"
    # Daily starts above 3.0
    assert _phrase_purchase_frequency(3.5) == "almost daily"
    assert _phrase_purchase_frequency(10.0) == "almost daily"


def test_very_rare_phrasing_in_default_phrasings():
    """DEFAULT_PHRASINGS now includes a very_rare key that the renderer uses."""
    from src.data.context_string import DEFAULT_PHRASINGS
    assert "very_rare" in DEFAULT_PHRASINGS["purchase_frequency"]
    assert DEFAULT_PHRASINGS["purchase_frequency"]["very_rare"] == "rarely"
