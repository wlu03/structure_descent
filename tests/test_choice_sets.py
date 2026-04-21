"""Tests for ``src/data/choice_sets.py`` (Wave 9, design doc §5).

v1 port behavior preservation + v2 augmentation (z_d / c_d / alt_texts) +
the fit-on-train assertion + an Amazon-fixture end-to-end integration
test + Wave-11 dedup guard (post-dryrun fix).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.choice_sets import build_choice_sets


REPO_ROOT = Path(__file__).resolve().parent.parent
EVENTS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_events_100.csv"
PERSONS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_persons_100.csv"
AMAZON_YAML = REPO_ROOT / "configs" / "datasets" / "amazon.yaml"


# --------------------------------------------------------------------------- #
# Stub adapter (duck-typed Protocol, no real src.data.adapter required).
# --------------------------------------------------------------------------- #


class _StubAdapter:
    """Minimal adapter used by the unit tests.

    Satisfies the two Protocol methods used by ``build_choice_sets``.
    Tracks ``alt_text`` call counts for the "uses adapter" test.
    """

    def __init__(self, suppress_fields: tuple[str, ...] = ()):
        self._suppress = tuple(suppress_fields)
        self.alt_text_calls: list[dict] = []

    def suppress_fields_for_c_d(self) -> tuple[str, ...]:
        return self._suppress

    def alt_text(self, alt_event_row) -> dict:
        d = {
            "title": alt_event_row.get("title", ""),
            "category": alt_event_row.get("category", ""),
            "price": alt_event_row.get("price", 0.0),
            "popularity": alt_event_row.get("popularity", 0),
        }
        self.alt_text_calls.append(d)
        return d


# --------------------------------------------------------------------------- #
# Tiny synthetic fixtures (hand-built; trivial to reason about).
# --------------------------------------------------------------------------- #


def _make_events(
    n_customers: int = 6,
    n_events_per_customer: int = 10,
    n_asins: int = 20,
    n_categories: int = 3,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """Build a deterministic small events frame with full required columns."""
    rows = []
    rng = np.random.default_rng(0)
    for c in range(n_customers):
        for e in range(n_events_per_customer):
            # Spread events across time so temporal filter has something to do.
            d = pd.to_datetime(start_date) + pd.Timedelta(days=c * 100 + e)
            asin_idx = (c * 7 + e * 3) % n_asins
            cat_idx = asin_idx % n_categories
            rows.append(
                dict(
                    customer_id=f"C{c:02d}",
                    order_date=d,
                    asin=f"A{asin_idx:03d}",
                    category=f"CAT{cat_idx}",
                    price=5.0 + (asin_idx % 10),
                    title=f"title A{asin_idx:03d} word{e}",
                    routine=int(e > 0),
                    novelty=int(e == 0),
                    recency_days=999 if e == 0 else 7,
                    cat_affinity=e,
                    brand="brand_x",
                )
            )
    df = pd.DataFrame(rows)
    # Popularity: count per asin.
    pop = df.groupby("asin").size().rename("popularity").reset_index()
    df = df.merge(pop, on="asin", how="left")
    # Split: earliest 6 train, 7-8 val, 9-10 test per customer (deterministic).
    df = df.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    splits = []
    for _, grp in df.groupby("customer_id", sort=False):
        n = len(grp)
        n_test = max(1, int(n * 0.2))
        n_val = max(1, int(n * 0.2))
        if n_test + n_val >= n:
            n_test, n_val = 1, 0
        n_train = n - n_test - n_val
        splits.extend(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
    df["split"] = splits
    return df


def _persons_for(events: pd.DataFrame) -> pd.DataFrame:
    """Build a persons_canonical frame covering every train customer.

    Mirrors what adapter.translate_z_d() would produce: customer_id +
    the 10 canonical z_d columns. The first few synthetic rows are rigged
    to exercise every age/income/city vocabulary slot so that
    ``fit_person_features`` sees the full 6+5+4 one-hot widths and z_d
    comes out shape (26,).
    """
    train_customers = sorted(
        events.loc[events["split"] == "train", "customer_id"].unique().tolist()
    )
    # Full canonical vocabularies (mirror DEFAULT_PHRASINGS in context_string.py).
    full_age = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    full_income = ["<25k", "25-50k", "50-100k", "100-150k", "150k+"]
    full_city = ["rural", "small", "medium", "large"]
    rows = []
    for i, cid in enumerate(train_customers):
        rows.append(
            dict(
                customer_id=cid,
                age_bucket=full_age[i % len(full_age)],
                income_bucket=full_income[i % len(full_income)],
                household_size=1 + (i % 4),
                has_kids=int(i % 2),
                city_size=full_city[i % len(full_city)],
                education=3,
                health_rating=4,
                risk_tolerance=0.0,
                purchase_frequency=2.0,
                novelty_rate=0.3,
            )
        )
    # If we have fewer train customers than vocab slots, append synthetic
    # "padding" rows carrying the missing vocab values so fit_person_features
    # learns the full vocabularies. These extra rows are then dropped before
    # return (they carry customer ids that won't appear in events).
    # Instead of padding (which would fail the fit-on-train assertion),
    # append vocabulary-coverage into the existing rows by cycling through
    # all vocabs across the first few rows. We already cycle, so as long as
    # we have >= max(len(full_age), len(full_income), len(full_city)) = 6
    # customers, every slot is hit. For smaller cases, widen the cycle.
    df = pd.DataFrame(rows)
    return df


# --------------------------------------------------------------------------- #
# v1 behavior preservation
# --------------------------------------------------------------------------- #


def test_port_preserves_v1_output_shape():
    events = _make_events()
    persons = _persons_for(events)
    adapter = _StubAdapter()
    records = build_choice_sets(
        events, persons, adapter, n_negatives=9, seed=42, n_resamples=1
    )
    assert len(records) == len(events)
    v1_keys = {
        "customer_id",
        "order_date",
        "category",
        "chosen_asin",
        "choice_asins",
        "chosen_idx",
        "chosen_features",
        "metadata",
    }
    for r in records:
        assert v1_keys.issubset(set(r.keys()))
        assert len(r["choice_asins"]) == 10  # n_negatives + 1
        # chosen_idx points at the chosen_asin inside choice_asins.
        assert r["choice_asins"][r["chosen_idx"]] == r["chosen_asin"]


def test_temporal_availability_filter():
    """Sampled negatives must have first_seen strictly < event.order_date."""
    events = _make_events(n_customers=4, n_events_per_customer=8, n_asins=30)
    persons = _persons_for(events)
    adapter = _StubAdapter()
    records = build_choice_sets(
        events, persons, adapter, n_negatives=9, seed=7
    )
    first_seen = events.groupby("asin")["order_date"].min()
    # Early events — only a handful of asins exist before them — should
    # never pull a future asin.
    for r in records:
        ev_date = pd.Timestamp(r["order_date"])
        chosen = r["chosen_asin"]
        for alt in r["choice_asins"]:
            if alt == chosen:
                continue
            # Each non-chosen alt must have been seen strictly before ev_date.
            # (If avail_len==0 the v1 code falls back to the chosen_code,
            # in which case alt == chosen and we skipped above.)
            assert first_seen.loc[alt] < ev_date, (
                f"alt {alt} first_seen {first_seen.loc[alt]} is not strictly "
                f"earlier than event date {ev_date}"
            )


def test_n_resamples_produces_nested_structure():
    events = _make_events()
    persons = _persons_for(events)
    adapter = _StubAdapter()
    K = 3
    records = build_choice_sets(
        events, persons, adapter, n_negatives=9, seed=42, n_resamples=K
    )
    for r in records:
        assert isinstance(r["choice_asins"], list) and len(r["choice_asins"]) == K
        assert isinstance(r["chosen_idx"], list) and len(r["chosen_idx"]) == K
        for inner in r["choice_asins"]:
            assert len(inner) == 10
        # alt_texts is also list-of-lists in this mode.
        assert isinstance(r["alt_texts"], list) and len(r["alt_texts"]) == K
        for inner in r["alt_texts"]:
            assert len(inner) == 10


def test_deterministic_under_seed():
    events = _make_events()
    persons = _persons_for(events)
    a1 = _StubAdapter()
    a2 = _StubAdapter()
    r1 = build_choice_sets(events, persons, a1, seed=42)
    r2 = build_choice_sets(events, persons, a2, seed=42)
    assert len(r1) == len(r2)
    for x, y in zip(r1, r2):
        assert x["choice_asins"] == y["choice_asins"]
        assert x["chosen_idx"] == y["chosen_idx"]


# --------------------------------------------------------------------------- #
# v2 augmentation
# --------------------------------------------------------------------------- #


def test_v2_record_has_z_d_c_d_alt_texts():
    events = _make_events()
    persons = _persons_for(events)
    adapter = _StubAdapter()
    records = build_choice_sets(events, persons, adapter, seed=42)
    for r in records:
        assert "z_d" in r
        assert "c_d" in r
        assert "alt_texts" in r


def test_z_d_shape_and_dtype():
    events = _make_events()
    persons = _persons_for(events)
    adapter = _StubAdapter()
    records = build_choice_sets(events, persons, adapter, seed=42)
    for r in records:
        assert isinstance(r["z_d"], np.ndarray)
        assert r["z_d"].shape == (26,)
        assert r["z_d"].dtype == np.float32


def test_c_d_is_string_and_nonempty():
    events = _make_events()
    persons = _persons_for(events)
    adapter = _StubAdapter()
    records = build_choice_sets(events, persons, adapter, seed=42)
    for r in records:
        assert isinstance(r["c_d"], str)
        assert len(r["c_d"]) > 0


def test_alt_texts_length_matches_J():
    events = _make_events()
    persons = _persons_for(events)
    adapter = _StubAdapter()
    records = build_choice_sets(
        events, persons, adapter, n_negatives=9, seed=42
    )
    for r in records:
        assert len(r["alt_texts"]) == 10  # n_negatives + 1
        for alt in r["alt_texts"]:
            assert isinstance(alt, dict)
            assert set(alt.keys()) == {"title", "category", "price", "popularity"}


def test_v2_alt_texts_uses_adapter():
    events = _make_events()
    persons = _persons_for(events)
    adapter = _StubAdapter()
    n_negatives = 9
    J = n_negatives + 1
    records = build_choice_sets(
        events, persons, adapter, n_negatives=n_negatives, seed=42, n_resamples=1
    )
    assert len(adapter.alt_text_calls) == len(records) * J


# --------------------------------------------------------------------------- #
# Fit-on-train assertion
# --------------------------------------------------------------------------- #


def test_fit_on_train_assertion_fires():
    """persons_canonical with a customer that's not in train -> AssertionError."""
    events = _make_events()
    persons = _persons_for(events)
    # Add an orphan customer B that is NOT in the train split (or any split).
    orphan = persons.iloc[0:1].copy()
    orphan.loc[:, "customer_id"] = "B_NOT_IN_EVENTS"
    persons_bad = pd.concat([persons, orphan], ignore_index=True)
    adapter = _StubAdapter()
    with pytest.raises(AssertionError) as exc:
        build_choice_sets(events, persons_bad, adapter, seed=42)
    assert "B_NOT_IN_EVENTS" in str(exc.value)


def test_fit_on_train_passes_when_persons_match_train():
    events = _make_events()
    persons = _persons_for(events)  # exactly the train customers
    adapter = _StubAdapter()
    records = build_choice_sets(events, persons, adapter, seed=42)
    assert len(records) == len(events)


# --------------------------------------------------------------------------- #
# ValueError from v1 (NaN in asin/category)
# --------------------------------------------------------------------------- #


def test_nan_in_asin_or_category_raises():
    events = _make_events()
    events.loc[0, "asin"] = np.nan
    persons = _persons_for(events)
    adapter = _StubAdapter()
    with pytest.raises(ValueError, match="NaN"):
        build_choice_sets(events, persons, adapter, seed=42)


# --------------------------------------------------------------------------- #
# Amazon fixture end-to-end
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not EVENTS_FIXTURE.exists() or not PERSONS_FIXTURE.exists(),
    reason="Amazon fixtures not available",
)
def test_amazon_fixture_end_to_end():
    """load -> clean -> survey_join -> state_features -> split ->
    attach_train_popularity -> translate_z_d -> build_choice_sets.

    Uses a minimal in-test adapter (suppress_fields matches the Amazon YAML's
    constant/composite fields) to avoid hard-requiring the Wave-9
    ``src.data.adapter`` module (which is co-developed in a sibling
    subagent task).
    """
    from src.data.clean import clean_events
    from src.data.load import load
    from src.data.schema_map import load_schema, translate_persons
    from src.data.split import temporal_split
    from src.data.state_features import (
        attach_train_popularity,
        compute_state_features,
    )
    from src.data.survey_join import join_survey

    schema = load_schema(AMAZON_YAML)
    events_raw, persons_raw = load(
        schema, events_path=EVENTS_FIXTURE, persons_path=PERSONS_FIXTURE
    )

    cleaned = clean_events(events_raw, schema)
    joined = join_survey(cleaned, persons_raw, schema)
    with_state = compute_state_features(joined)
    split_df = temporal_split(with_state, schema)
    with_pop = attach_train_popularity(split_df)

    # translate_persons needs training_events for the derived_from_events
    # fields (purchase_frequency / novelty_rate).
    train_events = with_pop.loc[with_pop["split"] == "train"]
    z_df_all = translate_persons(persons_raw, schema, training_events=train_events)
    # Restrict persons_canonical to customers that survive into train.
    train_customers = set(train_events["customer_id"].unique())
    z_df = z_df_all.loc[z_df_all["customer_id"].isin(train_customers)].reset_index(
        drop=True
    )
    # Upstream responsibility: filter events to customers we have a
    # canonical persons row for (others have survey refusals / drop_on_unknown
    # pruning and should not participate in choice-set building).
    known = set(z_df["customer_id"].unique())
    events_for_cs = with_pop.loc[with_pop["customer_id"].isin(known)].reset_index(
        drop=True
    )

    # Amazon adapter policy (NOTES.md Wave 8): has_kids, city_size,
    # health_rating, risk_tolerance are sentinel / constant for Amazon
    # and carry no per-customer signal, so they are suppressed from c_d.
    adapter = _StubAdapter(
        suppress_fields=("has_kids", "city_size", "health_rating", "risk_tolerance")
    )

    records = build_choice_sets(
        events_for_cs, z_df, adapter, n_negatives=9, seed=42, n_resamples=1
    )

    assert len(records) > 0
    # The 100-person fixture under-samples age / income / city vocabularies
    # (a fraction of the full 6+5+4 = 15 one-hot slots are missing),
    # so the fit-on-fixture z_d width is <= 26. The test asserts the
    # width is stable across records and fully-populated slots + the
    # 11 always-present scalar/binary dims are present.
    expected_width = records[0]["z_d"].shape[0]
    assert 15 <= expected_width <= 26
    for r in records[:20]:  # cheap spot-check
        assert r["z_d"].shape == (expected_width,)
        assert r["z_d"].dtype == np.float32
        assert isinstance(r["c_d"], str) and len(r["c_d"]) > 0
        assert len(r["alt_texts"]) == 10


# --------------------------------------------------------------------------- #
# Wave 11 dedup guard (post-dryrun fix).
#
# The 1-customer real-LLM dry-run showed alt[0] and alt[2] returning the
# identical outcome because the same ASIN appeared twice in a choice set
# (small catalog + popularity-weighted sampling-with-replacement). At
# scale this is vanishingly rare, but the guard eliminates it altogether
# and records a ``metadata["dedup_fallback"]`` bool when the temporally-
# available pool is smaller than J.
# --------------------------------------------------------------------------- #


def _make_tiny_catalog_events(
    n_events: int = 12,
    n_asins: int = 4,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """Build a 1-customer events frame with an intentionally tiny catalog.

    With ``n_asins=4`` and ``n_negatives=9``, the temporally-available
    pool can never reach J=10 distinct alternatives, so the dedup guard
    MUST fall back on cyclic padding and flag the record.
    """
    rows = []
    for e in range(n_events):
        d = pd.to_datetime(start_date) + pd.Timedelta(days=e)
        asin_idx = e % n_asins
        rows.append(
            dict(
                customer_id="C00",
                order_date=d,
                asin=f"A{asin_idx:03d}",
                category="CAT0",
                price=5.0 + asin_idx,
                title=f"title A{asin_idx:03d} word{e}",
                routine=int(e > 0),
                novelty=int(e == 0),
                recency_days=999 if e == 0 else 7,
                cat_affinity=e,
                brand="brand_x",
            )
        )
    df = pd.DataFrame(rows)
    pop = df.groupby("asin").size().rename("popularity").reset_index()
    df = df.merge(pop, on="asin", how="left")
    df = df.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    # Single-customer split: mark the tail events val/test so the train
    # partition has enough rows for persons_canonical.
    n = len(df)
    n_test = max(1, int(n * 0.2))
    n_val = max(1, int(n * 0.2))
    if n_test + n_val >= n:
        n_test, n_val = 1, 0
    n_train = n - n_test - n_val
    df["split"] = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    return df


def _make_scale_events(
    n_customers: int = 8,
    n_asins: int = 200,
    n_categories: int = 4,
    n_events_per_customer: int = 12,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """Build an events frame with a LARGE catalog so every event has
    ``available_pool >> J``. The first ``n_asins`` days of "seed" events
    spread one-asin-per-day front-load the first-seen dates so all
    real-customer events happen after every ASIN has first-seen.
    """
    rows = []
    # Seed phase: one event per ASIN, earliest dates, on a single seed
    # customer that we drop from the persons frame (they won't be in the
    # train split when we re-split below... actually we need them to be
    # train so build_choice_sets is happy). Simpler: embed the seed
    # events into customer 0's early history.
    seed_start = pd.to_datetime(start_date)
    # Customer 0 seeds the catalog across the first n_asins days.
    for a in range(n_asins):
        rows.append(
            dict(
                customer_id="C_SEED",
                order_date=seed_start + pd.Timedelta(days=a),
                asin=f"A{a:04d}",
                category=f"CAT{a % n_categories}",
                price=5.0 + (a % 10),
                title=f"title A{a:04d}",
                routine=0,
                novelty=1,
                recency_days=999,
                cat_affinity=0,
                brand="brand_x",
            )
        )
    # Real customers purchase AFTER all seed events.
    real_start = seed_start + pd.Timedelta(days=n_asins + 1)
    for c in range(n_customers):
        for e in range(n_events_per_customer):
            d = real_start + pd.Timedelta(days=c * n_events_per_customer + e)
            asin_idx = (c * 13 + e * 5) % n_asins
            rows.append(
                dict(
                    customer_id=f"C{c:02d}",
                    order_date=d,
                    asin=f"A{asin_idx:04d}",
                    category=f"CAT{asin_idx % n_categories}",
                    price=5.0 + (asin_idx % 10),
                    title=f"title A{asin_idx:04d} word{e}",
                    routine=int(e > 0),
                    novelty=int(e == 0),
                    recency_days=999 if e == 0 else 7,
                    cat_affinity=e,
                    brand="brand_x",
                )
            )
    df = pd.DataFrame(rows)
    pop = df.groupby("asin").size().rename("popularity").reset_index()
    df = df.merge(pop, on="asin", how="left")
    df = df.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    splits = []
    for _, grp in df.groupby("customer_id", sort=False):
        n = len(grp)
        n_test = max(1, int(n * 0.2))
        n_val = max(1, int(n * 0.2))
        if n_test + n_val >= n:
            n_test, n_val = 1, 0
        n_train = n - n_test - n_val
        splits.extend(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
    df["split"] = splits
    return df


def test_dedup_at_scale():
    """Large catalog: every record has J distinct ASINs and no fallback.

    200 ASINs front-loaded by seed events, J=10 → every real-customer
    event sees a pool >> J and the dedup loop converges without the
    fallback branch.
    """
    events = _make_scale_events(n_customers=6, n_asins=200)
    persons = _persons_for(events)
    adapter = _StubAdapter()
    records = build_choice_sets(
        events, persons, adapter, n_negatives=9, seed=42, n_resamples=1
    )
    # Skip the ``C_SEED`` records — those are fixture scaffolding with a
    # deliberately sparse early-catalog and may legitimately fall back.
    real_records = [r for r in records if r["customer_id"] != "C_SEED"]
    assert len(real_records) > 0
    for r in real_records:
        assert len(r["choice_asins"]) == 10
        assert len(set(r["choice_asins"])) == 10, (
            f"duplicate ASIN in choice set for customer={r['customer_id']}: "
            f"{r['choice_asins']}"
        )
        assert r["metadata"]["dedup_fallback"] is False, (
            f"unexpected fallback for customer={r['customer_id']}"
        )


def test_dedup_fallback_small_catalog(caplog):
    """Tiny catalog: pool < J triggers cyclic padding AND a WARNING."""
    events = _make_tiny_catalog_events(n_events=12, n_asins=4)
    persons = _persons_for(events)
    adapter = _StubAdapter()
    with caplog.at_level(logging.WARNING, logger="src.data.choice_sets"):
        records = build_choice_sets(
            events, persons, adapter, n_negatives=9, seed=42, n_resamples=1
        )

    # At least one record should have fallen back (the later events have
    # 4 available ASINs but J=10).
    fallback_records = [r for r in records if r["metadata"]["dedup_fallback"]]
    assert len(fallback_records) > 0, "expected at least one dedup fallback"

    # Every record has exactly J=10 entries (padding preserves shape).
    for r in records:
        assert len(r["choice_asins"]) == 10

    # At least one WARNING logged for the dedup fallback path.
    warn_messages = [rec.message for rec in caplog.records if rec.levelno == logging.WARNING]
    assert any("dedup fallback" in m for m in warn_messages), (
        f"expected dedup-fallback WARNING; got: {warn_messages}"
    )


def test_dedup_preserves_chosen():
    """Chosen ASIN always sits at choice_asins[chosen_idx], fallback or not."""
    events = _make_tiny_catalog_events(n_events=12, n_asins=4)
    persons = _persons_for(events)
    adapter = _StubAdapter()
    records = build_choice_sets(
        events, persons, adapter, n_negatives=9, seed=42, n_resamples=1
    )
    for r in records:
        assert r["choice_asins"][r["chosen_idx"]] == r["chosen_asin"]


def test_dedup_determinism():
    """Same seed + same events produce byte-identical dedup output."""
    events = _make_tiny_catalog_events(n_events=12, n_asins=4)
    persons = _persons_for(events)
    r1 = build_choice_sets(events, persons, _StubAdapter(), seed=42)
    r2 = build_choice_sets(events, persons, _StubAdapter(), seed=42)
    assert len(r1) == len(r2)
    for x, y in zip(r1, r2):
        assert x["choice_asins"] == y["choice_asins"]
        assert x["chosen_idx"] == y["chosen_idx"]
        assert x["metadata"]["dedup_fallback"] == y["metadata"]["dedup_fallback"]


def test_dedup_does_not_change_large_fixture():
    """100-customer Amazon fixture: most records are dedup_fallback==False.

    Don't assert zero — small-catalog categories inside the fixture can
    still trigger it legitimately. Just confirm the dominant record path
    is the non-fallback branch AND every resulting choice set has J
    entries.
    """
    if not EVENTS_FIXTURE.exists() or not PERSONS_FIXTURE.exists():
        pytest.skip("Amazon fixtures not available")

    from src.data.clean import clean_events
    from src.data.load import load
    from src.data.schema_map import load_schema, translate_persons
    from src.data.split import temporal_split
    from src.data.state_features import (
        attach_train_popularity,
        compute_state_features,
    )
    from src.data.survey_join import join_survey

    schema = load_schema(AMAZON_YAML)
    events_raw, persons_raw = load(
        schema, events_path=EVENTS_FIXTURE, persons_path=PERSONS_FIXTURE
    )
    cleaned = clean_events(events_raw, schema)
    joined = join_survey(cleaned, persons_raw, schema)
    with_state = compute_state_features(joined)
    split_df = temporal_split(with_state, schema)
    with_pop = attach_train_popularity(split_df)
    train_events = with_pop.loc[with_pop["split"] == "train"]
    z_df_all = translate_persons(persons_raw, schema, training_events=train_events)
    train_customers = set(train_events["customer_id"].unique())
    z_df = z_df_all.loc[z_df_all["customer_id"].isin(train_customers)].reset_index(
        drop=True
    )
    known = set(z_df["customer_id"].unique())
    events_for_cs = with_pop.loc[with_pop["customer_id"].isin(known)].reset_index(
        drop=True
    )
    adapter = _StubAdapter(
        suppress_fields=("has_kids", "city_size", "health_rating", "risk_tolerance")
    )
    records = build_choice_sets(
        events_for_cs, z_df, adapter, n_negatives=9, seed=42, n_resamples=1
    )

    # Every record has J=10 entries.
    for r in records:
        assert len(r["choice_asins"]) == 10

    # Dominant path is non-fallback: > 50% of records clean. (We do not
    # assert zero — the Amazon fixture has sparse categories that can
    # trigger the guard for very-early events.)
    n_clean = sum(1 for r in records if not r["metadata"]["dedup_fallback"])
    assert n_clean / len(records) > 0.5, (
        f"expected majority of records to be dedup-clean; "
        f"got {n_clean}/{len(records)} clean"
    )
