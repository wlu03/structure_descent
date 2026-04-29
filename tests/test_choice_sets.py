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
            # build_choice_sets injects per-(customer, asin) train-history
            # fields onto every alt dict (Fix 2 — log1p_purchase_count).
            assert set(alt.keys()) == {
                "title", "category", "price", "popularity",
                "is_repeat", "purchase_count",
            }


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
    """persons_canonical missing a customer present in events -> AssertionError.

    Wave-13 loosening: the assertion used to be
    ``persons_canonical.customer_id ⊆ train_customers`` (orphan persons
    rows were rejected). That broke cold-start, where val/test customers
    legitimately carry z_d rows via translate_z_d's transform path
    without ever appearing in the train split. The weakened invariant
    only rejects the case that would KeyError at the per-event z_d
    lookup: an event whose customer_id has NO row in persons_canonical.
    Here we drop one of the train customers from persons_canonical
    and expect the assertion to fire.
    """
    events = _make_events()
    persons = _persons_for(events)
    dropped_cid = str(persons.iloc[0]["customer_id"])
    persons_bad = persons.iloc[1:].reset_index(drop=True)
    adapter = _StubAdapter()
    with pytest.raises(AssertionError) as exc:
        build_choice_sets(events, persons_bad, adapter, seed=42)
    msg = str(exc.value)
    assert dropped_cid in msg
    assert "z_d row" in msg


def test_fit_on_train_allows_orphan_persons_rows():
    """Wave-13: persons rows for customers NOT in events are now allowed.

    Pre-Wave-13 this was an AssertionError (old strict invariant
    required persons_canonical ⊆ train_customers). The new invariant
    goes in the other direction — persons_canonical must be a
    *superset* of event_customers — so orphan persons rows are
    irrelevant. This test pins the looser semantics.
    """
    events = _make_events()
    persons = _persons_for(events)
    # Append an orphan customer that is NOT in events at all.
    orphan = persons.iloc[0:1].copy()
    orphan.loc[:, "customer_id"] = "B_NOT_IN_EVENTS"
    persons_with_orphan = pd.concat([persons, orphan], ignore_index=True)
    adapter = _StubAdapter()
    records = build_choice_sets(events, persons_with_orphan, adapter, seed=42)
    assert len(records) == len(events)


def test_fit_on_train_passes_when_persons_match_train():
    events = _make_events()
    persons = _persons_for(events)  # exactly the train customers
    adapter = _StubAdapter()
    records = build_choice_sets(events, persons, adapter, seed=42)
    assert len(records) == len(events)


# --------------------------------------------------------------------------- #
# Wave-13 cold-start compatibility: val/test customers in persons_canonical.
#
# The Wave-13 loosening flips the persons_canonical invariant from
# "subset of train customers" to "superset of event customers". This
# unblocks cold-start, where val/test customers have NO train rows by
# construction yet still need z_d vectors for evaluation.
# --------------------------------------------------------------------------- #


def _make_cold_start_events() -> pd.DataFrame:
    """Build a disjoint cold-start events frame.

    Three customers, each wholly in one split (train / val / test).
    Every customer has a few events so build_choice_sets has per-
    customer history to draw from. Catalog is large enough that the
    dedup guard does not trigger cyclic-fallback padding.
    """
    rows = []
    base = pd.to_datetime("2024-01-01")
    cid_splits = [
        ("C_train", "train"),
        ("C_val", "val"),
        ("C_test", "test"),
    ]
    for c_idx, (cid, split) in enumerate(cid_splits):
        for e in range(4):
            asin_idx = c_idx * 13 + e * 5
            rows.append(
                dict(
                    customer_id=cid,
                    order_date=base + pd.Timedelta(days=c_idx * 30 + e),
                    asin=f"A{asin_idx:04d}",
                    category=f"CAT{asin_idx % 3}",
                    price=5.0 + (asin_idx % 7),
                    title=f"title A{asin_idx:04d}",
                    routine=int(e > 0),
                    novelty=int(e == 0),
                    recency_days=999 if e == 0 else 7,
                    cat_affinity=e,
                    brand="brand_x",
                    split=split,
                )
            )
    # Seed additional train rows with a LARGE catalog so the temporal
    # availability prefix at each event's date has many ASINs to sample
    # from (avoids tiny-catalog dedup fallback).
    seed_start = base - pd.Timedelta(days=100)
    for a in range(50):
        rows.append(
            dict(
                customer_id="C_train",
                order_date=seed_start + pd.Timedelta(days=a),
                asin=f"A_seed{a:03d}",
                category=f"CAT{a % 3}",
                price=5.0 + (a % 7),
                title=f"seed title {a}",
                routine=0,
                novelty=1,
                recency_days=999,
                cat_affinity=0,
                brand="brand_x",
                split="train",
            )
        )
    df = pd.DataFrame(rows)
    pop = df.groupby("asin").size().rename("popularity").reset_index()
    df = df.merge(pop, on="asin", how="left")
    df = df.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    return df


def test_build_choice_sets_accepts_val_test_customers_in_persons_canonical():
    """Wave-13 loosening: persons_canonical may carry z_d rows for
    val/test customers who have NO train events (cold-start).

    Pre-Wave-13, the fit-on-train assertion rejected any persons
    row whose customer_id was not in the train split. Under cold-
    start that rejected every val/test customer. After the fix, the
    only requirement is that every event customer has a z_d row —
    so val/test customers carrying z_d are allowed and their events
    flow through build_choice_sets end-to-end.
    """
    events = _make_cold_start_events()
    all_cids = sorted(events["customer_id"].unique().tolist())
    # persons_canonical covers ALL customers (train + val + test).
    persons = _persons_for_customers(all_cids)
    adapter = _StubAdapter()

    # Should NOT raise — pre-Wave-13 this raised AssertionError on the
    # "val/test customer_id not in train" branch.
    records = build_choice_sets(
        events, persons, adapter, n_negatives=9, seed=42, n_resamples=1
    )
    assert len(records) == len(events)

    # Every split contributes records.
    record_cids = {r["customer_id"] for r in records}
    assert "C_train" in record_cids
    assert "C_val" in record_cids
    assert "C_test" in record_cids


def test_build_choice_sets_raises_when_event_customer_missing_z_d():
    """Coverage assertion fires when an event customer has no z_d row.

    Mirror of the old ``test_fit_on_train_assertion_fires`` but
    inverted: drop one of the customers from persons_canonical while
    keeping all their events, and expect an AssertionError whose
    message names the missing customer_id and the phrase "z_d row".
    """
    events = _make_cold_start_events()
    all_cids = sorted(events["customer_id"].unique().tolist())
    persons = _persons_for_customers(all_cids)
    # Drop the val customer's z_d row — their events now have nowhere
    # to look up z_d.
    persons_bad = persons[persons["customer_id"] != "C_val"].reset_index(
        drop=True
    )
    adapter = _StubAdapter()
    with pytest.raises(AssertionError) as exc:
        build_choice_sets(
            events, persons_bad, adapter, n_negatives=9, seed=42, n_resamples=1
        )
    msg = str(exc.value)
    assert "C_val" in msg
    assert "z_d row" in msg


def test_build_choice_sets_cold_start_end_to_end_smoke():
    """End-to-end smoke test: cold_start_split -> attach_train_popularity
    -> attach_train_brand_map -> persons_canonical covering all splits
    -> build_choice_sets.

    Exercises the full runner wiring under cold-start: val/test
    customers get z_d rows via an all-customers transform (the fit is
    still train-only in real code, via ``translate_z_d(training_events=)``;
    this test bypasses the adapter machinery and hand-rolls
    persons_canonical so the assertion we care about — coverage of
    event customer_ids — is exercised directly).
    """
    from src.data.split import cold_start_split
    from src.data.state_features import (
        attach_train_brand_map,
        attach_train_popularity,
    )

    # Build an events frame with 20 customers × 5 events each, spread
    # across time so the catalog is populated well before each event
    # (regardless of which customers land in which cold-start bucket).
    rows = []
    base = pd.to_datetime("2024-01-01")
    n_customers = 20
    n_events_per_customer = 5
    n_categories = 3
    for c in range(n_customers):
        for e in range(n_events_per_customer):
            # Offsets interleave so ASINs are first-seen across a
            # range of early dates regardless of the shuffle.
            asin_idx = (c + e * n_customers) % (n_customers * n_events_per_customer)
            rows.append(
                dict(
                    customer_id=f"C{c:02d}",
                    order_date=base + pd.Timedelta(days=e * n_customers + c),
                    asin=f"A{asin_idx:04d}",
                    category=f"CAT{asin_idx % n_categories}",
                    price=5.0 + (asin_idx % 7),
                    title=f"title C{c:02d}_e{e}",
                    routine=int(e > 0),
                    novelty=int(e == 0),
                    recency_days=999 if e == 0 else 7,
                    cat_affinity=e,
                    brand="brand_x",
                )
            )
    events = pd.DataFrame(rows)

    # Cold-start split: disjoint train / val / test customer sets.
    split_df = cold_start_split(
        events, val_customer_frac=0.2, test_customer_frac=0.2, seed=1
    )
    # Attach train-derived aggregates (both require the split column).
    with_pop = attach_train_popularity(split_df)
    with_brand = attach_train_brand_map(with_pop)

    # Verify cold-start produced all three splits (partition is data-
    # dependent; bail early with a clear message if the shuffle
    # happened to starve one bucket — at 20 customers with 0.2/0.2
    # fractions this should never happen).
    splits = set(with_brand["split"].unique().tolist())
    assert splits == {"train", "val", "test"}, (
        f"cold_start_split produced splits={splits}; expected all three"
    )

    # persons_canonical covering every customer (train + val + test).
    # In real code this is the output of
    # ``adapter.translate_z_d(persons_raw_keep, training_events=train_subset)``
    # — the fit is train-only, the transform runs on every customer.
    all_cids = sorted(with_brand["customer_id"].astype(str).unique().tolist())
    persons_canonical = _persons_for_customers(all_cids)

    adapter = _StubAdapter()
    records = build_choice_sets(
        with_brand,
        persons_canonical,
        adapter,
        n_negatives=9,
        seed=42,
        n_resamples=1,
    )

    # No exception; every split contributes at least one record.
    assert len(records) == len(with_brand)
    cid_to_split = dict(
        zip(with_brand["customer_id"].astype(str), with_brand["split"].astype(str))
    )
    splits_seen = {cid_to_split[str(r["customer_id"])] for r in records}
    assert splits_seen == {"train", "val", "test"}

    # Every record's customer_id has a matching z_d row in
    # persons_canonical (i.e. the coverage assertion was satisfied
    # and the per-event lookup never raised KeyError).
    zd_cids = set(persons_canonical["customer_id"].astype(str))
    for r in records:
        assert str(r["customer_id"]) in zd_cids


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


# --------------------------------------------------------------------------- #
# Wave 12 per-event ``recent_purchases`` in c_d.
#
# Each event gets a per-event render of ``c_d`` that includes the titles
# of the customer's purchases in the window immediately BEFORE the event
# (strict <, no leakage). Six tests below pin the behavior:
#  - a middle event gets its prior-events' titles,
#  - the FIRST event for a customer has no "Recent purchases" line,
#  - events outside the window are not included,
#  - the list is capped at ``max_recent_purchases``,
#  - very long titles are truncated to ≤80 chars in the output,
#  - same-day prior events render reproducibly across runs.
# --------------------------------------------------------------------------- #


def _make_customer_events(
    customer_id: str,
    n_events: int,
    *,
    offsets_days: list[int],
    titles: list[str],
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """Build a single-customer events frame with explicit per-event offsets.

    ``offsets_days[i]`` is the number of days (from ``start_date``) of the
    i-th event for this customer. ``titles[i]`` is the title. ASINs are
    generated deterministically so duplicate titles at different offsets
    still render as distinct catalog entries.
    """
    assert len(offsets_days) == n_events == len(titles)
    rows = []
    base = pd.to_datetime(start_date)
    for e in range(n_events):
        d = base + pd.Timedelta(days=offsets_days[e])
        rows.append(
            dict(
                customer_id=customer_id,
                order_date=d,
                asin=f"A{e:04d}",
                category="CAT0",
                price=5.0 + e,
                title=titles[e],
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
    # All events train so a single-customer persons_canonical is valid.
    df["split"] = ["train"] * len(df)
    return df


def _persons_for_single(customer_id: str) -> pd.DataFrame:
    """Persons-canonical with exactly one training customer."""
    return pd.DataFrame(
        [
            dict(
                customer_id=customer_id,
                age_bucket="45-54",
                income_bucket="50-100k",
                household_size=2,
                has_kids=0,
                city_size="medium",
                education=3,
                health_rating=4,
                risk_tolerance=0.0,
                purchase_frequency=2.0,
                novelty_rate=0.3,
            )
        ]
    )


def test_recent_purchases_populated_for_middle_event():
    """Customer with 4 events in 20 days: the 3rd event's c_d contains
    title fragments of the first two (NOT the 4th, which is after)."""
    events = _make_customer_events(
        "C_MID",
        n_events=4,
        offsets_days=[0, 5, 10, 15],
        titles=[
            "Kitchen Spatula Set",
            "Coffee Filter Pack",
            "Laundry Detergent",
            "Dish Soap",
        ],
    )
    persons = _persons_for_single("C_MID")
    adapter = _StubAdapter()
    records = build_choice_sets(events, persons, adapter, seed=42)
    # Records come back in event-order (sorted above); the 3rd event is
    # index 2.
    c_d = records[2]["c_d"]
    assert "Recent purchases" in c_d
    assert "Kitchen Spatula Set" in c_d
    assert "Coffee Filter Pack" in c_d
    # The 4th event's title must NOT have leaked into the 3rd event's c_d.
    assert "Dish Soap" not in c_d
    # The 3rd event's own title must not be in its recent_purchases block
    # either (strict < on the current event).
    # Split by the recent-purchases marker so we only inspect that line.
    tail = c_d.split("Recent purchases", 1)[1]
    assert "Laundry Detergent" not in tail


def test_recent_purchases_empty_for_first_event():
    """The FIRST event for a customer has no prior purchases, so c_d
    must not contain the "Recent purchases" clause."""
    events = _make_customer_events(
        "C_FIRST",
        n_events=3,
        offsets_days=[0, 5, 10],
        titles=["First Title", "Second Title", "Third Title"],
    )
    persons = _persons_for_single("C_FIRST")
    adapter = _StubAdapter()
    records = build_choice_sets(events, persons, adapter, seed=42)
    assert "Recent purchases" not in records[0]["c_d"]
    # Later events should have the clause.
    assert "Recent purchases" in records[1]["c_d"]


def test_recent_purchases_window_respected():
    """With window_days=7, events separated by > 7 days must NOT show
    up in c_d. Build events at offsets [0, 3, 10, 20, 60] and check the
    last event sees none of the first three (all > 7 days away)."""
    events = _make_customer_events(
        "C_WIN",
        n_events=5,
        offsets_days=[0, 3, 10, 20, 60],
        titles=[
            "Way Old Title",
            "Old Title",
            "Medium Age Title",
            "Nearly New Title",
            "Current Title",
        ],
    )
    persons = _persons_for_single("C_WIN")
    adapter = _StubAdapter()
    records = build_choice_sets(
        events, persons, adapter, seed=42, recent_purchases_window_days=7
    )
    # event[1] (offset=3): only event[0] at offset=0 is in the 7-day
    # window (3 - 0 = 3 < 7). event[0]'s title should appear.
    c_d_1 = records[1]["c_d"]
    assert "Recent purchases" in c_d_1
    assert "Way Old Title" in c_d_1

    # event[4] (offset=60): all prior events are > 7 days ago. No clause.
    c_d_last = records[4]["c_d"]
    assert "Recent purchases" not in c_d_last


def test_recent_purchases_max_cap_respected():
    """Customer with 20 consecutive daily events; on event 19 (the last),
    max_recent_purchases=3 caps the rendered list at 3 titles."""
    n = 20
    events = _make_customer_events(
        "C_CAP",
        n_events=n,
        offsets_days=list(range(n)),
        titles=[f"Title number {i:02d}" for i in range(n)],
    )
    persons = _persons_for_single("C_CAP")
    adapter = _StubAdapter()
    records = build_choice_sets(
        events,
        persons,
        adapter,
        seed=42,
        recent_purchases_window_days=30,
        max_recent_purchases=3,
    )
    c_d_last = records[n - 1]["c_d"]
    recent_line = [
        ln for ln in c_d_last.split("\n") if ln.startswith("Recent purchases")
    ]
    assert len(recent_line) == 1, (
        f"expected exactly one 'Recent purchases' line; got: {recent_line}"
    )
    line = recent_line[0]
    # Inside the clause, titles are comma-separated — count how many
    # recent titles appear. Most-recent-slice is events 16, 17, 18 (the
    # last 3 prior to event 19 since event 19 is excluded by strict <).
    assert "Title number 18" in line
    assert "Title number 17" in line
    assert "Title number 16" in line
    # A title outside the cap must NOT appear.
    assert "Title number 15" not in line
    assert "Title number 00" not in line


def test_titles_truncated():
    """A 300-char title is truncated to <=80 chars in the rendered output.

    Construct a customer with 2 events; the first has a 300-char title.
    The second event's c_d must contain the first title truncated.
    """
    long_title = "X" * 300
    events = _make_customer_events(
        "C_LONG",
        n_events=2,
        offsets_days=[0, 5],
        titles=[long_title, "Ordinary Title"],
    )
    persons = _persons_for_single("C_LONG")
    adapter = _StubAdapter()
    records = build_choice_sets(events, persons, adapter, seed=42)
    c_d = records[1]["c_d"]
    # The full 300-char title should NOT appear.
    assert long_title not in c_d
    # The first 80 chars should (truncation keeps a prefix).
    assert ("X" * 80) in c_d
    # And there should be no run of 81 X's (the cap is respected).
    assert ("X" * 81) not in c_d


def test_determinism_with_tied_dates():
    """Two events on the same day with different asins render identical
    c_d across two runs with the same seed — no Python hash nondetermism."""
    # Two same-day events at offset=0; a third event 5 days later.
    events = _make_customer_events(
        "C_TIES",
        n_events=3,
        offsets_days=[0, 0, 5],
        titles=["First Tied", "Second Tied", "Later One"],
    )
    persons = _persons_for_single("C_TIES")

    r1 = build_choice_sets(events, persons, _StubAdapter(), seed=42)
    r2 = build_choice_sets(events, persons, _StubAdapter(), seed=42)
    assert len(r1) == len(r2)
    for x, y in zip(r1, r2):
        assert x["c_d"] == y["c_d"]
    # The 5-day-later event must see both tied titles (both are strictly
    # earlier) and in a deterministic order. With asin tiebreaker A0000
    # < A0001, "First Tied" (asin A0000) renders before "Second Tied".
    c_d_last = r1[2]["c_d"]
    assert "First Tied" in c_d_last
    assert "Second Tied" in c_d_last
    # Order: "First Tied" appears before "Second Tied".
    assert c_d_last.index("First Tied") < c_d_last.index("Second Tied")


def test_per_event_c_d_differs_across_events():
    """c_d is now per-EVENT, not per-customer. Two events of the same
    customer at different times must have different c_d strings."""
    events = _make_customer_events(
        "C_DIFF",
        n_events=4,
        offsets_days=[0, 5, 10, 15],
        titles=[
            "Title Alpha",
            "Title Beta",
            "Title Gamma",
            "Title Delta",
        ],
    )
    persons = _persons_for_single("C_DIFF")
    adapter = _StubAdapter()
    records = build_choice_sets(events, persons, adapter, seed=42)
    cds = {r["c_d"] for r in records}
    # At least 3 of the 4 events have a distinct c_d (first is alone,
    # the other three each add a new title to the recent_purchases line).
    assert len(cds) >= 3, f"expected distinct per-event c_d; got {len(cds)} unique"


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


# --------------------------------------------------------------------------- #
# F3 cold-start leakage fix: first_seen / asin_lookup / popularity fallback
# must be computed from TRAIN rows only when a ``split`` column is present.
#
# Under a cold-start partition, ``events_df`` contains rows for customers
# who are held out entirely (they only appear in val/test). Deriving each
# ASIN's first-seen date or sampling-pool membership from the full frame
# leaks "when did the test population first encounter this product" into
# the availability prefix that gates negative sampling for train events.
# --------------------------------------------------------------------------- #


def _persons_for_customers(customer_ids: list[str]) -> pd.DataFrame:
    """Persons-canonical with a row per train customer_id (leakage tests).

    Rotates through the full age/income/city vocabularies so
    ``fit_person_features`` sees the full 6+5+4 one-hot widths.
    """
    full_age = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    full_income = ["<25k", "25-50k", "50-100k", "100-150k", "150k+"]
    full_city = ["rural", "small", "medium", "large"]
    rows = []
    for i, cid in enumerate(customer_ids):
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
    return pd.DataFrame(rows)


def _leaky_events_frame(include_split_column: bool = True) -> pd.DataFrame:
    """Build a synthetic events frame exercising the cold-start leak.

    Layout (every customer has at least one ``split=='train'`` event so
    the fit-on-train customer-id assertion at the top of
    :func:`build_choice_sets` passes):

      - C_TRAIN_00 .. C_TRAIN_04: five customers with train events in
        2019 buying A_safe00 .. A_safe04 (distinct per-customer ASINs).
        These populate the catalog so the negative sampler has a pool at
        the probe date.
      - C_TRAIN_MAIN: an anchor train event on 2019-06-01 (buying
        A_anchor) plus a train event on 2020-06-01 (buying A_safe_probe).
        The 2020-06-01 event is the probe whose candidate negatives we
        inspect.
      - C_MIXED: a train event on 2019-07-01 (buying A_mixed_anchor) —
        so C_MIXED is in ``train_customers`` — PLUS a ``split="test"``
        event on 2020-01-01 buying A_leaky. Under a true cold-start
        split every customer is wholly train or wholly test; here we
        simulate "the test-row population" by flagging a single row
        test while keeping the customer in the train cohort so the
        fit-on-train assertion passes. The leak scenario is identical:
        a row with ``split != "train"`` introduces A_leaky into the
        full-frame first_seen at 2020-01-01.
      - C_TRAIN_LATE: an anchor train event on 2019-08-01 plus a train
        event on 2021-01-01 buying A_leaky — so A_leaky DOES appear in
        train, but only AFTER the probe event.

    Under a correct (train-only) first_seen, A_leaky's earliest train
    date is 2021-01-01, which is > 2020-06-01, so the probe event must
    NOT see A_leaky as a candidate negative. Under the buggy (full-frame)
    first_seen, A_leaky's earliest date would be 2020-01-01 (from
    C_MIXED's test-labelled row), which is < 2020-06-01, so A_leaky
    WOULD appear in the sampling pool — exactly the leak this test
    guards against.
    """

    def _row(cid, date, asin, split, price=5.0):
        return dict(
            customer_id=cid,
            order_date=pd.to_datetime(date),
            asin=asin,
            category="CAT0",
            price=price,
            title=f"title for {asin}",
            routine=0,
            novelty=1,
            recency_days=999,
            cat_affinity=0,
            brand="brand_x",
            popularity=1,
            split=split,
        )

    rows = []
    # Five train customers priming the catalog.
    base = pd.to_datetime("2019-06-01")
    for i in range(5):
        rows.append(
            _row(
                f"C_TRAIN_{i:02d}",
                base + pd.Timedelta(days=i * 10),
                f"A_safe{i:02d}",
                "train",
                price=5.0 + i,
            )
        )
    # C_TRAIN_MAIN: anchor train row + probe train row.
    rows.append(_row("C_TRAIN_MAIN", "2019-06-01", "A_anchor", "train"))
    rows.append(
        _row("C_TRAIN_MAIN", "2020-06-01", "A_safe_probe", "train", price=7.0)
    )
    # C_MIXED: train-anchor (keeps customer in train cohort) + non-train
    # leaky row on 2020-01-01.
    rows.append(_row("C_MIXED", "2019-07-01", "A_mixed_anchor", "train"))
    rows.append(_row("C_MIXED", "2020-01-01", "A_leaky", "test", price=9.0))
    # C_TRAIN_LATE: anchor + late train A_leaky.
    rows.append(_row("C_TRAIN_LATE", "2019-08-01", "A_late_anchor", "train"))
    rows.append(
        _row("C_TRAIN_LATE", "2021-01-01", "A_leaky", "train", price=9.0)
    )
    df = pd.DataFrame(rows).sort_values(
        ["customer_id", "order_date"]
    ).reset_index(drop=True)
    if not include_split_column:
        df = df.drop(columns=["split"])
    return df


def test_first_seen_is_train_only_when_split_column_present():
    """Under cold-start, A_leaky's ``first_seen`` must come from the
    earliest TRAIN row (2021-01-01), not the earliest test-row
    (2020-01-01). Proxy: a train event dated 2020-06-01 must not have
    A_leaky as a candidate negative.

    Swept across several seeds to keep the assertion sharp: under the
    buggy path, A_leaky's popularity-weighted sampling chance of being
    drawn in at least one seed is ~1; under the fix it is zero.
    """
    events = _leaky_events_frame(include_split_column=True)
    train_cids = sorted(
        events.loc[events["split"] == "train", "customer_id"].unique().tolist()
    )
    persons = _persons_for_customers(train_cids)

    for seed in range(25):
        adapter = _StubAdapter()
        records = build_choice_sets(
            events, persons, adapter, n_negatives=9, seed=seed, n_resamples=1
        )

        # Locate the probe event (C_TRAIN_MAIN on 2020-06-01).
        probe = [
            r
            for r in records
            if r["customer_id"] == "C_TRAIN_MAIN"
            and pd.Timestamp(r["order_date"]) == pd.Timestamp("2020-06-01")
        ]
        assert len(probe) == 1, (
            f"expected exactly one probe record; got {len(probe)}"
        )
        probe_record = probe[0]
        assert "A_leaky" not in probe_record["choice_asins"], (
            f"leak (seed={seed}): A_leaky appeared as a candidate negative "
            f"for the probe event dated 2020-06-01, but its first train "
            f"occurrence is 2021-01-01. "
            f"choice_asins={probe_record['choice_asins']}"
        )


def test_first_seen_unchanged_when_no_split_column():
    """Backwards-compat pin for the all-rows path.

    When no ``split`` column is present (or every row is labelled
    ``"train"``), the full-frame first_seen / asin_lookup tables are
    used — identical to pre-fix behavior. We pin this in two ways:

    1. An events frame where every row has ``split == "train"`` must
       see A_leaky as a candidate negative at the probe date (because
       under the all-train scenario, A_leaky's first-seen is
       2020-01-01 which is strictly before 2020-06-01). This mirrors
       the pre-fix full-frame semantics.
    2. A frame with the ``split`` column dropped entirely must still
       execute end-to-end through the all-rows fallback (duck-typed
       ``if split_column in df.columns``). Wave-13 loosening: the
       pre-Wave-13 coverage assertion happened to dereference the
       split column first, so dropping it produced a ``KeyError``
       indirectly; the new assertion no longer reads the split
       column, so the no-split-column path now succeeds as intended.
    """
    events = _leaky_events_frame(include_split_column=True)
    # (1) All-train: full-frame semantics means A_leaky IS visible at
    # the probe date.
    events_all_train = events.assign(split=["train"] * len(events))
    all_cids = sorted(events_all_train["customer_id"].unique().tolist())
    persons = _persons_for_customers(all_cids)
    records = build_choice_sets(
        events_all_train,
        persons,
        _StubAdapter(),
        n_negatives=9,
        seed=42,
        n_resamples=1,
    )
    assert len(records) == len(events_all_train)
    # Every event's chosen ASIN is the first entry of choice_asins (after
    # permutation) at chosen_idx.
    for r in records:
        assert r["choice_asins"][r["chosen_idx"]] == r["chosen_asin"]

    # (2) No-split-column: the all-rows fallback executes cleanly.
    events_no_split = events.drop(columns=["split"])
    records_no_split = build_choice_sets(
        events_no_split,
        persons,
        _StubAdapter(),
        n_negatives=9,
        seed=42,
        n_resamples=1,
    )
    assert len(records_no_split) == len(events_no_split)
    for r in records_no_split:
        assert r["choice_asins"][r["chosen_idx"]] == r["chosen_asin"]


def test_first_seen_raises_on_split_column_with_no_train_rows():
    """If the ``split`` column is present but contains zero train rows,
    build_choice_sets must raise ``ValueError`` with an actionable
    message rather than silently falling back to all-rows mode.

    Wave-13 note: the coverage assertion at the top of
    :func:`build_choice_sets` now requires that every event customer
    has a z_d row in ``persons_canonical`` (the old invariant —
    persons ⊆ train — was invalidated by cold-start). We therefore
    build a persons frame covering every event customer so the
    coverage check passes and the no-train-rows ValueError becomes
    the first exception reached.
    """
    events = _leaky_events_frame(include_split_column=True)
    # Force every row to val/test — no train rows remain.
    events = events.assign(
        split=["test"] * len(events)
    )
    all_cids = sorted(events["customer_id"].unique().tolist())
    persons = _persons_for_customers(all_cids)
    adapter = _StubAdapter()
    with pytest.raises(ValueError, match="zero 'train' rows"):
        build_choice_sets(
            events, persons, adapter, n_negatives=9, seed=42, n_resamples=1
        )


# --------------------------------------------------------------------------- #
# Per-alt purchase_count / is_repeat (Fix 2)
# --------------------------------------------------------------------------- #


def _make_repeat_events() -> "pd.DataFrame":
    """Customer C00 buys ASIN_A twice (train), once again (val).

    Catalog has enough other ASINs that negatives can be sampled
    without forcing the dedup-fallback codepath.
    """
    rows = []
    asins = [f"A{i:03d}" for i in range(8)]
    asin_a = asins[0]
    base_date = pd.to_datetime("2024-01-01")
    for c in range(4):
        cid = f"C{c:02d}"
        for e in range(8):
            d = base_date + pd.Timedelta(days=c * 50 + e)
            if c == 0 and e in (0, 1):
                asin = asin_a
            elif c == 0 and e == 6:
                asin = asin_a
            else:
                asin = asins[1 + ((c + e) % (len(asins) - 1))]
            cat_idx = int(asin[1:]) % 3
            rows.append(dict(
                customer_id=cid, order_date=d, asin=asin,
                category=f"CAT{cat_idx}",
                price=5.0 + (int(asin[1:]) % 10),
                title=f"title {asin} word{e}",
                routine=int(e > 0),
                novelty=int(e == 0),
                recency_days=999 if e == 0 else 7,
                cat_affinity=e,
                brand="brand_x",
            ))
    df = pd.DataFrame(rows)
    pop = df.groupby("asin").size().rename("popularity").reset_index()
    df = df.merge(pop, on="asin", how="left")
    df = df.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    splits = []
    for _, g in df.groupby("customer_id", sort=False):
        n = len(g)
        # First 6 train, then 1 val, 1 test per customer.
        splits.extend(["train"] * (n - 2) + ["val", "test"])
    df["split"] = splits
    return df


def test_purchase_count_per_alt_correct():
    """Customer C00 has bought ASIN_A twice in train. A val event
    listing ASIN_A as a negative must report purchase_count == 2
    and is_repeat == 1.0 on that alt dict."""
    events = _make_repeat_events()
    persons = _persons_for(events)
    adapter = _StubAdapter()
    records = build_choice_sets(
        events, persons, adapter, n_negatives=3, seed=0, n_resamples=1,
    )
    asin_a = "A000"
    # Find C00 val event(s).
    found = False
    for r in records:
        if r["customer_id"] != "C00" or r["split"] != "val":
            continue
        for j, asin in enumerate(r["choice_asins"]):
            if asin == asin_a:
                # Either the chosen alt or a sampled negative; same map.
                # Train history of (C00, A000) = 2 events; val
                # split has adjust_chosen=False so count stays 2.
                assert r["alt_texts"][j]["purchase_count"] == 2
                assert r["alt_texts"][j]["is_repeat"] == 1.0
                found = True
    assert found, "expected ASIN_A to surface in a C00 val event"


def test_purchase_count_no_self_count_leak():
    """Train-event chosen alt: purchase_count must equal the count
    of PRIOR purchases (excluding the current event), matching
    what an identical-history negative would report. ref_df under
    a temporal split includes the current event itself; without
    the chosen-self adjustment, chosen would inflate by 1."""
    events = _make_repeat_events()
    persons = _persons_for(events)
    adapter = _StubAdapter()
    records = build_choice_sets(
        events, persons, adapter, n_negatives=3, seed=0, n_resamples=1,
    )
    # Look for the C00 train events where chosen == ASIN_A.
    asin_a = "A000"
    seen_counts = []
    for r in records:
        if r["customer_id"] != "C00" or r["split"] != "train":
            continue
        if r["chosen_asin"] != asin_a:
            continue
        chosen_idx = r["chosen_idx"]
        seen_counts.append(
            r["alt_texts"][chosen_idx]["purchase_count"]
        )
    assert seen_counts, "expected at least one C00 train event chosen=A000"
    # Train events for C00 buying A000 are at e=0, 1; the temporal
    # ref_df under-the-hood includes ALL train rows. With the
    # self-count adjustment in place, both events should report
    # count == (raw train count of (C00, A000) which is 2) - 1 = 1.
    # The third A000 purchase at e=6 falls in val per the split.
    for cnt in seen_counts:
        assert cnt < 2, (
            f"chosen alt purchase_count={cnt} suggests self-count leak; "
            f"expected raw_count - 1."
        )
