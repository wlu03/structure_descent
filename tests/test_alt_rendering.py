"""Unit tests for ``src/data/alt_rendering.py`` (Wave 10, design doc §2)."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import pandas as pd
import pytest

from src.data.adapter import AmazonAdapter
from src.data.alt_rendering import (
    BAND_LABELS,
    build_popularity_percentile_fn,
    register_on_adapter,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
EVENTS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_events_100.csv"
PERSONS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_persons_100.csv"


# --------------------------------------------------------------------------- #
# build_popularity_percentile_fn
# --------------------------------------------------------------------------- #


def test_percentile_fn_bands_monotone():
    """Popularities exactly 1..100 → textbook quantiles.

    With asin popularities 1..100 (unique), numpy quantile at 0.95 = 95.05,
    at 0.75 = 75.25, at 0.50 = 50.5. So:
      100 >= 95.05  → "top 5%"
      80  >= 75.25  → "top 25%"
      60  >= 50.5   → "top 50%"
      20  <  50.5   → "bottom 50%"
    """
    df = pd.DataFrame(
        {"asin": [f"A{i:03d}" for i in range(1, 101)], "popularity": list(range(1, 101))}
    )
    fn = build_popularity_percentile_fn(df)
    assert fn(100) == "top 5%"
    assert fn(80) == "top 25%"
    assert fn(60) == "top 50%"
    assert fn(20) == "bottom 50%"


def test_percentile_fn_uses_unique_asins():
    """Event-frequency must NOT weight the distribution.

    Construct events where asin "A" appears 100 times with popularity=1
    and asin "B" appears once with popularity=100. De-duplication on
    asin means the quantile computation sees popularities {1, 100} (two
    rows), not {1 × 100, 100} (101 rows where 1 dominates).

    Over {1, 100}: q95 = 95.05, q75 = 75.25, q50 = 50.5.
    So 100 → "top 5%"; 1 → "bottom 50%".
    """
    rows = [{"asin": "A", "popularity": 1}] * 100 + [{"asin": "B", "popularity": 100}]
    df = pd.DataFrame(rows)
    fn = build_popularity_percentile_fn(df)
    assert fn(100) == "top 5%"
    assert fn(1) == "bottom 50%"


def test_percentile_fn_empty_raises():
    """Empty DataFrame → ValueError, message references popularity col."""
    df = pd.DataFrame({"asin": [], "popularity": []})
    with pytest.raises(ValueError, match="popularity"):
        build_popularity_percentile_fn(df)


def test_percentile_fn_missing_column_raises():
    """Missing popularity column → KeyError."""
    df = pd.DataFrame({"asin": ["A", "B", "C"], "other": [1, 2, 3]})
    with pytest.raises(KeyError, match="popularity"):
        build_popularity_percentile_fn(df)


def test_percentile_fn_all_equal_does_not_crash():
    """When all asins have the same popularity, quantiles collapse.

    q95 == q75 == q50 == 1; any ``n >= 1`` falls through to the
    top-most matching clause. The spec only requires that the function
    returns a valid band string without crashing — we don't assert
    which band.
    """
    df = pd.DataFrame(
        {"asin": [f"A{i}" for i in range(20)], "popularity": [1] * 20}
    )
    fn = build_popularity_percentile_fn(df)
    band = fn(1)
    assert band in BAND_LABELS
    # Also check a query above / below the equal value.
    assert fn(0) in BAND_LABELS
    assert fn(5) in BAND_LABELS


def test_percentile_fn_is_picklable():
    """``pickle.dumps / loads`` round-trips a usable function.

    Callers may snapshot the adapter state (including the percentile
    function) for reproducibility manifests, so the closure must be
    picklable across a fresh-interpreter round-trip.
    """
    df = pd.DataFrame(
        {"asin": [f"A{i}" for i in range(1, 101)], "popularity": list(range(1, 101))}
    )
    fn = build_popularity_percentile_fn(df)

    # pickle round-trip.
    payload = pickle.dumps(fn)
    fn2 = pickle.loads(payload)
    assert fn2(100) == "top 5%"
    assert fn2(80) == "top 25%"
    assert fn2(60) == "top 50%"
    assert fn2(20) == "bottom 50%"

    # copy.deepcopy round-trip.
    fn3 = copy.deepcopy(fn)
    assert fn3(100) == "top 5%"
    assert fn3(20) == "bottom 50%"


# --------------------------------------------------------------------------- #
# register_on_adapter
# --------------------------------------------------------------------------- #


def test_register_on_adapter_updates_alt_text():
    """Before register: Wave-9 stub. After register: a valid band string."""
    adapter = AmazonAdapter()

    event_row = {
        "title": "x",
        "category": "y",
        "price": 1.0,
        "popularity": 100,
    }

    # Pre-register: stub wording.
    assert adapter._popularity_percentile_fn is None
    pre = adapter.alt_text(event_row)
    assert pre["popularity_rank"] == "popularity score 100"

    # Register.
    train_events = pd.DataFrame(
        {"asin": [f"A{i}" for i in range(1, 101)], "popularity": list(range(1, 101))}
    )
    register_on_adapter(adapter, train_events)
    assert adapter._popularity_percentile_fn is not None

    # Post-register: one of the four bands.
    post = adapter.alt_text(event_row)
    assert post["popularity_rank"] in BAND_LABELS
    assert post["title"] == "x"
    assert post["category"] == "y"
    assert post["price"] == 1.0
    # For popularity=100 on the 1..100 distribution: "top 5%".
    assert post["popularity_rank"] == "top 5%"


@pytest.mark.skipif(
    not EVENTS_FIXTURE.exists() or not PERSONS_FIXTURE.exists(),
    reason="Amazon fixtures not available",
)
def test_amazon_fixture_end_to_end():
    """load -> clean -> state_features -> split -> popularity -> register.

    Runs the full Wave-8 preprocessing pipeline against the 100-row
    Amazon fixture, then wires the popularity-percentile fn onto a
    ``YamlAdapter`` pointing at the same fixture, and verifies a
    random event row's ``alt_text`` yields a valid band label.
    """
    import tempfile
    import yaml

    from src.data.adapter import YamlAdapter
    from src.data.clean import clean_events
    from src.data.split import temporal_split
    from src.data.state_features import (
        attach_train_popularity,
        compute_state_features,
    )
    from src.data.survey_join import join_survey

    # Redirect the Amazon YAML to the fixture paths (same pattern as the
    # adapter / vocab-fix tests).
    with (REPO_ROOT / "configs" / "datasets" / "amazon.yaml").open("r") as f:
        cfg = yaml.safe_load(f)
    cfg["dataset"]["events"]["path"] = str(EVENTS_FIXTURE)
    cfg["dataset"]["persons"]["path"] = str(PERSONS_FIXTURE)
    # Keep the external_lookup path absolute so suppress logic stays consistent.
    for col_spec in cfg["dataset"]["persons"]["z_d_mapping"].values():
        if isinstance(col_spec, dict) and col_spec.get("kind") == "external_lookup":
            p = col_spec.get("lookup_path")
            if p is not None and not Path(p).is_absolute():
                col_spec["lookup_path"] = str(REPO_ROOT / p)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_yaml = Path(tmp) / "amazon_fixture.yaml"
        tmp_yaml.write_text(yaml.dump(cfg))
        adapter = YamlAdapter(tmp_yaml)

        events_raw = adapter.load_events()
        persons_raw = adapter.load_persons()
        cleaned = clean_events(events_raw, adapter.schema)
        joined = join_survey(cleaned, persons_raw, adapter.schema)
        with_state = compute_state_features(joined)
        split_df = temporal_split(with_state, adapter.schema)
        final = attach_train_popularity(split_df)

        train_df = final.loc[final["split"] == "train"].copy()
        assert len(train_df) > 0
        assert "popularity" in train_df.columns

        # Wire the popularity-percentile fn.
        register_on_adapter(adapter, train_df)
        assert adapter._popularity_percentile_fn is not None

        # Pick a random event row (deterministic via a fixed seed) and
        # render alt_text. The popularity must map to a valid band.
        sample = final.sample(n=1, random_state=0).iloc[0]
        event_row = {
            "title": sample.get("title", ""),
            "category": sample.get("category", ""),
            "price": float(sample.get("price", 0.0) or 0.0),
            "popularity": int(sample.get("popularity", 0) or 0),
        }
        out = adapter.alt_text(event_row)
        assert out["popularity_rank"] in BAND_LABELS
