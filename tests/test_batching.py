"""Tests for ``src/data/batching.py`` (Wave 10 glue module).

Covers:

* Happy-path shape contract (synthetic records + stubs).
* L2-norm-on-last-axis invariant on ``E``.
* Price tensor round-trip from ``alt_texts``.
* Cache wiring (miss-then-hit on the outcomes cache;
  encoder.encode called exactly once per ``assemble_batch``).
* Invariants (z_d width drift, NaN in E, c_star OOB, negative prices).
* Diversity-filter forwarding.
* Amazon 100-row fixture end-to-end.
* ``iter_to_torch_batches`` forwards ``prices``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.batching import AssembledBatch, assemble_batch, iter_to_torch_batches
from src.outcomes.encode import StubEncoder
from src.outcomes.generate import StubLLMClient, GenerationResult, OutcomesPayload


REPO_ROOT = Path(__file__).resolve().parent.parent
EVENTS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_events_100.csv"
PERSONS_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "amazon_persons_100.csv"
AMAZON_YAML = REPO_ROOT / "configs" / "datasets" / "amazon.yaml"


# --------------------------------------------------------------------------- #
# Helpers: synthetic records with the same schema as build_choice_sets output.
# --------------------------------------------------------------------------- #


def _make_records(
    N: int = 4,
    J: int = 5,
    p: int = 26,
    *,
    z_d_widths: list[int] | None = None,
    prices: list[list[float]] | None = None,
    chosen_idx: list[int] | None = None,
) -> list[dict]:
    """Build ``N`` hand-crafted records matching the build_choice_sets schema.

    Every record carries the keys documented in
    ``src/data/choice_sets.py``: ``customer_id``, ``chosen_asin``,
    ``choice_asins``, ``chosen_idx``, ``z_d``, ``c_d``, ``alt_texts``
    plus a minimal ``chosen_features`` / ``metadata`` / ``order_date``
    / ``category`` shim (unused by the batcher but present for
    realism).
    """
    rng = np.random.default_rng(0)
    records: list[dict] = []
    for i in range(N):
        width = p if z_d_widths is None else z_d_widths[i]
        z = rng.standard_normal(width).astype(np.float32)
        cs_idx = (i % J) if chosen_idx is None else chosen_idx[i]
        choice_asins = [f"A{i}_{j:02d}" for j in range(J)]
        chosen_asin = choice_asins[cs_idx]
        alt_texts = []
        for j in range(J):
            pr = (5.0 + j) if prices is None else prices[i][j]
            alt_texts.append(
                {
                    "title": f"title {chosen_asin if j == cs_idx else choice_asins[j]}",
                    "category": "cat",
                    "price": pr,
                    "popularity_rank": "top 50%",
                }
            )
        records.append(
            {
                "customer_id": f"C{i:03d}",
                "chosen_asin": chosen_asin,
                "choice_asins": choice_asins,
                "chosen_idx": cs_idx,
                "z_d": z,
                "c_d": f"Person context for C{i:03d}. Lives somewhere.",
                "alt_texts": alt_texts,
                # v1 padding fields (not consumed by batcher)
                "order_date": pd.Timestamp("2024-01-01"),
                "category": "cat",
                "chosen_features": {},
                "metadata": {},
            }
        )
    return records


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


def test_assemble_shapes():
    """4 records x J=5 x K=3 x d_e=768 x p=26."""
    records = _make_records(N=4, J=5, p=26)
    llm = StubLLMClient()
    enc = StubEncoder(d_e=768)
    batch = assemble_batch(
        records,
        adapter=None,
        llm_client=llm,
        encoder=enc,
        outcomes_cache=None,
        embeddings_cache=None,
        K=3,
    )
    assert isinstance(batch, AssembledBatch)
    assert len(batch) == 4
    assert tuple(batch.z_d.shape) == (4, 26)
    assert tuple(batch.E.shape) == (4, 5, 3, 768)
    assert tuple(batch.c_star.shape) == (4,)
    assert tuple(batch.omega.shape) == (4,)
    assert tuple(batch.prices.shape) == (4, 5)
    # dtypes
    assert batch.z_d.dtype == torch.float32
    assert batch.E.dtype == torch.float32
    assert batch.c_star.dtype == torch.int64
    assert batch.omega.dtype == torch.float32
    assert batch.prices.dtype == torch.float32
    # Default omega: all ones.
    assert torch.allclose(batch.omega, torch.ones(4))


def test_assemble_with_stubs_end_to_end():
    """Every tensor shape valid + E rows L2-normalized on last dim."""
    records = _make_records(N=3, J=4, p=26)
    batch = assemble_batch(
        records,
        adapter=None,
        llm_client=StubLLMClient(),
        encoder=StubEncoder(d_e=768),
        outcomes_cache=None,
        embeddings_cache=None,
        K=3,
        seed=42,
    )
    # Last-axis L2-norm invariant on E (StubEncoder produces L2-normalized rows).
    norms = batch.E.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # Diagnostics shapes.
    assert len(batch.customer_ids) == 3
    assert len(batch.chosen_asins) == 3
    assert len(batch.outcomes_nested) == 3
    assert all(len(alt_list) == 4 for alt_list in batch.outcomes_nested)
    assert all(len(k_list) == 3 for alt_list in batch.outcomes_nested for k_list in alt_list)


def test_c_star_in_range():
    """Every c_star value must sit in [0, J)."""
    records = _make_records(N=6, J=5, p=26)
    batch = assemble_batch(
        records,
        adapter=None,
        llm_client=StubLLMClient(),
        encoder=StubEncoder(d_e=768),
        outcomes_cache=None,
        embeddings_cache=None,
        K=3,
    )
    assert int(batch.c_star.min().item()) >= 0
    assert int(batch.c_star.max().item()) < 5


def test_prices_from_alt_texts():
    """batch.prices[i, j] == records[i]['alt_texts'][j]['price']."""
    price_grid = [
        [1.5, 2.5, 3.5, 4.5],
        [10.0, 20.0, 30.0, 40.0],
    ]
    records = _make_records(N=2, J=4, p=26, prices=price_grid)
    batch = assemble_batch(
        records,
        adapter=None,
        llm_client=StubLLMClient(),
        encoder=StubEncoder(d_e=768),
        outcomes_cache=None,
        embeddings_cache=None,
        K=3,
    )
    for i in range(2):
        for j in range(4):
            assert float(batch.prices[i, j].item()) == pytest.approx(price_grid[i][j])


# --------------------------------------------------------------------------- #
# Cache wiring
# --------------------------------------------------------------------------- #


class _CountingLLMClient:
    """Wrap StubLLMClient and tally generate() calls."""

    def __init__(self) -> None:
        self._inner = StubLLMClient()
        self.n_calls = 0

    def generate(self, *args: Any, **kwargs: Any) -> GenerationResult:
        self.n_calls += 1
        return self._inner.generate(*args, **kwargs)


def test_cache_miss_then_hit(tmp_path: Path):
    from src.outcomes.cache import OutcomesCache, EmbeddingsCache

    records = _make_records(N=3, J=4, p=26)
    o_cache = OutcomesCache(tmp_path / "out.sqlite")
    e_cache = EmbeddingsCache(tmp_path / "emb.sqlite")

    llm = _CountingLLMClient()
    enc = StubEncoder(d_e=768)

    assemble_batch(
        records,
        adapter=None,
        llm_client=llm,
        encoder=enc,
        outcomes_cache=o_cache,
        embeddings_cache=e_cache,
        K=3,
        seed=99,
    )
    # First assemble: exactly N * J = 12 generate calls.
    assert llm.n_calls == 3 * 4

    # Second assemble with the same (customer_id, asin, seed, prompt_version)
    # keys must hit the outcomes cache every time -> zero new LLM calls.
    llm.n_calls = 0
    assemble_batch(
        records,
        adapter=None,
        llm_client=llm,
        encoder=enc,
        outcomes_cache=o_cache,
        embeddings_cache=e_cache,
        K=3,
        seed=99,
    )
    assert llm.n_calls == 0


class _CountingEncoder:
    """Wrap StubEncoder and tally encode() calls."""

    def __init__(self, d_e: int = 64) -> None:
        self._inner = StubEncoder(d_e=d_e)
        self.n_calls = 0

    @property
    def encoder_id(self) -> str:
        return self._inner.encoder_id

    @property
    def d_e(self) -> int:
        return self._inner.d_e

    def encode(self, texts: list[str]) -> np.ndarray:
        self.n_calls += 1
        return self._inner.encode(texts)


def test_encoder_called_once_per_assemble():
    records = _make_records(N=4, J=3, p=26)
    enc = _CountingEncoder(d_e=64)
    assemble_batch(
        records,
        adapter=None,
        llm_client=StubLLMClient(),
        encoder=enc,
        outcomes_cache=None,
        embeddings_cache=None,
        K=3,
    )
    assert enc.n_calls == 1


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #


def test_invariant_z_d_width_mismatch():
    """One record with a different z_d width -> AssertionError."""
    records = _make_records(N=4, J=3, p=26, z_d_widths=[26, 26, 24, 26])
    with pytest.raises(AssertionError) as exc:
        assemble_batch(
            records,
            adapter=None,
            llm_client=StubLLMClient(),
            encoder=StubEncoder(d_e=64),
            outcomes_cache=None,
            embeddings_cache=None,
            K=3,
        )
    assert "z_d_width_uniform" in str(exc.value)


class _NaNEncoder:
    """Return rows with a NaN in the last position."""

    encoder_id = "nan-encoder"
    d_e = 64

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.random.default_rng(0).standard_normal(
            (len(texts), self.d_e)
        ).astype(np.float32)
        if len(texts) > 0:
            out[0, -1] = np.nan
        # L2-normalize the non-NaN rows best-effort so the shape check passes.
        return out


def test_invariant_nan_in_E():
    records = _make_records(N=2, J=3, p=26)
    with pytest.raises(AssertionError) as exc:
        assemble_batch(
            records,
            adapter=None,
            llm_client=StubLLMClient(),
            encoder=_NaNEncoder(),
            outcomes_cache=None,
            embeddings_cache=None,
            K=3,
        )
    assert "E_no_nan" in str(exc.value)


def test_invariant_c_star_out_of_range():
    records = _make_records(N=2, J=5, p=26)
    # Rewrite chosen_idx on records[0] to J (out of range).
    records[0]["chosen_idx"] = 5
    with pytest.raises(AssertionError) as exc:
        assemble_batch(
            records,
            adapter=None,
            llm_client=StubLLMClient(),
            encoder=StubEncoder(d_e=64),
            outcomes_cache=None,
            embeddings_cache=None,
            K=3,
        )
    assert "c_star_in_range" in str(exc.value)


def test_invariant_bad_prices_shape():
    """Corrupt one alt_texts entry's price to negative -> AssertionError."""
    records = _make_records(N=2, J=3, p=26)
    records[1]["alt_texts"][2]["price"] = -1.0
    with pytest.raises(AssertionError) as exc:
        assemble_batch(
            records,
            adapter=None,
            llm_client=StubLLMClient(),
            encoder=StubEncoder(d_e=64),
            outcomes_cache=None,
            embeddings_cache=None,
            K=3,
        )
    assert "prices_non_negative" in str(exc.value)


# --------------------------------------------------------------------------- #
# Diversity filter forwarding
# --------------------------------------------------------------------------- #


def test_diversity_filter_forwarded():
    """Custom diversity_filter callable must be invoked per (record, alt)."""
    records = _make_records(N=2, J=3, p=26)
    call_log: list[list[str]] = []

    def my_filter(outs: list[str]) -> tuple[list[str], bool]:
        call_log.append(list(outs))
        return outs, True

    assemble_batch(
        records,
        adapter=None,
        llm_client=StubLLMClient(),
        encoder=StubEncoder(d_e=64),
        outcomes_cache=None,
        embeddings_cache=None,
        K=3,
        diversity_filter=my_filter,
    )
    # Called at least once per (record, alt) -> N*J = 6 times.
    assert len(call_log) == 2 * 3


# --------------------------------------------------------------------------- #
# omega handling
# --------------------------------------------------------------------------- #


def test_omega_passthrough():
    records = _make_records(N=3, J=2, p=26)
    omega = np.array([0.5, 2.0, 1.0], dtype=np.float32)
    batch = assemble_batch(
        records,
        adapter=None,
        llm_client=StubLLMClient(),
        encoder=StubEncoder(d_e=64),
        outcomes_cache=None,
        embeddings_cache=None,
        K=3,
        omega=omega,
    )
    assert torch.allclose(batch.omega, torch.tensor([0.5, 2.0, 1.0]))


def test_omega_shape_mismatch_raises():
    records = _make_records(N=3, J=2, p=26)
    omega = np.ones(5, dtype=np.float32)
    with pytest.raises(AssertionError) as exc:
        assemble_batch(
            records,
            adapter=None,
            llm_client=StubLLMClient(),
            encoder=StubEncoder(d_e=64),
            outcomes_cache=None,
            embeddings_cache=None,
            K=3,
            omega=omega,
        )
    assert "omega_shape" in str(exc.value)


# --------------------------------------------------------------------------- #
# iter_to_torch_batches forwards prices
# --------------------------------------------------------------------------- #


def test_iter_to_torch_batches_forwards_prices():
    records = _make_records(N=6, J=4, p=26)
    batch = assemble_batch(
        records,
        adapter=None,
        llm_client=StubLLMClient(),
        encoder=StubEncoder(d_e=64),
        outcomes_cache=None,
        embeddings_cache=None,
        K=3,
    )
    seen_batches = 0
    for mb in iter_to_torch_batches(batch, batch_size=2, shuffle=False):
        assert "prices" in mb
        assert mb["prices"].shape[1] == 4  # J
        assert {"z_d", "E", "c_star", "omega", "prices"} <= set(mb.keys())
        seen_batches += 1
    assert seen_batches == 3  # 6 / 2


# --------------------------------------------------------------------------- #
# Amazon fixture end-to-end
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not EVENTS_FIXTURE.exists()
    or not PERSONS_FIXTURE.exists()
    or not AMAZON_YAML.exists(),
    reason="Amazon fixtures / config not available",
)
def test_amazon_fixture_small_end_to_end():
    """load -> clean -> survey_join -> state_features -> split ->
    attach_train_popularity -> translate_z_d -> build_choice_sets
    (20-record subset) -> assemble_batch (stubs)."""
    from src.data.adapter import YamlAdapter
    from src.data.choice_sets import build_choice_sets
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
    z_df_all = translate_persons(
        persons_raw, schema, training_events=train_events
    )
    train_customers = set(train_events["customer_id"].unique())
    z_df = z_df_all.loc[
        z_df_all["customer_id"].isin(train_customers)
    ].reset_index(drop=True)
    known = set(z_df["customer_id"].unique())
    events_for_cs = with_pop.loc[
        with_pop["customer_id"].isin(known)
    ].reset_index(drop=True)

    adapter = YamlAdapter(AMAZON_YAML)
    # Wave-10 brief: pre-alt_rendering.register_on_adapter, monkeypatch
    # _popularity_percentile_fn so alt_text renders a human-readable
    # "top 50%"-style string even in the smoke test.
    adapter._popularity_percentile_fn = lambda n: "top 50%"

    records = build_choice_sets(
        events_for_cs, z_df, adapter, n_negatives=4, seed=42, n_resamples=1
    )
    # 20-record subset.
    records = records[:20]
    assert len(records) == 20

    batch = assemble_batch(
        records,
        adapter=adapter,
        llm_client=StubLLMClient(),
        encoder=StubEncoder(d_e=64),  # shrink d_e to keep the test fast
        outcomes_cache=None,
        embeddings_cache=None,
        K=3,
        seed=0,
    )
    N = 20
    J = records[0]["choice_asins"].__len__()
    p = records[0]["z_d"].shape[0]
    assert tuple(batch.z_d.shape) == (N, p)
    assert tuple(batch.E.shape) == (N, J, 3, 64)
    assert tuple(batch.prices.shape) == (N, J)
    assert tuple(batch.c_star.shape) == (N,)
    assert torch.isfinite(batch.z_d).all().item()
    assert not torch.isnan(batch.E).any().item()
