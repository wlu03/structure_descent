"""Tests for ``src.outcomes.cache`` (KVStore, OutcomesCache, EmbeddingsCache).

Covers the checklist in the wave-1 task card: round-trips, key stability,
persistence across reopen, dtype/shape validation for embeddings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.outcomes.cache import EmbeddingsCache, KVStore, OutcomesCache


# ----------------------------------------------------------------------
# KVStore
# ----------------------------------------------------------------------
def test_kvstore_roundtrip(tmp_cache_dir: Path) -> None:
    path = tmp_cache_dir / "kv.sqlite"
    kv = KVStore(path)
    try:
        kv.put("k", b"hello")
        assert kv.get("k") == b"hello"
    finally:
        kv.close()


def test_kvstore_len_and_has(tmp_cache_dir: Path) -> None:
    kv = KVStore(tmp_cache_dir / "kv.sqlite")
    try:
        assert len(kv) == 0
        assert kv.has("missing") is False

        kv.put("a", b"1")
        kv.put("b", b"2")
        assert len(kv) == 2
        assert kv.has("a") is True
        assert kv.has("b") is True
        assert kv.has("c") is False

        # Idempotent replace — does not bump length.
        kv.put("a", b"11")
        assert len(kv) == 2
        assert kv.get("a") == b"11"
    finally:
        kv.close()


def test_kvstore_delete(tmp_cache_dir: Path) -> None:
    kv = KVStore(tmp_cache_dir / "kv.sqlite")
    try:
        kv.put("x", b"v")
        assert kv.has("x")
        kv.delete("x")
        assert not kv.has("x")
        assert kv.get("x") is None
        # Deleting a missing key is a no-op.
        kv.delete("never-existed")
    finally:
        kv.close()


def test_kvstore_context_manager(tmp_cache_dir: Path) -> None:
    path = tmp_cache_dir / "kv.sqlite"
    with KVStore(path) as kv:
        kv.put("k", b"v")
        assert kv.get("k") == b"v"
    # After the context closes the connection is released; re-opening works.
    with KVStore(path) as kv2:
        assert kv2.get("k") == b"v"


# ----------------------------------------------------------------------
# OutcomesCache
# ----------------------------------------------------------------------
def test_outcomes_key_stable() -> None:
    k1 = OutcomesCache.outcomes_key("cust-1", "B0001", 7, "v1")
    k2 = OutcomesCache.outcomes_key("cust-1", "B0001", 7, "v1")
    assert k1 == k2
    assert len(k1) == 64  # sha256 hex digest

    # Varying any single field must change the digest.
    assert OutcomesCache.outcomes_key("cust-2", "B0001", 7, "v1") != k1
    assert OutcomesCache.outcomes_key("cust-1", "B0002", 7, "v1") != k1
    assert OutcomesCache.outcomes_key("cust-1", "B0001", 8, "v1") != k1
    assert OutcomesCache.outcomes_key("cust-1", "B0001", 7, "v2") != k1


def test_outcomes_roundtrip(tmp_cache_dir: Path) -> None:
    outcomes = [
        "I save roughly twelve dollars this month.",
        "My commute feels a few minutes shorter.",
        "I feel calmer about the next two weeks.",
    ]
    metadata = {
        "temperature": 0.8,
        "model_id": "stub-7b",
        "finish_reason": "stop",
        "seed": 7,
        "prompt_version": "v1",
        "timestamp": 1_700_000_000.0,
    }

    with OutcomesCache(tmp_cache_dir / "outcomes.sqlite") as cache:
        assert cache.get_outcomes("cust-1", "B0001", 7, "v1") is None
        cache.put_outcomes("cust-1", "B0001", 7, "v1", outcomes, metadata)
        got = cache.get_outcomes("cust-1", "B0001", 7, "v1")

    assert got is not None
    assert got["outcomes"] == outcomes
    assert got["metadata"] == metadata


# ----------------------------------------------------------------------
# EmbeddingsCache
# ----------------------------------------------------------------------
def test_embedding_key_stable() -> None:
    k1 = EmbeddingsCache.embedding_key("a narrative", "mpnet-v2")
    k2 = EmbeddingsCache.embedding_key("a narrative", "mpnet-v2")
    assert k1 == k2
    assert len(k1) == 64

    assert EmbeddingsCache.embedding_key("other", "mpnet-v2") != k1
    assert EmbeddingsCache.embedding_key("a narrative", "other-enc") != k1


def test_embedding_roundtrip(tmp_cache_dir: Path) -> None:
    rng = np.random.default_rng(0)
    vec = rng.standard_normal(768).astype(np.float32)

    with EmbeddingsCache(tmp_cache_dir / "embeddings.sqlite") as cache:
        assert cache.get_embedding("string-A", "mpnet-v2") is None
        cache.put_embedding("string-A", "mpnet-v2", vec)
        got = cache.get_embedding("string-A", "mpnet-v2")

    assert got is not None
    assert got.dtype == np.float32
    assert got.shape == (768,)
    assert np.allclose(got, vec)


def test_embedding_wrong_dtype_raises(tmp_cache_dir: Path) -> None:
    with EmbeddingsCache(tmp_cache_dir / "embeddings.sqlite") as cache:
        bad = np.zeros(768, dtype=np.float64)
        with pytest.raises(ValueError):
            cache.put_embedding("s", "enc", bad)


def test_embedding_2d_raises(tmp_cache_dir: Path) -> None:
    with EmbeddingsCache(tmp_cache_dir / "embeddings.sqlite") as cache:
        bad = np.zeros((4, 192), dtype=np.float32)
        with pytest.raises(ValueError):
            cache.put_embedding("s", "enc", bad)


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def test_cache_persists_across_reopen(tmp_cache_dir: Path) -> None:
    outcomes_path = tmp_cache_dir / "outcomes.sqlite"
    embeddings_path = tmp_cache_dir / "embeddings.sqlite"

    outcomes = ["I save a few dollars.", "I feel slightly less stressed.", "My evening is simpler."]
    metadata = {"model_id": "stub", "temperature": 0.8, "seed": 1, "prompt_version": "v1"}

    vec = np.arange(768, dtype=np.float32)

    # First session — write.
    with OutcomesCache(outcomes_path) as oc:
        oc.put_outcomes("cust-A", "ASIN-1", 1, "v1", outcomes, metadata)
    with EmbeddingsCache(embeddings_path) as ec:
        ec.put_embedding("outcome string", "encoder-Z", vec)

    # Second session — read back from disk.
    with OutcomesCache(outcomes_path) as oc:
        got = oc.get_outcomes("cust-A", "ASIN-1", 1, "v1")
    with EmbeddingsCache(embeddings_path) as ec:
        got_vec = ec.get_embedding("outcome string", "encoder-Z")

    assert got is not None
    assert got["outcomes"] == outcomes
    assert got["metadata"] == metadata

    assert got_vec is not None
    assert got_vec.dtype == np.float32
    assert got_vec.shape == (768,)
    assert np.allclose(got_vec, vec)
