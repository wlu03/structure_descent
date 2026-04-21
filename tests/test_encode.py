"""Tests for :mod:`src.outcomes.encode` (redesign.md §4).

Every test uses :class:`~src.outcomes.encode.StubEncoder` to keep the
suite hermetic — :class:`~src.outcomes.encode.SentenceTransformersEncoder`
is explicitly out of scope for the test suite (it would require loading
a real model at test time).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.outcomes.cache import EmbeddingsCache
from src.outcomes.encode import (
    StubEncoder,
    encode_batch,
    encode_outcomes_tensor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class _CountingEncoder:
    """Wraps a real encoder and counts the total strings it encodes."""

    def __init__(self, inner: StubEncoder) -> None:
        self._inner = inner
        self.calls = 0            # number of encode() invocations
        self.encoded_count = 0    # total strings encoded across calls

    @property
    def encoder_id(self) -> str:
        return self._inner.encoder_id

    @property
    def d_e(self) -> int:
        return self._inner.d_e

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        self.encoded_count += len(texts)
        return self._inner.encode(texts)


# ---------------------------------------------------------------------------
# StubEncoder basics
# ---------------------------------------------------------------------------


def test_stub_shape_and_dtype() -> None:
    enc = StubEncoder()
    out = enc.encode(["a", "b", "c"])
    assert out.shape == (3, 768)
    assert out.dtype == np.float32


def test_stub_unit_norm() -> None:
    enc = StubEncoder()
    out = enc.encode(
        ["hello", "the quick brown fox", "", "PO-LEU is an acronym"]
    )
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_stub_deterministic() -> None:
    enc = StubEncoder()
    texts = ["alpha", "beta", "gamma"]
    a = enc.encode(texts)
    b = enc.encode(texts)
    assert np.allclose(a, b)


def test_stub_distinguishes_strings() -> None:
    enc = StubEncoder()
    out = enc.encode(["cat", "the quick brown fox"])
    sim = _cosine(out[0], out[1])
    assert sim < 0.999, f"expected cos sim < 0.999, got {sim}"


# ---------------------------------------------------------------------------
# encoder_id enters the cache key
# ---------------------------------------------------------------------------


def test_encoder_id_changes_cache_key() -> None:
    enc_a = StubEncoder(encoder_id="stub-hash-768")
    enc_b = StubEncoder(encoder_id="stub-hash-768-v2")

    key_a = EmbeddingsCache.embedding_key("any outcome string", enc_a.encoder_id)
    key_b = EmbeddingsCache.embedding_key("any outcome string", enc_b.encoder_id)

    assert key_a != key_b


# ---------------------------------------------------------------------------
# encode_batch cache semantics
# ---------------------------------------------------------------------------


def _make_cache(tmp_path: Path) -> EmbeddingsCache:
    return EmbeddingsCache(tmp_path / "emb.sqlite")


def test_encode_batch_cache_miss_then_hit(tmp_path: Path) -> None:
    client = _CountingEncoder(StubEncoder())
    cache = _make_cache(tmp_path)
    texts = ["alpha", "beta", "gamma", "delta"]

    first = encode_batch(texts, client=client, cache=cache)
    assert first.shape == (4, 768)
    assert client.encoded_count == 4

    second = encode_batch(texts, client=client, cache=cache)
    assert second.shape == (4, 768)
    # All four texts should now be pure cache hits; no new encodings.
    assert client.encoded_count == 4
    assert np.allclose(first, second)

    cache.close()


def test_encode_batch_partial_cache_hit(tmp_path: Path) -> None:
    inner = StubEncoder()
    client = _CountingEncoder(inner)
    cache = _make_cache(tmp_path)

    texts = ["one", "two", "three", "four", "five"]
    # Pre-populate cache with indices 1 and 3 by encoding them solo and
    # writing the vectors under the encoder_id used by `client`.
    pre_idx = [1, 3]
    pre_texts = [texts[i] for i in pre_idx]
    pre_vecs = inner.encode(pre_texts)
    for t, v in zip(pre_texts, pre_vecs):
        cache.put_embedding(t, client.encoder_id, v.astype(np.float32))

    result = encode_batch(texts, client=client, cache=cache)

    assert result.shape == (5, 768)
    # Only the three cache misses should have been encoded.
    assert client.encoded_count == 3

    # Rows 1 and 3 must equal the pre-populated vectors exactly.
    assert np.allclose(result[1], pre_vecs[0])
    assert np.allclose(result[3], pre_vecs[1])

    # And every row must match a fresh encode_batch call (same encoder,
    # deterministic), confirming original input order is preserved.
    reference = inner.encode(texts)
    assert np.allclose(result, reference)

    cache.close()


def test_encode_batch_preserves_order(tmp_path: Path) -> None:
    client = _CountingEncoder(StubEncoder())
    cache = _make_cache(tmp_path)
    texts = ["zeta", "eta", "theta", "iota", "kappa", "lambda"]

    baseline = StubEncoder().encode(texts)
    out = encode_batch(texts, client=client, cache=cache)

    assert out.shape == baseline.shape
    for i in range(len(texts)):
        assert np.allclose(out[i], baseline[i])

    # Reverse the order and re-check.
    rev_texts = list(reversed(texts))
    rev_out = encode_batch(rev_texts, client=client, cache=cache)
    for i, t in enumerate(rev_texts):
        baseline_idx = texts.index(t)
        assert np.allclose(rev_out[i], baseline[baseline_idx])

    cache.close()


# ---------------------------------------------------------------------------
# encode_outcomes_tensor
# ---------------------------------------------------------------------------


def _make_outcomes(B: int, J: int, K: int) -> list[list[list[str]]]:
    return [
        [
            [f"b{b}_j{j}_k{k}" for k in range(K)]
            for j in range(J)
        ]
        for b in range(B)
    ]


def test_encode_outcomes_tensor_shape() -> None:
    client = StubEncoder()
    outcomes = _make_outcomes(B=2, J=3, K=2)
    tensor = encode_outcomes_tensor(outcomes, client=client)
    assert tensor.shape == (2, 3, 2, 768)
    assert tensor.dtype == np.float32


def test_encode_outcomes_tensor_rectangular_check() -> None:
    client = StubEncoder()

    # Ragged on J: first event has 2 alternatives, second has 3.
    ragged_J: list[list[list[str]]] = [
        [["a", "b"], ["c", "d"]],
        [["e", "f"], ["g", "h"], ["i", "j"]],
    ]
    with pytest.raises(ValueError):
        encode_outcomes_tensor(ragged_J, client=client)

    # Ragged on K: first alternative has 2 outcomes, second has 3.
    ragged_K: list[list[list[str]]] = [
        [["a", "b"], ["c", "d", "e"]],
    ]
    with pytest.raises(ValueError):
        encode_outcomes_tensor(ragged_K, client=client)


def test_encode_outcomes_tensor_l2_norm() -> None:
    client = StubEncoder()
    outcomes = _make_outcomes(B=2, J=4, K=3)
    tensor = encode_outcomes_tensor(outcomes, client=client)
    assert tensor.shape == (2, 4, 3, 768)

    norms = np.linalg.norm(tensor, axis=-1)
    # Every (B, J, K) slice must be L2-normalized on the last dim.
    assert np.allclose(norms, 1.0, atol=1e-5)
