"""Tests for src/outcomes/diversity_filter.py (redesign.md §3.5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.outcomes.diversity_filter import (
    EmbeddingFn,
    HashEmbedder,
    diversity_filter,
    find_paraphrase_pair,
    pairwise_cosine,
)


# ---------------------------------------------------------------------------
# HashEmbedder
# ---------------------------------------------------------------------------


def test_hash_embedder_deterministic() -> None:
    emb = HashEmbedder(dim=64, seed=0)
    texts = ["hello world", "quantum physics today", "coffee on a rainy evening"]
    a = emb(texts)
    b = emb(texts)
    assert np.allclose(a, b)


def test_hash_embedder_unit_norm() -> None:
    emb = HashEmbedder(dim=64, seed=0)
    texts = [
        "a",
        "ab",
        "hello world",
        "this outcome sentence is a bit longer than the others",
    ]
    out = emb(texts)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_hash_embedder_distinguishes_strings() -> None:
    """Clearly different strings must not look like paraphrases."""
    emb = HashEmbedder(dim=64, seed=0)
    out = emb(["hello world", "quantum physics today"])
    sim = float(out[0] @ out[1])
    # Two strings with near-disjoint character 3-grams should have cosine
    # well below the §3.5 paraphrase threshold of 0.9.
    assert sim < 0.9
    # Additionally check the (much tighter) L2-distance spec: ≥ 1e-3.
    dist = float(np.linalg.norm(out[0] - out[1]))
    assert dist >= 1e-3


def test_hash_embedder_implements_protocol() -> None:
    """HashEmbedder should satisfy the EmbeddingFn runtime protocol."""
    assert isinstance(HashEmbedder(), EmbeddingFn)


def test_hash_embedder_dtype_and_shape() -> None:
    emb = HashEmbedder(dim=32, seed=7)
    out = emb(["a", "b", "c"])
    assert out.shape == (3, 32)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# pairwise_cosine
# ---------------------------------------------------------------------------


def test_pairwise_cosine_shape() -> None:
    emb = HashEmbedder(dim=64, seed=0)
    out = emb(["one", "two", "three", "four"])
    sim = pairwise_cosine(out)
    assert sim.shape == (4, 4)


def test_pairwise_cosine_symmetric_and_diag_one() -> None:
    emb = HashEmbedder(dim=64, seed=0)
    out = emb(["alpha", "beta", "gamma"])
    sim = pairwise_cosine(out)
    # Symmetry
    assert np.allclose(sim, sim.T, atol=1e-6)
    # Diagonal ≈ 1
    assert np.allclose(np.diag(sim), 1.0, atol=1e-5)


def test_pairwise_cosine_rejects_unnormalized() -> None:
    bad = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        pairwise_cosine(bad)


# ---------------------------------------------------------------------------
# find_paraphrase_pair
# ---------------------------------------------------------------------------


def test_find_paraphrase_pair_identical() -> None:
    outcomes = ["I save money this month.", "I save money this month."]
    pair = find_paraphrase_pair(outcomes)
    assert pair == (0, 1)


def test_find_paraphrase_pair_distinct() -> None:
    outcomes = [
        "I save money this month.",
        "The kids sleep better tonight.",
        "Getting dinner on the table takes five fewer minutes.",
    ]
    assert find_paraphrase_pair(outcomes) is None


def test_find_paraphrase_pair_threshold_tunable() -> None:
    outcomes = [
        "I save money this month.",
        "The kids sleep better tonight.",
    ]
    # At a near-impossible threshold, no pair exceeds it.
    assert find_paraphrase_pair(outcomes, threshold=0.99) is None
    # At a very permissive threshold, any non-orthogonal pair triggers.
    pair = find_paraphrase_pair(outcomes, threshold=0.1)
    # Either (0, 1) triggers or (conceivably) the pair is below 0.1; the
    # spec requires the former for "same list passes at 0.99 but fails at
    # 0.1". Assert the stronger condition.
    assert pair == (0, 1)


def test_find_paraphrase_pair_short_list_returns_none() -> None:
    assert find_paraphrase_pair([]) is None
    assert find_paraphrase_pair(["only one"]) is None


def test_find_paraphrase_pair_custom_embedder() -> None:
    """Caller-supplied EmbeddingFn is honored."""
    # A custom embedder that maps everything to the same vector: any pair
    # beyond the first trips the filter.
    class ConstantEmbedder:
        def __call__(self, texts: list[str]) -> np.ndarray:
            out = np.zeros((len(texts), 4), dtype=np.float32)
            out[:, 0] = 1.0
            return out

    pair = find_paraphrase_pair(
        ["a", "b", "c"], threshold=0.5, embed_fn=ConstantEmbedder()
    )
    assert pair == (0, 1)


# ---------------------------------------------------------------------------
# diversity_filter
# ---------------------------------------------------------------------------


def test_diversity_filter_ok_shape() -> None:
    outcomes = ["foo", "bar", "baz"]
    result = diversity_filter(outcomes)
    assert isinstance(result, tuple)
    assert len(result) == 2
    returned, ok = result
    assert returned is outcomes  # identity-preserving
    assert isinstance(ok, bool)


def test_diversity_filter_reject_paraphrase() -> None:
    outcomes = ["identical line", "identical line", "something else entirely"]
    returned, ok = diversity_filter(outcomes)
    assert returned == outcomes
    assert ok is False


def test_diversity_filter_accepts_distinct() -> None:
    outcomes = [
        "I save forty dollars this month on groceries.",
        "The kids sleep much better tonight.",
        "Dinner prep takes five fewer minutes than before.",
    ]
    returned, ok = diversity_filter(outcomes)
    assert returned == outcomes
    assert ok is True
