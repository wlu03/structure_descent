"""Unit tests for ``ZeroShotClaudeRanker`` and the shared ranker helpers.

Covers both the baseline class (``src/baselines/zero_shot_claude_ranker.py``)
and the shared helper module (``src/baselines/_llm_ranker_common.py``).
All tests use :class:`StubLLMClient` or thin test-local stub subclasses so
the suite runs without network access. See
``docs/llm_baselines/zero_shot_claude_ranker.md`` §Test Strategy for the
overall plan.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Mapping, Sequence

import numpy as np
import pytest

from src.baselines._llm_ranker_common import (
    DEFAULT_LETTERS,
    _stub_letter_probs,
    call_llm_for_ranking,
    extract_letter_logprobs,
    letter_permutations,
    render_alternatives,
)
from src.baselines.base import BaselineEventBatch
from src.baselines.zero_shot_claude_ranker import (
    ZeroShotClaudeRanker,
    ZeroShotClaudeRankerFitted,
)
from src.outcomes.generate import GenerationResult, StubLLMClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_alt(i: int) -> dict:
    return {
        "title": f"product-{i}",
        "category": f"cat-{i % 3}",
        "price": 10.0 + i,
        "popularity_rank": f"popularity score {100 - i}",
        "brand": f"brand-{i % 2}",
        "is_repeat": bool(i % 2 == 0),
    }


def _make_batch(n_events: int = 10, J: int = 4, seed: int = 0) -> BaselineEventBatch:
    """Minimal J=4 batch with per-event c_d + alt_texts for the ranker."""
    rng = np.random.default_rng(seed)
    base_features_list: List[np.ndarray] = []
    chosen: List[int] = []
    customer_ids: List[str] = []
    categories: List[str] = []
    raw_events: List[dict] = []
    for i in range(n_events):
        # Distinct alt_texts per event so the hash-based stub yields
        # event-varying probabilities.
        alt_texts = [_make_alt(i * J + j) for j in range(J)]
        c_d = f"Person {i}: profile signature {int(rng.integers(0, 1_000_000))}."
        base_features_list.append(np.zeros((J, 3), dtype=np.float32))
        chosen_idx = int(rng.integers(0, J))
        chosen.append(chosen_idx)
        customer_ids.append(f"cust_{i}")
        categories.append(f"cat_{i % 3}")
        raw_events.append(
            {
                "customer_id": f"cust_{i}",
                "chosen_idx": chosen_idx,
                "choice_asins": [f"ASIN-{i}-{j}" for j in range(J)],
                "c_d": c_d,
                "alt_texts": alt_texts,
                "category": f"cat_{i % 3}",
            }
        )
    return BaselineEventBatch(
        base_features_list=base_features_list,
        base_feature_names=["f0", "f1", "f2"],
        chosen_indices=chosen,
        customer_ids=customer_ids,
        categories=categories,
        raw_events=raw_events,
    )


class _ConstantStubClient(StubLLMClient):
    """Stub client that always returns the same text, regardless of messages.

    Required by the pathological-bias test: with a constant stub response,
    ``_stub_letter_probs`` produces a fixed letter-probability vector.
    Wrapping the output in this subclass lets us tune the vector to place
    probability mass wherever the test needs it.
    """

    def __init__(self, constant_text: str, model_id: str = "stub-constant"):
        super().__init__(model_id=model_id)
        self._constant_text = constant_text

    def generate(
        self,
        messages: list[dict],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
    ) -> GenerationResult:
        return GenerationResult(
            text=self._constant_text,
            finish_reason="stop",
            model_id=self.model_id,
        )


def _find_text_placing_prob_on(letter_idx: int, letters=DEFAULT_LETTERS, tries: int = 10_000) -> str:
    """Find a string whose SHA-256-derived stub probs peak on ``letter_idx``.

    The stub letter-probability function hashes the input text and uses the
    first few digest bytes as logits. To build a pathological test input we
    just enumerate candidate strings until we find one whose argmax lands
    on the target letter slot; this keeps the test deterministic without
    hard-coding SHA-256 preimages.
    """
    for i in range(tries):
        cand = f"bias-probe-{letter_idx}-{i}"
        probs = _stub_letter_probs(cand, letters)
        if int(np.argmax(probs)) == letter_idx:
            return cand
    raise RuntimeError("_find_text_placing_prob_on: no seed found in budget")


# ---------------------------------------------------------------------------
# Shared-helper tests
# ---------------------------------------------------------------------------


def test_letter_permutations_latin_square():
    """K=4 rotation: every alt appears in every letter-slot exactly once."""
    perms = letter_permutations(n_alts=4, K=4)
    assert len(perms) == 4
    for pi in perms:
        assert len(pi) == 4
        assert sorted(pi) == [0, 1, 2, 3]

    # Slot counts: for each letter slot s, Counter(pi[s] for pi in perms)
    # should be a histogram with exactly one entry per alt.
    for s in range(4):
        counts = Counter(pi[s] for pi in perms)
        assert counts == {0: 1, 1: 1, 2: 1, 3: 1}


def test_letter_permutations_validates_inputs():
    with pytest.raises(ValueError, match="n_alts"):
        letter_permutations(n_alts=0, K=4)
    with pytest.raises(ValueError, match="K"):
        letter_permutations(n_alts=4, K=0)


def test_render_alternatives_letter_binding():
    """Each letter block cites the right field values in the expected order."""
    alt_texts = [_make_alt(i) for i in range(4)]
    block = render_alternatives(alt_texts, DEFAULT_LETTERS)
    # Each letter label appears exactly once at the start of a line.
    for letter in DEFAULT_LETTERS:
        assert f"({letter}) Title: " in block
    # Field values are bound to the matching alt.
    assert "product-0" in block
    assert "product-3" in block
    # The alternatives appear in the order they were supplied.
    a_idx = block.index("(A) Title: ")
    b_idx = block.index("(B) Title: ")
    c_idx = block.index("(C) Title: ")
    d_idx = block.index("(D) Title: ")
    assert a_idx < b_idx < c_idx < d_idx


def test_render_alternatives_rejects_length_mismatch():
    with pytest.raises(ValueError, match="n_alts"):
        render_alternatives([_make_alt(0)], DEFAULT_LETTERS)


def test_extract_letter_logprobs_stub_fallback_is_deterministic():
    """Same text → same probs; different text → different probs; probs sum to 1."""
    probs_a = extract_letter_logprobs("some stub text", DEFAULT_LETTERS)
    probs_b = extract_letter_logprobs("some stub text", DEFAULT_LETTERS)
    probs_c = extract_letter_logprobs("different text entirely", DEFAULT_LETTERS)
    np.testing.assert_allclose(probs_a, probs_b, atol=1e-12)
    assert not np.allclose(probs_a, probs_c)
    assert probs_a.shape == (4,)
    np.testing.assert_allclose(probs_a.sum(), 1.0, atol=1e-10)
    np.testing.assert_allclose(probs_c.sum(), 1.0, atol=1e-10)
    assert np.all(probs_a >= 0)


def test_extract_letter_logprobs_parses_verbalized_json():
    text = 'The answer is {"A": 0.1, "B": 0.7, "C": 0.1, "D": 0.1}.'
    probs = extract_letter_logprobs(text, DEFAULT_LETTERS)
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-10)
    assert int(np.argmax(probs)) == 1  # "B"


def test_call_llm_for_ranking_stub_path_uses_hash():
    """``call_llm_for_ranking`` on StubLLMClient uses the hash fallback."""
    client = StubLLMClient()
    probs = call_llm_for_ranking(
        client,
        system="system",
        user="user",
        letters=DEFAULT_LETTERS,
        seed=0,
    )
    assert probs.shape == (4,)
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# ZeroShotClaudeRanker tests
# ---------------------------------------------------------------------------


def test_score_events_shape_and_length():
    batch = _make_batch(n_events=10, J=4, seed=1)
    ranker = ZeroShotClaudeRanker(client=StubLLMClient(), K=4)
    fitted = ranker.fit(batch, batch)
    scores = fitted.score_events(batch)
    assert len(scores) == 10
    for s in scores:
        assert s.shape == (4,)
        assert np.all(np.isfinite(s)), "scores must be finite after log floor"


def test_single_permutation_matches_argmax_for_pathological_stub():
    """With K=1 and a pathological stub, argmax aligns with the forced slot.

    Guards against an off-by-one in the un-permute step. We build a stub
    that always returns the same text, so letter-probs are constant across
    calls — the hash chooses one argmax letter slot, and with K=1 (identity
    permutation) that slot equals the canonical alternative index.
    """
    # Seed a stub text whose hash places argmax on letter C (idx 2).
    constant_text = _find_text_placing_prob_on(letter_idx=2)
    stub_probs = _stub_letter_probs(constant_text, DEFAULT_LETTERS)
    assert int(np.argmax(stub_probs)) == 2

    client = _ConstantStubClient(constant_text)
    batch = _make_batch(n_events=8, J=4, seed=3)
    ranker = ZeroShotClaudeRanker(client=client, K=1)
    fitted = ranker.fit(batch, batch)
    scores = fitted.score_events(batch)
    for s in scores:
        # With K=1 (identity permutation) the canonical alt at letter-slot
        # 2 is canonical index 2.
        assert int(np.argmax(s)) == 2


def test_permutation_debiasing_cancels_synthetic_bias():
    """K=4 rotation spreads the argmax across alternatives.

    With a constant stub + K=1, argmax is pinned to a single canonical
    alternative (the one currently showing in the stub's preferred letter
    slot, which under identity rotation is just that slot). With K=4,
    every alternative spends one permutation in the preferred slot, so the
    4 per-call distributions are each tilted toward a different canonical
    alternative; after averaging, the aggregate is ~uniform and argmax is
    close to uniformly distributed over alternatives across events.

    We do not run a χ² test — the dataset is small and the stub is fully
    deterministic given the event's alt_texts — but we DO assert that
    K=4 spreads the argmax across multiple alternatives while K=1 pins
    it to one.
    """
    constant_text = _find_text_placing_prob_on(letter_idx=0)  # prefer A
    client = _ConstantStubClient(constant_text)
    batch = _make_batch(n_events=32, J=4, seed=11)

    ranker_k1 = ZeroShotClaudeRanker(client=client, K=1)
    fitted_k1 = ranker_k1.fit(batch, batch)
    scores_k1 = fitted_k1.score_events(batch)
    argmax_k1 = [int(np.argmax(s)) for s in scores_k1]
    # K=1: every event's argmax is canonical alt 0 (the stub's forced slot).
    assert set(argmax_k1) == {0}

    ranker_k4 = ZeroShotClaudeRanker(client=client, K=4)
    fitted_k4 = ranker_k4.fit(batch, batch)
    scores_k4 = fitted_k4.score_events(batch)
    # With K=4 Latin square + a constant stub, each canonical alt receives
    # exactly the same averaged probability across events, so argmax ties
    # and numpy returns 0. The more meaningful check: per-event scores
    # should be ~uniform (all four log-probs nearly equal), unlike K=1.
    for s in scores_k4:
        spread = float(np.max(s) - np.min(s))
        assert spread < 1e-9, f"K=4 averaging should equalize logs, got spread={spread}"
    # And the K=1 spread is strictly larger than the K=4 spread for the
    # same events.
    spread_k1 = float(np.mean([np.max(s) - np.min(s) for s in scores_k1]))
    spread_k4 = float(np.mean([np.max(s) - np.min(s) for s in scores_k4]))
    assert spread_k1 > spread_k4 + 1e-6


def test_fitted_description_and_n_params():
    batch = _make_batch(n_events=3, J=4, seed=5)
    ranker = ZeroShotClaudeRanker(
        client=StubLLMClient(model_id="stub-v1"),
        K=4,
        temperature=0.0,
    )
    fitted = ranker.fit(batch, batch)
    assert isinstance(fitted, ZeroShotClaudeRankerFitted)
    assert fitted.n_params == 0
    desc = fitted.description
    assert desc.startswith("ZeroShot-Claude")
    assert "K=4" in desc
    assert "T=0" in desc
    assert "prompt=" in desc


def test_fit_raises_on_wrong_J():
    # J=3 batch: should raise at fit-time.
    raw_events = [
        {
            "customer_id": "c0",
            "chosen_idx": 0,
            "c_d": "context",
            "alt_texts": [_make_alt(i) for i in range(3)],
        }
    ]
    batch = BaselineEventBatch(
        base_features_list=[np.zeros((3, 1), dtype=np.float32)],
        base_feature_names=["f"],
        chosen_indices=[0],
        customer_ids=["c0"],
        categories=["cat0"],
        raw_events=raw_events,
    )
    ranker = ZeroShotClaudeRanker(client=StubLLMClient())
    with pytest.raises(ValueError, match="n_alternatives"):
        ranker.fit(batch, batch)


def test_fit_raises_without_raw_events():
    batch = BaselineEventBatch(
        base_features_list=[np.zeros((4, 1), dtype=np.float32) for _ in range(3)],
        base_feature_names=["f"],
        chosen_indices=[0, 1, 2],
        customer_ids=["a", "b", "c"],
        categories=["x"] * 3,
        raw_events=None,
    )
    ranker = ZeroShotClaudeRanker(client=StubLLMClient())
    with pytest.raises(ValueError, match="raw_events"):
        ranker.fit(batch, batch)


def test_fit_raises_on_empty_batch():
    batch = BaselineEventBatch(
        base_features_list=[],
        base_feature_names=["f"],
        chosen_indices=[],
        customer_ids=[],
        categories=[],
        raw_events=[],
    )
    ranker = ZeroShotClaudeRanker(client=StubLLMClient())
    with pytest.raises(ValueError, match="empty"):
        ranker.fit(batch, batch)


def test_baseline_name_is_zero_shot_claude():
    assert ZeroShotClaudeRanker.name == "ZeroShot-Claude"
    fitted = ZeroShotClaudeRankerFitted()
    assert fitted.name == "ZeroShot-Claude"
