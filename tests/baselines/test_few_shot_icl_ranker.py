"""Unit tests for :class:`FewShotICLRanker` and its ICL plumbing.

Mirrors the structure of ``tests/baselines/test_zero_shot_claude_ranker.py``
and enumerates the test list in the design doc §11. All tests use
:class:`StubLLMClient` or thin test-local subclasses so the suite runs
without network access.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd
import pytest

from src.baselines._llm_ranker_common import (
    DEFAULT_LETTERS,
    ICLExample,
    build_customer_timeline,
    letter_permutations,
)
from src.baselines.base import BaselineEventBatch
from src.baselines.few_shot_icl_ranker import (
    ICL_HEADER,
    ICL_SPLIT_MARKER,
    FewShotICLRanker,
    FewShotICLRankerFitted,
    build_user_prompt_with_icl,
    select_icl_examples,
)
from src.baselines.zero_shot_claude_ranker import USER_TEMPLATE
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


def _make_event(
    customer_id: str,
    order_date: pd.Timestamp,
    seed: int,
    J: int = 4,
    chosen_idx: int = 0,
) -> dict:
    return {
        "customer_id": customer_id,
        "order_date": order_date,
        "c_d": f"Person {customer_id} at {order_date.isoformat()} seed={seed}",
        "alt_texts": [_make_alt(seed * J + j) for j in range(J)],
        "chosen_idx": int(chosen_idx),
        "choice_asins": [f"ASIN-{seed}-{j}" for j in range(J)],
        "category": "cat_0",
    }


def _make_batch_from_events(events: List[dict], J: int = 4) -> BaselineEventBatch:
    return BaselineEventBatch(
        base_features_list=[np.zeros((J, 3), dtype=np.float32) for _ in events],
        base_feature_names=["f0", "f1", "f2"],
        chosen_indices=[int(e["chosen_idx"]) for e in events],
        customer_ids=[str(e["customer_id"]) for e in events],
        categories=[str(e.get("category", "cat_0")) for e in events],
        raw_events=events,
    )


class _RecordingStubClient(StubLLMClient):
    """Stub that records every user prompt it sees.

    The underlying ``StubLLMClient.generate`` is still invoked to build a
    deterministic return value (so the hash-based letter-probs stay
    deterministic); we just capture the messages along the way.
    """

    def __init__(self, model_id: str = "stub-recording"):
        super().__init__(model_id=model_id)
        self.calls: List[Dict[str, str]] = []

    def generate(
        self,
        messages: list[dict],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
    ) -> GenerationResult:
        record: Dict[str, str] = {}
        for m in messages:
            role = m.get("role", "")
            record[role] = str(m.get("content", ""))
        self.calls.append(record)
        return super().generate(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# §11: timeline + selection tests
# ---------------------------------------------------------------------------


def test_builds_customer_timeline_from_raw_events():
    """2 customers × 5 events each; timeline must be per-customer and sorted."""
    dates_a = [pd.Timestamp("2025-03-15") - pd.Timedelta(days=i) for i in range(5)]
    dates_b = [pd.Timestamp("2025-04-01") - pd.Timedelta(days=i) for i in range(5)]
    events: List[dict] = []
    for i, d in enumerate(dates_a):
        events.append(_make_event("A", d, seed=10 + i))
    for i, d in enumerate(dates_b):
        events.append(_make_event("B", d, seed=100 + i))

    batch = _make_batch_from_events(events)
    timeline = build_customer_timeline(batch)

    assert set(timeline.keys()) == {"A", "B"}
    assert len(timeline["A"]) == 5
    assert len(timeline["B"]) == 5
    # Ascending order per customer.
    for cid in ("A", "B"):
        ts = [ex.order_date for ex in timeline[cid]]
        assert ts == sorted(ts)
        for ex in timeline[cid]:
            assert isinstance(ex, ICLExample)
            assert len(ex.alt_texts) == 4


def test_builds_customer_timeline_returns_empty_when_raw_events_none():
    batch = BaselineEventBatch(
        base_features_list=[np.zeros((4, 1), dtype=np.float32) for _ in range(3)],
        base_feature_names=["f"],
        chosen_indices=[0, 1, 2],
        customer_ids=["a", "b", "c"],
        categories=["x"] * 3,
        raw_events=None,
    )
    assert build_customer_timeline(batch) == {}


def test_chronological_order_enforced():
    """Training events at t=1/2/3 and a test at t=0 → no ICL examples returned."""
    t0 = pd.Timestamp("2025-01-01")
    training = [
        ICLExample(
            order_date=t0 + pd.Timedelta(days=i),
            c_d=f"ctx-{i}",
            alt_texts=[_make_alt(j) for j in range(4)],
            chosen_idx=0,
        )
        for i in (1, 2, 3)
    ]
    timeline = {"A": training}
    selected = select_icl_examples(
        customer_id="A",
        test_order_date=t0,
        timeline=timeline,
        n_shots=3,
    )
    assert selected == []

    # A tied timestamp must also be excluded (strict `<`).
    selected_tie = select_icl_examples(
        customer_id="A",
        test_order_date=training[0].order_date,
        timeline=timeline,
        n_shots=3,
    )
    assert selected_tie == []


def test_n_shots_truncates_to_most_recent():
    t0 = pd.Timestamp("2025-01-01")
    training = [
        ICLExample(
            order_date=t0 + pd.Timedelta(days=i),
            c_d=f"ctx-{i}",
            alt_texts=[_make_alt(i * 4 + j) for j in range(4)],
            chosen_idx=(i % 4),
        )
        for i in range(10)
    ]
    timeline = {"A": training}
    test_date = t0 + pd.Timedelta(days=100)
    selected = select_icl_examples("A", test_date, timeline, n_shots=3)
    assert len(selected) == 3
    # The three most-recent are days 7, 8, 9.
    assert [ex.order_date for ex in selected] == [
        t0 + pd.Timedelta(days=7),
        t0 + pd.Timedelta(days=8),
        t0 + pd.Timedelta(days=9),
    ]


def test_select_icl_examples_unseen_customer_returns_empty():
    t0 = pd.Timestamp("2025-01-01")
    timeline = {
        "A": [
            ICLExample(
                order_date=t0,
                c_d="ctx",
                alt_texts=[_make_alt(j) for j in range(4)],
                chosen_idx=0,
            )
        ]
    }
    assert (
        select_icl_examples(
            "B", t0 + pd.Timedelta(days=1), timeline, n_shots=3
        )
        == []
    )


# ---------------------------------------------------------------------------
# §11: cold-start / raw_events=None fallbacks
# ---------------------------------------------------------------------------


def test_cold_start_falls_back_to_zero_shot():
    """Scoring customer B after fitting only on A → zero-shot prompt, cold_start tallied."""
    t_train = pd.Timestamp("2025-01-01")
    train_events = [
        _make_event("A", t_train + pd.Timedelta(days=i), seed=i) for i in range(3)
    ]
    train_batch = _make_batch_from_events(train_events)

    test_events = [
        _make_event(
            "B", t_train + pd.Timedelta(days=100), seed=999, chosen_idx=1
        )
    ]
    test_batch = _make_batch_from_events(test_events)

    client = _RecordingStubClient()
    ranker = FewShotICLRanker(letters=DEFAULT_LETTERS[:4], n_shots=3, llm_client=client, n_permutations=1)
    fitted = ranker.fit(train_batch, train_batch)
    scores = fitted.score_events(test_batch)

    assert len(scores) == 1
    assert fitted._cold_start_count == 1
    assert fitted._total_events == 1
    assert "cold_start=1/1" in fitted.description

    # Exactly one call (n_permutations=1). Prompt must NOT contain the ICL
    # header or sentinel, and must be byte-identical to the zero-shot
    # template.
    assert len(client.calls) == 1
    user_prompt = client.calls[0]["user"]
    assert ICL_HEADER.strip() not in user_prompt
    assert ICL_SPLIT_MARKER not in user_prompt
    # Byte-identity vs zero-shot template.
    from src.baselines.zero_shot_claude_ranker import USER_TEMPLATE as ZT
    from src.baselines._llm_ranker_common import render_alternatives

    expected = ZT.format(
        c_d=test_events[0]["c_d"],
        alternatives=render_alternatives(test_events[0]["alt_texts"], DEFAULT_LETTERS[:4]),
    )
    assert user_prompt == expected


def test_raw_events_none_graceful_degradation():
    """raw_events=None at fit: every test event cold-starts."""
    train_raw_events = [_make_event("A", pd.Timestamp("2025-01-01"), seed=0)]
    # Synthetic batch with raw_events=None for *training*.
    train_batch_no_raw = BaselineEventBatch(
        base_features_list=[np.zeros((4, 1), dtype=np.float32)],
        base_feature_names=["f"],
        chosen_indices=[0],
        customer_ids=["A"],
        categories=["cat_0"],
        raw_events=None,
    )

    # Test batch must have raw_events (we still score them).
    test_events = [
        _make_event("A", pd.Timestamp("2025-02-01") + pd.Timedelta(days=i), seed=10 + i)
        for i in range(3)
    ]
    test_batch = _make_batch_from_events(test_events)

    ranker = FewShotICLRanker(letters=DEFAULT_LETTERS[:4], 
        n_shots=3, llm_client=StubLLMClient(), n_permutations=1
    )
    fitted = ranker.fit(train_batch_no_raw, train_batch_no_raw)
    assert fitted.timeline == {}

    scores = fitted.score_events(test_batch)
    assert len(scores) == 3
    assert fitted._cold_start_count == 3
    assert fitted._total_events == 3


# ---------------------------------------------------------------------------
# §11: permutation count + call discipline
# ---------------------------------------------------------------------------


def test_score_events_respects_permutation_count():
    """n_permutations=4 × 3 events = 12 LLM calls."""
    t0 = pd.Timestamp("2025-01-01")
    train_events = [
        _make_event("A", t0 + pd.Timedelta(days=i), seed=i) for i in range(5)
    ]
    test_events = [
        _make_event("A", t0 + pd.Timedelta(days=10 + i), seed=100 + i)
        for i in range(3)
    ]
    train_batch = _make_batch_from_events(train_events)
    test_batch = _make_batch_from_events(test_events)

    client = _RecordingStubClient()
    ranker = FewShotICLRanker(letters=DEFAULT_LETTERS[:4], n_shots=3, llm_client=client, n_permutations=4)
    fitted = ranker.fit(train_batch, train_batch)
    scores = fitted.score_events(test_batch)

    assert len(scores) == 3
    for s in scores:
        assert s.shape == (4,)
        assert np.all(np.isfinite(s))
    assert len(client.calls) == 4 * 3

    # With n_permutations=1 we get exactly one call per event, and the
    # per-event log p_hat equals the hash-derived letter logprobs.
    client2 = _RecordingStubClient()
    ranker2 = FewShotICLRanker(letters=DEFAULT_LETTERS[:4], n_shots=3, llm_client=client2, n_permutations=1)
    fitted2 = ranker2.fit(train_batch, train_batch)
    scores2 = fitted2.score_events(test_batch)
    assert len(client2.calls) == 3
    for s in scores2:
        assert s.shape == (4,)


# ---------------------------------------------------------------------------
# §11: lockstep letter rotation (CRITICAL failsafe — design doc §6)
# ---------------------------------------------------------------------------


def test_lockstep_letter_rotation_in_icl_examples():
    """Golden-prompt test: the CHOSEN letter in ICL examples rotates with pi.

    For a single ICL example with ``chosen_idx=0``, under each of the
    K=4 rotations the emitted ``CHOSEN: {letter}`` must correspond to
    the slot s where ``pi[s] == 0``.
    """
    ex = ICLExample(
        order_date=pd.Timestamp("2025-01-01"),
        c_d="icl-context",
        alt_texts=[_make_alt(j) for j in range(4)],
        chosen_idx=0,
    )
    # Test alt_texts — distinct from the ICL example so we can tell the
    # two apart in the rendered prompt.
    test_alt_texts = [_make_alt(100 + j) for j in range(4)]
    c_d = "test-context"

    perms = letter_permutations(n_alts=4, K=4)
    # Expected CHOSEN letter for chosen_idx=0 under each rotation:
    #   k=0 pi=(0,1,2,3) → slot 0 → A
    #   k=1 pi=(1,2,3,0) → slot 3 → D
    #   k=2 pi=(2,3,0,1) → slot 2 → C
    #   k=3 pi=(3,0,1,2) → slot 1 → B
    expected_letters = []
    for pi in perms:
        for s, canonical in enumerate(pi):
            if canonical == 0:
                expected_letters.append(DEFAULT_LETTERS[s])
                break
    assert expected_letters == ["A", "D", "C", "B"]

    seen_letters: List[str] = []
    for pi in perms:
        permuted_test = [test_alt_texts[pi[s]] for s in range(4)]
        prompt = build_user_prompt_with_icl(
            c_d=c_d,
            alt_texts_permuted=permuted_test,
            icl_examples=[ex],
            pi=pi,
            letters=DEFAULT_LETTERS[:4],
            max_prefix_tokens=12_000,
        )
        # Extract "CHOSEN: X" from the prompt.
        marker = "CHOSEN: "
        idx = prompt.index(marker)
        letter = prompt[idx + len(marker)]
        seen_letters.append(letter)
        # Structural assertions: header + sentinel present; test section
        # follows the sentinel.
        assert ICL_HEADER.strip() in prompt
        assert ICL_SPLIT_MARKER in prompt
        # Sentinel must precede the PERSON block of the test suffix.
        assert prompt.index(ICL_SPLIT_MARKER) < prompt.index("PERSON:")

    assert seen_letters == expected_letters


def test_icl_alt_texts_rotate_with_permutation():
    """The ICL ALTERNATIVES block is permuted in lockstep too.

    Under pi=(1,2,3,0), letter A should show the alt at canonical index 1
    (``product-1``), not canonical index 0 (``product-0``).
    """
    ex = ICLExample(
        order_date=pd.Timestamp("2025-01-01"),
        c_d="icl-context",
        alt_texts=[_make_alt(j) for j in range(4)],  # titles product-0..3
        chosen_idx=0,
    )
    pi = (1, 2, 3, 0)
    prompt = build_user_prompt_with_icl(
        c_d="test-context",
        alt_texts_permuted=[_make_alt(100 + pi[s]) for s in range(4)],
        icl_examples=[ex],
        pi=pi,
        letters=DEFAULT_LETTERS[:4],
    )
    # The ICL block is between header and sentinel; it must show
    # product-1 under (A), product-2 under (B), etc.
    icl_segment = prompt[: prompt.index(ICL_SPLIT_MARKER)]
    assert "(A) Title: product-1" in icl_segment
    assert "(B) Title: product-2" in icl_segment
    assert "(C) Title: product-3" in icl_segment
    assert "(D) Title: product-0" in icl_segment


# ---------------------------------------------------------------------------
# §11: context-overflow truncation
# ---------------------------------------------------------------------------


def test_context_overflow_triggers_truncation():
    """With a tiny token budget, only the most-recent ICL examples survive."""
    t0 = pd.Timestamp("2025-01-01")
    examples = [
        ICLExample(
            order_date=t0 + pd.Timedelta(days=i),
            c_d=(f"context-{i} " * 50),  # long-ish to pressure the budget
            alt_texts=[_make_alt(i * 4 + j) for j in range(4)],
            chosen_idx=(i % 4),
        )
        for i in range(5)
    ]
    # Baseline unbounded prompt.
    pi = (0, 1, 2, 3)
    test_alt_texts = [_make_alt(999 + j) for j in range(4)]
    full = build_user_prompt_with_icl(
        c_d="test-ctx",
        alt_texts_permuted=test_alt_texts,
        icl_examples=examples,
        pi=pi,
        letters=DEFAULT_LETTERS[:4],
        max_prefix_tokens=100_000,
    )
    # Large budget keeps all 5 example headers.
    for i in range(5):
        assert f"=== Example {i + 1} ===" in full

    # With a tight budget, at least some oldest examples must be dropped.
    # Pick a budget empirically sized to keep 1-3 of the five (≈800 chars
    # per example × 5 = 4000 chars ≈ 1000 tokens; 600 keeps ~1-2).
    tight = build_user_prompt_with_icl(
        c_d="test-ctx",
        alt_texts_permuted=test_alt_texts,
        icl_examples=examples,
        pi=pi,
        letters=DEFAULT_LETTERS[:4],
        max_prefix_tokens=600,  # forces truncation
    )
    # Some examples must have been dropped.
    n_kept = sum(f"context-{i} " in tight for i in range(5))
    assert n_kept < 5, f"expected some truncation, kept {n_kept}/5"
    # Oldest-first drop policy: whichever examples survive must be the
    # most-recent contiguous suffix (i.e. if context-k is present, so is
    # every context-j with j > k).
    present = [f"context-{i} " in tight for i in range(5)]
    for i in range(4):
        if present[i]:
            assert present[i + 1], (
                f"truncation not oldest-first: context-{i} kept but "
                f"context-{i + 1} dropped"
            )

    # Extreme: tiny budget → the prompt degrades to zero-shot.
    degraded = build_user_prompt_with_icl(
        c_d="test-ctx",
        alt_texts_permuted=test_alt_texts,
        icl_examples=examples,
        pi=pi,
        letters=DEFAULT_LETTERS[:4],
        max_prefix_tokens=10,
    )
    assert ICL_SPLIT_MARKER not in degraded
    assert ICL_HEADER.strip() not in degraded


# ---------------------------------------------------------------------------
# Plumbing: fit-time validation + description + names
# ---------------------------------------------------------------------------


def test_fit_raises_on_empty_batch():
    empty_batch = BaselineEventBatch(
        base_features_list=[],
        base_feature_names=["f"],
        chosen_indices=[],
        customer_ids=[],
        categories=[],
        raw_events=[],
    )
    ranker = FewShotICLRanker(letters=DEFAULT_LETTERS[:4], llm_client=StubLLMClient())
    with pytest.raises(ValueError, match="empty"):
        ranker.fit(empty_batch, empty_batch)


def test_fit_raises_on_wrong_J():
    raw_events = [
        {
            "customer_id": "c0",
            "chosen_idx": 0,
            "order_date": pd.Timestamp("2025-01-01"),
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
    ranker = FewShotICLRanker(letters=DEFAULT_LETTERS[:4], llm_client=StubLLMClient())
    with pytest.raises(ValueError, match="n_alternatives"):
        ranker.fit(batch, batch)


def test_fitted_description_and_n_params():
    t0 = pd.Timestamp("2025-01-01")
    events = [_make_event("A", t0 + pd.Timedelta(days=i), seed=i) for i in range(4)]
    batch = _make_batch_from_events(events)
    ranker = FewShotICLRanker(letters=DEFAULT_LETTERS[:4], 
        n_shots=3,
        llm_client=StubLLMClient(model_id="stub-v1"),
        n_permutations=4,
        seed=0,
    )
    fitted = ranker.fit(batch, batch)
    assert isinstance(fitted, FewShotICLRankerFitted)
    assert fitted.n_params == 0
    desc = fitted.description
    assert desc.startswith("FewShot-ICL-Claude")
    assert "n_shots=3" in desc
    assert "K=4" in desc
    assert "cold_start=0/0" in desc


def test_baseline_name_is_few_shot_icl_claude():
    assert FewShotICLRanker.name == "FewShot-ICL-Claude"
    fitted = FewShotICLRankerFitted()
    assert fitted.name == "FewShot-ICL-Claude"


def test_score_events_raises_without_raw_events():
    train_events = [_make_event("A", pd.Timestamp("2025-01-01"), seed=0)]
    train_batch = _make_batch_from_events(train_events)
    ranker = FewShotICLRanker(letters=DEFAULT_LETTERS[:4], llm_client=StubLLMClient())
    fitted = ranker.fit(train_batch, train_batch)

    no_raw = BaselineEventBatch(
        base_features_list=[np.zeros((4, 1), dtype=np.float32)],
        base_feature_names=["f"],
        chosen_indices=[0],
        customer_ids=["A"],
        categories=["x"],
        raw_events=None,
    )
    with pytest.raises(ValueError, match="raw_events"):
        fitted.score_events(no_raw)
