"""Unit tests for the ST+MLP ablation baseline.

Hermetic: every test uses :class:`~src.outcomes.encode.StubEncoder` so
no sentence-transformers weights are downloaded. The real
:class:`SentenceTransformersEncoder` is exercised only via
``scripts/run_baselines.py`` in production.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.baselines import BaselineEventBatch, FittedBaseline
from src.baselines.st_mlp_ablation import (
    STMLPChoice,
    STMLPFitted,
    _is_repeat_one_hot_on_chosen,
    _mlp_param_count,
    _render_alt_text,
)
from src.outcomes.encode import EncoderClient, StubEncoder


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _default_alt(
    *,
    title: str = "Acme Cup",
    category: str = "Kitchen",
    price: float = 9.99,
    popularity_rank: Any = "popularity score 100",
    brand: str = "Acme",
    is_repeat: bool = False,
    state: str = "CA",
) -> Dict[str, Any]:
    """7-key alt dict matching :func:`src.data.adapter.alt_text`'s contract."""
    return {
        "title": title,
        "category": category,
        "price": price,
        "popularity_rank": popularity_rank,
        "brand": brand,
        "is_repeat": is_repeat,
        "state": state,
    }


def _make_batch(
    *,
    n_events: int,
    J: int,
    chosen_indices: Sequence[int],
    alt_texts_per_event: Sequence[Sequence[Dict[str, Any]]],
    metadata: Optional[Sequence[Dict[str, Any]]] = None,
    with_raw_events: bool = True,
) -> BaselineEventBatch:
    """Build a minimal :class:`BaselineEventBatch` carrying ``alt_texts``.

    ``base_features_list`` is populated with zeros since ST+MLP does
    not consume it — we just need the batch to satisfy the
    :class:`BaselineEventBatch` post-init invariants.
    """
    base_list: List[np.ndarray] = [
        np.zeros((J, 3), dtype=np.float32) for _ in range(n_events)
    ]
    raw_events: Optional[List[dict]] = None
    if with_raw_events:
        raw_events = []
        for e, alts in enumerate(alt_texts_per_event):
            raw_events.append(
                {
                    "customer_id": f"c{e}",
                    "chosen_idx": int(chosen_indices[e]),
                    "alt_texts": list(alts),
                    "category": str(alts[0].get("category", "")),
                }
            )
    kwargs: Dict[str, Any] = dict(
        base_features_list=base_list,
        base_feature_names=["f0", "f1", "f2"],
        chosen_indices=list(chosen_indices),
        customer_ids=[f"c{e}" for e in range(n_events)],
        categories=[
            str(alt_texts_per_event[e][0].get("category", ""))
            for e in range(n_events)
        ],
        raw_events=raw_events,
    )
    if metadata is not None:
        kwargs["metadata"] = list(metadata)
    return BaselineEventBatch(**kwargs)


def _planted_signal_batch(
    *,
    n_events: int,
    J: int,
    keyword: str,
    seed: int,
    leak_is_repeat: bool = False,
) -> BaselineEventBatch:
    """Construct a batch with a **constant** signal on the chosen alt.

    The :class:`StubEncoder` is a deterministic hash, so it produces
    near-orthogonal vectors for distinct strings and *identical*
    vectors for identical strings. To plant a learnable signal we make
    every chosen alt render to the same ``"CHOSEN-<keyword>"`` title
    and every non-chosen alt render to a fixed ``"OTHER"`` title. An
    MLP can then learn a linear separator between two clusters. The
    chosen-alt position is randomized per event so a position prior
    cannot trivially solve the task.
    """
    rng = np.random.default_rng(seed)
    alt_texts_per_event: List[List[Dict[str, Any]]] = []
    chosen_indices: List[int] = []
    chosen_title = f"CHOSEN-{keyword}"
    other_title = "OTHER"
    for e in range(n_events):
        chosen = int(rng.integers(0, J))
        chosen_indices.append(chosen)
        alts: List[Dict[str, Any]] = []
        for j in range(J):
            title = chosen_title if j == chosen else other_title
            alts.append(
                _default_alt(
                    title=title,
                    is_repeat=(leak_is_repeat and j == chosen),
                    popularity_rank="popularity score 50",
                )
            )
        alt_texts_per_event.append(alts)
    return _make_batch(
        n_events=n_events,
        J=J,
        chosen_indices=chosen_indices,
        alt_texts_per_event=alt_texts_per_event,
    )


# ---------------------------------------------------------------------------
# _render_alt_text
# ---------------------------------------------------------------------------


def test_render_alt_text_format():
    """All 7 fields render; no ``None`` / ``nan`` substrings."""
    alt = _default_alt(
        title="Hydro Flask 32oz",
        category="Drinkware",
        price=44.95,
        popularity_rank="popularity score 7",
        brand="Hydro Flask",
        is_repeat=True,
        state="NY",
    )
    out = _render_alt_text(alt)
    assert "Hydro Flask 32oz" in out
    assert "Category: Drinkware" in out
    assert "Brand: Hydro Flask" in out
    assert "Price: $44.9" in out or "Price: $45.0" in out  # {:.1f} round
    assert "Popularity: popularity score 7" in out
    assert "Repeat purchase: True" in out
    assert "State: NY" in out
    assert "None" not in out
    assert "nan" not in out.lower()


def test_render_alt_text_missing_fields_fall_back():
    """Missing / ``None`` keys collapse to stable literals."""
    out = _render_alt_text(
        {
            "title": None,
            "category": None,
            "price": None,
            "popularity_rank": None,
            "brand": None,
            "state": None,
        }
    )
    assert "unknown title" in out
    assert "unknown category" in out
    assert "unknown_brand" in out
    assert "Price: $0.0" in out
    assert "popularity score 0" in out
    assert "None" not in out


def test_render_alt_text_stable_across_calls():
    """Same dict → byte-identical string (cache-key stability)."""
    alt = _default_alt()
    assert _render_alt_text(alt) == _render_alt_text(alt)
    assert _render_alt_text(alt, drop_is_repeat=True) == _render_alt_text(
        alt, drop_is_repeat=True
    )


def test_render_alt_text_drop_is_repeat_omits_sentence():
    """``drop_is_repeat=True`` omits the Repeat-purchase sentence entirely."""
    alt = _default_alt(is_repeat=True)
    with_repeat = _render_alt_text(alt, drop_is_repeat=False)
    without_repeat = _render_alt_text(alt, drop_is_repeat=True)
    assert "Repeat purchase" in with_repeat
    assert "Repeat purchase" not in without_repeat


def test_render_alt_text_price_coerces_nonnumeric():
    """Non-numeric ``price`` coerces to ``0.0`` rather than raising."""
    alt = _default_alt()
    alt["price"] = "not a number"
    out = _render_alt_text(alt)
    assert "Price: $0.0" in out


# ---------------------------------------------------------------------------
# n_params
# ---------------------------------------------------------------------------


def test_n_params_matches_mlp_param_count():
    """``n_params == (d_e + 1) * hidden + (hidden + 1)``."""
    d_e = 64
    hidden = 16
    encoder = StubEncoder(d_e=d_e)
    batch = _planted_signal_batch(n_events=6, J=3, keyword="alpha", seed=0)
    fitted = STMLPChoice(
        hidden_dim=hidden,
        n_epochs=1,
        batch_size=4,
        patience=1,
        encoder=encoder,
        seed=0,
    ).fit(batch, batch)
    assert fitted.n_params == _mlp_param_count(d_e, hidden)
    # Sanity: also matches the torch state dict numel.
    import torch as _t
    total = sum(int(p.numel()) for p in fitted.state_dict.values() if isinstance(p, _t.Tensor))
    assert fitted.n_params == total


# ---------------------------------------------------------------------------
# score_events — shape / probability sum
# ---------------------------------------------------------------------------


def test_score_events_shape_and_probabilities_sum_to_one_after_softmax():
    """Returns ``List[np.ndarray]`` of shape ``(J,)``; softmax sums to 1."""
    encoder = StubEncoder(d_e=32)
    batch = _planted_signal_batch(n_events=5, J=4, keyword="beta", seed=11)
    fitted = STMLPChoice(
        hidden_dim=8,
        n_epochs=1,
        batch_size=4,
        patience=1,
        encoder=encoder,
        seed=0,
    ).fit(batch, batch)

    scores = fitted.score_events(batch)
    assert isinstance(scores, list)
    assert len(scores) == batch.n_events
    for s in scores:
        assert isinstance(s, np.ndarray)
        assert s.shape == (batch.n_alternatives,)
        # Stable softmax → probabilities sum to 1.
        e = np.exp(s - s.max())
        probs = e / e.sum()
        assert abs(float(probs.sum()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_given_seed():
    """Two ``fit`` calls with the same seed → ``torch.allclose`` state dicts."""
    encoder_a = StubEncoder(d_e=32)
    encoder_b = StubEncoder(d_e=32)
    batch = _planted_signal_batch(n_events=12, J=3, keyword="gamma", seed=77)
    kwargs = dict(
        hidden_dim=8,
        n_epochs=3,
        batch_size=4,
        patience=2,
        seed=123,
    )
    fit_a = STMLPChoice(encoder=encoder_a, **kwargs).fit(batch, batch)
    fit_b = STMLPChoice(encoder=encoder_b, **kwargs).fit(batch, batch)
    import torch as _t
    for k in fit_a.state_dict:
        assert _t.allclose(fit_a.state_dict[k], fit_b.state_dict[k])
    # Top-level scalar summaries also agree.
    assert math.isclose(fit_a.train_nll, fit_b.train_nll, rel_tol=0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Signal learning
# ---------------------------------------------------------------------------


def test_fit_learns_synthetic_signal():
    """Planted keyword in chosen-alt titles → train NLL beats uniform."""
    encoder = StubEncoder(d_e=32)
    train = _planted_signal_batch(n_events=80, J=4, keyword="delta", seed=1)
    val = _planted_signal_batch(n_events=40, J=4, keyword="delta", seed=2)

    fitted = STMLPChoice(
        hidden_dim=16,
        lr=5e-3,
        n_epochs=40,
        batch_size=16,
        patience=20,
        encoder=encoder,
        seed=0,
    ).fit(train, val)

    J = train.n_alternatives
    uniform_nll = math.log(J)
    assert fitted.train_nll < uniform_nll - 0.1, (
        f"train_nll={fitted.train_nll:.3f} should beat log(J)={uniform_nll:.3f}"
    )

    # Top-1 on held-out should beat chance by a clear margin.
    scores = fitted.score_events(val)
    preds = np.array([int(np.argmax(s)) for s in scores])
    truth = np.asarray(val.chosen_indices)
    top1 = float((preds == truth).mean())
    assert top1 > 1.0 / J + 0.1, f"top1={top1:.3f} should beat chance 1/J={1/J:.3f}"


# ---------------------------------------------------------------------------
# is_repeat auto-drop
# ---------------------------------------------------------------------------


def test_is_repeat_auto_drop_when_one_hot_on_chosen():
    """1-hot ``is_repeat`` pattern → fitted model drops the sentence."""
    encoder = StubEncoder(d_e=16)
    batch = _planted_signal_batch(
        n_events=10, J=3, keyword="epsilon", seed=5, leak_is_repeat=True
    )
    # Sanity: the detector sees the pattern.
    alt_texts = [ev["alt_texts"] for ev in batch.raw_events]  # type: ignore[index]
    assert _is_repeat_one_hot_on_chosen(alt_texts, batch.chosen_indices) is True

    fitted = STMLPChoice(
        hidden_dim=8,
        n_epochs=1,
        batch_size=4,
        patience=1,
        encoder=encoder,
        seed=0,
        drop_is_repeat="auto",
    ).fit(batch, batch)
    assert fitted.drop_is_repeat is True

    # And the rendered string used for scoring omits the sentence.
    sample = _render_alt_text(
        batch.raw_events[0]["alt_texts"][0],  # type: ignore[index]
        drop_is_repeat=fitted.drop_is_repeat,
    )
    assert "Repeat purchase" not in sample


def test_is_repeat_auto_keeps_sentence_when_not_leaking():
    """Non-leak pattern (all False) → ``drop_is_repeat`` stays False."""
    encoder = StubEncoder(d_e=16)
    batch = _planted_signal_batch(
        n_events=10, J=3, keyword="zeta", seed=6, leak_is_repeat=False
    )
    fitted = STMLPChoice(
        hidden_dim=8,
        n_epochs=1,
        batch_size=4,
        patience=1,
        encoder=encoder,
        seed=0,
        drop_is_repeat="auto",
    ).fit(batch, batch)
    assert fitted.drop_is_repeat is False


def test_is_repeat_explicit_false_overrides_auto_detection():
    """``drop_is_repeat=False`` keeps the sentence even if the pattern trips."""
    encoder = StubEncoder(d_e=16)
    batch = _planted_signal_batch(
        n_events=10, J=3, keyword="eta", seed=8, leak_is_repeat=True
    )
    fitted = STMLPChoice(
        hidden_dim=8,
        n_epochs=1,
        batch_size=4,
        patience=1,
        encoder=encoder,
        seed=0,
        drop_is_repeat=False,
    ).fit(batch, batch)
    assert fitted.drop_is_repeat is False


# ---------------------------------------------------------------------------
# Raw-events / robustness
# ---------------------------------------------------------------------------


def test_requires_raw_events():
    """Batch without ``raw_events`` → actionable ``ValueError`` on fit."""
    J = 3
    batch = BaselineEventBatch(
        base_features_list=[np.zeros((J, 3), dtype=np.float32) for _ in range(4)],
        base_feature_names=["f0", "f1", "f2"],
        chosen_indices=[0, 1, 2, 0],
        customer_ids=[f"c{i}" for i in range(4)],
        categories=["x"] * 4,
        raw_events=None,
    )
    with pytest.raises(ValueError, match="raw_events"):
        STMLPChoice(encoder=StubEncoder(d_e=8), n_epochs=1).fit(batch, batch)


def test_requires_alt_texts_field_on_every_event():
    """Missing ``alt_texts`` key → same actionable error."""
    encoder = StubEncoder(d_e=8)
    batch = _planted_signal_batch(n_events=3, J=2, keyword="a", seed=0)
    # Nuke alt_texts on one record.
    assert batch.raw_events is not None
    del batch.raw_events[0]["alt_texts"]
    with pytest.raises(ValueError, match="raw_events"):
        STMLPChoice(encoder=encoder, n_epochs=1).fit(batch, batch)


def test_missing_alt_texts_fields_graceful():
    """Records lacking ``state`` / ``brand`` / ``popularity_rank`` fit cleanly."""
    encoder = StubEncoder(d_e=16)
    n_events, J = 4, 3
    alt_texts_per_event: List[List[Dict[str, Any]]] = []
    for e in range(n_events):
        alts: List[Dict[str, Any]] = []
        for j in range(J):
            alts.append(
                {
                    "title": f"item {e}-{j}",
                    "category": "cat",
                    "price": 1.0 + j,
                    # popularity_rank / brand / state / is_repeat absent.
                }
            )
        alt_texts_per_event.append(alts)
    batch = _make_batch(
        n_events=n_events,
        J=J,
        chosen_indices=[0] * n_events,
        alt_texts_per_event=alt_texts_per_event,
    )
    fitted = STMLPChoice(
        hidden_dim=8,
        n_epochs=1,
        batch_size=2,
        patience=1,
        encoder=encoder,
        seed=0,
    ).fit(batch, batch)
    scores = fitted.score_events(batch)
    assert len(scores) == n_events
    assert all(s.shape == (J,) for s in scores)


# ---------------------------------------------------------------------------
# Encoder call-counting (local cache behaviour)
# ---------------------------------------------------------------------------


class _CountingEncoder:
    """Wraps a :class:`StubEncoder` and records every ``encode`` call.

    Used to check that duplicate texts within a batch are not encoded
    twice — the per-fit in-memory dict (§9 layer 2) must absorb them.
    """

    def __init__(self, inner: EncoderClient) -> None:
        self._inner = inner
        self.call_counts: List[int] = []
        self.total_texts = 0

    @property
    def encoder_id(self) -> str:
        return self._inner.encoder_id

    @property
    def d_e(self) -> int:
        return self._inner.d_e

    def encode(self, texts: list[str]) -> np.ndarray:
        self.call_counts.append(len(texts))
        self.total_texts += len(texts)
        return self._inner.encode(texts)


def test_score_events_does_not_re_encode_when_same_batch():
    """Calling ``score_events`` on the fit batch encodes 0 new texts.

    The per-fit local dict is populated during ``fit`` and reused on the
    first ``score_events`` call against the same batch.
    """
    inner = StubEncoder(d_e=16)
    spy = _CountingEncoder(inner)
    batch = _planted_signal_batch(n_events=5, J=3, keyword="theta", seed=13)
    fitted = STMLPChoice(
        hidden_dim=8,
        n_epochs=1,
        batch_size=4,
        patience=1,
        encoder=spy,
        seed=0,
    ).fit(batch, batch)
    before = spy.total_texts
    _ = fitted.score_events(batch)
    after = spy.total_texts
    assert after == before, (
        f"score_events should hit the local cache entirely; "
        f"encoded {after - before} fresh texts."
    )


def test_score_events_probabilities_sum_to_one_after_softmax_regression():
    """Regression guard for finite logits on an unseen batch."""
    encoder = StubEncoder(d_e=16)
    train = _planted_signal_batch(n_events=6, J=3, keyword="iota", seed=21)
    test = _planted_signal_batch(n_events=4, J=3, keyword="iota", seed=22)
    fitted = STMLPChoice(
        hidden_dim=8,
        n_epochs=2,
        batch_size=2,
        patience=1,
        encoder=encoder,
        seed=0,
    ).fit(train, train)
    scores = fitted.score_events(test)
    for s in scores:
        assert np.all(np.isfinite(s))
        e = np.exp(s - s.max())
        probs = e / e.sum()
        assert abs(float(probs.sum()) - 1.0) < 1e-6
        # After softmax every coord must sit in [0, 1].
        assert float(probs.min()) >= 0.0
        assert float(probs.max()) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# FittedBaseline protocol conformance
# ---------------------------------------------------------------------------


def test_fitted_satisfies_fitted_baseline_protocol():
    """``STMLPFitted`` ducks into :class:`FittedBaseline`."""
    encoder = StubEncoder(d_e=16)
    batch = _planted_signal_batch(n_events=4, J=2, keyword="kappa", seed=3)
    fitted = STMLPChoice(
        hidden_dim=4,
        n_epochs=1,
        batch_size=2,
        patience=1,
        encoder=encoder,
        seed=0,
    ).fit(batch, batch)
    assert isinstance(fitted, FittedBaseline)
    assert isinstance(fitted.description, str) and fitted.description
    assert fitted.name == "ST+MLP"
