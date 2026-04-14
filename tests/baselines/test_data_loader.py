"""
Tests for the baseline data loader.

Exercises both entry points:
  - make_baseline_batch (pure conversion)
  - load_from_checkpoints (round-trip via save_baseline_batch + load)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from src.baselines import BaselineEventBatch
from src.baselines.data import (
    REQUIRED_CHECKPOINT_FILES,
    load_from_checkpoints,
    make_baseline_batch,
    save_baseline_batch,
)


def _toy_choices(n: int = 5, n_alts: int = 4) -> list:
    return [
        {
            "customer_id": f"c{i}",
            "category": f"cat{i % 2}",
            "chosen_idx": i % n_alts,
            "chosen_asin": f"a{i}",
            "choice_asins": [f"a{i}_{k}" for k in range(n_alts)],
            "metadata": {"is_repeat": bool(i % 2), "price": 10.0 + i},
        }
        for i in range(n)
    ]


def _toy_features(n: int = 5, n_alts: int = 4, n_base: int = 3) -> list:
    rng = np.random.default_rng(0)
    return [rng.normal(size=(n_alts, n_base)) for _ in range(n)]


def test_make_baseline_batch_minimal():
    choices = _toy_choices(n=5, n_alts=4)
    feats = _toy_features(n=5, n_alts=4, n_base=3)
    names = ["a", "b", "c"]

    batch = make_baseline_batch(choices, feats, names)

    assert isinstance(batch, BaselineEventBatch)
    assert batch.n_events == 5
    assert batch.n_alternatives == 4
    assert batch.n_base_terms == 3
    assert batch.base_feature_names == names
    assert batch.chosen_indices == [0, 1, 2, 3, 0]
    assert batch.customer_ids == ["c0", "c1", "c2", "c3", "c4"]
    assert batch.categories == ["cat0", "cat1", "cat0", "cat1", "cat0"]
    assert batch.raw_events == choices  # defensive copy of the same content
    assert batch.metadata[1]["is_repeat"] is True


def test_make_baseline_batch_rejects_misaligned_lengths():
    choices = _toy_choices(n=5)
    feats = _toy_features(n=4)
    with pytest.raises(ValueError, match="parallel"):
        make_baseline_batch(choices, feats, ["a", "b", "c"])


def test_make_baseline_batch_rejects_missing_chosen_idx():
    choices = _toy_choices(n=2)
    del choices[0]["chosen_idx"]
    feats = _toy_features(n=2)
    with pytest.raises(KeyError, match="chosen_idx"):
        make_baseline_batch(choices, feats, ["a", "b", "c"])


def test_make_baseline_batch_rejects_multi_resample():
    choices = _toy_choices(n=2)
    choices[0]["chosen_idx"] = [0, 1]  # multi-resample sentinel
    feats = _toy_features(n=2)
    with pytest.raises(ValueError, match="Multi-resample"):
        make_baseline_batch(choices, feats, ["a", "b", "c"])


def test_make_baseline_batch_rejects_wrong_feature_shape():
    choices = _toy_choices(n=3)
    feats = _toy_features(n=3, n_base=4)
    with pytest.raises(ValueError, match="n_alts"):
        make_baseline_batch(choices, feats, ["a", "b", "c"])  # n_base=3 but feats have 4


def test_load_from_checkpoints_missing_files_actionable_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as exc:
        load_from_checkpoints(str(tmp_path))
    msg = str(exc.value)
    # All six files should be listed
    for required in REQUIRED_CHECKPOINT_FILES:
        assert required in msg
    assert "make_baseline_batch" in msg


def test_save_and_load_roundtrip(tmp_path: Path):
    """Writing each split via save_baseline_batch then reading via
    load_from_checkpoints should round-trip identically."""
    n_alts = 4
    n_base = 3
    names = ["a", "b", "c"]

    for split in ("train", "val", "test"):
        choices = _toy_choices(n=6, n_alts=n_alts)
        feats = _toy_features(n=6, n_alts=n_alts, n_base=n_base)
        batch = make_baseline_batch(choices, feats, names)
        save_baseline_batch(
            batch,
            features_path=tmp_path / f"{split}_base_features.pkl",
            choices_path=tmp_path / f"{split}_choices.pkl",
        )

    train, val, test = load_from_checkpoints(str(tmp_path))

    for batch, label in [(train, "train"), (val, "val"), (test, "test")]:
        assert batch.n_events == 6
        assert batch.n_alternatives == n_alts
        assert batch.n_base_terms == n_base
        assert batch.base_feature_names == names
        # chosen_indices should match our toy generator
        assert batch.chosen_indices == [0, 1, 2, 3, 0, 1]


def test_load_from_checkpoints_payload_validation(tmp_path: Path):
    """A features pickle missing required keys should raise KeyError."""
    for split in ("train", "val", "test"):
        with open(tmp_path / f"{split}_choices.pkl", "wb") as f:
            pickle.dump(_toy_choices(n=2), f)
        with open(tmp_path / f"{split}_base_features.pkl", "wb") as f:
            pickle.dump({"oops": "wrong schema"}, f)

    with pytest.raises(KeyError, match="features_list"):
        load_from_checkpoints(str(tmp_path))
