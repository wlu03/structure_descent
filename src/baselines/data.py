"""
Data loader for the baseline suite.

Converts the existing Structure Descent pipeline's choice events and base
features into BaselineEventBatch instances that every baseline can consume.

There are two entry points:

1. `make_baseline_batch(choices, base_features_per_event, base_feature_names)`
   The pure conversion function. The caller supplies the choice events
   (output of src.data_prep.build_choice_sets) and a parallel list of
   per-event base feature matrices (one [n_alts, n_base_terms] array per
   event). This is the format-agnostic path; use it from a notebook cell
   after computing base features.

2. `load_from_checkpoints(data_dir)`
   The convenience path. Reads pickled splits from the project's standard
   checkpoint directory if they exist. Expected files:

       <data_dir>/train_choices.pkl
       <data_dir>/val_choices.pkl
       <data_dir>/test_choices.pkl
       <data_dir>/train_base_features.pkl
       <data_dir>/val_base_features.pkl
       <data_dir>/test_base_features.pkl

   Each *_base_features.pkl is a dict with keys:
       'features_list'  : list of np.ndarray, shape (n_alts, n_base_terms)
       'feature_names'  : list of str

   IMPORTANT: notebook 02_dsl_features.ipynb currently saves *structure*
   features (post DSLStructure projection), not *base* features, and only
   for the train split. To use load_from_checkpoints you need to extend
   notebook 02 to additionally save the raw 12-column base feature
   matrices for all three splits under the filenames above. This is a
   one-cell change; see the migration note in the project README or
   contact the maintainer.

   If the files are missing, this function raises FileNotFoundError with
   an actionable message instead of silently using a degraded format.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .base import BaselineEventBatch


REQUIRED_CHECKPOINT_FILES = (
    "train_choices.pkl",
    "val_choices.pkl",
    "test_choices.pkl",
    "train_base_features.pkl",
    "val_base_features.pkl",
    "test_base_features.pkl",
)


def make_baseline_batch(
    choices: List[dict],
    base_features_per_event: List[np.ndarray],
    base_feature_names: List[str],
) -> BaselineEventBatch:
    """
    Convert choice events + parallel base-feature matrices into a
    BaselineEventBatch.

    Parameters
    ----------
    choices : list of dict
        Output of src.data_prep.build_choice_sets. Each dict must contain
        at least: 'customer_id', 'category', 'chosen_idx'. Conventional
        additional keys: 'chosen_asin', 'choice_asins', 'metadata'.
    base_features_per_event : list of np.ndarray
        One (n_alts, n_base_terms) array per event, parallel to `choices`.
        These are the raw DSL base features (Layer 1 + Layer 2), not the
        per-structure projections.
    base_feature_names : list of str
        Column names for the matrices, length == n_base_terms.

    Returns
    -------
    BaselineEventBatch
    """
    n_choices = len(choices)
    n_features = len(base_features_per_event)
    if n_choices != n_features:
        raise ValueError(
            f"choices ({n_choices}) and base_features_per_event "
            f"({n_features}) must be parallel lists of the same length."
        )
    if n_choices == 0:
        raise ValueError("Cannot build a BaselineEventBatch from zero events.")

    n_base = len(base_feature_names)
    first = np.asarray(base_features_per_event[0])
    if first.ndim != 2 or first.shape[1] != n_base:
        raise ValueError(
            f"Each base feature matrix must have shape (n_alts, {n_base}), "
            f"got {first.shape}."
        )

    chosen_indices: List[int] = []
    customer_ids: List[str] = []
    categories: List[str] = []
    metadata: List[dict] = []

    for ev in choices:
        if "chosen_idx" not in ev:
            raise KeyError(
                "Each choice event must have a 'chosen_idx' key. "
                "Missing in event: " + repr({k: ev.get(k) for k in list(ev)[:3]})
            )
        idx = ev["chosen_idx"]
        # build_choice_sets supports a multi-resample mode where chosen_idx
        # is a list. The baseline suite operates on single-resample data.
        if isinstance(idx, (list, tuple, np.ndarray)):
            raise ValueError(
                "Multi-resample choice events (chosen_idx is a list) are "
                "not supported by the baseline loader. Re-export with "
                "n_resamples=1 in build_choice_sets."
            )
        chosen_indices.append(int(idx))
        customer_ids.append(str(ev["customer_id"]))
        categories.append(str(ev["category"]))
        metadata.append(dict(ev.get("metadata", {})))

    return BaselineEventBatch(
        base_features_list=[np.asarray(m) for m in base_features_per_event],
        base_feature_names=list(base_feature_names),
        chosen_indices=chosen_indices,
        customer_ids=customer_ids,
        categories=categories,
        metadata=metadata,
        raw_events=list(choices),
    )


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _check_checkpoint_dir(data_dir: Path) -> None:
    missing = [f for f in REQUIRED_CHECKPOINT_FILES if not (data_dir / f).exists()]
    if missing:
        msg = (
            f"Missing baseline checkpoint files in {data_dir}/:\n"
            + "\n".join(f"  - {f}" for f in missing)
            + "\n\nTo produce these files, extend notebook "
            "02_dsl_features.ipynb so that it saves the raw 12-column base\n"
            "feature matrices (not the per-structure projection) for ALL\n"
            "three splits, under the filenames listed above.\n\n"
            "Alternatively, build BaselineEventBatch instances directly\n"
            "from a notebook cell with:\n"
            "    from src.baselines.data import make_baseline_batch\n"
            "    train_batch = make_baseline_batch(train_choices,\n"
            "                                     train_base_features,\n"
            "                                     base_feature_names)\n"
        )
        raise FileNotFoundError(msg)


def load_from_checkpoints(
    data_dir: str = "data",
) -> Tuple[BaselineEventBatch, BaselineEventBatch, BaselineEventBatch]:
    """
    Load (train, val, test) BaselineEventBatch from the project's checkpoint
    directory. Raises FileNotFoundError with an actionable message if any
    required file is missing.
    """
    d = Path(data_dir)
    _check_checkpoint_dir(d)

    train_choices = _load_pickle(d / "train_choices.pkl")
    val_choices = _load_pickle(d / "val_choices.pkl")
    test_choices = _load_pickle(d / "test_choices.pkl")

    train_features = _load_pickle(d / "train_base_features.pkl")
    val_features = _load_pickle(d / "val_base_features.pkl")
    test_features = _load_pickle(d / "test_base_features.pkl")

    def _bundle(features_payload: dict, choices: list) -> BaselineEventBatch:
        if "features_list" not in features_payload or "feature_names" not in features_payload:
            raise KeyError(
                "Base features pickle must contain keys 'features_list' "
                "and 'feature_names'. Got: " + repr(list(features_payload.keys()))
            )
        return make_baseline_batch(
            choices=choices,
            base_features_per_event=features_payload["features_list"],
            base_feature_names=features_payload["feature_names"],
        )

    return (
        _bundle(train_features, train_choices),
        _bundle(val_features, val_choices),
        _bundle(test_features, test_choices),
    )


def save_baseline_batch(
    batch: BaselineEventBatch,
    features_path: Path | str,
    choices_path: Optional[Path | str] = None,
) -> None:
    """
    Persist a BaselineEventBatch to the on-disk format expected by
    load_from_checkpoints. Useful for caching computed batches between
    notebook runs.
    """
    fp = Path(features_path)
    payload = {
        "features_list": batch.base_features_list,
        "feature_names": batch.base_feature_names,
    }
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "wb") as f:
        pickle.dump(payload, f)

    if choices_path is not None and batch.raw_events is not None:
        cp = Path(choices_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        with open(cp, "wb") as f:
            pickle.dump(batch.raw_events, f)
