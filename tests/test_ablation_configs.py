"""Tests for the §11 ablation YAML set.

Covers the Wave 7 "Ablation YAML set" deliverable in
`configs/ablation_*.yaml`. Each file is expected to be a self-contained
standalone YAML (not a diff) that inherits the structure of
`configs/default.yaml` and overrides only what the ablation calls for.
"""

from __future__ import annotations

import glob
import os
from typing import Any

import pytest
import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")


def _ablation_paths() -> list[str]:
    paths = sorted(glob.glob(os.path.join(CONFIGS_DIR, "ablation_*.yaml")))
    assert paths, f"no ablation_*.yaml files found in {CONFIGS_DIR}"
    return paths


def _load(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"{path} did not parse to a dict"
    return data


def _load_by_id(ablation_id: str) -> dict[str, Any]:
    filename = f"ablation_{ablation_id}.yaml"
    path = os.path.join(CONFIGS_DIR, filename)
    assert os.path.exists(path), f"missing expected config: {filename}"
    return _load(path)


# ---------------------------------------------------------------------------
# global structural invariants
# ---------------------------------------------------------------------------


def test_every_ablation_yaml_parses() -> None:
    required_top_level = {"model", "outcomes", "train", "ablation"}
    for path in _ablation_paths():
        cfg = _load(path)
        missing = required_top_level - set(cfg.keys())
        assert not missing, f"{os.path.basename(path)} missing keys: {missing}"


def test_ablation_ids_are_unique() -> None:
    ids: list[str] = []
    for path in _ablation_paths():
        cfg = _load(path)
        ablation_block = cfg.get("ablation")
        assert isinstance(ablation_block, dict), (
            f"{os.path.basename(path)}: `ablation` must be a mapping"
        )
        aid = ablation_block.get("id")
        assert isinstance(aid, str) and aid, (
            f"{os.path.basename(path)}: ablation.id must be a non-empty string"
        )
        ids.append(aid)
    assert len(ids) == len(set(ids)), f"duplicate ablation.id values: {ids}"


def test_all_ablation_files_have_description() -> None:
    for path in _ablation_paths():
        cfg = _load(path)
        desc = cfg["ablation"].get("description")
        assert isinstance(desc, str) and desc.strip(), (
            f"{os.path.basename(path)}: ablation.description must be non-empty"
        )


# ---------------------------------------------------------------------------
# primary ablations (A1..A8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ablation_id, expected_M",
    [("M3", 3), ("M10", 10), ("M20", 20)],
)
def test_M_sweep_values(ablation_id: str, expected_M: int) -> None:
    cfg = _load_by_id(ablation_id)
    assert cfg["model"]["M"] == expected_M


def test_softplus_weights_override() -> None:
    cfg = _load_by_id("softplus_weights")
    assert cfg["model"]["weight_net"]["normalization"] == "softplus"


def test_uniform_salience_flag() -> None:
    cfg = _load_by_id("uniform_salience")
    assert cfg["model"]["uniform_salience"] is True


def test_concat_and_film_backbone() -> None:
    concat_cfg = _load_by_id("concat_utility")
    assert concat_cfg["model"]["backbone"] == "concat_utility"

    film_cfg = _load_by_id("film_utility")
    assert film_cfg["model"]["backbone"] == "film_utility"


# ---------------------------------------------------------------------------
# secondary ablations (K, tau, encoder, generator, no-subsample)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ablation_id, expected_K",
    [("K1", 1), ("K5", 5), ("K7", 7)],
)
def test_K_sweep_consistency(ablation_id: str, expected_K: int) -> None:
    cfg = _load_by_id(ablation_id)
    assert cfg["model"]["K"] == expected_K
    assert cfg["outcomes"]["K"] == expected_K
    assert cfg["model"]["K"] == cfg["outcomes"]["K"]


@pytest.mark.parametrize(
    "ablation_id, expected_tau",
    [("tau_0p5", 0.5), ("tau_2p0", 2.0)],
)
def test_tau_sweep_values(ablation_id: str, expected_tau: float) -> None:
    cfg = _load_by_id(ablation_id)
    assert cfg["model"]["temperature"] == pytest.approx(expected_tau)


def test_encoder_swap_changes_d_e() -> None:
    cfg = _load_by_id("encoder_1024")
    assert cfg["model"]["d_e"] == 1024
    # the encoder model_id should also have been swapped away from default
    default_cfg = _load(os.path.join(CONFIGS_DIR, "default.yaml"))
    assert (
        cfg["outcomes"]["encoder"]["model_id"]
        != default_cfg["outcomes"]["encoder"]["model_id"]
    )


def test_generator_swap_changes_generator_id() -> None:
    cfg = _load_by_id("generator_alt")
    default_cfg = _load(os.path.join(CONFIGS_DIR, "default.yaml"))
    assert (
        cfg["outcomes"]["generator"]["model_id"]
        != default_cfg["outcomes"]["generator"]["model_id"]
    )
    # everything else in the outcomes block should be identical
    for key in ("temperature", "top_p", "max_tokens", "prompt_version"):
        assert (
            cfg["outcomes"]["generator"][key]
            == default_cfg["outcomes"]["generator"][key]
        ), f"generator.{key} should not change for the generator-swap ablation"


def test_no_subsample_is_explicitly_false() -> None:
    cfg = _load_by_id("no_subsample")
    assert cfg["subsample"]["enabled"] is False
