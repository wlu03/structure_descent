"""Lightweight sanity check for the PO-LEU config tree.

Loads every YAML under `configs/`, asserts it parses, and for each
`ablation_*.yaml` asserts its structure matches the same top-level keys
as `configs/default.yaml` (i.e. ablation files are standalone, not diffs)
while still carrying the required `ablation` self-documentation block.

Run:
    python scripts/validate_configs.py
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Any

import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")


# Ablation files may add exactly these extra model-level keys beyond
# whatever default.yaml carries. Everything else must either already be
# present in default.yaml or is a known structural override.
ALLOWED_EXTRA_MODEL_KEYS: set[str] = {
    "backbone",   # A7/A8 ConcatUtility / FiLMUtility switch
}
ALLOWED_EXTRA_ATTR_HEAD_KEYS: set[str] = {
    "names",              # A1/A2/A3 named-vs-latent attributes
    "person_dependent",   # A5 flag (config-only at this stage)
}


def _load(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a dict")
    return data


def validate() -> int:
    default_path = os.path.join(CONFIGS_DIR, "default.yaml")
    default_cfg = _load(default_path)
    default_top = set(default_cfg.keys())
    default_model_keys = set(default_cfg["model"].keys())
    default_attr_head_keys = set(
        default_cfg["model"].get("attribute_heads", {}).keys()
    )

    ablation_paths = sorted(glob.glob(os.path.join(CONFIGS_DIR, "ablation_*.yaml")))
    if not ablation_paths:
        print(f"[validate_configs] no ablation files found in {CONFIGS_DIR}",
              file=sys.stderr)
        return 1

    seen_ids: set[str] = set()
    errors: list[str] = []
    for path in ablation_paths:
        name = os.path.basename(path)
        try:
            cfg = _load(path)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{name}: parse failure: {exc}")
            continue

        # top-level keys: ablation file must carry everything default has
        # plus the `ablation` self-documentation block.
        missing = default_top - set(cfg.keys())
        if missing:
            errors.append(f"{name}: missing top-level keys: {sorted(missing)}")
        if "ablation" not in cfg:
            errors.append(f"{name}: missing `ablation` block")
            continue

        extra_top = set(cfg.keys()) - default_top - {"ablation"}
        if extra_top:
            errors.append(
                f"{name}: unexpected top-level keys beyond default: {sorted(extra_top)}"
            )

        ablation_id = cfg["ablation"].get("id")
        description = cfg["ablation"].get("description")
        if not isinstance(ablation_id, str) or not ablation_id:
            errors.append(f"{name}: ablation.id must be a non-empty string")
        elif ablation_id in seen_ids:
            errors.append(f"{name}: duplicate ablation.id {ablation_id!r}")
        else:
            seen_ids.add(ablation_id)
        if not isinstance(description, str) or not description.strip():
            errors.append(f"{name}: ablation.description must be a non-empty string")

        # model keys: only allow-listed additions beyond default
        model_block = cfg.get("model") or {}
        extra_model = set(model_block.keys()) - default_model_keys
        bad_model = extra_model - ALLOWED_EXTRA_MODEL_KEYS
        if bad_model:
            errors.append(
                f"{name}: unexpected model-level keys: {sorted(bad_model)}"
            )

        # attribute_heads keys: only allow-listed additions beyond default
        attr_head_block = model_block.get("attribute_heads") or {}
        extra_attr_heads = set(attr_head_block.keys()) - default_attr_head_keys
        bad_attr_heads = extra_attr_heads - ALLOWED_EXTRA_ATTR_HEAD_KEYS
        if bad_attr_heads:
            errors.append(
                f"{name}: unexpected model.attribute_heads keys: "
                f"{sorted(bad_attr_heads)}"
            )

        # For K sweeps, enforce model.K == outcomes.K inline.
        if ablation_id and ablation_id.startswith("K"):
            m_k = model_block.get("K")
            o_k = (cfg.get("outcomes") or {}).get("K")
            if m_k != o_k:
                errors.append(
                    f"{name}: K-sweep requires model.K == outcomes.K, got "
                    f"{m_k!r} vs {o_k!r}"
                )

    if errors:
        for err in errors:
            print(f"[validate_configs] ERROR {err}", file=sys.stderr)
        return 1

    print(f"[validate_configs] OK: {len(ablation_paths)} ablation files validated.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(validate())
