"""Shared fixtures for the PO-LEU test suite.

`synthetic_batch` is the canonical batch used by every module's shape and
gradient tests. Dimensions match redesign.md defaults: B=4, J=10, K=3,
d_e=768, p=26.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SyntheticBatch:
    """A seeded synthetic batch matching the default PO-LEU dimensions."""

    z_d: torch.Tensor       # (B, p)
    E: torch.Tensor         # (B, J, K, d_e), L2-normalized along last dim
    c_star: torch.Tensor    # (B,) int64, chosen index in {0, ..., J-1}
    omega: torch.Tensor     # (B,) float, importance weights (ones by default)

    B: int = 4
    J: int = 10
    K: int = 3
    d_e: int = 768
    p: int = 26


@pytest.fixture(scope="session")
def synthetic_batch() -> SyntheticBatch:
    """Seeded synthetic batch for all module tests."""
    g = torch.Generator().manual_seed(0)

    B, J, K, d_e, p = 4, 10, 3, 768, 26

    z_d = torch.randn(B, p, generator=g)

    E_raw = torch.randn(B, J, K, d_e, generator=g)
    E = torch.nn.functional.normalize(E_raw, p=2, dim=-1)

    c_star = torch.randint(0, J, (B,), generator=g, dtype=torch.int64)
    omega = torch.ones(B)

    return SyntheticBatch(z_d=z_d, E=E, c_star=c_star, omega=omega,
                          B=B, J=J, K=K, d_e=d_e, p=p)


@pytest.fixture(scope="session")
def default_config() -> dict:
    """Load configs/default.yaml once per session."""
    import yaml
    with open(REPO_ROOT / "configs" / "default.yaml") as fh:
        return yaml.safe_load(fh)


@pytest.fixture()
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Isolated cache dir per test for cache-touching modules."""
    d = tmp_path / "cache"
    d.mkdir()
    return d
