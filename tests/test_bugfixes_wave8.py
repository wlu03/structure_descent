"""Wave 8 bug-fix tests.

Two silent-degradation patterns fixed in Wave 8:

1. ``src.train.loop.try_import_subsample_weights`` silently swallowed both
   ImportError and runtime failures. It now emits a ``WARNING`` log on
   each fallback path with a message that distinguishes the two cases.

2. ``src.train.loop._compute_batch_loss`` unconditionally passed
   ``prices=None`` to the regularizer, so the §9.2 monotonicity term was
   dead code regardless of ``cfg.monotonicity_enabled``. ``iter_batches``
   now accepts an optional ``prices`` tensor and ``_compute_batch_loss``
   reads ``batch.get("prices")`` from the batch dict.
"""

from __future__ import annotations

import importlib
import logging
import sys

import pandas as pd
import pytest
import torch

from src.train import loop as loop_mod
from src.train.loop import (
    _compute_batch_loss,
    iter_batches,
    try_import_subsample_weights,
)


# ---------------------------------------------------------------------------
# Bug fix 1: try_import_subsample_weights warnings
# ---------------------------------------------------------------------------


def test_subsample_warns_on_import_failure(monkeypatch, caplog):
    """ImportError path emits a WARNING that names the import error."""
    # Force the import to fail by stashing ``src.train.subsample`` as None
    # in sys.modules; Python then raises ImportError on `from X import Y`.
    monkeypatch.setitem(sys.modules, "src.train.subsample", None)

    caplog.set_level(logging.WARNING, logger="src.train.loop")
    selected, weights = try_import_subsample_weights(
        pd.DataFrame({"customer_id": ["x"]}), n_customers=10
    )

    assert selected is None and weights is None
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1, f"expected 1 WARNING, got {len(warns)}: {warns!r}"
    assert "not importable" in warns[0].getMessage()
    assert "ω_t=1" in warns[0].getMessage()


def test_subsample_warns_on_runtime_failure(monkeypatch, caplog):
    """Runtime failure path emits a DIFFERENT WARNING that names the exception."""
    # Install a fake subsample module whose subsample_customers raises.
    fake_mod = type(sys)("src.train.subsample")

    def _boom(*_args, **_kwargs):
        raise KeyError("routine")  # classic "state_features didn't run" signal

    fake_mod.subsample_customers = _boom  # type: ignore[attr-defined]
    fake_mod.apply_subsample = lambda *a, **k: (None, None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.train.subsample", fake_mod)

    caplog.set_level(logging.WARNING, logger="src.train.loop")
    selected, weights = try_import_subsample_weights(
        pd.DataFrame({"customer_id": ["x"]}), n_customers=10
    )

    assert selected is None and weights is None
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    msg = warns[0].getMessage()
    # Must distinguish from the import-failure path.
    assert "raised" in msg
    assert "KeyError" in msg or "routine" in msg
    assert "state_features" in msg


def test_subsample_no_warn_when_n_customers_is_none(caplog):
    """The n_customers=None short-circuit must not log — it's the normal path."""
    caplog.set_level(logging.WARNING, logger="src.train.loop")
    selected, weights = try_import_subsample_weights(
        pd.DataFrame({"customer_id": ["x"]}), n_customers=None
    )
    assert selected is None and weights is None
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warns == [], f"unexpected WARNING(s): {warns!r}"


# ---------------------------------------------------------------------------
# Bug fix 2: prices threaded through iter_batches + _compute_batch_loss
# ---------------------------------------------------------------------------


def _tiny_synthetic(n: int = 4, j: int = 10, k: int = 3, d_e: int = 768, p: int = 26):
    torch.manual_seed(0)
    z_d = torch.randn(n, p)
    E = torch.nn.functional.normalize(torch.randn(n, j, k, d_e), dim=-1)
    c_star = torch.randint(0, j, (n,))
    omega = torch.ones(n)
    return z_d, E, c_star, omega


def test_iter_batches_omits_prices_when_none():
    """Omitting the prices kwarg means batches lack a 'prices' key (default)."""
    z_d, E, c_star, omega = _tiny_synthetic(n=4)
    batches = list(iter_batches(z_d, E, c_star, omega, batch_size=2, shuffle=False))
    assert len(batches) == 2
    for b in batches:
        assert set(b.keys()) == {"z_d", "E", "c_star", "omega"}
        assert "prices" not in b


def test_iter_batches_threads_prices_when_supplied():
    """Supplying a prices tensor means every yielded batch has a 'prices' key of shape (batch, J)."""
    z_d, E, c_star, omega = _tiny_synthetic(n=4)
    prices = torch.arange(40, dtype=torch.float32).reshape(4, 10)
    batches = list(
        iter_batches(
            z_d, E, c_star, omega, batch_size=2, shuffle=False, prices=prices
        )
    )
    assert len(batches) == 2
    for b in batches:
        assert "prices" in b
        assert b["prices"].shape == (2, 10)
    # And the values should slice the original row-wise.
    torch.testing.assert_close(batches[0]["prices"], prices[:2])
    torch.testing.assert_close(batches[1]["prices"], prices[2:])


def test_iter_batches_rejects_malformed_prices():
    """Wrong-shape prices raise a clear ValueError."""
    z_d, E, c_star, omega = _tiny_synthetic(n=4)
    with pytest.raises(ValueError, match="prices first dim"):
        list(
            iter_batches(
                z_d, E, c_star, omega,
                batch_size=2, shuffle=False,
                prices=torch.zeros(5, 10),  # wrong N
            )
        )
    with pytest.raises(ValueError, match="prices must be 2-D"):
        list(
            iter_batches(
                z_d, E, c_star, omega,
                batch_size=2, shuffle=False,
                prices=torch.zeros(4),  # wrong rank
            )
        )


def test_monotonicity_regularizer_active_when_prices_present():
    """With prices + enabled=True the monotonicity term contributes; with enabled=False it is zero."""
    from src.model.po_leu import POLEU
    from src.train.regularizers import RegularizerConfig, combined_regularizer

    torch.manual_seed(0)
    model = POLEU()

    z_d, E, c_star, omega = _tiny_synthetic(n=4)
    # Construct a batch where the chosen alternative has the highest price
    # per row — the steepest ascending price path, which should produce a
    # nonzero monotonicity penalty if the financial head's price-gradient
    # goes the wrong way for any random-init model.
    prices = torch.zeros(4, 10)
    for b in range(4):
        order = torch.randperm(10)
        prices[b] = torch.linspace(1.0, 100.0, 10)[order]
    batch = {
        "z_d": z_d, "E": E, "c_star": c_star, "omega": omega, "prices": prices,
    }

    # Use a large λ so the (naturally small at random init) monotonicity
    # penalty is trivially detectable against the ~log(J) ≈ 2.3 data loss.
    LAMBDA = 1e6

    cfg_on = RegularizerConfig(
        weight_l2=0.0,
        salience_entropy=0.0,
        monotonicity_enabled=True,
        monotonicity=LAMBDA,
        diversity=0.0,
    )
    loss_on = _compute_batch_loss(model, batch, cfg_on)

    cfg_off = RegularizerConfig(
        weight_l2=0.0,
        salience_entropy=0.0,
        monotonicity_enabled=False,
        monotonicity=LAMBDA,
        diversity=0.0,
    )
    loss_off = _compute_batch_loss(model, batch, cfg_off)

    # With λ=1e6 the on/off gap is at least ~1e-1 on random-init; if it
    # comes out identical the term is still dead code.
    diff = (loss_on - loss_off).abs().item()
    assert diff > 1e-4, (
        f"monotonicity term appears inert even with enabled=True + prices; "
        f"loss_on={loss_on.item()}, loss_off={loss_off.item()}, diff={diff:.3e}"
    )


def test_monotonicity_regularizer_inert_when_prices_absent():
    """No prices in the batch dict -> monotonicity term is zero even with enabled=True."""
    from src.model.po_leu import POLEU
    from src.train.regularizers import RegularizerConfig

    torch.manual_seed(0)
    model = POLEU()
    z_d, E, c_star, omega = _tiny_synthetic(n=4)
    batch = {"z_d": z_d, "E": E, "c_star": c_star, "omega": omega}  # no prices

    cfg_on = RegularizerConfig(
        weight_l2=0.0,
        salience_entropy=0.0,
        monotonicity_enabled=True,
        monotonicity=1.0,
        diversity=0.0,
    )
    cfg_off = RegularizerConfig(
        weight_l2=0.0,
        salience_entropy=0.0,
        monotonicity_enabled=False,
        monotonicity=1.0,
        diversity=0.0,
    )
    loss_on = _compute_batch_loss(model, batch, cfg_on)
    loss_off = _compute_batch_loss(model, batch, cfg_off)

    # Without prices there's nothing for the monotonicity term to act on,
    # so enabling vs. disabling must produce the same loss.
    torch.testing.assert_close(loss_on, loss_off)
