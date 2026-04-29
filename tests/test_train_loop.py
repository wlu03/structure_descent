"""Tests for :mod:`src.train.loop` (redesign.md §9.1, §9.3, §9.4).

Synthetic-only: B=64 events with the default shapes from Appendix B.
Nothing in this suite loads real data or trains beyond 2 epochs.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src.model.po_leu import POLEU
from src.train import loop as loop_mod
from src.train.loop import (
    TrainConfig,
    TrainState,
    evaluate_nll,
    fit,
    iter_batches,
    make_optimizer_and_scheduler,
    train_one_epoch,
    try_import_subsample_weights,
)


# --- Synthetic data helpers -------------------------------------------------


def _make_synthetic(
    *,
    N: int = 64,
    J: int = 10,
    K: int = 3,
    d_e: int = 768,
    p: int = 26,
    seed: int = 0,
) -> dict:
    """Build a synthetic in-memory dataset sized for fast CPU tests."""
    g = torch.Generator().manual_seed(seed)
    z_d = torch.randn(N, p, generator=g)
    E_raw = torch.randn(N, J, K, d_e, generator=g)
    E = torch.nn.functional.normalize(E_raw, p=2, dim=-1)
    c_star = torch.randint(0, J, (N,), generator=g, dtype=torch.int64)
    omega = torch.ones(N)
    return {
        "z_d": z_d,
        "E": E,
        "c_star": c_star,
        "omega": omega,
        "N": N,
        "J": J,
        "K": K,
        "d_e": d_e,
        "p": p,
    }


def _make_model() -> POLEU:
    """Default-config POLEU for training-loop tests."""
    return POLEU()


def _batches_fn(data: dict, batch_size: int, shuffle: bool = False):
    """Factory for fresh iterators (fit expects zero-arg callables)."""

    def _fn():
        return iter_batches(
            data["z_d"],
            data["E"],
            data["c_star"],
            data["omega"],
            batch_size=batch_size,
            shuffle=shuffle,
        )

    return _fn


# --- iter_batches -----------------------------------------------------------


def test_iter_batches_shapes():
    data = _make_synthetic(N=64)
    batches = list(
        iter_batches(
            data["z_d"],
            data["E"],
            data["c_star"],
            data["omega"],
            batch_size=16,
            shuffle=False,
        )
    )
    assert len(batches) == 4
    for b in batches:
        assert set(b.keys()) == {"z_d", "E", "c_star", "omega"}
        assert b["z_d"].shape == (16, data["p"])
        assert b["E"].shape == (16, data["J"], data["K"], data["d_e"])
        assert b["c_star"].shape == (16,)
        assert b["omega"].shape == (16,)


def test_iter_batches_omega_ones_default():
    data = _make_synthetic(N=10)
    batches = list(
        iter_batches(
            data["z_d"],
            data["E"],
            data["c_star"],
            omega=None,
            batch_size=10,
            shuffle=False,
        )
    )
    assert len(batches) == 1
    assert torch.allclose(batches[0]["omega"], torch.ones(10))


def test_iter_batches_last_partial_batch():
    """N=10, batch_size=4 → batches of size 4, 4, 2."""
    data = _make_synthetic(N=10)
    sizes = [
        b["z_d"].shape[0]
        for b in iter_batches(
            data["z_d"],
            data["E"],
            data["c_star"],
            data["omega"],
            batch_size=4,
            shuffle=False,
        )
    ]
    assert sizes == [4, 4, 2]


# --- TrainConfig ------------------------------------------------------------


def test_train_config_from_default():
    cfg = TrainConfig.from_default()
    # Values come from configs/default.yaml; confirm §9.3/Appendix B.
    assert cfg.batch_size == 128
    assert cfg.lr == pytest.approx(1e-3)
    assert cfg.lr_min == pytest.approx(1e-4)
    assert cfg.optimizer == "adam"
    assert cfg.beta1 == pytest.approx(0.9)
    assert cfg.beta2 == pytest.approx(0.999)
    assert cfg.max_epochs == 30
    assert cfg.early_stopping_patience == 9999  # disabled in default.yaml; runs full max_epochs
    assert cfg.grad_clip == pytest.approx(1.0)


# --- make_optimizer_and_scheduler ------------------------------------------


def test_optimizer_is_adam_with_cosine_schedule():
    model = _make_model()
    cfg = TrainConfig(lr=1e-3, lr_min=1e-4, beta1=0.9, beta2=0.999)

    opt, sched = make_optimizer_and_scheduler(model, cfg, total_steps=10)

    assert isinstance(opt, torch.optim.Adam)
    # Initial LR and betas
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert opt.param_groups[0]["betas"] == (0.9, 0.999)
    # Cosine-annealing schedule with correct eta_min
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert sched.T_max == 10
    assert sched.eta_min == pytest.approx(1e-4)


def test_make_optimizer_rejects_non_adam():
    model = _make_model()
    cfg = TrainConfig(optimizer="sgd")
    with pytest.raises(ValueError):
        make_optimizer_and_scheduler(model, cfg, total_steps=1)


# --- subsample helper -------------------------------------------------------


def test_try_import_subsample_weights_graceful(monkeypatch):
    """If apply_subsample raises (e.g., missing columns), caller gets (None, None)."""
    # Empty dataframe → build_customer_profiles will fail; the helper must
    # swallow that and return (None, None) rather than crash.
    df = pd.DataFrame()
    ids, w = try_import_subsample_weights(df, n_customers=5, seed=0)
    assert ids is None
    assert w is None


def test_try_import_subsample_weights_n_none():
    """n_customers=None short-circuits to (None, None) without touching subsample."""
    df = pd.DataFrame({"customer_id": [1, 2, 3]})
    ids, w = try_import_subsample_weights(df, n_customers=None)
    assert ids is None and w is None


def test_try_import_subsample_weights_import_failure(monkeypatch):
    """If the subsample module is unimportable, fall back to (None, None)."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "src.train.subsample":
            raise ImportError("synthetic")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    df = pd.DataFrame({"customer_id": [1]})
    ids, w = try_import_subsample_weights(df, n_customers=5)
    assert ids is None and w is None


# --- train_one_epoch --------------------------------------------------------


def test_train_one_epoch_loss_drops_over_steps():
    torch.manual_seed(0)
    data = _make_synthetic(N=64, seed=0)
    model = _make_model()
    cfg = TrainConfig(batch_size=16, lr=1e-2, lr_min=1e-4, grad_clip=1.0)

    # total_steps across the 2 epochs so cosine schedule doesn't wrap.
    total_steps = 2 * (data["N"] // cfg.batch_size)
    opt, sched = make_optimizer_and_scheduler(model, cfg, total_steps)

    mk = _batches_fn(data, cfg.batch_size, shuffle=False)

    ep1 = train_one_epoch(model, mk(), opt, sched, None, cfg)
    ep2 = train_one_epoch(model, mk(), opt, sched, None, cfg)

    assert np.isfinite(ep1["mean_loss"]) and np.isfinite(ep2["mean_loss"])
    # Mean loss should decrease across epochs on synthetic data at lr=1e-2.
    assert ep2["mean_loss"] < ep1["mean_loss"]


def test_train_one_epoch_respects_grad_clip():
    """Grads must be clipped to ‖g‖₂ ≤ grad_clip BEFORE optimizer.step.

    We patch ``torch.nn.utils.clip_grad_norm_`` at the ``loop`` module's
    import site: after clip_grad_norm_ runs we inspect the global grad
    norm across all model params. Then we delegate to the real
    clip_grad_norm_ so optimizer.step and scheduler.step keep their
    normal ordering (avoiding the "step before scheduler" warning that
    appears when optimizer.step itself is patched).
    """
    import warnings

    torch.manual_seed(0)
    data = _make_synthetic(N=32, seed=0)
    model = _make_model()
    cfg = TrainConfig(batch_size=16, lr=1e-1, lr_min=1e-4, grad_clip=0.5)

    total_steps = data["N"] // cfg.batch_size
    opt, sched = make_optimizer_and_scheduler(model, cfg, total_steps)

    observed_norms: list[float] = []
    real_clip = loop_mod.clip_grad_norm_

    def spy_clip(params, max_norm, *a, **kw):
        total_norm = real_clip(params, max_norm, *a, **kw)
        # Measure the *post-clip* grad norm directly.
        post_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                post_sq += float(p.grad.detach().pow(2).sum().item())
        observed_norms.append(post_sq ** 0.5)
        return total_norm

    # Patch in-module so train_one_epoch picks it up.
    original = loop_mod.clip_grad_norm_
    loop_mod.clip_grad_norm_ = spy_clip  # type: ignore[assignment]
    try:
        mk = _batches_fn(data, cfg.batch_size, shuffle=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            train_one_epoch(model, mk(), opt, sched, None, cfg)
    finally:
        loop_mod.clip_grad_norm_ = original  # type: ignore[assignment]

    assert len(observed_norms) >= 1
    for n in observed_norms:
        assert n <= cfg.grad_clip + 1e-3, (n, cfg.grad_clip)


# --- evaluate_nll -----------------------------------------------------------


def test_evaluate_nll_finite_and_positive():
    torch.manual_seed(0)
    data = _make_synthetic(N=32, seed=1)
    model = _make_model()
    mk = _batches_fn(data, 16)

    nll = evaluate_nll(model, mk())

    assert np.isfinite(nll)
    assert nll > 0.0
    # Upper bound: initial softmax on 10-way classification has NLL≈log(10)
    # ≈ 2.30, with some slack for randomness.
    assert nll < 5.0


# --- fit --------------------------------------------------------------------


def test_fit_runs_without_crashing():
    torch.manual_seed(0)
    data = _make_synthetic(N=32, seed=0)
    model = _make_model()
    cfg = TrainConfig(
        batch_size=16,
        lr=1e-3,
        lr_min=1e-4,
        max_epochs=2,
        early_stopping_patience=10,
        grad_clip=1.0,
    )

    total_steps = cfg.max_epochs * (data["N"] // cfg.batch_size)
    state = fit(
        model,
        train_batches_fn=_batches_fn(data, cfg.batch_size),
        val_batches_fn=_batches_fn(data, cfg.batch_size),
        train_cfg=cfg,
        reg_cfg=None,
        total_steps=total_steps,
        seed=0,
    )

    assert isinstance(state, TrainState)
    assert state.val_nll is not None
    assert np.isfinite(state.val_nll)
    assert state.epoch == cfg.max_epochs - 1  # 2 epochs → last index is 1.


def test_a1_reload_fires_without_early_stop() -> None:
    """A.1 best-checkpoint reload must fire even when patience is huge.

    Regression for the bug observed in the Apr-27 sweep: the reload was
    gated on ``state.stopped_early=True``, so with the production default
    ``early_stopping_patience=9999`` the reload never fired and runs
    silently kept their final-epoch (worse) weights. After the fix, the
    reload fires whenever a best snapshot exists and the final-epoch val
    NLL is worse than the best.
    """
    torch.manual_seed(0)
    data = _make_synthetic(N=16, seed=0)
    model = _make_model()
    cfg = TrainConfig(
        batch_size=8,
        lr=5e-3,                       # nonzero updates → val NLL moves around
        lr_min=1e-6,
        max_epochs=4,
        early_stopping_patience=9999,  # disabled, like production
        grad_clip=1.0,
    )
    total_steps = cfg.max_epochs * (data["N"] // cfg.batch_size)
    state = fit(
        model,
        train_batches_fn=_batches_fn(data, cfg.batch_size),
        val_batches_fn=_batches_fn(data, cfg.batch_size),
        train_cfg=cfg,
        reg_cfg=None,
        total_steps=total_steps,
        seed=0,
    )
    # Did NOT stop early (the patience gate is huge).
    assert not state.stopped_early
    # If the loop ever saw an improvement, A.1 should have fired and the
    # final val_nll should equal best_val_nll up to floating slack.
    if state.best_val_nll < float("inf") and state.best_val_nll < state.val_nll + 1e-9:
        # ``state.val_nll`` after reload should match ``best_val_nll`` exactly
        # (we re-evaluate to be defensive but on a fixed batch it matches).
        assert state.reloaded_best_checkpoint or state.val_nll == state.best_val_nll


def test_fit_early_stopping_triggers():
    """With patience=1 and val that never improves after epoch 1, stop early."""
    torch.manual_seed(0)
    data = _make_synthetic(N=16, seed=0)
    model = _make_model()
    cfg = TrainConfig(
        batch_size=8,
        lr=1e-9,                        # negligible updates ⇒ val NLL ~constant
        lr_min=1e-10,
        max_epochs=10,
        early_stopping_patience=1,
        grad_clip=1.0,
    )

    total_steps = cfg.max_epochs * (data["N"] // cfg.batch_size)

    seen: list[int] = []

    def on_end(s: TrainState) -> None:
        seen.append(s.epoch)

    state = fit(
        model,
        train_batches_fn=_batches_fn(data, cfg.batch_size),
        val_batches_fn=_batches_fn(data, cfg.batch_size),
        train_cfg=cfg,
        reg_cfg=None,
        total_steps=total_steps,
        seed=0,
        on_epoch_end=on_end,
    )

    assert isinstance(state, TrainState)
    # Must stop before running all max_epochs.
    assert len(seen) < cfg.max_epochs
    assert state.stopped_early


# --- regularizer integration ------------------------------------------------


def test_regularizers_integrate(monkeypatch):
    """fit() runs cleanly both with reg_cfg=None and with a provided reg_cfg.

    The regularizers sibling module may or may not exist at test time;
    we monkey-patch ``loop_mod.combined_regularizer`` with a stub so the
    test is hermetic regardless.
    """
    torch.manual_seed(0)
    data = _make_synthetic(N=16, seed=0)
    cfg = TrainConfig(
        batch_size=8, lr=1e-3, lr_min=1e-4, max_epochs=1,
        early_stopping_patience=10, grad_clip=1.0,
    )
    total_steps = data["N"] // cfg.batch_size

    # Path A: reg_cfg=None.
    model_a = _make_model()
    state_a = fit(
        model_a,
        train_batches_fn=_batches_fn(data, cfg.batch_size),
        val_batches_fn=_batches_fn(data, cfg.batch_size),
        train_cfg=cfg,
        reg_cfg=None,
        total_steps=total_steps,
        seed=0,
    )
    assert np.isfinite(state_a.val_nll)

    # Path B: reg_cfg provided. Monkey-patch the regularizer to a small,
    # differentiable term so we exercise the integration code path.
    call_count = {"n": 0}

    def stub_reg(model, intermediates, E, prices, reg_cfg, **kwargs):
        # Group-2 added z_d as a kwarg; absorb any future kwargs too.
        call_count["n"] += 1
        # Small L2 on the weight-net parameters for gradient flow.
        term = sum((p * p).sum() for p in model.weight_net.parameters())
        return 1e-6 * term

    monkeypatch.setattr(loop_mod, "combined_regularizer", stub_reg)

    reg_cfg = SimpleNamespace(weight_l2=1e-4)  # duck-typed stand-in
    model_b = _make_model()
    state_b = fit(
        model_b,
        train_batches_fn=_batches_fn(data, cfg.batch_size),
        val_batches_fn=_batches_fn(data, cfg.batch_size),
        train_cfg=cfg,
        reg_cfg=reg_cfg,
        total_steps=total_steps,
        seed=0,
    )
    assert np.isfinite(state_b.val_nll)
    assert call_count["n"] >= 1


# --- determinism ------------------------------------------------------------


def test_deterministic_under_seed():
    """Two fit() calls with the same seed and non-shuffling batches match."""
    data = _make_synthetic(N=32, seed=0)
    cfg = TrainConfig(
        batch_size=16, lr=1e-3, lr_min=1e-4, max_epochs=2,
        early_stopping_patience=10, grad_clip=1.0,
    )
    total_steps = cfg.max_epochs * (data["N"] // cfg.batch_size)

    results = []
    for _ in range(2):
        # Re-instantiate the model inside fit() so the seed controls init.
        torch.manual_seed(123)
        model = _make_model()
        state = fit(
            model,
            train_batches_fn=_batches_fn(data, cfg.batch_size, shuffle=False),
            val_batches_fn=_batches_fn(data, cfg.batch_size, shuffle=False),
            train_cfg=cfg,
            reg_cfg=None,
            total_steps=total_steps,
            seed=123,
        )
        results.append(state.val_nll)

    assert results[0] == pytest.approx(results[1], rel=0, abs=1e-7)
