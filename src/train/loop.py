"""Mini-batch training loop for PO-LEU (redesign.md §9.1, §9.3, §9.4).

Responsibilities
----------------
* Config dataclass (:class:`TrainConfig`) mirroring Appendix B / §9.3 knobs.
* Optimizer + cosine-annealing LR schedule (§9.3).
* Per-batch iterator with importance weights (§9.1 ``ω_t``).
* Per-epoch training and pure-NLL validation passes (§9.1).
* ``fit`` driver with early stopping on val NLL (§9.3, patience=5).

The regularizer integration follows §9.2: when a :class:`RegularizerConfig`
is supplied, the training loop calls
``combined_regularizer(model, intermediates, E, None, reg_cfg)`` from
:mod:`src.train.regularizers` and adds the returned scalar to the data
loss. That module is a sibling wave — we import it defensively so this
module is importable even before it lands.

The subsample-weights helper follows Appendix C: if
:mod:`src.train.subsample` is available and accepts the provided dataframe,
we reuse leverage-score sampling; otherwise we gracefully fall back to
``ω = 1`` (spec §9.1).

Nothing here does file I/O; checkpointing is a separate concern (not in
scope for this module — see the orchestrator plan).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from src.model.po_leu import cross_entropy_loss

logger = logging.getLogger(__name__)


# --- Optional regularizer import --------------------------------------------
# The sibling module ``src.train.regularizers`` ships the §9.2 helpers.
# This module must stay importable even before that module lands, and the
# training loop simply skips the regularizer term when ``reg_cfg is None``.
try:  # pragma: no cover - import-time branch, exercised in tests via monkey-patch
    from src.train.regularizers import (  # type: ignore
        RegularizerConfig,
        combined_regularizer,
    )
except ImportError:  # pragma: no cover - fallback for out-of-order waves
    RegularizerConfig = None  # type: ignore[assignment,misc]

    def combined_regularizer(*args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore[misc]
        """Sentinel regularizer used when :mod:`src.train.regularizers` is
        not yet available. Returns a zero scalar.

        Tests that exercise the regularizer branch monkey-patch this symbol
        on the module (``src.train.loop.combined_regularizer``) rather than
        depending on the sibling module.
        """
        return torch.zeros(())


__all__ = [
    "TrainConfig",
    "TrainState",
    "make_optimizer_and_scheduler",
    "iter_batches",
    "try_import_subsample_weights",
    "train_one_epoch",
    "evaluate_nll",
    "fit",
]


# --- Config -----------------------------------------------------------------


@dataclass
class TrainConfig:
    """Training hyperparameters (§9.3, Appendix B).

    Defaults match ``configs/default.yaml`` under the ``train:`` block.
    """

    batch_size: int = 128
    lr: float = 1e-3
    lr_min: float = 1e-4
    optimizer: str = "adam"
    beta1: float = 0.9
    beta2: float = 0.999
    max_epochs: int = 30
    early_stopping_patience: int = 5
    grad_clip: float = 1.0

    @classmethod
    def from_default(cls) -> "TrainConfig":
        """Load defaults from ``configs/default.yaml`` under ``train:``.

        Uses the repo-root-relative path, which matches the single
        top-level ``configs/default.yaml`` (see NOTES.md scaffold entry).
        """
        import yaml  # deferred import; pyyaml is already a project dep.

        # Walk up from this file: src/train/loop.py -> src/train -> src ->
        # repo root. This avoids hard-coding an absolute path.
        repo_root = Path(__file__).resolve().parents[2]
        cfg_path = repo_root / "configs" / "default.yaml"
        with cfg_path.open("r") as fh:
            cfg = yaml.safe_load(fh)

        train_block = cfg.get("train", {}) or {}
        return cls(
            batch_size=int(train_block.get("batch_size", cls.batch_size)),
            lr=float(train_block.get("lr", cls.lr)),
            lr_min=float(train_block.get("lr_min", cls.lr_min)),
            optimizer=str(train_block.get("optimizer", cls.optimizer)),
            beta1=float(train_block.get("beta1", cls.beta1)),
            beta2=float(train_block.get("beta2", cls.beta2)),
            max_epochs=int(train_block.get("max_epochs", cls.max_epochs)),
            early_stopping_patience=int(
                train_block.get(
                    "early_stopping_patience", cls.early_stopping_patience
                )
            ),
            grad_clip=float(train_block.get("grad_clip", cls.grad_clip)),
        )


@dataclass
class TrainState:
    """Bookkeeping passed to ``on_epoch_end`` and returned by :func:`fit`.

    Attributes
    ----------
    step:
        Global optimizer step count (batches seen across all epochs).
    epoch:
        Zero-indexed epoch counter of the most recently completed epoch.
    train_loss:
        Last training batch's total loss (data + regularizers).
    val_nll:
        Pure-data NLL on the validation set after the last epoch; ``None``
        before the first validation pass.
    best_val_nll:
        Best (lowest) validation NLL observed so far. Initialized to
        ``+inf``.
    patience_counter:
        Epochs since ``best_val_nll`` last improved.
    stopped_early:
        ``True`` iff early stopping triggered (patience exhausted).
    """

    step: int = 0
    epoch: int = 0
    train_loss: float = float("nan")
    val_nll: float | None = None
    best_val_nll: float = float("inf")
    patience_counter: int = 0
    stopped_early: bool = False


# --- Optimizer + scheduler --------------------------------------------------


def make_optimizer_and_scheduler(
    model: nn.Module,
    cfg: TrainConfig,
    total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Build Adam + cosine-annealing LR schedule (§9.3).

    Parameters
    ----------
    model:
        Module whose ``parameters()`` will be trained.
    cfg:
        Training config. Uses ``lr``, ``beta1``, ``beta2``, ``lr_min``.
    total_steps:
        Cosine schedule's ``T_max`` (§9.3 "full training horizon"). The
        LR linearly follows a cosine from ``cfg.lr`` down to ``cfg.lr_min``
        across ``total_steps`` optimizer steps.

    Returns
    -------
    (optimizer, scheduler)
        The optimizer is ``torch.optim.Adam`` and the scheduler is
        ``CosineAnnealingLR``.
    """
    if cfg.optimizer.lower() != "adam":
        # §9.3 is Adam-only; accept the field for future expansion but
        # refuse silently-incorrect behavior.
        raise ValueError(
            f"Only Adam is implemented per §9.3; got optimizer={cfg.optimizer!r}."
        )
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1, got {total_steps}.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=cfg.lr_min,
    )
    return optimizer, scheduler


# --- Batch iterator ---------------------------------------------------------


def iter_batches(
    z_d: torch.Tensor,
    E: torch.Tensor,
    c_star: torch.Tensor,
    omega: torch.Tensor | None,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
    prices: torch.Tensor | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield mini-batches of a full-in-memory dataset.

    Shape contract
    --------------
    ``z_d``: ``(N, p)``
    ``E``: ``(N, J, K, d_e)``
    ``c_star``: ``(N,)`` int64
    ``omega``: ``(N,)`` float, or ``None`` → treated as ones (§9.1 fallback)
    ``prices``: optional ``(N, J)`` float tensor of per-alternative prices;
        when supplied, each yielded batch includes a ``"prices"`` key used
        by the §9.2 monotonicity regularizer. When ``None``, no ``"prices"``
        key is emitted and the monotonicity term is inert.

    Each yielded dict has keys ``"z_d", "E", "c_star", "omega"`` and,
    when ``prices`` is provided, ``"prices"``. First-axis is reduced to
    ``batch_size`` (last batch may be shorter).

    When ``shuffle=True`` a permutation is drawn with the provided
    ``generator`` (or the default generator if ``None``); when ``False``
    the natural index order is used.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}.")

    N = int(z_d.shape[0])
    if E.shape[0] != N or c_star.shape[0] != N:
        raise ValueError(
            "z_d, E, c_star must share the first dimension; "
            f"got {z_d.shape=}, {E.shape=}, {c_star.shape=}."
        )
    if omega is None:
        omega_t = torch.ones(N, dtype=torch.float32)
    else:
        if omega.shape[0] != N:
            raise ValueError(
                f"omega first dim must equal N={N}; got {omega.shape=}."
            )
        omega_t = omega

    if prices is not None:
        if prices.shape[0] != N:
            raise ValueError(
                f"prices first dim must equal N={N}; got {prices.shape=}."
            )
        if prices.dim() != 2:
            raise ValueError(
                f"prices must be 2-D (N, J); got {prices.shape=}."
            )

    if shuffle:
        perm = torch.randperm(N, generator=generator)
    else:
        perm = torch.arange(N)

    for start in range(0, N, batch_size):
        idx = perm[start : start + batch_size]
        batch = {
            "z_d": z_d.index_select(0, idx),
            "E": E.index_select(0, idx),
            "c_star": c_star.index_select(0, idx),
            "omega": omega_t.index_select(0, idx),
        }
        if prices is not None:
            batch["prices"] = prices.index_select(0, idx)
        yield batch


# --- Optional subsampling integration ---------------------------------------


def try_import_subsample_weights(
    train_df: Any,
    n_customers: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Attempt Appendix-C leverage-score subsampling; fall back to ``(None, None)``.

    The v2.0 subsampling module lives at :mod:`src.train.subsample` and
    exposes ``subsample_customers`` / ``apply_subsample`` (see §C.6). If
    either the import fails or the call raises, we return
    ``(None, None)`` and the caller should treat ``ω_t = 1`` per §9.1.

    Parameters
    ----------
    train_df:
        Training-split dataframe (duck-typed; only the subsample module
        inspects its columns).
    n_customers:
        Budget for the subsample (see Appendix C). If ``None`` the
        function returns ``(None, None)`` without invoking subsample
        code — Appendix C §C.6 spec: "Set ``n_customers=None`` (or skip
        the call) to train on the full population with ``ω_t = 1``".
    seed:
        RNG seed forwarded to ``subsample_customers``.

    Returns
    -------
    (selected_ids, importance_weights) or (None, None)
        The successful path returns a ``(selected_ids, per_event_weights)``
        pair. ``per_event_weights`` is the event-level weight vector
        produced by ``apply_subsample`` (so it aligns with the filtered
        dataframe).
    """
    if n_customers is None:
        return (None, None)

    try:
        from src.train.subsample import (  # type: ignore
            apply_subsample,
            subsample_customers,
        )
    except ImportError:
        logger.warning(
            "src.train.subsample not importable — falling back to ω_t=1. "
            "This is expected if subsampling is disabled; if you enabled "
            "subsample.enabled=true in the config, this is a real bug."
        )
        return (None, None)

    try:
        selected_ids, customer_weights = subsample_customers(
            train_df, n_customers=n_customers, seed=seed
        )
        _filtered, event_weights = apply_subsample(
            train_df, selected_ids, customer_weights
        )
    except Exception as exc:
        # Any runtime failure (e.g., missing columns, degenerate SVD,
        # KMeans warning elevated to error) falls back to ω=1 per §9.1.
        logger.warning(
            "src.train.subsample.subsample_customers raised %r — "
            "falling back to ω_t=1. Most likely cause: state_features "
            "did not produce the required columns (routine, recency_days, "
            "novelty) on this DataFrame. Run compute_state_features first.",
            exc,
        )
        return (None, None)

    return (selected_ids, event_weights)


# --- Per-epoch loops --------------------------------------------------------


def _compute_batch_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    reg_cfg: Any | None,
) -> torch.Tensor:
    """Forward pass + loss assembly for one batch.

    Returns the scalar total loss with autograd graph attached.

    The ``reg_cfg`` path delegates to ``combined_regularizer(model,
    intermediates, E, prices, reg_cfg)`` per §9.2. ``prices`` is read from
    the batch dict when present (populated by :func:`iter_batches` when
    the caller supplied a ``prices`` tensor) and is ``None`` otherwise.
    When ``prices is None`` the §9.2 monotonicity term is inert regardless
    of ``cfg.monotonicity_enabled`` — the caller must thread real prices
    to enable it.
    """
    z_d = batch["z_d"]
    E = batch["E"]
    c_star = batch["c_star"]
    omega = batch["omega"]
    prices = batch.get("prices")  # optional; None disables monotonicity term

    logits, intermediates = model(z_d, E)
    loss = cross_entropy_loss(logits, c_star, omega)

    if reg_cfg is not None:
        reg_term = combined_regularizer(model, intermediates, E, prices, reg_cfg)
        loss = loss + reg_term

    return loss


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    reg_cfg: Any | None,
    train_cfg: TrainConfig,
) -> dict[str, float]:
    """Run one training epoch.

    For each batch:

    1. Forward pass through ``model`` → ``(logits, intermediates)``.
    2. Data loss from :func:`cross_entropy_loss` with the batch's
       ``omega`` (§9.1).
    3. If ``reg_cfg`` is provided, add the §9.2 regularizer term.
    4. Zero grads, backward, gradient-clip to ``train_cfg.grad_clip``
       (**before** ``optimizer.step``), step optimizer, step scheduler.

    Returns
    -------
    dict with ``"mean_loss"``, ``"last_loss"``, ``"n_batches"``.
    """
    model.train()

    total_loss = 0.0
    last_loss = float("nan")
    n_batches = 0

    for batch in batches:
        loss = _compute_batch_loss(model, batch, reg_cfg)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # §9.3: grad clip at ‖·‖_2 ≤ 1.0 BEFORE optimizer.step.
        clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        last_loss = float(loss.detach().item())
        total_loss += last_loss
        n_batches += 1

    mean_loss = total_loss / n_batches if n_batches > 0 else float("nan")
    return {
        "mean_loss": mean_loss,
        "last_loss": last_loss,
        "n_batches": float(n_batches),
    }


@torch.no_grad()
def evaluate_nll(
    model: nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
) -> float:
    """Return the un-weighted, un-regularized mean NLL across ``batches``.

    This is the quantity fed into early stopping (§9.3 "val NLL") and
    reported by §13. Regularizers are excluded (they are training-only),
    and ``ω`` is ignored so the val metric is comparable across runs
    regardless of whether subsampling is on.

    Aggregation is a simple mean over per-event NLL: we weight each
    batch-mean by its batch size so the final number equals
    ``(1 / N_events) ∑_t ℓ_t`` exactly.
    """
    model.eval()

    total_weighted = 0.0
    total_count = 0

    for batch in batches:
        logits, _ = model(batch["z_d"], batch["E"])
        per_event = torch.nn.functional.cross_entropy(
            logits, batch["c_star"], reduction="none"
        )
        bsz = int(per_event.shape[0])
        total_weighted += float(per_event.sum().item())
        total_count += bsz

    if total_count == 0:
        return float("nan")
    return total_weighted / total_count


# --- Full fit loop ----------------------------------------------------------


def fit(
    model: nn.Module,
    train_batches_fn: Callable[[], Iterable[dict[str, torch.Tensor]]],
    val_batches_fn: Callable[[], Iterable[dict[str, torch.Tensor]]],
    *,
    train_cfg: TrainConfig,
    reg_cfg: Any | None = None,
    total_steps: int,
    seed: int = 0,
    on_epoch_end: Callable[[TrainState], None] | None = None,
) -> TrainState:
    """Train ``model`` with early stopping on val NLL (§9.3).

    Parameters
    ----------
    model:
        ``nn.Module`` to train; typically :class:`POLEU`.
    train_batches_fn, val_batches_fn:
        Zero-arg callables returning a fresh iterator each epoch (so
        generator-style batchers can be re-consumed).
    train_cfg:
        Hyperparameters (§9.3).
    reg_cfg:
        Optional :class:`RegularizerConfig` (§9.2); ``None`` skips
        regularizers.
    total_steps:
        ``T_max`` for the cosine schedule — typically
        ``n_batches_per_epoch * max_epochs``. See §9.3.
    seed:
        Seed for ``torch.manual_seed`` to make the run deterministic.
    on_epoch_end:
        Optional hook called with the current :class:`TrainState` after
        each epoch's val pass; use it for logging/checkpointing.

    Returns
    -------
    TrainState
        Final state at exit (either after ``max_epochs`` or early stop).
    """
    torch.manual_seed(int(seed))

    optimizer, scheduler = make_optimizer_and_scheduler(
        model, train_cfg, total_steps
    )

    state = TrainState()

    for epoch in range(train_cfg.max_epochs):
        train_stats = train_one_epoch(
            model,
            train_batches_fn(),
            optimizer,
            scheduler,
            reg_cfg,
            train_cfg,
        )

        val_nll = evaluate_nll(model, val_batches_fn())

        state.epoch = epoch
        state.step += int(train_stats["n_batches"])
        state.train_loss = float(train_stats["last_loss"])
        state.val_nll = float(val_nll)

        improved = val_nll < state.best_val_nll
        if improved:
            state.best_val_nll = float(val_nll)
            state.patience_counter = 0
        else:
            state.patience_counter += 1

        if on_epoch_end is not None:
            on_epoch_end(state)

        if state.patience_counter >= train_cfg.early_stopping_patience:
            state.stopped_early = True
            break

    return state
