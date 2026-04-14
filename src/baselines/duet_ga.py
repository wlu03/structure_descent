"""
DUET baseline: Economically-consistent ANN-based discrete choice model.

Reference
---------
Han, Y. et al. (2024). "DUET: Economically-consistent ANN-based discrete
choice model." arXiv:2404.13198.
https://arxiv.org/abs/2404.13198

Historical note
---------------
This module is named ``duet_ga.py`` for backwards-compatibility with the
prior (incorrect) GA-over-DSL implementation. The ``_ga`` suffix is
legacy — the model here is the faithful DUET ANN from the arXiv paper,
not a genetic algorithm. Back-compat aliases ``DuetGA`` / ``DuetGAFitted``
/ ``_structure_key`` continue to exist at the bottom of the module so
existing imports and notebooks keep working while callers migrate to the
preferred ``DUET`` / ``DUETFitted`` names.

Method
------
DUET combines two parallel branches whose outputs are summed into a
single utility:

    u(x_ia | theta) = beta^T x_ia + f_theta(x_ia)

where
    - ``beta^T x_ia``  is a linear ("interpretable") branch with a
      directly-inspectable coefficient per input feature, and
    - ``f_theta(x_ia)`` is a small tanh MLP ("flexible" branch) that
      captures residual nonlinearities.

Training minimizes the conditional-logit NLL over the choice set,
regularized by (i) a small L2 penalty on the neural weights and (ii) a
**soft monotonicity penalty** that enforces economic consistency on
specific inputs. The arXiv paper enforces:

    dU / d(price)  <= 0     (negative price sensitivity)
    dU / d(rating) >= 0     (positive rating sensitivity)

as soft hinge penalties evaluated at every training point via
``torch.autograd.grad``. Violations above zero are squared and summed.
This follows the paper's formulation (§3, Eq. 6) where the penalty
strength lambda controls the trade-off between fit and consistency.

Loss::

    L(beta, theta) =
        - sum_i log softmax(u_i(x))[chosen_i]
        + l2 * ||theta_NN||^2
        + lam_mono * sum_{i, j in MONO}
                     sum_a max(0, sign_expected_j * du/dx_j)^2

Domain adaptation (Amazon e-commerce)
-------------------------------------
Input features come from ``BaselineEventBatch.base_features_list`` and
``base_feature_names``. The monotonicity targets are looked up by name:

    price_sensitivity  (expected sign: negative)
    rating_signal      (expected sign: positive)

If either name is absent in the batch the corresponding constraint is
silently dropped with a warning at fit time.

Architecture defaults
---------------------
Flexible branch: 2 hidden layers of 32 units, tanh activation. Small by
design — this is a baseline, not a competitor to deep choice models.

Interface: implements Baseline / FittedBaseline from src/baselines/base.py.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .base import BaselineEventBatch, FittedBaseline


# Expected monotonicity: feature name -> sign (+1 or -1).
_DEFAULT_MONOTONIC_FEATURES: dict = {
    "price_sensitivity": -1,   # negative price sensitivity
    "rating_signal": +1,       # positive rating sensitivity
}


# -----------------------------------------------------------------------------
# Batch stacking
# -----------------------------------------------------------------------------


def _stack_batch(batch: BaselineEventBatch) -> Tuple[np.ndarray, np.ndarray]:
    """Stack per-event feature matrices into a single (E, A, F) tensor."""
    if batch.n_events == 0:
        raise ValueError("DUET received an empty batch")
    n_alts = batch.n_alternatives
    n_feats = batch.n_base_terms
    X = np.zeros((batch.n_events, n_alts, n_feats), dtype=np.float32)
    for e, feats in enumerate(batch.base_features_list):
        arr = np.asarray(feats, dtype=np.float32)
        if arr.shape != (n_alts, n_feats):
            raise ValueError(
                f"event {e}: shape {arr.shape} != expected ({n_alts}, {n_feats}); "
                "DUET requires a uniform choice-set size."
            )
        X[e] = arr
    y = np.asarray(batch.chosen_indices, dtype=np.int64)
    return X, y


def _resolve_mono_indices(
    feature_names: Sequence[str],
    targets: dict,
) -> List[Tuple[int, int]]:
    """Return [(feature_index, sign)] for each feature in targets that exists."""
    resolved: List[Tuple[int, int]] = []
    for name, sign in targets.items():
        if name in feature_names:
            resolved.append((feature_names.index(name), int(sign)))
        else:
            warnings.warn(
                f"DUET: monotonic feature {name!r} not present in "
                f"base_feature_names; dropping that constraint.",
                stacklevel=2,
            )
    return resolved


# -----------------------------------------------------------------------------
# Fitted object
# -----------------------------------------------------------------------------


@dataclass
class DUETFitted:
    """Fitted DUET baseline, conforming to the FittedBaseline protocol."""

    name: str
    beta: np.ndarray                 # linear branch coefficients, shape (F,)
    nn_state_dict: dict              # torch state_dict for the flexible branch
    nn_arch: Tuple[int, ...]         # hidden layer sizes
    feature_names: List[str]
    mono_constraints: List[Tuple[int, int]]
    n_params_total: int
    train_nll: float
    val_nll: float

    def score_events(self, batch: BaselineEventBatch) -> List[np.ndarray]:
        import torch
        X_np, _ = _stack_batch(batch)
        X = torch.as_tensor(X_np)

        nn = _build_mlp(X.shape[-1], self.nn_arch)
        nn.load_state_dict(self.nn_state_dict)
        nn.eval()

        beta = torch.as_tensor(self.beta, dtype=X.dtype)
        with torch.no_grad():
            linear = torch.matmul(X, beta)                       # (E, A)
            flexible = nn(X.reshape(-1, X.shape[-1])).reshape(X.shape[:-1])
            utils = (linear + flexible).cpu().numpy()            # (E, A)

        return [utils[e] for e in range(utils.shape[0])]

    @property
    def n_params(self) -> int:
        return int(self.n_params_total)

    @property
    def description(self) -> str:
        return (
            f"DUET linear+MLP{self.nn_arch} "
            f"mono={len(self.mono_constraints)} "
            f"params={self.n_params_total} "
            f"train_nll={self.train_nll:.3f} val_nll={self.val_nll:.3f}"
        )


def _build_mlp(input_dim: int, hidden: Tuple[int, ...]):
    import torch.nn as nn
    layers: List = []
    prev = input_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.Tanh())
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
# Baseline class
# -----------------------------------------------------------------------------


class DUET:
    """DUET baseline (Han et al. 2024, arXiv:2404.13198).

    Parameters
    ----------
    hidden : tuple of int
        Flexible-branch MLP hidden layer sizes. Default (32, 32).
    learning_rate : float
        Adam learning rate.
    n_epochs : int
        Maximum training epochs.
    batch_size : int
        Mini-batch size over events.
    l2 : float
        L2 regularization on the flexible-branch weights only (the linear
        branch is unregularized so coefficients remain interpretable).
    lam_mono : float
        Strength of the soft monotonicity penalty (paper's lambda).
    mono_targets : dict, optional
        Mapping {feature_name: sign} for which features to enforce
        monotonic utility on. Defaults to
        ``{"price_sensitivity": -1, "rating_signal": +1}``. Pass ``{}`` to
        disable all monotonicity.
    patience : int
        Early-stopping patience on validation NLL.
    seed : int
        PRNG seed.
    """

    name = "DUET"

    def __init__(
        self,
        hidden: Tuple[int, ...] = (32, 32),
        learning_rate: float = 5e-3,
        n_epochs: int = 100,
        batch_size: int = 64,
        l2: float = 1e-4,
        lam_mono: float = 1.0,
        mono_targets: Optional[dict] = None,
        patience: int = 15,
        seed: int = 0,
    ):
        self.hidden = tuple(hidden)
        self.learning_rate = float(learning_rate)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.l2 = float(l2)
        self.lam_mono = float(lam_mono)
        self.mono_targets = (
            dict(_DEFAULT_MONOTONIC_FEATURES) if mono_targets is None
            else dict(mono_targets)
        )
        self.patience = int(patience)
        self.seed = int(seed)

    def fit(
        self,
        train: BaselineEventBatch,
        val: BaselineEventBatch,
    ) -> DUETFitted:
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "DUET requires PyTorch. Install with `pip install torch`."
            ) from e

        torch.manual_seed(self.seed)

        X_tr_np, y_tr_np = _stack_batch(train)
        X_va_np, y_va_np = _stack_batch(val) if val.n_events > 0 else (None, None)

        X_tr = torch.as_tensor(X_tr_np)
        y_tr = torch.as_tensor(y_tr_np)
        if X_va_np is not None:
            X_va = torch.as_tensor(X_va_np)
            y_va = torch.as_tensor(y_va_np)
        else:
            X_va = None
            y_va = None

        n_events, n_alts, n_feats = X_tr.shape
        mono_idx = _resolve_mono_indices(train.base_feature_names, self.mono_targets)

        beta = torch.zeros(n_feats, dtype=torch.float32, requires_grad=True)
        nn_model = _build_mlp(n_feats, self.hidden)
        params = [beta] + list(nn_model.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        def utility(X):
            linear = torch.matmul(X, beta)                              # (B, A)
            flexible = nn_model(X.reshape(-1, n_feats)).reshape(X.shape[:-1])
            return linear + flexible

        def nll(X, y):
            utils = utility(X)                                          # (B, A)
            log_probs = F.log_softmax(utils, dim=-1)
            chosen_lp = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            return -chosen_lp.mean()

        def l2_penalty():
            return sum((p * p).sum() for p in nn_model.parameters())

        def monotonicity_penalty(X):
            if not mono_idx:
                return torch.zeros((), dtype=X.dtype)
            # Recompute utility on a leaf tensor so autograd can take dU/dX.
            X_leaf = X.detach().clone().requires_grad_(True)
            u = utility(X_leaf).sum()
            grads = torch.autograd.grad(
                u, X_leaf, create_graph=True, retain_graph=True
            )[0]                                                         # (B, A, F)
            total = torch.zeros((), dtype=X.dtype)
            for idx, sign in mono_idx:
                g_j = grads[..., idx]
                # violation: sign_expected * g_j must be >= 0, so penalize
                # hinge(-(sign * g_j)).
                violation = torch.clamp(-sign * g_j, min=0.0)
                total = total + (violation * violation).mean()
            return total

        best_val = float("inf")
        best_state: Optional[dict] = None
        best_beta: Optional[np.ndarray] = None
        patience_left = self.patience

        n_batches = max(1, (n_events + self.batch_size - 1) // self.batch_size)
        perm_rng = np.random.default_rng(self.seed)

        for epoch in range(self.n_epochs):
            nn_model.train()
            perm = perm_rng.permutation(n_events)
            for b in range(n_batches):
                idx = perm[b * self.batch_size : (b + 1) * self.batch_size]
                if len(idx) == 0:
                    continue
                Xb = X_tr[idx]
                yb = y_tr[idx]
                loss = (
                    nll(Xb, yb)
                    + self.l2 * l2_penalty()
                    + self.lam_mono * monotonicity_penalty(Xb)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if X_va is not None:
                nn_model.eval()
                with torch.no_grad():
                    v_nll = float(nll(X_va, y_va).item())
                if v_nll < best_val - 1e-6:
                    best_val = v_nll
                    best_state = {k: v.detach().clone() for k, v in nn_model.state_dict().items()}
                    best_beta = beta.detach().cpu().numpy().copy()
                    patience_left = self.patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

        if best_state is None:
            best_state = {k: v.detach().clone() for k, v in nn_model.state_dict().items()}
            best_beta = beta.detach().cpu().numpy().copy()
            best_val = float("inf")

        # Load best state for final metric reporting.
        nn_model.load_state_dict(best_state)
        nn_model.eval()
        with torch.no_grad():
            beta_t = torch.as_tensor(best_beta, dtype=X_tr.dtype)

            def final_nll(X, y):
                linear = torch.matmul(X, beta_t)
                flexible = nn_model(X.reshape(-1, n_feats)).reshape(X.shape[:-1])
                utils = linear + flexible
                log_probs = F.log_softmax(utils, dim=-1)
                chosen_lp = log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1)
                return -chosen_lp.sum().item()

            train_nll = final_nll(X_tr, y_tr)
            val_nll = final_nll(X_va, y_va) if X_va is not None else float("nan")

        nn_param_count = sum(int(p.numel()) for p in nn_model.parameters())
        n_params_total = int(n_feats + nn_param_count)

        return DUETFitted(
            name=self.name,
            beta=best_beta,
            nn_state_dict={k: v.cpu() for k, v in best_state.items()},
            nn_arch=self.hidden,
            feature_names=list(train.base_feature_names),
            mono_constraints=list(mono_idx),
            n_params_total=n_params_total,
            train_nll=float(train_nll),
            val_nll=float(val_nll),
        )


# -----------------------------------------------------------------------------
# Backwards-compatibility shims
# -----------------------------------------------------------------------------


# Legacy name the old GA baseline used. Callers import it via
# ``from src.baselines.duet_ga import DuetGA, DuetGAFitted``; we keep them
# as aliases so nothing outside this file has to change.
DuetGA = DUET
DuetGAFitted = DUETFitted


def _structure_key(structure) -> str:
    """Legacy helper from the GA implementation.

    Kept as an import-compat shim for tests and older notebooks. The DUET
    neural baseline has no notion of a DSL structure, so this simply
    canonicalizes whatever is passed in (if it has a ``.terms`` attribute)
    or falls back to ``repr``.
    """
    terms = getattr(structure, "terms", None)
    if terms is None:
        return repr(structure)
    return "|".join(sorted(repr(t) for t in terms))
