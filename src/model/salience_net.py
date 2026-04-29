"""Salience network for PO-LEU (redesign.md §7).

Implements ``s_k^{(j)}(e_k, z_d, c_d)``: a small MLP that consumes the
concatenation of an outcome embedding ``e_k`` with the person feature
vector ``z_d`` and a per-event category embedding, returning a scalar
per outcome which is then softmaxed across the ``K`` outcomes within
each alternative ``j``.

Shapes follow redesign.md §9.4 step 5: given ``E`` of shape
``(B, J, K, d_e)`` and ``z_d`` of shape ``(B, p)``, :class:`SalienceNet`
returns ``(B, J, K)``, with rows summing to 1 along the ``K`` axis for
each ``(b, j)``.

Group-2 fix (category injection)
--------------------------------
SalienceNet now optionally consumes a per-event category code ``c``
(int64 of shape ``(B,)``); a small ``nn.Embedding(n_categories, d_cat)``
lookup is concatenated alongside ``E`` and ``z_d``. Motivation: on
standardised-goods buckets (ABIS_BOOK, CELLULAR_PHONE_CASE, ...) the
popularity rank dominates the choice signal, but a category-blind
salience module can down-weight popularity-aligned outcome heads.
Threading the category lets salience up-weight the right heads for
those buckets without affecting the rest of the catalog.

Backward compat: ``n_categories`` defaults to 1 and ``c`` defaults to
``None``; the forward then synthesizes a (B,) zeros tensor and the
embedding becomes a single-bucket bias, preserving every legacy
caller's shape and gradient semantics.

This file also exposes :class:`UniformSalience`, the ablation A6
variant (§7.3, §11) that returns ``1/K`` per entry with zero trainable
parameters.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


# Module-level constants (redesign.md §7.1 + Appendix B).
DEFAULT_D_E: int = 768
DEFAULT_P: int = 26
DEFAULT_HIDDEN: int = 64
# Group-2 category-injection defaults. n_categories=1 + d_cat=8 keeps
# legacy callers single-bucket. fc1 in_dim widens by d_cat; param
# count goes from 50_945 → 51_465 (see below).
DEFAULT_N_CATEGORIES: int = 1
DEFAULT_D_CAT: int = 8

# fc1 in_dim = d_e + p + d_cat = 802 by default.
# fc1: 802*64 + 64 = 51_392
# fc2:  64*1  + 1  =     65
# emb:  1 * 8      =      8
# total            = 51_465
EXPECTED_PARAM_COUNT_DEFAULT: int = (
    (DEFAULT_D_E + DEFAULT_P + DEFAULT_D_CAT) * DEFAULT_HIDDEN
    + DEFAULT_HIDDEN
    + DEFAULT_HIDDEN * 1
    + 1
    + DEFAULT_N_CATEGORIES * DEFAULT_D_CAT
)


def _xavier_init_linear(layer: nn.Linear) -> None:
    """Xavier-uniform on weight, zero on bias (redesign.md §0)."""
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)


class SalienceNet(nn.Module):
    """Salience MLP ``s_k^{(j)}(e_k, z_d, c_d)`` (§7.1 + Group-2 cat injection).

    Layers:
        Linear(d_e + p + d_cat -> hidden) -> ReLU -> Linear(hidden -> 1)
        then softmax over ``K`` within each ``(b, j)``.

    No dropout / layernorm / residuals. Biases start at zero; weights
    are Xavier-uniform; the category embedding is also Xavier-uniform.
    Deterministic given a fixed seed.
    """

    def __init__(
        self,
        d_e: int = DEFAULT_D_E,
        p: int = DEFAULT_P,
        hidden: int = DEFAULT_HIDDEN,
        n_categories: int = DEFAULT_N_CATEGORIES,
        d_cat: int = DEFAULT_D_CAT,
    ) -> None:
        super().__init__()
        self.d_e = int(d_e)
        self.p = int(p)
        self.hidden = int(hidden)
        self.n_categories = int(n_categories)
        self.d_cat = int(d_cat)
        if self.n_categories < 1:
            raise ValueError(
                f"n_categories must be >= 1; got {self.n_categories}."
            )
        if self.d_cat < 1:
            raise ValueError(f"d_cat must be >= 1; got {self.d_cat}.")
        self.in_dim = self.d_e + self.p + self.d_cat
        self.cat_emb = nn.Embedding(self.n_categories, self.d_cat)
        # Xavier-uniform on the embedding so its scale matches z_d.
        nn.init.xavier_uniform_(self.cat_emb.weight)
        self.fc1 = nn.Linear(self.in_dim, self.hidden)
        self.fc2 = nn.Linear(self.hidden, 1)
        self.act = nn.ReLU()
        _xavier_init_linear(self.fc1)
        _xavier_init_linear(self.fc2)

    def forward(
        self,
        E: torch.Tensor,
        z_d: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute salience ``s_k^{(j)}`` per outcome, softmaxed over ``K``.

        Shape contract:
            E:   (B, J, K, d_e)
            z_d: (B, p)
            c:   (B,) int64 in [0, n_categories) — optional. When ``None``
                 we build a (B,) zeros tensor so the embedding picks the
                 0th row uniformly. Preserves legacy (n_categories=1)
                 behaviour for unmodified callers.
            out: (B, J, K), each row along K sums to 1.0.
        """
        if E.dim() != 4:
            raise ValueError(
                f"E must be 4-D (B, J, K, d_e); got shape {tuple(E.shape)}."
            )
        if z_d.dim() != 2:
            raise ValueError(
                f"z_d must be 2-D (B, p); got shape {tuple(z_d.shape)}."
            )
        B, J, K, d_e = E.shape
        if d_e != self.d_e:
            raise ValueError(
                f"E last dim ({d_e}) != configured d_e ({self.d_e})."
            )
        if z_d.shape[0] != B:
            raise ValueError(
                f"Batch mismatch: E has B={B} but z_d has B={z_d.shape[0]}."
            )
        if z_d.shape[1] != self.p:
            raise ValueError(
                f"z_d last dim ({z_d.shape[1]}) != configured p ({self.p})."
            )

        # Default category code: zeros — the legacy single-bucket path.
        if c is None:
            c = torch.zeros(B, dtype=torch.long, device=E.device)
        else:
            if c.dim() != 1:
                raise ValueError(
                    f"c must be 1-D (B,); got shape {tuple(c.shape)}."
                )
            if c.shape[0] != B:
                raise ValueError(
                    f"Batch mismatch: E has B={B} but c has B={c.shape[0]}."
                )
            if c.dtype != torch.long:
                raise ValueError(
                    f"c must be int64 (torch.long); got dtype {c.dtype}."
                )

        # Category embedding lookup -> (B, d_cat). Broadcast to (B, J, K, d_cat).
        cat_vec = self.cat_emb(c)
        cat_bcast = cat_vec[:, None, None, :].expand(-1, J, K, -1)

        # Broadcast z_d to (B, J, K, p) without copying where possible.
        z_bcast = z_d[:, None, None, :].expand(-1, J, K, -1)

        # Concat along feature axis -> (B, J, K, d_e + p + d_cat).
        x = torch.cat([E, z_bcast, cat_bcast], dim=-1)

        # MLP -> (B, J, K, 1) -> squeeze -> (B, J, K).
        raw = self.fc2(self.act(self.fc1(x))).squeeze(-1)

        # Softmax over K within each (b, j).
        return torch.softmax(raw, dim=-1)

    def num_params(self) -> int:
        """Total trainable parameter count (51_465 at defaults; §7.1 + cat-emb)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class UniformSalience(nn.Module):
    """Ablation A6: uniform salience ``s_k = 1/K`` (§7.3, §11).

    Matches the :class:`SalienceNet` call signature and output shape so
    the two modules are one-line swaps in the end-to-end forward pass
    (§9.4 step 5). No trainable parameters.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        E: torch.Tensor,
        z_d: torch.Tensor,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a uniform ``(B, J, K)`` tensor of ``1/K`` per entry.

        Shape contract:
            E:   (B, J, K, d_e)
            z_d: (B, p)      (unused; accepted for API parity)
            c:   (B,) int64  (unused; accepted for Group-2 API parity)
            out: (B, J, K), every entry exactly 1/K.
        """
        if E.dim() != 4:
            raise ValueError(
                f"E must be 4-D (B, J, K, d_e); got shape {tuple(E.shape)}."
            )
        B, J, K, _ = E.shape
        # dtype/device follow E so downstream elementwise ops just work.
        return torch.full(
            (B, J, K),
            fill_value=1.0 / float(K),
            dtype=E.dtype,
            device=E.device,
        )

    def num_params(self) -> int:
        """Always 0: this module has no trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
