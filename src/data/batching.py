"""Wave 10 glue: choice-set records -> training tensors.

This module is the bridge between the Wave-9 :func:`build_choice_sets`
record list and the tensor-shaped inputs that the Wave-6 training loop
consumes (:mod:`src.train.loop`). It drives outcome generation and
embedding through the existing :mod:`src.outcomes.generate` /
:mod:`src.outcomes.encode` APIs with the existing
:class:`OutcomesCache` / :class:`EmbeddingsCache` plumbing -- no
re-implementation of any of those.

Public surface
--------------
* :class:`AssembledBatch` -- materialized CPU tensors (``z_d``, ``E``,
  ``c_star``, ``omega``, ``prices``) + the diagnostics (customer ids,
  chosen asins, nested outcome strings) that the §12 reports and §13
  strata rely on.
* :func:`assemble_batch` -- build an :class:`AssembledBatch` from the
  output of :func:`build_choice_sets`.
* :func:`iter_to_torch_batches` -- thin wrapper over
  :func:`src.train.loop.iter_batches` that always forwards the
  ``prices`` tensor so the §9.2 monotonicity regularizer fires (the
  Wave-8 bug-fix: training-loop callers commonly forgot to thread
  prices through).

Invariants
----------
Every tensor is CPU-resident; the caller moves them to device. ``E``
is L2-normalized on the last axis because :func:`encode_batch` returns
L2-normalized rows. ``prices`` is non-negative by construction. Width
``p`` of ``z_d`` is derived from the first record, not hardcoded, and
is asserted to match across records; the same is true of ``J``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, List, Sequence

import numpy as np
import torch

from src.outcomes.encode import encode_batch
from src.outcomes.generate import generate_outcomes
from src.train.loop import iter_batches


logger = logging.getLogger(__name__)


__all__ = [
    "AssembledBatch",
    "assemble_batch",
    "iter_to_torch_batches",
]


# --------------------------------------------------------------------------- #
# Dataclass
# --------------------------------------------------------------------------- #


@dataclass
class AssembledBatch:
    """Materialized tensors + diagnostics for a full run slice.

    All tensors are CPU-resident. The training loop moves them to
    device. Diagnostic fields (``customer_ids``, ``chosen_asins``,
    ``outcomes_nested``) are pure Python structures that the §12
    interpretability reports and the §13 strata breakdowns consume;
    they are not touched by the training forward pass.
    """

    z_d: torch.Tensor
    """(N, p) float32. Canonical person-feature vectors."""

    E: torch.Tensor
    """(N, J, K, d_e) float32. L2-normalized on the last axis."""

    c_star: torch.Tensor
    """(N,) int64. Ground-truth chosen alternative index in ``[0, J)``."""

    omega: torch.Tensor
    """(N,) float32. Per-event importance weights; ones if no subsample."""

    prices: torch.Tensor
    """(N, J) float32. Per-alternative price used by §9.2 monotonicity."""

    # Diagnostics (not consumed by training):
    customer_ids: list[str] = field(default_factory=list)
    chosen_asins: list[str] = field(default_factory=list)
    outcomes_nested: list[list[list[str]]] = field(default_factory=list)
    """[N][J][K] strings; used by head-naming + per-decision reports."""

    def __len__(self) -> int:
        return int(self.z_d.shape[0])


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #


def assemble_batch(
    records: list[dict],
    adapter: Any,
    *,
    llm_client: Any,
    encoder: Any,
    outcomes_cache: Any | None,
    embeddings_cache: Any | None,
    K: int = 3,
    seed: int = 0,
    prompt_version: str = "v1",
    diversity_filter: Callable[[list[str]], tuple[list[str], bool]] | None = None,
    omega: np.ndarray | None = None,
    progress: bool = False,
) -> AssembledBatch:
    """Materialize training tensors from choice-set records.

    Shape contract
    --------------
    ``records``: list[dict], output of :func:`build_choice_sets`. Each
    record carries the keys documented in that module (``customer_id``,
    ``chosen_asin``, ``choice_asins``, ``chosen_idx``, ``z_d``,
    ``c_d``, ``alt_texts``, ...). ``z_d`` drives the width ``p`` of the
    output tensor; the first record's width wins, and every other
    record is asserted to match.

    ``adapter``: a :class:`DatasetAdapter`-shaped object. Only used for
    diagnostics; per the Wave-10 brief, ``records`` already carries
    everything needed.

    ``llm_client`` / ``encoder``: real or stub backends (see
    :class:`src.outcomes.generate.StubLLMClient` and
    :class:`src.outcomes.encode.StubEncoder`). The encoder is invoked
    exactly once per assemble_batch over the full flat list of
    ``N * J * K`` outcome strings (design invariant -- one encoder
    pass per batch, never per-record).

    ``outcomes_cache`` / ``embeddings_cache``: the existing SQLite
    caches. ``None`` disables caching (useful in tests).

    ``K``: outcomes per alternative; forwarded to
    :func:`generate_outcomes`. The Wave-10 brief spells out that this
    argument wins over any ``adapter.schema.K`` on the adapter.

    ``seed`` / ``prompt_version``: forwarded to
    :func:`generate_outcomes`. ``seed`` also seeds the stub LLM, so the
    whole assemble pipeline is deterministic.

    ``diversity_filter``: optional callable forwarded to
    :func:`generate_outcomes`.

    ``omega``: (N,) per-event importance weights. ``None`` -> all ones.

    ``progress``: if True, emit DEBUG progress markers per 10% of the
    ``N * J`` generation calls. Never emits a tqdm bar itself (the
    encoder pass already has its own; :mod:`src.outcomes.encode`
    owns it).

    Returns
    -------
    :class:`AssembledBatch`

    Invariants (asserted before return; AssertionError on any failure)
    -----------------------------------------------------------------
    * ``z_d.shape == (N, p)`` with every record sharing the same ``p``.
    * ``E.shape == (N, J, K, d_e)``.
    * ``torch.isnan(E).any()`` is False.
    * ``torch.isfinite(z_d).all()`` is True.
    * ``0 <= c_star.min() and c_star.max() < J``.
    * ``prices.shape == (N, J)`` and ``(prices >= 0).all()``.
    * ``omega.shape == (N,)``.
    """
    if not isinstance(records, list):
        raise TypeError(
            f"records must be a list of dicts, got {type(records).__name__}"
        )
    N = len(records)
    if N == 0:
        raise ValueError("records is empty; nothing to assemble.")

    # ----- derive p and J from the first record, assert rest match -------
    first = records[0]
    z0 = first["z_d"]
    if not isinstance(z0, np.ndarray):
        raise TypeError(
            f"records[0]['z_d'] must be np.ndarray, got {type(z0).__name__}"
        )
    if z0.ndim != 1:
        raise AssertionError(
            f"z_d_width_uniform: records[0]['z_d'].ndim={z0.ndim}, want 1"
        )
    p = int(z0.shape[0])

    choice0 = first["choice_asins"]
    if not isinstance(choice0, list):
        raise TypeError(
            "records[0]['choice_asins'] must be a list (this module does "
            "not support n_resamples>1 / list-of-lists mode)"
        )
    J = len(choice0)
    if J == 0:
        raise ValueError("records[0]['choice_asins'] is empty; J must be > 0.")

    # ----- cross-record uniformity -----------------------------------------
    for i, rec in enumerate(records):
        z = rec["z_d"]
        if not isinstance(z, np.ndarray) or z.ndim != 1 or int(z.shape[0]) != p:
            shape = getattr(z, "shape", None)
            raise AssertionError(
                f"z_d_width_uniform: records[{i}]['z_d'] shape {shape} "
                f"does not match records[0] width p={p}"
            )
        if len(rec["choice_asins"]) != J:
            raise AssertionError(
                f"J_uniform: records[{i}] has J={len(rec['choice_asins'])}, "
                f"want J={J} (from records[0])"
            )
        if len(rec["alt_texts"]) != J:
            raise AssertionError(
                f"J_uniform: records[{i}]['alt_texts'] has length "
                f"{len(rec['alt_texts'])}, want J={J}"
            )

    logger.info(
        "assembling batch of N=%d (J=%d, K=%d, p=%d, prompt_version=%r, seed=%d)",
        N,
        J,
        K,
        p,
        prompt_version,
        seed,
    )

    # ----- z_d, c_star, prices, diagnostics --------------------------------
    z_d_np = np.stack(
        [np.asarray(rec["z_d"], dtype=np.float32) for rec in records],
        axis=0,
    )
    c_star_np = np.asarray(
        [int(rec["chosen_idx"]) for rec in records],
        dtype=np.int64,
    )
    prices_np = np.zeros((N, J), dtype=np.float32)
    customer_ids: list[str] = []
    chosen_asins: list[str] = []
    outcomes_nested: list[list[list[str]]] = []

    for i, rec in enumerate(records):
        customer_ids.append(str(rec["customer_id"]))
        chosen_asins.append(str(rec["chosen_asin"]))
        for j, alt in enumerate(rec["alt_texts"]):
            # ``alt`` is an adapter-rendered alt-text dict with a
            # "price" key (see DatasetAdapter.alt_text docstring).
            prices_np[i, j] = float(alt.get("price", 0.0) or 0.0)

    # ----- outcome generation + flat encode --------------------------------
    # Collect outcome strings in row-major (i, j, k) order so the reshape
    # at the end preserves the (N, J, K) layout without an explicit
    # index-by-index fill.
    flat_texts: list[str] = []
    total_calls = N * J
    progress_mark = max(1, total_calls // 10) if progress else None

    calls_done = 0
    for i, rec in enumerate(records):
        per_alt_k_lists: list[list[str]] = []
        cid = str(rec["customer_id"])
        c_d = rec["c_d"]
        alt_texts_list = rec["alt_texts"]
        choice_asins = rec["choice_asins"]
        for j in range(J):
            payload = generate_outcomes(
                customer_id=cid,
                asin=str(choice_asins[j]),
                c_d=c_d,
                alt=alt_texts_list[j],
                K=K,
                seed=seed,
                prompt_version=prompt_version,
                client=llm_client,
                cache=outcomes_cache,
                diversity_filter=diversity_filter,
            )
            outcome_strs = list(payload.outcomes)
            if len(outcome_strs) != K:
                raise AssertionError(
                    f"outcomes_length: records[{i}] alt[{j}] returned "
                    f"{len(outcome_strs)} outcomes, expected K={K}"
                )
            per_alt_k_lists.append(outcome_strs)
            flat_texts.extend(outcome_strs)
            calls_done += 1
            if progress_mark is not None and calls_done % progress_mark == 0:
                logger.debug(
                    "assemble_batch: %d / %d outcome calls done (%.0f%%)",
                    calls_done,
                    total_calls,
                    100.0 * calls_done / total_calls,
                )
        outcomes_nested.append(per_alt_k_lists)

    # One encoder pass per batch (Wave-10 design invariant).
    flat_vecs = encode_batch(flat_texts, client=encoder, cache=embeddings_cache)
    d_e = int(encoder.d_e)
    expected = (N * J * K, d_e)
    if flat_vecs.shape != expected:
        raise AssertionError(
            f"encode_shape: encode_batch returned {flat_vecs.shape!r}, "
            f"expected {expected!r}"
        )
    E_np = flat_vecs.reshape(N, J, K, d_e)

    # ----- omega -----------------------------------------------------------
    if omega is None:
        omega_np = np.ones(N, dtype=np.float32)
    else:
        omega_np = np.asarray(omega, dtype=np.float32).reshape(-1)
        if omega_np.shape != (N,):
            raise AssertionError(
                f"omega_shape: omega has shape {omega_np.shape}, want {(N,)}"
            )

    # ----- to torch --------------------------------------------------------
    z_d_t = torch.from_numpy(z_d_np).to(torch.float32)
    E_t = torch.from_numpy(E_np.astype(np.float32, copy=False))
    c_star_t = torch.from_numpy(c_star_np).to(torch.int64)
    omega_t = torch.from_numpy(omega_np).to(torch.float32)
    prices_t = torch.from_numpy(prices_np).to(torch.float32)

    # ----- final invariant checks (boundary assertions) --------------------
    if z_d_t.shape != (N, p):
        raise AssertionError(
            f"z_d_shape: {tuple(z_d_t.shape)} != {(N, p)}"
        )
    if E_t.shape != (N, J, K, d_e):
        raise AssertionError(
            f"E_shape: {tuple(E_t.shape)} != {(N, J, K, d_e)}"
        )
    if torch.isnan(E_t).any().item():
        # Find an offending slice for diagnostics.
        nan_any_ijk = torch.isnan(E_t).any(dim=-1)
        idx = torch.nonzero(nan_any_ijk, as_tuple=False)[0].tolist()
        raise AssertionError(
            f"E_no_nan: E contains NaN at (n={idx[0]}, j={idx[1]}, k={idx[2]})"
        )
    if not torch.isfinite(z_d_t).all().item():
        bad_rows = (~torch.isfinite(z_d_t).all(dim=-1)).nonzero(as_tuple=False).flatten().tolist()
        raise AssertionError(
            f"z_d_finite: non-finite rows at indices {bad_rows[:5]}"
        )
    if c_star_t.numel() > 0:
        cs_min = int(c_star_t.min().item())
        cs_max = int(c_star_t.max().item())
        if cs_min < 0 or cs_max >= J:
            raise AssertionError(
                f"c_star_in_range: c_star range [{cs_min}, {cs_max}] "
                f"violates [0, J={J})"
            )
    if prices_t.shape != (N, J):
        raise AssertionError(
            f"prices_shape: {tuple(prices_t.shape)} != {(N, J)}"
        )
    if not (prices_t >= 0).all().item():
        bad = (prices_t < 0).nonzero(as_tuple=False)[0].tolist()
        raise AssertionError(
            f"prices_non_negative: negative price at (n={bad[0]}, j={bad[1]})"
        )
    if omega_t.shape != (N,):
        raise AssertionError(
            f"omega_shape: {tuple(omega_t.shape)} != {(N,)}"
        )

    return AssembledBatch(
        z_d=z_d_t,
        E=E_t,
        c_star=c_star_t,
        omega=omega_t,
        prices=prices_t,
        customer_ids=customer_ids,
        chosen_asins=chosen_asins,
        outcomes_nested=outcomes_nested,
    )


# --------------------------------------------------------------------------- #
# Torch batch iterator that always plumbs prices
# --------------------------------------------------------------------------- #


def iter_to_torch_batches(
    batch: AssembledBatch,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Thin wrapper over :func:`src.train.loop.iter_batches` that
    forwards :attr:`AssembledBatch.prices` so the §9.2 monotonicity
    regularizer fires (the Wave-8 bug-fix: training-loop callers used
    to drop prices silently).

    Shape contract matches :func:`src.train.loop.iter_batches`; each
    yielded dict always has a ``"prices"`` key of shape
    ``(batch_size, J)`` (``batch_size`` trimmed on the last partial
    batch).
    """
    yield from iter_batches(
        batch.z_d,
        batch.E,
        batch.c_star,
        batch.omega,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        prices=batch.prices,
    )
