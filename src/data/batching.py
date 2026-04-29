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

import concurrent.futures
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, List, Optional, Sequence

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
    "DEFAULT_TABULAR_FEATURE_NAMES",
]


# --------------------------------------------------------------------------- #
# Strategy B tabular-feature column list
# --------------------------------------------------------------------------- #

# Default per-alternative columns for the PO-LEU Strategy B linear
# residual. ``popularity_rank`` is intentionally excluded — see the
# rationale in :mod:`src.model.po_leu` (the encoder text already renders
# popularity, so adding it here creates an identifiability fight with the
# neural branch). Mirrors the price-only subset of
# ``src.baselines.data_adapter.BUILTIN_FEATURE_NAMES``.
DEFAULT_TABULAR_FEATURE_NAMES: tuple[str, ...] = (
    "price",
    "log1p_price",
    "price_rank",
)


# Mapping from band-label popularity_rank strings (what the adapter emits
# for the LLM prompt) to a coarse [0, 1] numeric. Used as a fallback when
# popularity_rank is requested as a tabular feature; the *dense* signal
# the residual really wants is popularity_count below.
_POPULARITY_BAND_TO_NUMBER: dict[str, float] = {
    "top 5%": 0.95,
    "top 25%": 0.75,
    "top 50%": 0.5,
    "bottom 50%": 0.25,
}


SUPPORTED_TABULAR_FEATURES: tuple[str, ...] = (
    # Price-family (computed from prices_np alone, no record lookup needed).
    "price",
    "log1p_price",
    "price_rank",
    # Per-alt fields read from rec["alt_texts"][j]. Sifringer L-MNL: only
    # add features the encoder genuinely cannot see.
    #
    # popularity_rank: band-label string ("top 5%" etc.) — mapped to a
    # numeric via _POPULARITY_BAND_TO_NUMBER; numeric values pass through.
    # Coarser than popularity_count; kept for back-compat.
    "popularity_rank",
    "log1p_popularity_rank",
    # popularity_count: raw integer popularity (train-only count). Dense
    # numeric the encoder cannot extract from a coarse band label.
    "popularity_count",
    "log1p_popularity_count",
    "is_repeat",
    "log1p_purchase_count",
)


def _build_x_tab_matrix(
    prices_np: np.ndarray,
    feature_names: tuple[str, ...],
    records: list[dict] | None = None,
) -> np.ndarray:
    """Build the per-event ``(N, J, F)`` tabular tensor.

    Supported columns (see :data:`SUPPORTED_TABULAR_FEATURES`):

    Price-family (computed from ``prices_np``):
        * ``price``        — pass-through (already non-negative).
        * ``log1p_price``  — ``log1p(max(price, 0))``.
        * ``price_rank``   — within-event dense rank of ``price`` scaled to
                             ``[0, 1]``; constant within events of size 1.

    Record-family (read from ``records[i]["alt_texts"][j]``):
        * ``popularity_rank``       — band-label string from the adapter
                                      (``"top 5%"``, ``"top 25%"``,
                                      ``"top 50%"``, ``"bottom 50%"``)
                                      mapped to a coarse numeric via
                                      ``_POPULARITY_BAND_TO_NUMBER``.
                                      Numeric values pass through.
                                      Unknown / missing → 0.0.
        * ``log1p_popularity_rank`` — log1p of the band-mapped numeric.
        * ``popularity_count``      — raw integer popularity (train-only
                                      count); the dense numeric the
                                      encoder cannot extract from the
                                      coarse band label.
        * ``log1p_popularity_count``— log1p(popularity_count).
        * ``is_repeat``             — 1.0 if the customer has bought
                                      this ASIN in train history, else
                                      0.0. Read from
                                      ``alt_texts[j]["is_repeat"]``
                                      (per-alt train-history map);
                                      falls back to legacy chosen-only
                                      ``metadata["is_repeat"]``.
        * ``log1p_purchase_count``  — log1p of per-(customer, asin)
                                      train-history count from
                                      ``alt_texts[j]["purchase_count"]``.

    Adding a record-family feature requires non-None ``records``; passing
    ``records=None`` while requesting one raises ``ValueError`` at the
    caller's boundary so a typo in YAML surfaces here rather than as a
    silent column of zeros downstream.
    """
    N, J = prices_np.shape
    F = len(feature_names)
    out = np.zeros((N, J, F), dtype=np.float32)

    # Pre-compute price-derived columns lazily.
    log1p_prices: np.ndarray | None = None
    price_ranks: np.ndarray | None = None

    # Record-family features need rec["alt_texts"][j].<key>. Validate up
    # front so a typo or stale records list doesn't silently produce zeros.
    record_features = {
        "popularity_rank", "log1p_popularity_rank",
        "popularity_count", "log1p_popularity_count",
        "is_repeat",
        "log1p_purchase_count",
    }
    needs_records = any(name in record_features for name in feature_names)
    if needs_records and records is None:
        raise ValueError(
            "tabular feature(s) "
            + ", ".join(repr(n) for n in feature_names if n in record_features)
            + " require ``records`` to be passed; got None."
        )
    if needs_records and len(records) != N:
        raise ValueError(
            f"records length ({len(records)}) does not match prices N ({N})."
        )

    def _alt_field(rec: dict, j: int, key: str) -> float:
        """Extract a numeric per-alt field from rec['alt_texts'][j]; 0 on miss."""
        try:
            alts = rec.get("alt_texts") or []
            if j >= len(alts):
                return 0.0
            v = alts[j].get(key)
            if v is None:
                return 0.0
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def _alt_popularity_rank(rec: dict, j: int) -> float:
        """popularity_rank is a band-label string in production
        ("top 5%" / "top 25%" / "top 50%" / "bottom 50%") emitted by the
        adapter for the LLM prompt. Map it to a coarse numeric via
        _POPULARITY_BAND_TO_NUMBER. Numeric values (legacy / synthetic
        fixtures) pass through unchanged. Unknown strings -> 0.0.
        """
        try:
            alts = rec.get("alt_texts") or []
            if j >= len(alts):
                return 0.0
            v = alts[j].get("popularity_rank")
            if v is None:
                return 0.0
            if isinstance(v, str):
                key = v.strip().lower()
                for band_key, band_val in _POPULARITY_BAND_TO_NUMBER.items():
                    if key == band_key.lower():
                        return float(band_val)
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return 0.0
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def _record_is_repeat(rec: dict, j: int) -> float:
        """Read is_repeat for alt j.

        Preferred path: rec["alt_texts"][j]["is_repeat"], populated by
        choice_sets per-(customer, asin) train-history map. Fallback:
        rec.get("alt_is_repeat_per_alt"). Final fallback: legacy
        chosen-only rec["metadata"]["is_repeat"] when j is the chosen
        index. Else 0.0.
        """
        try:
            alts = rec.get("alt_texts") or []
            if j < len(alts):
                v = alts[j].get("is_repeat")
                if v is not None:
                    return float(bool(v)) if isinstance(v, bool) else float(v)
            per_alt = rec.get("alt_is_repeat_per_alt")
            if per_alt is not None and j < len(per_alt):
                vv = per_alt[j]
                return float(bool(vv)) if isinstance(vv, bool) else float(vv)
            chosen_idx = rec.get("chosen_idx")
            if isinstance(chosen_idx, int) and j == chosen_idx:
                meta = rec.get("metadata") or {}
                v = meta.get("is_repeat")
                if v is not None:
                    return float(bool(v)) if isinstance(v, bool) else float(v)
            return 0.0
        except (TypeError, ValueError):
            return 0.0


    for f, name in enumerate(feature_names):
        if name == "price":
            out[:, :, f] = prices_np
        elif name == "log1p_price":
            if log1p_prices is None:
                log1p_prices = np.log1p(np.maximum(prices_np, 0.0)).astype(
                    np.float32
                )
            out[:, :, f] = log1p_prices
        elif name == "price_rank":
            if price_ranks is None:
                price_ranks = np.zeros((N, J), dtype=np.float32)
                if J > 1:
                    for i in range(N):
                        order = np.argsort(prices_np[i], kind="stable")
                        ranks = np.empty(J, dtype=np.float32)
                        ranks[order] = (
                            np.arange(J, dtype=np.float32) / float(J - 1)
                        )
                        price_ranks[i] = ranks
                # else: J=1 leaves the column at zero, which is the
                # documented behaviour of _price_rank in data_adapter.
            out[:, :, f] = price_ranks
        elif name == "popularity_rank":
            for i, rec in enumerate(records):
                for j in range(J):
                    out[i, j, f] = _alt_popularity_rank(rec, j)
        elif name == "log1p_popularity_rank":
            for i, rec in enumerate(records):
                for j in range(J):
                    pr = _alt_popularity_rank(rec, j)
                    out[i, j, f] = float(np.log1p(max(pr, 0.0)))
        elif name == "popularity_count":
            for i, rec in enumerate(records):
                for j in range(J):
                    out[i, j, f] = _alt_field(rec, j, "popularity_count")
        elif name == "log1p_popularity_count":
            for i, rec in enumerate(records):
                for j in range(J):
                    pc = _alt_field(rec, j, "popularity_count")
                    out[i, j, f] = float(np.log1p(max(pc, 0.0)))
        elif name == "is_repeat":
            for i, rec in enumerate(records):
                for j in range(J):
                    out[i, j, f] = _record_is_repeat(rec, j)
        elif name == "log1p_purchase_count":
            for i, rec in enumerate(records):
                for j in range(J):
                    pc = _alt_field(rec, j, "purchase_count")
                    out[i, j, f] = float(np.log1p(max(pc, 0.0)))
        else:
            raise ValueError(
                f"Unsupported tabular feature {name!r}; supported: "
                + ", ".join(SUPPORTED_TABULAR_FEATURES) + "."
            )

    return out


# --------------------------------------------------------------------------- #
# Stub-contamination guard
# --------------------------------------------------------------------------- #


def _is_stub_client(client: Any) -> bool:
    """Return True if ``client`` looks like a stub LLM client.

    Two signals, short-circuited in order:

    1. ``client._is_stub`` — the explicit marker set on
       :class:`src.outcomes.generate.StubLLMClient`. Preferred because
       it survives subclassing and test-local wrappers that keep the
       attribute exposed.
    2. ``type(client).__name__`` starts with ``"Stub"`` — fallback for
       third-party stubs / test doubles (``_ThreadSafeCountingLLMClient``
       wrapping a :class:`StubLLMClient`, ad-hoc subclasses in test
       modules) that don't thread the attribute through.

    Used by :func:`assemble_batch` to decide whether a stub-origin
    outcome (``metadata['model_id']`` starting with ``"stub"``) is
    expected (test mode) or indicates cache contamination (production
    mode, must raise).
    """
    if getattr(client, "_is_stub", False):
        return True
    return type(client).__name__.startswith("Stub")


# --------------------------------------------------------------------------- #
# Thread-safety wrapper for SQLite-backed OutcomesCache
# --------------------------------------------------------------------------- #


class _LockedOutcomesCache:
    """Thread-safe shim around an :class:`OutcomesCache` instance.

    ``generate_outcomes`` only touches the cache via ``get_outcomes`` /
    ``put_outcomes`` so we override just those two methods. We wrap *only*
    SQLite I/O, leaving the slow network-bound LLM call fully parallel
    across workers.

    Lock-granularity choice
    -----------------------
    The brief recommended option (a): a single ``threading.Lock`` around
    ``get_outcomes``/``put_outcomes`` delegating to the caller-supplied
    cache. That is correct in principle for a WAL-mode SQLite backend,
    but Python's ``sqlite3`` module enforces ``check_same_thread=True``
    by default — a connection opened on thread T raises
    ``ProgrammingError`` when touched from any other thread, *even
    under a lock*. Since ``cache.py`` owns that constructor and the
    brief forbids modifying it, a plain lock is insufficient.

    The minimal hybrid fix: lazily open a fresh :class:`OutcomesCache`
    bound to the same DB file per worker thread (tracked in
    ``threading.local``), and still keep a single ``threading.Lock``
    around the actual read/write so writers serialise the same way
    they would under option (a). Read/write of the caller's cache is
    avoided from any non-owning thread. This keeps option (a)'s
    lock-around-I/O semantics while satisfying ``check_same_thread``
    without patching ``cache.py``.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._path = getattr(inner, "path", None)
        self._table = getattr(inner, "table", "kv")
        self._lock = threading.Lock()
        self._tls = threading.local()
        # Track every per-thread connection we open so we can close them
        # when the batch finishes; worker threads are pooled and may
        # outlive a single assemble_batch call.
        self._opened: list[Any] = []
        self._opened_lock = threading.Lock()

    def _thread_local_cache(self) -> Any:
        c = getattr(self._tls, "cache", None)
        if c is not None:
            return c
        if self._path is None:
            # No path metadata; fall back to the inner instance. This is
            # only safe when the caller happens to be on the same thread
            # that constructed it — SQLite will raise otherwise, which
            # is the correct (loud) failure mode.
            c = self._inner
        else:
            # Local import to keep the cache.py import lazy.
            from src.outcomes.cache import OutcomesCache

            c = OutcomesCache(self._path, table=self._table, create=False)
            with self._opened_lock:
                self._opened.append(c)
        self._tls.cache = c
        return c

    def get_outcomes(self, *args: Any, **kwargs: Any) -> Any:
        local = self._thread_local_cache()
        with self._lock:
            return local.get_outcomes(*args, **kwargs)

    def put_outcomes(self, *args: Any, **kwargs: Any) -> Any:
        local = self._thread_local_cache()
        with self._lock:
            return local.put_outcomes(*args, **kwargs)

    def close(self) -> None:
        with self._opened_lock:
            opened = list(self._opened)
            self._opened.clear()
        for c in opened:
            try:
                c.close()
            except Exception:
                pass


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

    x_tab: torch.Tensor | None = None
    """(N, J, F) float32 or ``None``.

    Per-alternative tabular features for the Strategy B linear residual
    on PO-LEU. Columns follow ``tabular_feature_names`` (default:
    ``("price", "log1p_price", "price_rank")``). ``None`` when the
    caller did not request the residual; the training loop and PO-LEU
    forward both treat ``None`` as "skip the residual entirely", which
    keeps existing runs bit-identical.
    """

    c: torch.Tensor | None = None
    """(N,) int64 or ``None``.

    Per-event category code consumed by PO-LEU's SalienceNet
    category embedding (Group-2 fix). Populated from
    ``records[i]["category_code"]`` when present; ``None`` when the
    upstream choice-set builder did not emit category codes (legacy
    fixtures, hand-crafted records). The training loop reads
    ``batch.get("c")`` and treats ``None`` as "single-bucket fallback",
    which preserves bit-identical behaviour for unmodified callers.
    """

    category_vocab: tuple[str, ...] = ()
    """Ordered category names parallel to :attr:`c`'s integer codes.

    Empty tuple when records did not carry a ``category_vocab`` key.
    Length sets ``n_categories`` for the SalienceNet embedding when
    PO-LEU is constructed downstream of this batch.
    """

    tabular_feature_names: tuple[str, ...] = ()
    """Names of the columns in :attr:`x_tab` (parallel to its last axis).

    Empty tuple when :attr:`x_tab` is ``None``; populated when the
    residual features are materialised. Preserved alongside the tensor
    so the model and downstream interpretability code can label
    ``β`` coefficients without re-deriving the column order.
    """

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
    prompt_version_cascade: Sequence[str] | None = None,
    diversity_filter: Callable[[list[str]], tuple[list[str], bool]] | None = None,
    omega: np.ndarray | None = None,
    progress: bool = False,
    max_concurrent_llm_calls: int = 8,
    tabular_feature_names: Sequence[str] | None = None,
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

    # Group-2: per-event category codes for the SalienceNet category
    # embedding. Falls back to all-zeros when records predate the
    # build_choice_sets that stashes "category_code" — preserves legacy
    # single-bucket behaviour.
    cat_codes_np = np.asarray(
        [int(rec.get("category_code", 0)) for rec in records],
        dtype=np.int64,
    )
    # Ordered category vocabulary parallel to cat_codes_np. Read off
    # records[0] (every record carries the same interned tuple); empty
    # tuple when absent.
    _vocab_raw = records[0].get("category_vocab", ()) if records else ()
    category_vocab: tuple[str, ...] = tuple(str(v) for v in _vocab_raw)

    # ----- outcome generation (parallel) + flat encode ---------------------
    # Dispatch every (i, j) LLM / cache call to a ThreadPoolExecutor so the
    # N*J network-bound calls overlap. Results are written back into a
    # pre-allocated (N, J) slot grid by index, so wall-time order of
    # completion doesn't affect the final flat_texts ordering.
    total_calls = N * J
    progress_mark = max(1, total_calls // 10) if progress else None

    # Wrap the outcomes_cache so SQLite I/O is serialized across threads.
    # The (slow) LLM call inside generate_outcomes runs outside the lock.
    if outcomes_cache is not None:
        thread_safe_cache: Any = _LockedOutcomesCache(outcomes_cache)
    else:
        thread_safe_cache = None

    # Separate lock for the progress counter; int increments aren't atomic
    # under the GIL in the general case, and we want predictable logging.
    progress_lock = threading.Lock()
    calls_done = 0

    # Precompute once — we use it inside every worker call to decide
    # whether a stub-origin outcome is contamination or expected.
    caller_is_stub = _is_stub_client(llm_client)

    # Curriculum-refinement cascade: when ``prompt_version_cascade`` is
    # provided, prefer cache hits under earlier versions (e.g. "v2_refined")
    # and only fall through to ``generate_outcomes`` (which may LLM-call)
    # under the LAST version. This lets the third PO-LEU training round
    # pick up refined outcomes for failure events while reusing the v1
    # cache for everything else — no surprise regeneration cost.
    cascade: tuple[str, ...] = (
        tuple(prompt_version_cascade)
        if prompt_version_cascade
        else (prompt_version,)
    )
    # The version under which generate_outcomes is allowed to write a NEW
    # entry on full miss. Always the last in the cascade.
    fallback_pv = cascade[-1]

    def _try_cache_only(
        rec: dict, j: int, pv: str
    ) -> Optional[list[str]]:
        """Cache-only lookup for one (event, alt, prompt_version)."""
        if thread_safe_cache is None:
            return None
        from src.outcomes.generate import build_cache_prompt_version
        cache_pv = build_cache_prompt_version(
            prompt_version=pv,
            K=K,
            model_id=getattr(llm_client, "model_id", "unknown"),
            c_d=rec["c_d"],
        )
        cached = thread_safe_cache.get_outcomes(
            str(rec["customer_id"]),
            str(rec["choice_asins"][j]),
            seed,
            cache_pv,
        )
        if cached is None:
            return None
        outs = list(cached.get("outcomes", []))
        if len(outs) != int(K):
            return None  # legacy K mismatch — let generate_outcomes regen
        return outs

    def _generate_one(i: int, j: int) -> tuple[int, int, list[str]]:
        rec = records[i]
        # Try every preferred version in the cascade FIRST; only the last
        # version is permitted to actually call the LLM.
        for pv_pref in cascade[:-1]:
            preferred = _try_cache_only(rec, j, pv_pref)
            if preferred is not None:
                if len(preferred) != K:
                    raise AssertionError(
                        f"cascade hit at {pv_pref!r} for records[{i}] alt[{j}] "
                        f"returned {len(preferred)} outcomes, expected K={K}"
                    )
                return (i, j, preferred)
        payload = generate_outcomes(
            customer_id=str(rec["customer_id"]),
            asin=str(rec["choice_asins"][j]),
            c_d=rec["c_d"],
            alt=rec["alt_texts"][j],
            K=K,
            seed=seed,
            prompt_version=fallback_pv,
            client=llm_client,
            cache=thread_safe_cache,
            diversity_filter=diversity_filter,
        )
        # Defense-in-depth: if generate_outcomes handed us a payload whose
        # metadata says a stub produced it, but the caller's llm_client is
        # itself not a stub, the cache has been contaminated by a prior
        # stub run under a key that a real run now happens to hit. Refuse
        # to proceed — silently training on stub outputs is the failure
        # mode that motivated the model_id fold in the cache_prompt_version
        # composite. The model_id fold alone would prevent fresh collisions;
        # this guard catches pre-existing entries that pre-date the fold.
        payload_model_id = str(payload.metadata.get("model_id", ""))
        if (
            payload_model_id.lower().startswith("stub")
            and not caller_is_stub
        ):
            raise RuntimeError(
                "Refusing to proceed: generate_outcomes returned a stub "
                f"outcome (model_id={payload_model_id!r}) but "
                f"llm_client={llm_client!r} is not a stub. This indicates "
                "stub-contaminated cache entries. Clean the cache and retry."
            )
        outcome_strs = list(payload.outcomes)
        if len(outcome_strs) != K:
            raise AssertionError(
                f"outcomes_length: records[{i}] alt[{j}] returned "
                f"{len(outcome_strs)} outcomes, expected K={K}"
            )
        # Progress logging: increment under lock, log on 10% milestones.
        nonlocal calls_done
        if progress_mark is not None:
            with progress_lock:
                calls_done += 1
                done_now = calls_done
            if done_now % progress_mark == 0:
                logger.debug(
                    "assemble_batch: %d / %d outcome calls done (%.0f%%)",
                    done_now,
                    total_calls,
                    100.0 * done_now / total_calls,
                )
        return (i, j, outcome_strs)

    # Pre-allocate the (N, J) grid so workers can fill by index.
    outcomes_grid: list[list[list[str] | None]] = [
        [None for _ in range(J)] for _ in range(N)
    ]

    # max_workers must be at least 1. If the caller passes <=0, clamp to 1.
    workers = max(1, int(max_concurrent_llm_calls))

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_generate_one, i, j)
                for i in range(N)
                for j in range(J)
            ]
            # as_completed surfaces errors from any worker via .result();
            # we deliberately do NOT catch them — silent failures in
            # outcome generation would corrupt the training batch.
            for future in concurrent.futures.as_completed(futures):
                i, j, outcome_strs = future.result()
                outcomes_grid[i][j] = outcome_strs
    finally:
        # Shut down the single-thread cache I/O executor (if any); the
        # caller still owns the underlying outcomes_cache connection.
        if isinstance(thread_safe_cache, _LockedOutcomesCache):
            thread_safe_cache.close()

    # Flatten in deterministic row-major (i, j, k) order so the downstream
    # encode_batch reshape to (N, J, K, d_e) is correct regardless of the
    # order in which workers finished.
    flat_texts: list[str] = []
    for i in range(N):
        per_alt_k_lists: list[list[str]] = []
        for j in range(J):
            outcome_strs = outcomes_grid[i][j]
            # _generate_one always writes a list; None here would indicate
            # a bug above (missed future).
            assert outcome_strs is not None, (
                f"internal: outcomes_grid[{i}][{j}] was never filled"
            )
            per_alt_k_lists.append(outcome_strs)
            flat_texts.extend(outcome_strs)
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
        # V4 fix: omega is an Appendix-C leverage weight and must be
        # non-negative. A negative entry would silently flip the sign
        # of the per-event loss contribution, so fail loud at the
        # boundary rather than corrupt training.
        if not np.all(np.isfinite(omega_np)):
            bad_idx = int(np.argmax(~np.isfinite(omega_np)))
            raise AssertionError(
                f"omega_finite: non-finite omega at index {bad_idx} "
                f"(value={omega_np[bad_idx]!r})"
            )
        if not np.all(omega_np >= 0.0):
            bad_idx = int(np.argmax(omega_np < 0.0))
            raise AssertionError(
                f"omega_non_negative: negative omega at index {bad_idx} "
                f"(value={float(omega_np[bad_idx])!r})"
            )

    # ----- to torch --------------------------------------------------------
    z_d_t = torch.from_numpy(z_d_np).to(torch.float32)
    E_t = torch.from_numpy(E_np.astype(np.float32, copy=False))
    c_star_t = torch.from_numpy(c_star_np).to(torch.int64)
    omega_t = torch.from_numpy(omega_np).to(torch.float32)
    prices_t = torch.from_numpy(prices_np).to(torch.float32)
    # Group-2 category code tensor (None when records carry no codes;
    # cat_codes_np is always a length-N int64 array, but if the
    # category_vocab is empty we treat the per-event "0" as legacy
    # zero-bucket — both downstream and the loop default to that).
    c_t: torch.Tensor | None = (
        torch.from_numpy(cat_codes_np).to(torch.int64) if len(records) > 0 else None
    )

    # ----- Strategy B x_tab ------------------------------------------------
    x_tab_t: torch.Tensor | None = None
    tab_names: tuple[str, ...] = ()
    if tabular_feature_names is not None:
        tab_names = tuple(tabular_feature_names)
        if tab_names:  # empty tuple => skip silently (config off-switch)
            x_tab_np = _build_x_tab_matrix(prices_np, tab_names, records)
            x_tab_t = torch.from_numpy(x_tab_np).to(torch.float32)
            if x_tab_t.shape != (N, J, len(tab_names)):
                raise AssertionError(
                    f"x_tab_shape: {tuple(x_tab_t.shape)} != "
                    f"{(N, J, len(tab_names))}"
                )
            if not torch.isfinite(x_tab_t).all().item():
                raise AssertionError(
                    "x_tab_finite: tabular feature tensor contains "
                    "non-finite entries (likely upstream price/NaN bug)."
                )

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
    # V4 fix: the §4 encoder contract promises L2-normalized embeddings
    # on the last axis (SentenceTransformers returns unit-norm vectors
    # when ``normalize_embeddings=True``). Verify at the boundary so a
    # misconfigured encoder can't silently poison the value function
    # with unnormalised ``e`` vectors.
    if E_t.numel() > 0:
        norms = torch.linalg.vector_norm(E_t, ord=2, dim=-1)
        if not torch.allclose(
            norms, torch.ones_like(norms), atol=1e-3, rtol=0.0
        ):
            bad = (
                (norms - 1.0).abs().argmax().item()
            )
            flat_norms = norms.reshape(-1)
            n_stride = J * K
            k_stride = K
            n_idx = bad // n_stride
            rem = bad % n_stride
            j_idx = rem // k_stride
            k_idx = rem % k_stride
            raise AssertionError(
                f"E_l2_normalized: |E[{n_idx},{j_idx},{k_idx}]|_2 = "
                f"{float(flat_norms[bad])!r} (expected ~1.0, atol=1e-3)"
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
        x_tab=x_tab_t,
        c=c_t,
        category_vocab=category_vocab,
        tabular_feature_names=tab_names,
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

    Also forwards :attr:`AssembledBatch.x_tab` when populated, so the
    Strategy B tabular residual on PO-LEU sees per-alternative numeric
    features. When ``x_tab`` is ``None`` the yielded dicts have no
    ``"x_tab"`` key — the training loop reads ``batch.get("x_tab")``
    and treats absence as "skip the residual", preserving bit-identical
    behaviour for runs that don't enable the feature.

    Shape contract matches :func:`src.train.loop.iter_batches`; each
    yielded dict always has a ``"prices"`` key of shape
    ``(batch_size, J)`` (``batch_size`` trimmed on the last partial
    batch). When applicable, ``"x_tab"`` has shape
    ``(batch_size, J, F)`` where ``F = len(batch.tabular_feature_names)``.
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
        x_tab=batch.x_tab,
        c=batch.c,
    )
