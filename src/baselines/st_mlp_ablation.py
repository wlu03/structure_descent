"""ST+MLP Ablation — "PO-LEU minus narratives".

Embeds the 5-key alt metadata dict with the same frozen encoder PO-LEU
uses (``sentence-transformers/all-mpnet-base-v2``) and trains a small
MLP ``(d_e -> 64 -> 1)``. The resulting per-alt scalar is softmaxed
across ``J`` alternatives and trained against ``c_star`` under
cross-entropy.

What this isolates
------------------
PO-LEU's pipeline is (LLM narratives) -> (encoder) -> (PO-LEU heads).
This baseline removes the first stage *and* replaces the third with a
generic MLP. It answers: *how much of PO-LEU's win comes from the
frozen encoder alone?*

What it does NOT isolate
------------------------
It does not separate narratives from heads. For the pure head
ablation (narratives kept, heads replaced), see
``src/model/ablations.py::variant_A6``. If ST+MLP's top-1 / NLL / MRR
land within 1 SE of PO-LEU's, the LLM-narrative step is not adding
signal beyond the encoder — report as a negative result.

Variant choice (design doc §3): **B (independent MLP)** — generic
MLP on ``(N, J, d_e)``. The confound between "narrative-vs-metadata"
and "PO-LEU-head-vs-MLP" is called out explicitly; the goal is a
competitive frozen-LLM baseline, not a pure internal ablation of
PO-LEU's heads.

Integration (do NOT auto-edit registries; the design doc forbids it)
--------------------------------------------------------------------
* Registry entry (to be added manually to ``src/baselines/run_all.py``)::

      ("ST-MLP", "src.baselines.st_mlp_ablation", "STMLPChoice"),

* ``__init__.py`` export (to be added manually)::

      from .st_mlp_ablation import STMLPChoice, STMLPFitted
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from src.outcomes.cache import EmbeddingsCache
from src.outcomes.encode import EncoderClient, encode_batch

from .base import BaselineEventBatch, FittedBaseline


# ---------------------------------------------------------------------------
# Text rendering (design doc §4)
# ---------------------------------------------------------------------------


def _render_alt_text(
    alt: Mapping[str, Any],
    *,
    drop_popularity: bool = False,
) -> str:
    """Render a 5-key ``alt_texts`` dict into a single sentence.

    Canonical key ordering; every event uses the same template so the
    frozen encoder sees a stationary prompt. Missing / ``None`` values
    collapse to sensible literals so the output is always non-empty and
    contains no ``None`` / ``nan`` substrings.

    Per-alternative leakage policy: every key consumed here is per-ASIN
    constant, so the same ASIN renders identically whether it appears as
    the chosen alt or as a sampled negative. ``state`` and ``is_repeat``
    are NOT rendered (they were per-customer / per-customer-per-ASIN
    fields whose chosen-vs-negative asymmetry leaked the chosen position
    to the encoder); see :mod:`src.data.adapter.YamlAdapter.alt_text`.

    Parameters
    ----------
    alt:
        Mapping with the canonical keys
        ``title, category, price, popularity_rank, brand``. Produced by
        :func:`src.data.adapter.alt_text`.
    drop_popularity:
        If ``True`` the popularity rank is replaced with the literal
        ``"unknown"`` so the encoder cannot read its numeric value.
        Diagnostic for suspected popularity-as-text leakage.

    Notes
    -----
    ``c_d`` is intentionally not rendered. PO-LEU's encoder only sees
    ``c_d`` indirectly through generated narratives; injecting ``c_d``
    here would give ST+MLP a signal PO-LEU lacks at the encoder layer
    and bias the comparison.
    """
    title = str(alt.get("title") or "unknown title").strip()
    category = str(alt.get("category") or "unknown category").strip()
    brand = str(alt.get("brand") or "unknown_brand").strip()

    price_raw = alt.get("price")
    try:
        price = float(price_raw if price_raw is not None else 0.0)
    except (TypeError, ValueError):
        price = 0.0

    if drop_popularity:
        pop_str = "unknown"
    else:
        pop_raw = alt.get("popularity_rank", 0)
        if pop_raw is None:
            pop_str = "popularity score 0"
        else:
            pop_str = str(pop_raw).strip() or "popularity score 0"

    parts: List[str] = [
        f"{title}.",
        f"Category: {category}.",
        f"Brand: {brand}.",
        f"Price: ${price:.1f}.",
        f"Popularity: {pop_str}.",
    ]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# MLP head (design doc §6)
# ---------------------------------------------------------------------------


def _build_mlp(d_e: int, hidden: int, dropout: float):
    """Construct ``Linear(d_e, hidden) -> GELU -> Dropout -> Linear(hidden, 1)``.

    Imported lazily so the module stays importable on torch-less hosts
    (the :class:`STMLPChoice.fit` path raises a clean ImportError then).
    """
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(d_e, hidden),
        nn.GELU(),
        nn.Dropout(float(dropout)),
        nn.Linear(hidden, 1),
    )


def _mlp_param_count(d_e: int, hidden: int) -> int:
    """Closed-form parameter count for the two-layer MLP.

    ``(d_e + 1) * hidden + (hidden + 1)``. See the design doc §6; at
    defaults (``d_e=768, hidden=64``) this is ``49,281``.
    """
    return (int(d_e) + 1) * int(hidden) + (int(hidden) + 1)


# ---------------------------------------------------------------------------
# raw_events access helpers
# ---------------------------------------------------------------------------


_RAW_EVENTS_MSG = (
    "STMLPChoice requires raw_events[e]['alt_texts']; re-produce the "
    "batch with records_to_baseline_batch."
)


def _extract_alt_texts(batch: BaselineEventBatch) -> List[List[Dict[str, Any]]]:
    """Pull per-event ``alt_texts`` lists out of ``batch.raw_events``.

    Raises a :class:`ValueError` with an actionable message if
    ``raw_events`` is absent or any event lacks ``alt_texts`` — the
    design doc §8 mandates this contract, mirroring the ASIN-mode
    popularity baseline's behaviour.
    """
    if batch.raw_events is None:
        raise ValueError(_RAW_EVENTS_MSG)
    J = batch.n_alternatives
    out: List[List[Dict[str, Any]]] = []
    for e, rec in enumerate(batch.raw_events):
        if not isinstance(rec, Mapping) or "alt_texts" not in rec:
            raise ValueError(_RAW_EVENTS_MSG)
        alts = rec["alt_texts"]
        try:
            alts_list = list(alts)
        except TypeError as exc:
            raise ValueError(_RAW_EVENTS_MSG) from exc
        if len(alts_list) != J:
            raise ValueError(
                f"STMLPChoice: event {e} has {len(alts_list)} alt_texts "
                f"but batch.n_alternatives={J}; uniform J required."
            )
        # Each alt entry must be a mapping so _render_alt_text can `.get`.
        for j, a in enumerate(alts_list):
            if not isinstance(a, Mapping):
                raise ValueError(
                    f"STMLPChoice: event {e} alt {j}: alt_texts entry "
                    f"must be a dict-like mapping, got {type(a).__name__}."
                )
        out.append([dict(a) for a in alts_list])
    return out


def _collect_omega(batch: BaselineEventBatch) -> Optional[np.ndarray]:
    """Return a length-N float array of ``omega`` weights, or ``None``.

    Omega is read from ``batch.metadata[e]['omega']`` (same contract as
    ``src.model.po_leu.cross_entropy_loss``). If *any* event lacks it,
    we return ``None`` — mixing weighted / unweighted events silently
    would bias the loss.
    """
    if not batch.metadata:
        return None
    vals: List[float] = []
    for meta in batch.metadata:
        if not isinstance(meta, Mapping) or "omega" not in meta:
            return None
        try:
            vals.append(float(meta["omega"]))
        except (TypeError, ValueError):
            return None
    return np.asarray(vals, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fitted object (design doc §8)
# ---------------------------------------------------------------------------


@dataclass
class STMLPFitted:
    """Fitted ST+MLP ablation, conforming to :class:`FittedBaseline`.

    Attributes
    ----------
    name
        Display name (matches :attr:`STMLPChoice.name`).
    state_dict
        ``torch`` state dict for the two-layer MLP.
    d_e
        Encoder output dimension (frozen at fit time from
        ``encoder.d_e``).
    hidden
        MLP hidden dimension.
    dropout
        Dropout probability (used only at train time).
    encoder
        Frozen :class:`EncoderClient` used for
        :meth:`score_events`.
    embeddings_cache
        Optional project-global :class:`EmbeddingsCache` forwarded to
        :func:`encode_batch` so scoring-time encodes hit the same cache
        PO-LEU uses.
    drop_popularity
        Whether the popularity field was rendered as ``"unknown"``
        during training. Scoring respects the same decision.
    n_params_total
        Count returned by :attr:`n_params`.
    train_nll, val_nll
        Per-event average NLL on the training / validation batches at
        the best checkpoint (or the final epoch if no val batch was
        supplied).
    """

    name: str
    state_dict: Dict[str, Any]
    d_e: int
    hidden: int
    dropout: float
    encoder: EncoderClient
    embeddings_cache: Optional[EmbeddingsCache]
    drop_popularity: bool
    n_params_total: int
    train_nll: float
    val_nll: float
    # Local in-memory embedding cache shared between fit and
    # score_events (design doc §9 layer 2). Not part of the public
    # contract; intentionally excluded from equality / hashing by the
    # default dataclass.
    local_embed_cache: Dict[str, np.ndarray] = field(default_factory=dict)

    # -- FittedBaseline protocol -----------------------------------------

    def score_events(self, batch: BaselineEventBatch) -> List[np.ndarray]:
        """Score each event as a length-``J`` logit vector.

        Re-renders every alternative through :func:`_render_alt_text`,
        routes the texts through :func:`encode_batch` (honouring the
        shared :class:`EmbeddingsCache`), reshapes to ``(N, J, d_e)``,
        and runs a single ``eval()`` MLP forward pass. The shared
        harness handles softmax / top-1 / MRR / NLL.
        """
        import torch

        alt_texts_per_event = _extract_alt_texts(batch)
        N = len(alt_texts_per_event)
        if N == 0:
            return []
        J = len(alt_texts_per_event[0])

        texts = [
            _render_alt_text(alt, drop_popularity=self.drop_popularity)
            for alts in alt_texts_per_event
            for alt in alts
        ]
        flat_vecs = _encode_with_local_cache(
            texts,
            encoder=self.encoder,
            cache=self.embeddings_cache,
            local=self.local_embed_cache,
        )
        if flat_vecs.shape != (N * J, self.d_e):
            raise RuntimeError(
                f"ST+MLP score_events: encoder returned shape "
                f"{flat_vecs.shape!r}; expected {(N * J, self.d_e)!r}."
            )
        E2 = flat_vecs.reshape(N, J, self.d_e).astype(np.float32, copy=False)

        model = _build_mlp(self.d_e, self.hidden, self.dropout)
        model.load_state_dict(self.state_dict)
        model.eval()
        with torch.no_grad():
            x = torch.as_tensor(E2)                                   # (N, J, d_e)
            logits = model(x.reshape(-1, self.d_e)).reshape(N, J)     # (N, J)
            out = logits.detach().cpu().numpy().astype(np.float32)

        return [out[i] for i in range(N)]

    @property
    def n_params(self) -> int:
        return int(self.n_params_total)

    @property
    def description(self) -> str:
        return (
            f"ST+MLP d_e={self.d_e} hidden={self.hidden} "
            f"dropout={self.dropout:g} drop_popularity={self.drop_popularity} "
            f"params={self.n_params_total} "
            f"train_nll={self.train_nll:.3f} val_nll={self.val_nll:.3f}"
        )


# ---------------------------------------------------------------------------
# Local cache helper (design doc §9 layer 2)
# ---------------------------------------------------------------------------


def _encode_with_local_cache(
    texts: Sequence[str],
    *,
    encoder: EncoderClient,
    cache: Optional[EmbeddingsCache],
    local: Dict[str, np.ndarray],
) -> np.ndarray:
    """Encode ``texts`` preferring a per-fit in-memory dict.

    The global :class:`EmbeddingsCache` is authoritative and honoured
    via :func:`encode_batch`; the local dict short-circuits duplicate
    strings within a single fit / score pass without round-tripping
    SQLite. Misses from both layers are encoded in a single batch and
    written back to both.
    """
    d_e = int(encoder.d_e)
    n = len(texts)
    if n == 0:
        return np.zeros((0, d_e), dtype=np.float32)

    out = np.zeros((n, d_e), dtype=np.float32)
    miss_indices: List[int] = []
    miss_texts: List[str] = []
    for i, t in enumerate(texts):
        vec = local.get(t)
        if vec is not None:
            out[i] = vec
        else:
            miss_indices.append(i)
            miss_texts.append(t)

    if miss_texts:
        encoded = encode_batch(miss_texts, client=encoder, cache=cache)
        if encoded.shape != (len(miss_texts), d_e):
            raise RuntimeError(
                f"encode_batch returned shape {encoded.shape!r}; "
                f"expected {(len(miss_texts), d_e)!r}."
            )
        for idx, text, vec in zip(miss_indices, miss_texts, encoded):
            vec32 = np.asarray(vec, dtype=np.float32)
            out[idx] = vec32
            local[text] = vec32

    return out


# ---------------------------------------------------------------------------
# Fit-time class (design doc §7, §8)
# ---------------------------------------------------------------------------


class STMLPChoice:
    """ST+MLP ablation — "PO-LEU minus narratives".

    Embeds the 5-key alt metadata dict with the same frozen encoder
    PO-LEU uses and trains a small MLP ``(d_e -> 64 -> 1)``. See module
    docstring for the full comparison note (design doc §11).

    Parameters
    ----------
    hidden_dim
        MLP hidden dimension (design doc default: 64).
    dropout
        Dropout probability on the hidden layer (default: 0.1).
    lr
        Adam learning rate (default: 1e-3).
    weight_decay
        Adam weight decay (default: 1e-4).
    n_epochs
        Maximum training epochs (default: 100).
    batch_size
        Mini-batch size over events (default: 64).
    patience
        Early-stopping patience on validation NLL (default: 15).
    seed
        PRNG seed for torch / numpy / DataLoader generator (default: 7).
    encoder
        Frozen :class:`EncoderClient`. Tests pass a
        :class:`~src.outcomes.encode.StubEncoder`; real runs use
        :class:`~src.outcomes.encode.SentenceTransformersEncoder`. If
        ``None``, defaults to a
        :class:`~src.outcomes.encode.SentenceTransformersEncoder`
        constructed lazily in :meth:`fit`.
    embeddings_cache
        Optional project-global :class:`EmbeddingsCache`. Shared with
        PO-LEU by design — the sentence format differs from narratives
        so keys do not collide.
    drop_popularity
        Diagnostic for popularity-as-text leakage. ``False`` (default)
        renders the popularity rank in the encoded text. ``True``
        replaces it with ``"unknown"`` so the encoder cannot read its
        numeric value. Toggle this and re-fit to test whether the
        encoder is exploiting popularity rather than learning
        preference structure.
    """

    name: str = "ST+MLP"

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        n_epochs: int = 100,
        batch_size: int = 64,
        patience: int = 15,
        seed: int = 7,
        encoder: Optional[EncoderClient] = None,
        embeddings_cache: Optional[EmbeddingsCache] = None,
        drop_popularity: bool = False,
    ) -> None:
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim!r}")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {dropout!r}"
            )
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr!r}")
        if n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {n_epochs!r}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size!r}")

        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.seed = int(seed)
        self.encoder = encoder
        self.embeddings_cache = embeddings_cache
        self.drop_popularity = bool(drop_popularity)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train: BaselineEventBatch,
        val: Optional[BaselineEventBatch] = None,
    ) -> STMLPFitted:
        """Fit the MLP on ``train`` with optional early-stopping on ``val``.

        The encoder is frozen throughout.
        """
        try:
            import torch
            import torch.nn.functional as F
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "STMLPChoice requires PyTorch. Install with `pip install torch`."
            ) from exc

        if train.n_events == 0:
            raise ValueError("STMLPChoice.fit received an empty train batch")

        # ---- encoder ---------------------------------------------------
        encoder = self.encoder
        if encoder is None:
            # Lazy default: the real SentenceTransformers encoder. Tests
            # always pass a StubEncoder so this import path is never hit
            # in the hermetic test suite.
            from src.outcomes.encode import SentenceTransformersEncoder
            encoder = SentenceTransformersEncoder()
        d_e = int(encoder.d_e)

        train_alt_texts = _extract_alt_texts(train)

        # ---- seeding (§7) ----------------------------------------------
        torch.manual_seed(self.seed)
        # The MLP is purely per-alt so there is no CUDA-specific RNG
        # path to seed; keeping the manual_seed call is sufficient for
        # deterministic init on CPU.
        perm_rng = np.random.default_rng(self.seed)

        # ---- embedding tensor for train + val -------------------------
        local_cache: Dict[str, np.ndarray] = {}
        E_train = self._encode_events(
            train_alt_texts,
            encoder=encoder,
            local_cache=local_cache,
            drop_popularity=self.drop_popularity,
            d_e=d_e,
        )
        y_train_np = np.asarray(train.chosen_indices, dtype=np.int64)
        omega_train = _collect_omega(train)

        if val is not None and val.n_events > 0:
            val_alt_texts = _extract_alt_texts(val)
            E_val = self._encode_events(
                val_alt_texts,
                encoder=encoder,
                local_cache=local_cache,
                drop_popularity=self.drop_popularity,
                d_e=d_e,
            )
            y_val_np = np.asarray(val.chosen_indices, dtype=np.int64)
        else:
            E_val = None
            y_val_np = None

        E_train_t = torch.as_tensor(E_train)
        y_train_t = torch.as_tensor(y_train_np)
        omega_train_t = (
            torch.as_tensor(omega_train) if omega_train is not None else None
        )

        if E_val is not None:
            E_val_t = torch.as_tensor(E_val)
            y_val_t = torch.as_tensor(y_val_np)
        else:
            E_val_t = None
            y_val_t = None

        # ---- model + optimiser -----------------------------------------
        model = _build_mlp(d_e, self.hidden_dim, self.dropout)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        def _forward(E: "torch.Tensor") -> "torch.Tensor":
            # E: (B, J, d_e) -> logits (B, J)
            B, J, D = E.shape
            flat = model(E.reshape(B * J, D)).reshape(B, J)
            return flat

        def _loss(
            logits: "torch.Tensor",
            y: "torch.Tensor",
            w: "Optional[torch.Tensor]",
        ) -> "torch.Tensor":
            per = F.cross_entropy(logits, y, reduction="none")
            if w is None:
                return per.mean()
            return (w * per).sum() / w.sum().clamp_min(1e-12)

        # ---- training loop (§7) ---------------------------------------
        N_tr = int(E_train_t.shape[0])
        best_val_nll = float("inf")
        best_state: Optional[Dict[str, Any]] = None
        patience_left = self.patience

        for _epoch in range(self.n_epochs):
            model.train()
            perm = perm_rng.permutation(N_tr)
            for start in range(0, N_tr, self.batch_size):
                idx = perm[start : start + self.batch_size]
                if len(idx) == 0:
                    continue
                idx_t = torch.as_tensor(idx, dtype=torch.long)
                Eb = E_train_t.index_select(0, idx_t)
                yb = y_train_t.index_select(0, idx_t)
                wb = (
                    omega_train_t.index_select(0, idx_t)
                    if omega_train_t is not None
                    else None
                )
                logits = _forward(Eb)
                loss = _loss(logits, yb, wb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if E_val_t is not None:
                model.eval()
                with torch.no_grad():
                    v_nll = float(F.cross_entropy(_forward(E_val_t), y_val_t).item())
                if v_nll < best_val_nll - 1e-6:
                    best_val_nll = v_nll
                    best_state = {
                        k: v.detach().clone() for k, v in model.state_dict().items()
                    }
                    patience_left = self.patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            train_nll = float(
                F.cross_entropy(_forward(E_train_t), y_train_t).item()
            )
            if E_val_t is not None:
                val_nll = float(
                    F.cross_entropy(_forward(E_val_t), y_val_t).item()
                )
            else:
                val_nll = float("nan")

        state_dict = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

        return STMLPFitted(
            name=self.name,
            state_dict=state_dict,
            d_e=d_e,
            hidden=self.hidden_dim,
            dropout=self.dropout,
            encoder=encoder,
            embeddings_cache=self.embeddings_cache,
            drop_popularity=self.drop_popularity,
            n_params_total=_mlp_param_count(d_e, self.hidden_dim),
            train_nll=float(train_nll),
            val_nll=float(val_nll),
            local_embed_cache=local_cache,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _encode_events(
        self,
        alt_texts_per_event: Sequence[Sequence[Mapping[str, Any]]],
        *,
        encoder: EncoderClient,
        local_cache: Dict[str, np.ndarray],
        drop_popularity: bool,
        d_e: int,
    ) -> np.ndarray:
        """Render + encode a batch into ``(N, J, d_e)`` float32."""
        N = len(alt_texts_per_event)
        if N == 0:
            return np.zeros((0, 0, d_e), dtype=np.float32)
        J = len(alt_texts_per_event[0])
        texts = [
            _render_alt_text(alt, drop_popularity=drop_popularity)
            for alts in alt_texts_per_event
            for alt in alts
        ]
        flat = _encode_with_local_cache(
            texts,
            encoder=encoder,
            cache=self.embeddings_cache,
            local=local_cache,
        )
        return flat.reshape(N, J, d_e).astype(np.float32, copy=False)


__all__ = [
    "STMLPChoice",
    "STMLPFitted",
    "_render_alt_text",
]
