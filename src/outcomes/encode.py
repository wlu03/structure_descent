"""Frozen sentence encoder for outcome narratives (redesign.md §4).

This module produces ``d_e = 768``-dim embedding vectors for outcome
strings (§4.1 default). Two concrete clients live here:

* :class:`StubEncoder` — a deterministic, hermetic stub backed by numpy
  + :mod:`hashlib`. Used in the test suite and any context that must not
  import heavy ML dependencies.
* :class:`SentenceTransformersEncoder` — the real client. Imports
  ``sentence_transformers`` (and transitively ``torch``) **lazily** at
  instantiation time so that ``import src.outcomes.encode`` remains free
  of ML dependencies and safe on ML-less environments.

Both clients satisfy the :class:`EncoderClient` :class:`~typing.Protocol`.
The top-level helpers :func:`encode_batch` and :func:`encode_outcomes_tensor`
layer the §4.3 :class:`~src.outcomes.cache.EmbeddingsCache` on top of any
``EncoderClient``; individual clients do not consult the cache themselves.

Shape contract
--------------
* ``encode(texts)`` → ``(N, d_e)`` float32, L2-normalized on the last dim.
* ``encode_batch(texts, client=..., cache=...)`` → ``(N, d_e)`` preserving
  the order of ``texts``.
* ``encode_outcomes_tensor(outcomes, client=..., cache=...)`` →
  ``(B, J, K, d_e)`` where ``B = len(outcomes)``, ``J =
  len(outcomes[0])``, and ``K = len(outcomes[0][0])``. The outer three
  dimensions must be rectangular (all inner lists equal length) or a
  :class:`ValueError` is raised.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np

from src.outcomes.cache import EmbeddingsCache


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EncoderClient(Protocol):
    """Frozen sentence-encoder contract.

    Attributes
    ----------
    encoder_id:
        Stable string identifier. Incorporated into the §4.3 cache key
        (``sha256(outcome_string || encoder_id)``) so swapping the client
        — or any hyperparameter that materially changes its outputs —
        invalidates cached embeddings.
    d_e:
        Output embedding dimension. Default for PO-LEU is ``768`` (§4.1).

    Methods
    -------
    encode(texts)
        Return an ``(N, d_e)`` float32 array, L2-normalized along the
        last dim. Implementations must not mutate ``texts`` and must
        not consult any cache — caching is layered on top by
        :func:`encode_batch`.
    """

    @property
    def encoder_id(self) -> str:  # pragma: no cover - protocol stub
        ...

    @property
    def d_e(self) -> int:  # pragma: no cover - protocol stub
        ...

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# Deterministic stub
# ---------------------------------------------------------------------------


class StubEncoder:
    """Deterministic hash-seeded encoder for tests.

    Strategy
    --------
    For each input string ``s``:

    1. Derive a 64-bit seed from :func:`hashlib.blake2b` over
       ``encoder_id || "\\x00" || s`` (truncated to 8 bytes, big-endian).
    2. Instantiate a fresh :class:`numpy.random.Generator` with that
       seed, draw ``d_e`` Gaussian samples, and L2-normalize.

    Two different strings differ in at least one byte of the blake2b
    digest with overwhelming probability, so their seeds (and therefore
    their vectors) diverge. ``encoder_id`` mixes into the seed so
    distinct stub encoders occupy effectively orthogonal subspaces —
    which also changes the §4.3 cache key.

    Parameters
    ----------
    encoder_id:
        Identifier baked into the hash seed and exposed via the
        :attr:`encoder_id` property. Default ``"stub-hash-768"``.
    d_e:
        Output dimension. Default ``768``.
    """

    def __init__(
        self,
        encoder_id: str = "stub-hash-768",
        d_e: int = 768,
    ) -> None:
        if d_e <= 0:
            raise ValueError(f"d_e must be positive, got {d_e}")
        self._encoder_id = str(encoder_id)
        self._d_e = int(d_e)

    # -- EncoderClient interface ------------------------------------------

    @property
    def encoder_id(self) -> str:
        return self._encoder_id

    @property
    def d_e(self) -> int:
        return self._d_e

    # -- helpers ----------------------------------------------------------

    def _seed_for(self, text: str) -> int:
        """Derive a deterministic 64-bit seed for ``text``.

        The ``encoder_id`` is mixed into the digest material so two
        ``StubEncoder`` instances with different ids produce different
        vectors for the same string — which is exactly the property
        ``test_encoder_id_changes_cache_key`` and the §4.3 cache-key
        rule rely on.
        """
        material = self._encoder_id.encode("utf-8") + b"\x00" + text.encode("utf-8")
        digest = hashlib.blake2b(material, digest_size=8).digest()
        return int.from_bytes(digest, "big", signed=False)

    # -- public API -------------------------------------------------------

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` into ``(N, d_e)`` float32, L2-normalized rows.

        Never mutates ``texts``; never touches any cache.
        """
        if not isinstance(texts, list):
            raise TypeError(
                f"texts must be a list of str, got {type(texts).__name__}"
            )
        n = len(texts)
        out = np.zeros((n, self._d_e), dtype=np.float32)
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"texts[{i}] must be str, got {type(text).__name__}"
                )
            seed = self._seed_for(text)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self._d_e).astype(np.float32, copy=False)
            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                vec /= norm
            else:
                # Astronomically unlikely (Gaussian draw landing on
                # exact zero in every coordinate); fall back to a
                # deterministic unit basis.
                vec = np.zeros(self._d_e, dtype=np.float32)
                vec[seed % self._d_e] = 1.0
            out[i] = vec
        return out


# ---------------------------------------------------------------------------
# Real sentence-transformers-backed client (lazy import)
# ---------------------------------------------------------------------------


class SentenceTransformersEncoder:
    """Frozen ``sentence-transformers`` encoder (redesign.md §4.1-§4.2).

    The model and its dependencies (``sentence_transformers``, ``torch``)
    are imported **lazily** inside :meth:`__init__`. Simply importing
    :mod:`src.outcomes.encode` never loads them, which keeps the module
    hermetic in environments where torch is unavailable.

    Parameters
    ----------
    model_id:
        Hugging Face repo id. Default
        ``"sentence-transformers/all-mpnet-base-v2"`` (§4.1).
    max_length:
        Token-truncation limit passed to the model. §4.2 step 2 pins
        this at 64.
    pooling:
        Pooling strategy. Only ``"mean"`` is exercised by the tests; the
        parameter is exposed so future pooling variants can slot in
        without a new class. Any other value passes through to the
        underlying ``encode()`` call unchanged, but correctness for
        other values is not covered here.
    device:
        Optional device string (``"cuda"``, ``"cpu"``, ...). ``None``
        delegates to ``sentence-transformers`` default detection.

    Notes
    -----
    * The model is **frozen**: §9.5 forbids fine-tuning it.
    * ``encoder_id`` is a deterministic function of
      ``(model_id, pooling, max_length)`` so changing any of those
      invalidates the §4.3 cache.
    """

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-mpnet-base-v2",
        *,
        max_length: int = 64,
        pooling: str = "mean",
        device: str | None = None,
    ) -> None:
        self._model_id = str(model_id)
        self._max_length = int(max_length)
        self._pooling = str(pooling)
        self._device = device

        try:
            # Lazy import: only incurred on instantiation, never on
            # ``import src.outcomes.encode``.
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - covered by env, not tests
            raise ImportError(
                "SentenceTransformersEncoder requires the 'sentence-transformers' "
                "package. Install it with `pip install sentence-transformers` or "
                "use StubEncoder for hermetic tests."
            ) from exc

        self._model = SentenceTransformer(self._model_id, device=self._device)
        # §4.2 step 2: 64-token truncation.
        self._model.max_seq_length = self._max_length

        # Sentence-transformers models expose their output dim; fall back
        # to 768 (§4.1 default) if an exotic variant omits the helper.
        try:
            self._d_e = int(self._model.get_sentence_embedding_dimension())
        except Exception:  # pragma: no cover - defensive
            self._d_e = 768

    # -- EncoderClient interface ------------------------------------------

    @property
    def encoder_id(self) -> str:
        """Cache-key identifier: ``model_id|pooling=<p>|max_length=<n>``.

        Swapping any of the three components yields a different id and
        therefore a different §4.3 cache key, so cached embeddings are
        not mixed across encoder configurations.
        """
        return (
            f"{self._model_id}|pooling={self._pooling}|max_length={self._max_length}"
        )

    @property
    def d_e(self) -> int:
        return self._d_e

    # -- public API -------------------------------------------------------

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` into ``(N, d_e)`` float32, L2-normalized rows."""
        if not isinstance(texts, list):
            raise TypeError(
                f"texts must be a list of str, got {type(texts).__name__}"
            )
        if len(texts) == 0:
            return np.zeros((0, self._d_e), dtype=np.float32)

        # sentence-transformers handles mean-pooling + L2 normalization
        # internally; §4.2 step 4 says normalization is on by default.
        arr = self._model.encode(
            list(texts),
            batch_size=min(len(texts), 64),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self._d_e:
            raise RuntimeError(
                f"encoder returned shape {arr.shape!r}; expected (N, {self._d_e})"
            )
        return arr


# ---------------------------------------------------------------------------
# Batched encoding with cache layer
# ---------------------------------------------------------------------------


def encode_batch(
    texts: list[str],
    *,
    client: EncoderClient,
    cache: EmbeddingsCache | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode ``texts`` to ``(N, d_e)`` with optional §4.3 cache layer.

    Behaviour:

    * When ``cache is None`` this is a pure pass-through to
      ``client.encode(texts)`` — no side effects, no caching.
    * When ``cache`` is provided, each text is looked up by
      ``EmbeddingsCache.embedding_key(text, client.encoder_id)``. The
      cache-miss subset is encoded in a **single** batch (to amortize
      model setup and batch kernels), the results are written back to
      the cache, and the full result tensor is reassembled in the
      original input order.

    Parameters
    ----------
    texts:
        Input strings. Not mutated.
    client:
        Any :class:`EncoderClient`. Must expose ``d_e`` and
        ``encoder_id``.
    cache:
        Optional :class:`~src.outcomes.cache.EmbeddingsCache`. When
        ``None``, the function is a pure pass-through.
    show_progress:
        If ``True``, wrap the cache-miss encoding in a :mod:`tqdm`
        progress bar. Silent by default so library use is quiet.

    Returns
    -------
    np.ndarray
        ``(N, d_e)`` float32. Rows are L2-normalized on the last dim.

    Raises
    ------
    TypeError
        If ``texts`` is not a list of ``str``.
    """
    if not isinstance(texts, list):
        raise TypeError(
            f"texts must be a list of str, got {type(texts).__name__}"
        )
    for i, t in enumerate(texts):
        if not isinstance(t, str):
            raise TypeError(
                f"texts[{i}] must be str, got {type(t).__name__}"
            )

    n = len(texts)
    d_e = int(client.d_e)

    if n == 0:
        return np.zeros((0, d_e), dtype=np.float32)

    # Pass-through when no cache: let the client handle everything.
    if cache is None:
        arr = np.asarray(client.encode(list(texts)), dtype=np.float32)
        if arr.shape != (n, d_e):
            raise RuntimeError(
                f"client.encode returned shape {arr.shape!r}; expected {(n, d_e)!r}"
            )
        return arr

    out = np.zeros((n, d_e), dtype=np.float32)
    miss_indices: list[int] = []
    miss_texts: list[str] = []

    for i, text in enumerate(texts):
        vec = cache.get_embedding(text, client.encoder_id)
        if vec is None:
            miss_indices.append(i)
            miss_texts.append(text)
        else:
            if vec.shape != (d_e,):
                raise RuntimeError(
                    f"cache returned embedding of shape {vec.shape!r} "
                    f"for client with d_e={d_e}; cache corruption or "
                    "encoder_id collision"
                )
            out[i] = vec.astype(np.float32, copy=False)

    if miss_texts:
        # One batch for the entire cache-miss subset — §16 warns that
        # redundant encoding is the usual throughput bottleneck.
        encoded = np.asarray(client.encode(list(miss_texts)), dtype=np.float32)
        if encoded.shape != (len(miss_texts), d_e):
            raise RuntimeError(
                f"client.encode returned shape {encoded.shape!r}; "
                f"expected {(len(miss_texts), d_e)!r}"
            )

        iterator: object = zip(miss_indices, miss_texts, encoded)
        if show_progress:
            try:
                from tqdm import tqdm  # type: ignore
            except ImportError:  # pragma: no cover - tqdm is a hard dep
                pass
            else:
                iterator = tqdm(
                    iterator,
                    total=len(miss_texts),
                    desc="encode_batch",
                    leave=False,
                )

        for idx, text, vec in iterator:
            vec32 = np.asarray(vec, dtype=np.float32)
            out[idx] = vec32
            cache.put_embedding(text, client.encoder_id, vec32)

    return out


# ---------------------------------------------------------------------------
# 4-D outcomes tensor builder
# ---------------------------------------------------------------------------


def encode_outcomes_tensor(
    outcomes: list[list[list[str]]],
    *,
    client: EncoderClient,
    cache: EmbeddingsCache | None = None,
) -> np.ndarray:
    """Encode a nested outcome list into a ``(B, J, K, d_e)`` tensor.

    Parameters
    ----------
    outcomes:
        Nested list with shape semantics ``[B][J][K]``. Every inner
        alternative must have the same ``J`` entries, and every outcome
        list must have the same ``K`` entries.
    client:
        Frozen encoder conforming to :class:`EncoderClient`.
    cache:
        Optional :class:`~src.outcomes.cache.EmbeddingsCache` forwarded
        to :func:`encode_batch`.

    Returns
    -------
    np.ndarray
        ``(B, J, K, d_e)`` float32. Rows along the last dim are
        L2-normalized.

    Raises
    ------
    ValueError
        If the outer three dimensions are ragged (alternatives of
        different J, or outcome lists of different K).
    """
    if not isinstance(outcomes, list):
        raise ValueError(
            f"outcomes must be list[list[list[str]]], got {type(outcomes).__name__}"
        )
    B = len(outcomes)
    if B == 0:
        return np.zeros((0, 0, 0, int(client.d_e)), dtype=np.float32)

    # Rectangular check on (J) level.
    if not all(isinstance(alt_list, list) for alt_list in outcomes):
        raise ValueError("outcomes[b] must each be a list of alternatives")
    J_values = {len(alt_list) for alt_list in outcomes}
    if len(J_values) != 1:
        raise ValueError(
            f"outcomes is ragged: alternatives-per-event counts = {sorted(J_values)}"
        )
    J = J_values.pop()
    if J == 0:
        return np.zeros((B, 0, 0, int(client.d_e)), dtype=np.float32)

    # Rectangular check on (K) level.
    K_values: set[int] = set()
    for b, alt_list in enumerate(outcomes):
        for j, k_list in enumerate(alt_list):
            if not isinstance(k_list, list):
                raise ValueError(
                    f"outcomes[{b}][{j}] must be a list of outcome strings"
                )
            K_values.add(len(k_list))
    if len(K_values) != 1:
        raise ValueError(
            f"outcomes is ragged: outcomes-per-alternative counts = {sorted(K_values)}"
        )
    K = K_values.pop()
    d_e = int(client.d_e)

    # Flatten in row-major order; reshape preserves it on the way back.
    flat: list[str] = []
    for alt_list in outcomes:
        for k_list in alt_list:
            for s in k_list:
                if not isinstance(s, str):
                    raise ValueError(
                        "outcomes leaves must be str, got "
                        f"{type(s).__name__}"
                    )
                flat.append(s)

    flat_vecs = encode_batch(flat, client=client, cache=cache)
    if flat_vecs.shape != (B * J * K, d_e):
        raise RuntimeError(
            f"encode_batch returned shape {flat_vecs.shape!r}; expected "
            f"{(B * J * K, d_e)!r}"
        )
    return flat_vecs.reshape(B, J, K, d_e)
