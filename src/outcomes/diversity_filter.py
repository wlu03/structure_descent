"""Post-hoc paraphrase filter for generated outcomes (redesign.md §3.5).

This module detects near-duplicate outcomes within a single alternative's
K-tuple using cosine similarity on encoder embeddings. It exposes a callable
with the signature expected by ``generate.py``'s ``diversity_filter`` parameter
plus small helpers used by the tests.

The real encoder is a Wave-3 deliverable. To keep this module testable today
we define an :class:`EmbeddingFn` :class:`~typing.Protocol` that matches the
Wave-3 interface and ship a deterministic stub :class:`HashEmbedder` that
relies only on numpy + stdlib.

Contract recap
--------------
* The filter never regenerates — it only signals failure. The retry loop is
  ``generate.py``'s responsibility (redesign.md §3.5: "at most 2 retries;
  after that, accept").
* Default threshold is ``0.9`` (redesign.md §3.5).
* All helpers are deterministic and have no side effects on import.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingFn(Protocol):
    """Callable that maps ``list[str] -> (N, d) float32, L2-normalized``.

    Wave 3 will ship a concrete implementation backed by a frozen sentence
    encoder (redesign.md §4). Any object satisfying this protocol — including
    :class:`HashEmbedder` below — is accepted by :func:`find_paraphrase_pair`
    and :func:`diversity_filter`.
    """

    def __call__(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# Stub encoder
# ---------------------------------------------------------------------------


class HashEmbedder:
    """Deterministic stdlib+numpy stub encoder for tests.

    The strategy is a signed hashing trick over character 3-grams:

    1. Extract all character 3-grams (sliding window) of each input string.
       Padding is applied with a sentinel so strings shorter than 3 chars
       still emit at least one 3-gram.
    2. Each 3-gram is hashed to a bucket in ``[0, dim)`` via SHA-256.
    3. A second, seeded SHA-256 hash over the same 3-gram + seed yields a
       sign in ``{-1, +1}``. This makes different hashing seeds produce
       (nearly) orthogonal embedding spaces, which matters for tests.
    4. Counts are accumulated as ``signed_count[bucket]`` and L2-normalized.

    Two distinct strings share 3-grams only to the extent that their
    characters overlap, so e.g. ``"hello world"`` and ``"quantum physics today"``
    land in nearly disjoint buckets and have cosine similarity well below the
    §3.5 threshold of ``0.9``.
    """

    def __init__(self, dim: int = 64, seed: int = 0) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = int(dim)
        self.seed = int(seed)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _char_ngrams(text: str, n: int = 3) -> list[str]:
        """Character n-grams with a sentinel pad so short strings still emit one.

        A single leading and trailing ``"\\x02"`` byte flanks the string; this
        also helps distinguish e.g. ``"ab"`` from ``"ba"`` when n=3.
        """
        padded = "\x02" + text + "\x03"
        if len(padded) < n:
            return [padded]
        return [padded[i : i + n] for i in range(len(padded) - n + 1)]

    def _bucket(self, token: str) -> int:
        """Hash ``token`` to a bucket in ``[0, dim)``."""
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        # Use the first 8 bytes as a big-endian unsigned integer.
        bucket = int.from_bytes(digest[:8], "big") % self.dim
        return bucket

    def _sign(self, token: str) -> int:
        """Seed-dependent sign in ``{-1, +1}`` for ``token``."""
        material = f"{self.seed}\x00{token}".encode("utf-8")
        digest = hashlib.sha256(material).digest()
        # Parity of the last byte: even -> +1, odd -> -1.
        return 1 if (digest[-1] & 1) == 0 else -1

    # -- public API --------------------------------------------------------

    def __call__(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` into a ``(N, dim)`` float32, L2-normalized matrix."""
        if not isinstance(texts, list):
            raise TypeError(f"texts must be a list of str, got {type(texts).__name__}")

        n = len(texts)
        out = np.zeros((n, self.dim), dtype=np.float32)

        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"texts[{i}] must be str, got {type(text).__name__}"
                )
            for gram in self._char_ngrams(text):
                b = self._bucket(gram)
                s = self._sign(gram)
                out[i, b] += s

            # L2-normalize row. Guard against all-zero rows (pathological
            # input where signs perfectly cancel in every bucket): fall back
            # to a deterministic unit vector so downstream invariants hold.
            norm = float(np.linalg.norm(out[i]))
            if norm > 0.0:
                out[i] /= norm
            else:
                # Pick a fallback bucket deterministically from the text.
                fallback_bucket = self._bucket("__fallback__\x00" + text)
                out[i, fallback_bucket] = 1.0

        return out


# ---------------------------------------------------------------------------
# Cosine similarity and paraphrase detection
# ---------------------------------------------------------------------------


def pairwise_cosine(emb: np.ndarray) -> np.ndarray:
    """Return the ``(N, N)`` cosine-similarity matrix of pre-normalized ``emb``.

    Parameters
    ----------
    emb:
        ``(N, d)`` array. **Every row must already be L2-normalized**; this is
        asserted with ``atol=1e-4``. Pre-normalization is cheap inside the
        embedder and avoids the per-call cost here.

    Returns
    -------
    np.ndarray
        ``(N, N)`` cosine similarities. Diagonal entries are ``≈ 1``.
    """
    if emb.ndim != 2:
        raise ValueError(f"emb must be 2-D, got shape {emb.shape!r}")

    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        bad = np.where(np.abs(norms - 1.0) > 1e-4)[0]
        raise ValueError(
            "pairwise_cosine expects L2-normalized rows; "
            f"{len(bad)} of {len(norms)} rows violate ||row||≈1"
        )

    return emb @ emb.T


def find_paraphrase_pair(
    outcomes: list[str],
    *,
    threshold: float = 0.9,
    embed_fn: EmbeddingFn | None = None,
) -> tuple[int, int] | None:
    """Return the first ``(i, j)`` (``i < j``) with ``cos_sim > threshold``.

    Self-similarity on the diagonal is ignored.

    Parameters
    ----------
    outcomes:
        The K outcome strings for a single alternative.
    threshold:
        Cosine-similarity cutoff. Default is ``0.9`` from redesign.md §3.5.
    embed_fn:
        Optional embedder conforming to :class:`EmbeddingFn`. If ``None``, a
        :class:`HashEmbedder(dim=64, seed=0)` is instantiated internally.

    Returns
    -------
    tuple[int, int] | None
        ``(i, j)`` for the first offending pair encountered in row-major order,
        or ``None`` if the tuple is paraphrase-free.
    """
    if len(outcomes) < 2:
        return None

    if embed_fn is None:
        embed_fn = HashEmbedder(dim=64, seed=0)

    emb = embed_fn(outcomes)
    sim = pairwise_cosine(emb)

    n = sim.shape[0]
    # Scan strictly upper-triangular entries in row-major order so the first
    # returned pair is the lexicographically smallest (i, j) with i < j.
    for i in range(n):
        for j in range(i + 1, n):
            if float(sim[i, j]) > threshold:
                return (i, j)
    return None


def diversity_filter(
    outcomes: list[str],
    *,
    threshold: float = 0.9,
    embed_fn: EmbeddingFn | None = None,
) -> tuple[list[str], bool]:
    """Signal whether ``outcomes`` contain a near-duplicate pair.

    Signature matches the callable ``generate.py`` injects at the retry
    boundary. The filter **does not regenerate**; the caller decides whether
    to retry.

    Parameters
    ----------
    outcomes:
        The K outcome strings for a single alternative.
    threshold, embed_fn:
        Forwarded to :func:`find_paraphrase_pair`.

    Returns
    -------
    tuple[list[str], bool]
        ``(outcomes_as_is, ok)`` where ``ok`` is ``True`` iff no pair exceeds
        the threshold. The outcomes list is returned unchanged (identity-
        preserving) so callers can chain retries without losing intermediate
        state.
    """
    pair = find_paraphrase_pair(outcomes, threshold=threshold, embed_fn=embed_fn)
    return outcomes, pair is None
