"""SQLite-backed key-value caches for PO-LEU outcome generation and embedding.

Implements two independent caches over a shared ``KVStore`` backend:

* :class:`OutcomesCache` — stores LLM-generated outcome narratives keyed by
  ``sha256(customer_id || asin || seed || prompt_version)`` (redesign.md §3.4).
* :class:`EmbeddingsCache` — stores frozen-encoder embeddings keyed by
  ``sha256(outcome_string || encoder_id)`` (redesign.md §4.3).

The two caches are independent artefacts; changing the generator prompt
invalidates outcomes but not embeddings for strings that did not change
(§15 reproducibility checklist).

Thread-safety
-------------
SQLite connections are **not** thread-safe. Open one :class:`KVStore`
instance per thread. No global state is kept; every bit of state lives on
the instance.

Serialization
-------------
* Outcomes values are JSON bytes (UTF-8).
* Embedding values are serialized via :func:`numpy.save` into an
  ``io.BytesIO`` buffer (``.npy`` format). This carries dtype and shape
  and is read back with :func:`numpy.load`. See ``NOTES.md``.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np


class KVStore:
    """Single-table SQLite key-value store with WAL journaling.

    Parameters
    ----------
    path:
        Location of the SQLite database file. Parent directories must
        already exist.
    table:
        Table name. Defaults to ``"kv"``.
    create:
        If ``True`` (default) create the table if it does not yet exist.
        If ``False`` the table must already exist.

    Notes
    -----
    The connection is opened in WAL mode (``PRAGMA journal_mode=WAL``)
    which permits concurrent readers while a writer holds the log.
    SQLite connections are not thread-safe; open one store per thread.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        table: str = "kv",
        create: bool = True,
    ) -> None:
        self.path = Path(path)
        self.table = table
        # check_same_thread=True (default) intentionally enforces
        # the single-thread contract documented in the module docstring.
        self._conn: sqlite3.Connection | None = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        if create:
            self._conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table} ("
                "key TEXT PRIMARY KEY, "
                "value BLOB NOT NULL, "
                "created_at REAL NOT NULL"
                ")"
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Core KV operations
    # ------------------------------------------------------------------
    def get(self, key: str) -> bytes | None:
        """Return the value bytes for ``key`` or ``None`` if absent."""
        cur = self._require_conn().execute(
            f"SELECT value FROM {self.table} WHERE key = ?", (key,)
        )
        row = cur.fetchone()
        return None if row is None else bytes(row[0])

    def put(self, key: str, value: bytes) -> None:
        """Idempotently write ``value`` under ``key`` (INSERT OR REPLACE)."""
        conn = self._require_conn()
        conn.execute(
            f"INSERT OR REPLACE INTO {self.table} (key, value, created_at) "
            "VALUES (?, ?, ?)",
            (key, sqlite3.Binary(value), time.time()),
        )
        conn.commit()

    def has(self, key: str) -> bool:
        """Return whether ``key`` is present in the store."""
        cur = self._require_conn().execute(
            f"SELECT 1 FROM {self.table} WHERE key = ? LIMIT 1", (key,)
        )
        return cur.fetchone() is not None

    def delete(self, key: str) -> None:
        """Remove ``key`` from the store (no error if absent)."""
        conn = self._require_conn()
        conn.execute(f"DELETE FROM {self.table} WHERE key = ?", (key,))
        conn.commit()

    def __len__(self) -> int:
        cur = self._require_conn().execute(f"SELECT COUNT(*) FROM {self.table}")
        return int(cur.fetchone()[0])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the underlying SQLite connection. Idempotent."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "KVStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("KVStore is closed")
        return self._conn


# ----------------------------------------------------------------------
# Outcomes cache (§3.4)
# ----------------------------------------------------------------------
class OutcomesCache(KVStore):
    """Cache of LLM-generated outcome narratives.

    Key:  ``sha256(customer_id || asin || seed || prompt_version)``.
    Value: JSON-encoded ``{"outcomes": [...], "metadata": {...}}``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        table: str = "kv",
        create: bool = True,
    ) -> None:
        super().__init__(path, table=table, create=create)

    @staticmethod
    def outcomes_key(
        customer_id: str,
        asin: str,
        seed: int,
        prompt_version: str,
    ) -> str:
        """Return the hex SHA-256 digest for an outcome cache entry.

        Field order matches redesign.md §3.4:
        ``customer_id || asin || seed || prompt_version``.
        A null byte (``\\x00``) separates the fields so concatenations
        cannot collide across field boundaries.
        """
        payload = b"\x00".join(
            [
                str(customer_id).encode("utf-8"),
                str(asin).encode("utf-8"),
                str(int(seed)).encode("utf-8"),
                str(prompt_version).encode("utf-8"),
            ]
        )
        return hashlib.sha256(payload).hexdigest()

    def get_outcomes(
        self,
        customer_id: str,
        asin: str,
        seed: int,
        prompt_version: str,
    ) -> dict | None:
        """Return the cached ``{"outcomes", "metadata"}`` dict, or ``None``."""
        raw = self.get(self.outcomes_key(customer_id, asin, seed, prompt_version))
        if raw is None:
            return None
        return json.loads(raw.decode("utf-8"))

    def put_outcomes(
        self,
        customer_id: str,
        asin: str,
        seed: int,
        prompt_version: str,
        outcomes: list[str],
        metadata: dict[str, Any],
    ) -> None:
        """Write outcomes + metadata under the derived key (idempotent)."""
        key = self.outcomes_key(customer_id, asin, seed, prompt_version)
        payload = {"outcomes": list(outcomes), "metadata": dict(metadata)}
        self.put(key, json.dumps(payload).encode("utf-8"))


# ----------------------------------------------------------------------
# Embeddings cache (§4.3)
# ----------------------------------------------------------------------
class EmbeddingsCache(KVStore):
    """Cache of frozen-encoder outcome embeddings.

    Key:  ``sha256(outcome_string || encoder_id)``.
    Value: ``numpy.save``-encoded 1-D float32 array of shape ``(d_e,)``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        table: str = "kv",
        create: bool = True,
    ) -> None:
        super().__init__(path, table=table, create=create)

    @staticmethod
    def embedding_key(outcome_string: str, encoder_id: str) -> str:
        """Return the hex SHA-256 digest for an embedding cache entry."""
        payload = b"\x00".join(
            [
                str(outcome_string).encode("utf-8"),
                str(encoder_id).encode("utf-8"),
            ]
        )
        return hashlib.sha256(payload).hexdigest()

    def get_embedding(
        self,
        outcome_string: str,
        encoder_id: str,
    ) -> np.ndarray | None:
        """Return the cached float32 vector or ``None`` if absent."""
        raw = self.get(self.embedding_key(outcome_string, encoder_id))
        if raw is None:
            return None
        buf = io.BytesIO(raw)
        arr = np.load(buf, allow_pickle=False)
        # Canonicalise: stored arrays are always 1-D float32.
        return np.asarray(arr, dtype=np.float32)

    def put_embedding(
        self,
        outcome_string: str,
        encoder_id: str,
        vec: np.ndarray,
    ) -> None:
        """Write ``vec`` (1-D float32) under the derived key.

        Raises
        ------
        ValueError
            If ``vec`` is not a 1-D float32 ndarray.
        """
        if not isinstance(vec, np.ndarray):
            raise ValueError(
                f"embedding must be a numpy.ndarray, got {type(vec).__name__}"
            )
        if vec.ndim != 1:
            raise ValueError(
                f"embedding must be 1-D, got shape {tuple(vec.shape)}"
            )
        if vec.dtype != np.float32:
            raise ValueError(
                f"embedding must be float32, got dtype {vec.dtype}"
            )

        buf = io.BytesIO()
        np.save(buf, vec, allow_pickle=False)
        key = self.embedding_key(outcome_string, encoder_id)
        self.put(key, buf.getvalue())
