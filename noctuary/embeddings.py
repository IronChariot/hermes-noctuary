"""Local embeddings and the derived vector index.

Embedder resolution for ``embeddingModel``:

  * ``"hash"``  — built-in character n-gram hashing embedder. No dependencies,
    deterministic, mediocre quality. Always available; also the fallback.
  * ``"auto"``  — try fastembed's default small model, then
    sentence-transformers ``all-MiniLM-L6-v2``, then hash.
  * anything else — treated as a model name for fastembed, then
    sentence-transformers, then hash (with a warning).

The index is derived data (SQLite, float32 blobs, brute-force cosine): it can
always be rebuilt from the Markdown files with ``hermes noctuary reindex``.
"""

from __future__ import annotations

import array
import hashlib
import logging
import math
import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_ST_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"
_FASTEMBED_DEFAULT = "BAAI/bge-small-en-v1.5"


class HashEmbedder:
    """Deterministic character n-gram hashing embedder (zero dependencies)."""

    name = "hash"
    dim = 512

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            lowered = " " + (text or "").lower() + " "
            for n in (3, 4, 5):
                for i in range(max(0, len(lowered) - n + 1)):
                    gram = lowered[i:i + n]
                    h = int.from_bytes(
                        hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(),
                        "little",
                    )
                    idx = h % self.dim
                    sign = 1.0 if (h >> 32) & 1 else -1.0
                    vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class _FastembedEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding  # noqa: import guarded by caller

        self.name = f"fastembed:{model_name}"
        self._model = TextEmbedding(model_name=model_name)
        probe = list(self._model.embed(["probe"]))[0]
        self.dim = len(probe)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [_normalized(list(map(float, v))) for v in self._model.embed(list(texts))]


class _SentenceTransformersEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # noqa: guarded

        self.name = f"st:{model_name}"
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]


def _normalized(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def get_embedder(model_setting: str = "auto"):
    """Resolve the configured embedder, falling back to the hash embedder."""
    setting = (model_setting or "auto").strip()
    if setting == "hash":
        return HashEmbedder()

    candidates: List[Tuple[str, str]] = []
    if setting == "auto":
        candidates = [("fastembed", _FASTEMBED_DEFAULT), ("st", _ST_DEFAULT)]
    else:
        candidates = [("fastembed", setting), ("st", setting)]

    for backend, model_name in candidates:
        try:
            if backend == "fastembed":
                return _FastembedEmbedder(model_name)
            return _SentenceTransformersEmbedder(model_name)
        except Exception as exc:
            logger.debug("noctuary: embedder %s/%s unavailable: %s",
                         backend, model_name, exc)

    if setting != "auto":
        logger.warning(
            "noctuary: embedding model %r not available; using hash fallback",
            setting,
        )
    return HashEmbedder()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class EmbeddingIndex:
    """SQLite-backed vector index over graph nodes (derived, rebuildable)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS nodes (
                 node_id TEXT PRIMARY KEY,
                 node_type TEXT NOT NULL,
                 pinned INTEGER NOT NULL DEFAULT 0,
                 text_hash TEXT NOT NULL,
                 dim INTEGER NOT NULL,
                 vec BLOB NOT NULL,
                 updated REAL NOT NULL
               )"""
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    # -- embedder signature guard --------------------------------------------

    def embedder_signature(self) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key='embedder'"
            ).fetchone()
        return row[0] if row else ""

    def reset_for_embedder(self, signature: str) -> None:
        """Drop all vectors when the embedder changed."""
        with self._lock:
            self._conn.execute("DELETE FROM nodes")
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('embedder', ?)",
                (signature,),
            )
            self._conn.commit()

    # -- rows ----------------------------------------------------------------

    def get_hash(self, node_id: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT text_hash FROM nodes WHERE node_id=?", (node_id,)
            ).fetchone()
        return row[0] if row else None

    def upsert(
        self,
        node_id: str,
        node_type: str,
        pinned: bool,
        content_hash: str,
        vec: Sequence[float],
    ) -> None:
        blob = array.array("f", vec).tobytes()
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO nodes "
                "(node_id, node_type, pinned, text_hash, dim, vec, updated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (node_id, node_type, 1 if pinned else 0, content_hash,
                 len(vec), blob, time.time()),
            )
            self._conn.commit()

    def remove_absent(self, keep_ids: Iterable[str]) -> int:
        keep = set(keep_ids)
        with self._lock:
            rows = self._conn.execute("SELECT node_id FROM nodes").fetchall()
            stale = [r[0] for r in rows if r[0] not in keep]
            for node_id in stale:
                self._conn.execute("DELETE FROM nodes WHERE node_id=?", (node_id,))
            self._conn.commit()
        return len(stale)

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    # -- search --------------------------------------------------------------

    def search(
        self,
        query_vec: Sequence[float],
        *,
        node_types: Optional[Sequence[str]] = None,
        include_pinned: bool = False,
        top_k: int = 10,
    ) -> List[Tuple[str, str, float]]:
        """Return ``(node_id, node_type, cosine)`` ranked best-first.

        ``node_types`` restricts the candidate set; ``include_pinned=True``
        adds pinned nodes of ANY type on top of that restriction (the
        passive-recall "surface + key pinned nodes" rule).
        """
        with self._lock:
            if node_types is None:
                rows = self._conn.execute(
                    "SELECT node_id, node_type, dim, vec FROM nodes"
                ).fetchall()
            else:
                marks = ",".join("?" for _ in node_types)
                clause = f"node_type IN ({marks})"
                params: List = list(node_types)
                if include_pinned:
                    clause += " OR pinned=1"
                rows = self._conn.execute(
                    f"SELECT node_id, node_type, dim, vec FROM nodes WHERE {clause}",
                    params,
                ).fetchall()

        q = list(query_vec)
        scored: List[Tuple[str, str, float]] = []
        try:
            import numpy as np

            qv = np.asarray(q, dtype="f4")
            for node_id, node_type, dim, blob in rows:
                if dim != len(q):
                    continue
                v = np.frombuffer(blob, dtype="f4")
                scored.append((node_id, node_type, float(qv.dot(v))))
        except ImportError:
            for node_id, node_type, dim, blob in rows:
                if dim != len(q):
                    continue
                v = array.array("f")
                v.frombytes(blob)
                scored.append(
                    (node_id, node_type, sum(a * b for a, b in zip(q, v)))
                )
        scored.sort(key=lambda item: item[2], reverse=True)
        return scored[:top_k]


def reindex(store, index: EmbeddingIndex, embedder) -> Tuple[int, int]:
    """Bring the index up to date with the graph. Returns (updated, removed)."""
    signature = f"{embedder.name}:{embedder.dim}"
    if index.embedder_signature() != signature:
        index.reset_for_embedder(signature)

    nodes = store.all_nodes()
    updated = 0
    pending: List = []
    for node in nodes:
        content_hash = text_hash(node.embed_text() + ("|p" if node.pinned else ""))
        if index.get_hash(node.id) != content_hash:
            pending.append((node, content_hash))
    if pending:
        vectors = embedder.embed([n.embed_text() for n, _ in pending])
        for (node, content_hash), vec in zip(pending, vectors):
            index.upsert(node.id, node.type, node.pinned, content_hash, vec)
            updated += 1
    removed = index.remove_absent([n.id for n in nodes])
    return updated, removed
