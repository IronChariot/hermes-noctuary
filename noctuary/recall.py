"""Passive recall and deliberate retrieval.

Passive recall (every non-trivial turn) searches ONLY the surface layer plus
pinned nodes, and produces a technical "recall packet": per hit a snippet or
short description, provenance, confidence, recall level, and the node id for
drill-down. Levels (requirements section 5):

  1. no recognition — nothing surfaced ("does not ring a bell")
  2. familiarity    — something related exists; details not at this level
  3. gist           — enough context for a natural reply

Levels 4–5 (deliberate recollection, source verification) are reached only
through the ``memory_recall`` / ``memory_verify`` / ``memory_search`` tools.

The live path never edits the graph: retrievals are logged to state.json and
folded into node frontmatter by the nightly librarian.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from .config import NoctuaryConfig
from .embeddings import EmbeddingIndex, get_embedder
from .store import Node, NoctuaryStore

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4  # rough budget estimate; deliberately conservative

PACKET_HEADER = (
    "## Noctuary passive recall\n"
    "Technical packet (levels: familiarity < gist; confidence 0..1). "
    "Drill down with memory_recall(node_id) or memory_verify. "
    "Nothing here means nothing surfaced — never that an event did not happen.\n"
)


@dataclass
class RecallHit:
    node: Node
    similarity: float
    score: float
    level: str  # "familiarity" | "gist"


@dataclass
class RecallPacket:
    text: str
    node_ids: List[str]

    @property
    def count(self) -> int:
        return len(self.node_ids)


class RecallEngine:
    """Embedding search + ranking over the graph, shared by prefetch and tools."""

    def __init__(self, store: NoctuaryStore, cfg: NoctuaryConfig):
        self.store = store
        self.cfg = cfg
        self.index = EmbeddingIndex(store.root / "index.sqlite")
        self._embedder = None
        self._embedder_lock = threading.Lock()

    def close(self) -> None:
        self.index.close()

    # -- embedder ------------------------------------------------------------

    @property
    def embedder(self):
        with self._embedder_lock:
            if self._embedder is None:
                self._embedder = get_embedder(self.cfg.get_str("embeddingModel"))
            return self._embedder

    @property
    def is_warm(self) -> bool:
        """True once the embedder is loaded (no lock; racy reads are fine)."""
        return self._embedder is not None

    def warm(self) -> None:
        """Load the embedding model ahead of the first prefetch."""
        try:
            _ = self.embedder
        except Exception as exc:
            logger.warning("noctuary: embedder warm-up failed: %s", exc)

    def reindex(self) -> Tuple[int, int]:
        from .embeddings import reindex
        return reindex(self.store, self.index, self.embedder)

    # -- passive recall ------------------------------------------------------

    def build_packet(self, query: str) -> RecallPacket:
        """Search surface + pinned nodes, rank, format under the token budget."""
        query = (query or "").strip()[:1500]
        if not query:
            return RecallPacket("", [])

        query_vec = self.embedder.embed([query])[0]
        raw = self.index.search(
            query_vec,
            node_types=["surface"],
            include_pinned=True,
            top_k=max(12, self.cfg.get_int("maxRecallEntries") * 2),
        )
        hits = self._rank(raw)
        if not hits:
            return RecallPacket("", [])

        packet = self._format_packet(hits)
        self.store.record_retrieval([h.node.id for h in hits])
        return packet

    def _rank(self, raw: Sequence[Tuple[str, str, float]]) -> List[RecallHit]:
        familiar_floor = self.cfg.get_float("familiaritySimilarity")
        gist_floor = self.cfg.get_float("gistSimilarity")
        max_entries = self.cfg.get_int("maxRecallEntries")

        hits: List[RecallHit] = []
        for node_id, _node_type, sim in raw:
            if sim < familiar_floor:
                continue
            node = self.store.load_node(node_id)
            if node is None:
                continue
            score = (
                0.55 * sim
                + 0.15 * node.salience
                + 0.15 * node.accessibility
                + 0.10 * _recency_weight(node.last_retrieved or node.modified)
                + 0.05 * min(1.0, len(node.link_targets()) / 8.0)
            )
            level = "gist" if sim >= gist_floor else "familiarity"
            hits.append(RecallHit(node=node, similarity=sim, score=score, level=level))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:max_entries]

    def _format_packet(self, hits: List[RecallHit]) -> RecallPacket:
        budget_chars = self.cfg.get_int("recallTokenBudget") * _CHARS_PER_TOKEN
        lines: List[str] = [PACKET_HEADER]
        used = len(PACKET_HEADER)
        node_ids: List[str] = []

        for hit in hits:
            node = hit.node
            provenance = node.time or (node.created or "")[:10] or "undated"
            if node.topics:
                provenance += " · " + ", ".join(node.topics[:4])
            if hit.level == "gist":
                snippet = _one_paragraph(node.body, 380)
                line = (
                    f"- [gist | conf {node.confidence:.2f} | node {node.id} "
                    f"({node.type})] {node.title or node.id} — {snippet} "
                    f"({provenance})"
                )
            else:
                line = (
                    f"- [familiarity | conf {node.confidence:.2f} | node {node.id} "
                    f"({node.type})] something related: {node.title or node.id} "
                    f"({provenance}); details not available at this level — "
                    f"use memory_recall if it matters"
                )
            if used + len(line) + 1 > budget_chars and node_ids:
                break
            lines.append(line)
            used += len(line) + 1
            node_ids.append(node.id)

        if not node_ids:
            return RecallPacket("", [])
        return RecallPacket("\n".join(lines), node_ids)

    # -- deliberate retrieval (tools) ----------------------------------------

    def search_nodes(
        self,
        query: str,
        *,
        node_types: Optional[Sequence[str]] = None,
        limit: int = 8,
    ) -> List[Tuple[Node, float]]:
        """Free semantic search over the graph (default: below-surface layers)."""
        if node_types is None:
            node_types = ["episode", "concept", "pattern"]
        query_vec = self.embedder.embed([(query or "")[:1500]])[0]
        raw = self.index.search(query_vec, node_types=list(node_types), top_k=limit)
        out: List[Tuple[Node, float]] = []
        for node_id, _t, sim in raw:
            node = self.store.load_node(node_id)
            if node is not None:
                out.append((node, sim))
        return out

    def expand_node(self, node_id: str) -> Optional[Dict]:
        """Node content plus graph neighbours, for ``memory_recall``."""
        node = self.store.load_node(node_id)
        if node is None:
            return None

        neighbours: List[Dict] = []
        seen = {node.id}
        for kind in ("broader", "narrower", "related", "supports", "contradicts"):
            for target in node.links.get(kind, []):
                if target in seen:
                    continue
                seen.add(target)
                linked = self.store.load_node(target)
                if linked is None:
                    neighbours.append({"id": target, "link": kind, "missing": True})
                else:
                    neighbours.append(_node_brief(linked, link=kind))
        for back in self.store.backlinks(node.id):
            if back.id in seen:
                continue
            seen.add(back.id)
            neighbours.append(_node_brief(back, link="backlink"))

        self.store.record_retrieval([node.id])
        return {
            "id": node.id,
            "type": node.type,
            "title": node.title,
            "confidence": node.confidence,
            "salience": node.salience,
            "accessibility": node.accessibility,
            "time": node.time,
            "topics": node.topics,
            "participants": node.participants,
            "sources": node.sources,
            "content": node.body[:6000],
            "neighbours": neighbours[:24],
        }


def _node_brief(node: Node, *, link: str) -> Dict:
    return {
        "id": node.id,
        "link": link,
        "type": node.type,
        "title": node.title,
        "confidence": node.confidence,
        "summary": _one_paragraph(node.body, 200),
    }


def _one_paragraph(text: str, max_chars: int) -> str:
    para = (text or "").strip().split("\n\n", 1)[0].replace("\n", " ").strip()
    if len(para) > max_chars:
        para = para[: max_chars - 1].rstrip() + "…"
    return para


def _recency_weight(stamp: Optional[str]) -> float:
    """1.0 for today, decaying toward 0 over ~90 days. Bad stamps score 0."""
    if not stamp:
        return 0.0
    try:
        then = datetime.fromisoformat(str(stamp))
        now = datetime.now(then.tzinfo) if then.tzinfo else datetime.now()
        days = max(0.0, (now - then).total_seconds() / 86400.0)
        return max(0.0, 1.0 - days / 90.0)
    except (ValueError, TypeError):
        return 0.0
