"""Noctuary — a human-like long-term memory provider for Hermes Agent.

A noctuary is a night journal: the system records the day, and a nightly
librarian turns the record into memory. See noctuary-requirements.md in the
repository for the full design.

Live path (this module):
  * ``sync_turn`` appends completed turns to an append-only daily raw log
    (background thread; the Discord conversation only).
  * ``prefetch`` injects a small recall packet built from the surface layer
    plus pinned nodes — graded recall, never full RAG dumps.
  * Tools ``memory_recall`` / ``memory_verify`` / ``memory_search`` provide
    deliberate recollection and exact source verification.

The live path never edits the graph. All interpretation happens in the
nightly librarian (``hermes noctuary consolidate``).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider, RecallStatus, is_trivial_prompt

from .config import NoctuaryConfig, load_config, save_config_values
from .store import NoctuaryStore

logger = logging.getLogger(__name__)

GLYPH = "🌙"

_SYSTEM_PROMPT_BLOCK = """# Noctuary memory
You have graded, human-like long-term memory. On each turn a passive recall
packet may appear with vague context at one of these levels:
1. no recognition — nothing surfaced ("does not ring a bell"; NEVER "it did not happen")
2. familiarity — something related exists, details unavailable at that level
3. gist — enough context for a natural reply
Deliberate levels via tools:
4. memory_recall / memory_search — expand nodes, search episodes and concepts
5. memory_verify — read the exact archived source messages

Tool economy: answer from the gist when the gist is sufficient. Call memory
tools only when specificity matters, when you are uncertain, or when Sam
explicitly asks for accurate recall. Tool calls are expensive; minimise them.
Never state a detail with more certainty than the packet or tool result
carries — vague memories stay vague."""

_TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "memory_recall",
        "description": (
            "Deliberate recollection (level 4). Expand a memory node by id "
            "(from a recall packet) or by query: returns its full content plus "
            "neighbouring nodes (same theme, same period, linked episodes). "
            "Use when the gist is not enough."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Node id to expand."},
                "query": {"type": "string",
                          "description": "Alternative: expand the best-matching node."},
            },
        },
    },
    {
        "name": "memory_verify",
        "description": (
            "Source verification (level 5). Return the exact archived original "
            "messages behind a memory: pass a source_ref (e.g. '2026-08-19/014') "
            "or a node_id whose sources should be read. Use only when exact "
            "wording or exact facts matter."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_ref": {"type": "string",
                               "description": "Archived turn ref YYYY-MM-DD/NNN."},
                "node_id": {"type": "string",
                            "description": "Node whose source refs to read."},
            },
        },
    },
    {
        "name": "memory_search",
        "description": (
            "Free semantic search over episode, concept, and pattern nodes "
            "(below the surface layer). Use when passive recall surfaced "
            "nothing but something might exist deeper."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "description": "Max results (default 8)."},
            },
            "required": ["query"],
        },
    },
]


def _tool_error(message: str) -> str:
    return json.dumps({"error": message})


class _TurnLogger:
    """Background writer that appends completed turns to the daily raw log."""

    def __init__(self, store: NoctuaryStore, platform: str):
        self._store = store
        self._platform = platform
        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="noctuary-turnlog", daemon=True
        )
        self._thread.start()

    def enqueue(self, user_text: str, assistant_text: str) -> None:
        self._queue.put((user_text, assistant_text, time.time()))

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            user_text, assistant_text, ts = item
            try:
                self._store.append_turn(
                    user_text, assistant_text, platform=self._platform, ts=ts
                )
            except Exception as exc:
                logger.warning("noctuary: failed to archive turn: %s", exc)
            finally:
                self._queue.task_done()

    def flush(self, timeout: float = 10.0) -> None:
        """Block until queued turns are on disk (context boundaries)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            # unfinished_tasks also covers the turn a worker is writing
            # right now, which empty() would miss.
            if getattr(self._queue, "unfinished_tasks", 0) == 0:
                return
            time.sleep(0.05)

    def stop(self) -> None:
        self.flush()
        self._queue.put(None)
        self._thread.join(timeout=5.0)


class NoctuaryProvider(MemoryProvider):
    """Memory provider implementing the Noctuary design."""

    def __init__(self, config: Optional[NoctuaryConfig] = None):
        self._cfg = config
        self._store: Optional[NoctuaryStore] = None
        self._engine = None  # RecallEngine, created lazily
        self._turn_logger: Optional[_TurnLogger] = None
        self._ingest_enabled = False
        self._platform = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_cache: Optional[tuple] = None  # (query, packet)
        self._last_recall_count = 0

    # -- identification ------------------------------------------------------

    @property
    def name(self) -> str:
        return "noctuary"

    def is_available(self) -> bool:
        # Local-only: no credentials, no network. PyYAML ships with Hermes.
        return True

    # -- lifecycle -----------------------------------------------------------

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home")
        self._cfg = load_config(hermes_home)
        self._store = NoctuaryStore(self._cfg.store_root)
        self._store.init()

        self._platform = str(kwargs.get("platform", "") or "").lower()
        agent_context = str(kwargs.get("agent_context", "primary") or "primary")
        allowed = [p.lower() for p in self._cfg.get_list("ingestPlatforms")]
        # Ingestion gate: only the primary agent on an allowed platform
        # (the single ongoing Discord conversation) writes to the archive.
        # Subagents, cron jobs, and other platforms read but never write.
        self._ingest_enabled = (
            agent_context in ("primary", "") and self._platform in allowed
        )
        if self._ingest_enabled:
            self._turn_logger = _TurnLogger(self._store, self._platform)
        else:
            logger.info(
                "noctuary: read-only for context=%s platform=%s",
                agent_context, self._platform,
            )

        from .recall import RecallEngine
        self._engine = RecallEngine(self._store, self._cfg)
        threading.Thread(
            target=self._engine.warm, name="noctuary-warm", daemon=True
        ).start()

    def shutdown(self) -> None:
        if self._turn_logger is not None:
            self._turn_logger.stop()
            self._turn_logger = None
        if self._engine is not None:
            self._engine.close()
            self._engine = None

    # -- system prompt & recall ----------------------------------------------

    def system_prompt_block(self) -> str:
        if self._store is None:
            return ""
        return _SYSTEM_PROMPT_BLOCK

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._last_recall_count = 0
        if self._engine is None or is_trivial_prompt(query):
            return ""
        packet = None
        with self._prefetch_lock:
            cached = self._prefetch_cache
            if cached and cached[0] == query:
                packet = cached[1]
                self._prefetch_cache = None
        if packet is None:
            # Latency guard: a cold embedding model can take longer than the
            # prefetch ceiling to load. Skip recall for this turn instead of
            # blocking the reply; the background warm-up will have finished
            # by the next turn.
            if not self._engine.is_warm:
                logger.info("noctuary: embedder still warming, skipping recall")
                return ""
            try:
                packet = self._engine.build_packet(query)
            except Exception as exc:
                logger.warning("noctuary: prefetch failed: %s", exc)
                return ""
        self._last_recall_count = packet.count
        return packet.text

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._engine is None or is_trivial_prompt(query):
            return

        def _work():
            try:
                packet = self._engine.build_packet(query)
                with self._prefetch_lock:
                    self._prefetch_cache = (query, packet)
            except Exception as exc:
                logger.debug("noctuary: background prefetch failed: %s", exc)

        threading.Thread(target=_work, name="noctuary-prefetch", daemon=True).start()

    def recall_status(self) -> Optional[RecallStatus]:
        if self._last_recall_count <= 0:
            return None
        return RecallStatus(
            provider_label="noctuary", count=self._last_recall_count, glyph=GLYPH
        )

    # -- ingestion -----------------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if self._turn_logger is None:
            return
        if not (user_content or "").strip() and not (assistant_content or "").strip():
            return
        self._turn_logger.enqueue(user_content or "", assistant_content or "")

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        # Turns are archived as they complete; just make sure nothing queued
        # is lost at the context boundary. The raw log is the extraction —
        # interpretation waits for the nightly librarian.
        if self._turn_logger is not None:
            self._turn_logger.flush()
        return ""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if self._turn_logger is not None:
            self._turn_logger.flush()

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        if self._turn_logger is not None:
            self._turn_logger.flush()

    # -- tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return _TOOL_SCHEMAS

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if self._engine is None or self._store is None:
            return _tool_error("noctuary is not initialized")
        try:
            if tool_name == "memory_recall":
                return self._tool_recall(args)
            if tool_name == "memory_verify":
                return self._tool_verify(args)
            if tool_name == "memory_search":
                return self._tool_search(args)
        except Exception as exc:
            logger.warning("noctuary: tool %s failed: %s", tool_name, exc)
            return _tool_error(str(exc))
        return _tool_error(f"unknown tool: {tool_name}")

    def _tool_recall(self, args: Dict[str, Any]) -> str:
        node_id = str(args.get("node_id") or "").strip()
        query = str(args.get("query") or "").strip()
        if not node_id and not query:
            return _tool_error("memory_recall needs node_id or query")
        if not node_id:
            matches = self._engine.search_nodes(query, limit=1)
            if not matches:
                return json.dumps({
                    "found": False,
                    "note": "nothing surfaced for this query — that means no "
                            "recognition, not that the event never happened",
                })
            node_id = matches[0][0].id
        expanded = self._engine.expand_node(node_id)
        if expanded is None:
            return json.dumps({
                "found": False,
                "note": f"no node '{node_id}' — nothing surfaced under that id",
            })
        expanded["found"] = True
        return json.dumps(expanded, ensure_ascii=False)

    def _tool_verify(self, args: Dict[str, Any]) -> str:
        source_ref = str(args.get("source_ref") or "").strip()
        node_id = str(args.get("node_id") or "").strip()
        refs: List[str] = []
        if source_ref:
            refs = [source_ref]
        elif node_id:
            node = self._store.load_node(node_id)
            if node is None:
                return _tool_error(f"no node '{node_id}'")
            if not node.sources:
                return json.dumps({
                    "found": False,
                    "note": f"node '{node_id}' has no source refs — it is "
                            "interpretation, not directly archived fact",
                })
            refs = node.sources[:6]
        else:
            return _tool_error("memory_verify needs source_ref or node_id")

        out = []
        for ref in refs:
            turn = self._store.get_source(ref)
            if turn is None:
                out.append({"ref": ref, "found": False})
            else:
                out.append({
                    "ref": ref,
                    "found": True,
                    "time": f"{turn.date} {turn.time}",
                    "platform": turn.platform,
                    "user": turn.user[:4000],
                    "assistant": turn.assistant[:4000],
                })
        return json.dumps(
            {"sources": out, "note": "exact archived text; the authority for "
                                     "exact recall"},
            ensure_ascii=False,
        )

    def _tool_search(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return _tool_error("memory_search needs a query")
        limit = min(20, max(1, int(args.get("limit", 8) or 8)))
        matches = self._engine.search_nodes(query, limit=limit)
        if not matches:
            return json.dumps({
                "results": [],
                "note": "nothing surfaced — no recognition, not proof of absence",
            })
        self._store.record_retrieval([n.id for n, _ in matches])
        results = [
            {
                "id": node.id,
                "type": node.type,
                "title": node.title,
                "similarity": round(sim, 3),
                "confidence": node.confidence,
                "time": node.time,
                "summary": node.body[:300],
            }
            for node, sim in matches
        ]
        return json.dumps({"results": results}, ensure_ascii=False)

    # -- config --------------------------------------------------------------

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "recallTokenBudget",
                "description": "Token budget for the passive recall packet",
                "type": "integer", "default": 1000, "minimum": 100, "maximum": 8000,
            },
            {
                "key": "embeddingModel",
                "description": "Local embedding model (auto, hash, or a "
                               "fastembed/sentence-transformers model name)",
                "default": "auto",
            },
            {
                "key": "librarianModel",
                "description": "Model for nightly consolidation (empty = agent's model)",
                "default": "",
            },
            {
                "key": "librarianProvider",
                "description": "Provider for the librarian model (empty = auto)",
                "default": "",
            },
            {
                "key": "surfacePageLimit",
                "description": "Target size of the surface (passive recall) layer",
                "type": "integer", "default": 12, "minimum": 3, "maximum": 50,
            },
            {
                "key": "minUserTurns",
                "description": "Activity gate: days with fewer user turns are skipped",
                "type": "integer", "default": 1, "minimum": 1,
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        save_config_values(values, hermes_home)


def register(ctx) -> None:
    """Register the Noctuary memory provider with the plugin system."""
    ctx.register_memory_provider(NoctuaryProvider())
