"""Backlog bootstrap — ``hermes noctuary ingest <path>``.

Ingests an existing long-running session log into daily source records, then
(optionally) runs the librarian day by day to build the initial graph:
episodes, first concept pages, first surface pages.

Supported inputs:

  * ``.db`` / ``.sqlite`` — a Hermes session database (the ``messages``
    table: session_id, role, content, timestamp). ``--session`` filters to
    one session id.
  * ``.json``  — a message list, or ``{"messages": [...]}``, each message
    ``{"role": ..., "content": ..., "timestamp"?: ...}``.
  * ``.jsonl`` — one such message object per line.
  * ``.md`` / ``.txt`` — plain transcript; lines starting with markers like
    ``User:`` / ``Sam:`` / ``Assistant:`` / ``Wren:`` switch speaker.

Messages without timestamps land on ``--date`` (default: the file's
modification date). Ingested turns are appended through the same append-only
source-record path as live capture.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .config import NoctuaryConfig
from .store import NoctuaryStore

logger = logging.getLogger(__name__)

_USER_MARKERS = re.compile(r"^\s*(?:\*\*)?(user|sam|human)(?:\*\*)?\s*[:>]", re.IGNORECASE)
_ASSISTANT_MARKERS = re.compile(
    r"^\s*(?:\*\*)?(assistant|wren|maomao|agent|ai|hermes)(?:\*\*)?\s*[:>]", re.IGNORECASE
)


@dataclass
class RawMessage:
    role: str            # "user" | "assistant"
    text: str
    ts: Optional[float]  # unix seconds, None when unknown


def parse_input(
    path: Path,
    *,
    fmt: str = "auto",
    session_id: Optional[str] = None,
) -> List[RawMessage]:
    suffix = path.suffix.lower()
    if fmt == "auto":
        if suffix in (".db", ".sqlite", ".sqlite3"):
            fmt = "hermes-db"
        elif suffix == ".jsonl":
            fmt = "jsonl"
        elif suffix == ".json":
            fmt = "json"
        else:
            fmt = "text"

    if fmt == "hermes-db":
        return _parse_hermes_db(path, session_id)
    if fmt == "json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("messages") or []
        return _messages_from_objects(data)
    if fmt == "jsonl":
        objects = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                objects.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("noctuary ingest: skipping bad JSONL line")
        return _messages_from_objects(objects)
    return _parse_text(path.read_text(encoding="utf-8"))


def _parse_hermes_db(path: Path, session_id: Optional[str]) -> List[RawMessage]:
    import sqlite3

    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        query = (
            "SELECT role, content, timestamp FROM messages "
            "WHERE role IN ('user', 'assistant')"
        )
        params: List = []
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY timestamp, id"
        out: List[RawMessage] = []
        for role, content, ts in conn.execute(query, params):
            text = _content_to_text(content)
            if text.strip():
                out.append(RawMessage(role=role, text=text,
                                      ts=float(ts) if ts else None))
        return out
    finally:
        conn.close()


def _content_to_text(content) -> str:
    if isinstance(content, str):
        # Multi-part content may arrive JSON-encoded.
        if content.startswith("[") or content.startswith("{"):
            try:
                return _content_to_text(json.loads(content))
            except (json.JSONDecodeError, RecursionError):
                return content
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return "" if content is None else str(content)


def _messages_from_objects(objects: List) -> List[RawMessage]:
    out: List[RawMessage] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        role = str(obj.get("role", "")).lower()
        if role not in ("user", "assistant"):
            continue
        text = _content_to_text(obj.get("content"))
        if not text.strip():
            continue
        ts = obj.get("timestamp") or obj.get("ts") or obj.get("created_at")
        ts_value: Optional[float] = None
        if isinstance(ts, (int, float)):
            ts_value = float(ts)
        elif isinstance(ts, str):
            try:
                ts_value = datetime.fromisoformat(ts).timestamp()
            except ValueError:
                ts_value = None
        out.append(RawMessage(role=role, text=text, ts=ts_value))
    return out


def _parse_text(text: str) -> List[RawMessage]:
    messages: List[RawMessage] = []
    role: Optional[str] = None
    buffer: List[str] = []

    def flush():
        nonlocal buffer
        if role and buffer:
            content = "\n".join(buffer).strip()
            if content:
                messages.append(RawMessage(role=role, text=content, ts=None))
        buffer = []

    for line in text.splitlines():
        if _USER_MARKERS.match(line):
            flush()
            role = "user"
            buffer = [_USER_MARKERS.sub("", line, count=1).strip()]
        elif _ASSISTANT_MARKERS.match(line):
            flush()
            role = "assistant"
            buffer = [_ASSISTANT_MARKERS.sub("", line, count=1).strip()]
        elif role:
            buffer.append(line)
    flush()

    if not messages and text.strip():
        # No speaker markers at all: archive the whole file as one user block
        # so the evidence is at least preserved.
        messages.append(RawMessage(role="user", text=text.strip(), ts=None))
    return messages


def archive_messages(
    store: NoctuaryStore,
    messages: List[RawMessage],
    *,
    default_date: Optional[str] = None,
    platform: str = "backlog",
) -> Dict[str, int]:
    """Pair user/assistant messages into turns and append them to source logs.

    Returns {date: turn_count}. Consecutive same-role messages are joined;
    an assistant message without a preceding user message becomes a turn
    with an empty user side.
    """
    if default_date:
        base = datetime.strptime(default_date, "%Y-%m-%d")
        default_ts = base.replace(hour=12).timestamp()
    else:
        default_ts = datetime.now().timestamp()

    counts: Dict[str, int] = {}
    pending_user: List[str] = []
    pending_ts: Optional[float] = None

    def emit(user_text: str, assistant_text: str, ts: Optional[float]):
        stamp = ts if ts else default_ts
        ref = store.append_turn(
            user_text, assistant_text, platform=platform, ts=stamp
        )
        date = ref.split("/", 1)[0]
        counts[date] = counts.get(date, 0) + 1

    for msg in messages:
        if msg.role == "user":
            pending_user.append(msg.text)
            pending_ts = pending_ts or msg.ts
        else:
            emit("\n\n".join(pending_user), msg.text, msg.ts or pending_ts)
            pending_user = []
            pending_ts = None
    if pending_user:
        emit("\n\n".join(pending_user), "", pending_ts)
    return counts


def run_ingest(
    store: NoctuaryStore,
    cfg: NoctuaryConfig,
    path: Path,
    *,
    fmt: str = "auto",
    session_id: Optional[str] = None,
    default_date: Optional[str] = None,
    consolidate_after: bool = True,
    max_days: Optional[int] = None,
    log: Callable[[str], None] = logger.info,
) -> Dict[str, int]:
    """Archive a backlog file and (optionally) consolidate the resulting days."""
    store.init()
    messages = parse_input(path, fmt=fmt, session_id=session_id)
    if not messages:
        log(f"noctuary ingest: no messages found in {path}")
        return {}

    if default_date is None and all(m.ts is None for m in messages):
        default_date = datetime.fromtimestamp(
            path.stat().st_mtime
        ).strftime("%Y-%m-%d")

    counts = archive_messages(store, messages, default_date=default_date)
    total = sum(counts.values())
    log(f"noctuary ingest: archived {total} turns across {len(counts)} days")
    store.git_commit(
        f"noctuary: ingest backlog {path.name} "
        f"({total} turns, {len(counts)} days)"
    )

    if consolidate_after:
        from .librarian import consolidate_day, pending_days
        days = pending_days(store)
        if max_days:
            days = days[:max_days]
        for day in days:
            consolidate_day(store, cfg, day, log=log)
    return counts
