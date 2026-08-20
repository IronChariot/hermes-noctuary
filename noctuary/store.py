"""Noctuary graph store.

Plain Markdown files with YAML frontmatter inside a git repository under
``$HERMES_HOME/noctuary/``:

    noctuary/
      graph/
        episodes/<id>.md     — bounded events, always with source refs
        concepts/<id>.md     — wiki pages for recurring subjects
        patterns/<id>.md     — tentative interpretations (supports/contradicts)
        surface/<id>.md      — the small passive-recall layer
      sources/<YYYY-MM-DD>.md — append-only daily raw logs (never edited)
      index.sqlite            — derived embedding index (gitignored)
      state.json              — runtime state: retrieval log, consolidated days
      consolidation.log       — one line per librarian run (gitignored)

Invariants enforced here:
  * source logs are append-only — there is no API that rewrites them;
  * graph edits happen through :meth:`NoctuaryStore.save_node`, and only the
    librarian and backlog ingestion call it;
  * every change set becomes a git commit via :meth:`git_commit`.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

NODE_TYPES = ("episode", "concept", "pattern", "surface")
_TYPE_DIRS = {
    "episode": "episodes",
    "concept": "concepts",
    "pattern": "patterns",
    "surface": "surface",
}
LINK_KINDS = ("broader", "narrower", "related", "supports", "contradicts")

# Source-log markers. A collision would need user text that reproduces this
# exact shape at column 0, which is acceptable for a personal chat archive.
_TURN_RE = re.compile(r"^### turn (\d{4}-\d{2}-\d{2}/\d{3,}) · (\S+) · (\S+)\s*$")
_ROLE_RE = re.compile(r"^\*\*(user|assistant)\*\*:\s*$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_GITIGNORE = "index.sqlite\nstate.json\nconsolidation.log\nconfig.json\n*.lock\n"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def slugify(text: str, *, max_len: int = 48) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:max_len].strip("-") or "node"


@dataclass
class Node:
    """One graph node — a Markdown file with YAML frontmatter."""

    id: str
    type: str
    title: str = ""
    body: str = ""
    created: str = ""
    modified: str = ""
    last_retrieved: Optional[str] = None
    confidence: float = 0.5
    salience: float = 0.5
    accessibility: float = 0.5
    pinned: bool = False
    time: str = ""
    participants: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    links: Dict[str, List[str]] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def link_targets(self) -> List[str]:
        out: List[str] = []
        for kind in LINK_KINDS:
            out.extend(self.links.get(kind, []))
        return out

    def add_link(self, kind: str, target: str) -> None:
        if kind not in LINK_KINDS or not target or target == self.id:
            return
        bucket = self.links.setdefault(kind, [])
        if target not in bucket:
            bucket.append(target)

    def to_markdown(self) -> str:
        import yaml

        front: Dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "created": self.created,
            "modified": self.modified,
            "last_retrieved": self.last_retrieved,
            "confidence": round(float(self.confidence), 3),
            "salience": round(float(self.salience), 3),
            "accessibility": round(float(self.accessibility), 3),
            "pinned": bool(self.pinned),
        }
        if self.time:
            front["time"] = self.time
        if self.participants:
            front["participants"] = self.participants
        if self.topics:
            front["topics"] = self.topics
        front["links"] = {k: self.links.get(k, []) for k in LINK_KINDS}
        front["sources"] = self.sources
        for key, value in self.extra.items():
            front.setdefault(key, value)
        header = yaml.safe_dump(front, sort_keys=False, allow_unicode=True).strip()
        return f"---\n{header}\n---\n\n{self.body.strip()}\n"

    @classmethod
    def from_markdown(cls, text: str) -> "Node":
        import yaml

        front: Dict[str, Any] = {}
        body = text
        if text.startswith("---"):
            parts = text.split("\n---", 2)
            if len(parts) >= 2:
                raw = parts[0].lstrip("-\n")
                try:
                    front = yaml.safe_load(raw) or {}
                except Exception:
                    front = {}
                body = parts[1]
                if len(parts) == 3:
                    body = parts[1] + "\n---" + parts[2]
                # parts[1] starts right after the closing --- line
                body = body.lstrip("\n")
                if body.startswith("-"):
                    # yaml separator remnants; strip a leading dashes-only line
                    first, _, rest = body.partition("\n")
                    if set(first.strip()) <= {"-"}:
                        body = rest.lstrip("\n")
        links = front.get("links") or {}
        if not isinstance(links, dict):
            links = {}
        node = cls(
            id=str(front.get("id", "")),
            type=str(front.get("type", "concept")),
            title=str(front.get("title", "")),
            body=body.strip(),
            created=str(front.get("created", "")),
            modified=str(front.get("modified", "")),
            last_retrieved=front.get("last_retrieved"),
            confidence=float(front.get("confidence", 0.5) or 0.0),
            salience=float(front.get("salience", 0.5) or 0.0),
            accessibility=float(front.get("accessibility", 0.5) or 0.0),
            pinned=bool(front.get("pinned", False)),
            time=str(front.get("time", "") or ""),
            participants=list(front.get("participants") or []),
            topics=list(front.get("topics") or []),
            links={k: list(links.get(k) or []) for k in LINK_KINDS},
            sources=[str(s) for s in (front.get("sources") or [])],
        )
        known = {
            "id", "type", "title", "created", "modified", "last_retrieved",
            "confidence", "salience", "accessibility", "pinned", "time",
            "participants", "topics", "links", "sources",
        }
        node.extra = {k: v for k, v in front.items() if k not in known}
        return node

    def embed_text(self, *, max_chars: int = 2000) -> str:
        parts = [self.title]
        if self.topics:
            parts.append(", ".join(self.topics))
        parts.append(self.body[:max_chars])
        return "\n".join(p for p in parts if p)


@dataclass
class Turn:
    """One archived conversation turn (user + assistant) in a source log."""

    ref: str          # "YYYY-MM-DD/NNN"
    time: str         # "HH:MM:SS"
    platform: str
    user: str
    assistant: str

    @property
    def date(self) -> str:
        return self.ref.split("/", 1)[0]


class NoctuaryStore:
    """Filesystem + git backend for the Noctuary graph."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.graph_dir = self.root / "graph"
        self.sources_dir = self.root / "sources"
        self.state_path = self.root / "state.json"
        self.log_path = self.root / "consolidation.log"
        self._state_lock = threading.Lock()
        self._append_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    def init(self) -> None:
        """Create the directory layout and the git repository (idempotent)."""
        for sub in _TYPE_DIRS.values():
            (self.graph_dir / sub).mkdir(parents=True, exist_ok=True)
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        gitignore = self.root / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(_GITIGNORE, encoding="utf-8")
        if not (self.root / ".git").exists() and self.git_available():
            if self._git("init") is not None:
                self.git_commit("noctuary: initialize store")

    # -- nodes ---------------------------------------------------------------

    def node_path(self, node_id: str, node_type: str) -> Path:
        return self.graph_dir / _TYPE_DIRS[node_type] / f"{node_id}.md"

    def find_node_path(self, node_id: str) -> Optional[Path]:
        # Guard against path escapes from model-produced ids.
        safe = slugify(node_id, max_len=80)
        if safe != node_id:
            node_id = safe
        for sub in _TYPE_DIRS.values():
            path = self.graph_dir / sub / f"{node_id}.md"
            if path.is_file():
                return path
        return None

    def exists(self, node_id: str) -> bool:
        return self.find_node_path(node_id) is not None

    def load_node(self, node_id: str) -> Optional[Node]:
        path = self.find_node_path(node_id)
        if path is None:
            return None
        try:
            node = Node.from_markdown(path.read_text(encoding="utf-8"))
            if not node.id:
                node.id = path.stem
            return node
        except Exception as exc:
            logger.warning("noctuary: failed to parse node %s: %s", path, exc)
            return None

    def save_node(self, node: Node) -> None:
        if node.type not in _TYPE_DIRS:
            raise ValueError(f"unknown node type: {node.type}")
        node.id = slugify(node.id, max_len=80)
        if not node.created:
            node.created = now_iso()
        node.modified = now_iso()
        path = self.node_path(node.id, node.type)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(node.to_markdown(), encoding="utf-8")

    def all_nodes(self, node_type: Optional[str] = None) -> List[Node]:
        types = [node_type] if node_type else list(NODE_TYPES)
        out: List[Node] = []
        for t in types:
            directory = self.graph_dir / _TYPE_DIRS[t]
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.md")):
                try:
                    node = Node.from_markdown(path.read_text(encoding="utf-8"))
                    if not node.id:
                        node.id = path.stem
                    node.type = t
                    out.append(node)
                except Exception as exc:
                    logger.warning("noctuary: skipping unreadable node %s: %s", path, exc)
        return out

    def unique_id(self, base: str, prefix: str = "") -> str:
        candidate = slugify(f"{prefix}{base}", max_len=80)
        if not self.exists(candidate):
            return candidate
        for n in range(2, 1000):
            alt = f"{candidate}-{n}"
            if not self.exists(alt):
                return alt
        return f"{candidate}-{int(time.time())}"

    def backlinks(self, node_id: str) -> List[Node]:
        """Nodes whose typed links point at *node_id* (O(graph) scan)."""
        return [n for n in self.all_nodes() if node_id in n.link_targets()]

    # -- source records (append-only) ----------------------------------------

    def source_path(self, date: str) -> Path:
        return self.sources_dir / f"{date}.md"

    def _next_seq(self, date: str) -> int:
        path = self.source_path(date)
        if not path.is_file():
            return 1
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if _TURN_RE.match(line):
                count += 1
        return count + 1

    def append_turn(
        self,
        user_text: str,
        assistant_text: str,
        *,
        platform: str = "unknown",
        ts: Optional[float] = None,
    ) -> str:
        """Append one completed turn to the daily raw log. Returns the ref."""
        moment = datetime.fromtimestamp(ts).astimezone() if ts else datetime.now().astimezone()
        date = moment.strftime("%Y-%m-%d")
        with self._append_lock:
            seq = self._next_seq(date)
            ref = f"{date}/{seq:03d}"
            path = self.source_path(date)
            path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not path.exists()
            with open(path, "a", encoding="utf-8", newline="\n") as f:
                if new_file:
                    f.write(f"# Raw log — {date}\n\n")
                    f.write("<!-- append-only source record; never edit -->\n\n")
                f.write(f"### turn {ref} · {moment.strftime('%H:%M:%S')} · {platform}\n\n")
                if user_text:
                    f.write("**user**:\n")
                    f.write(user_text.rstrip() + "\n\n")
                if assistant_text:
                    f.write("**assistant**:\n")
                    f.write(assistant_text.rstrip() + "\n\n")
            return ref

    def read_turns(self, date: str) -> List[Turn]:
        path = self.source_path(date)
        if not path.is_file():
            return []
        turns: List[Turn] = []
        current: Optional[Turn] = None
        role: Optional[str] = None
        for line in path.read_text(encoding="utf-8").splitlines():
            turn_match = _TURN_RE.match(line)
            if turn_match:
                if current:
                    turns.append(current)
                current = Turn(
                    ref=turn_match.group(1),
                    time=turn_match.group(2),
                    platform=turn_match.group(3),
                    user="",
                    assistant="",
                )
                role = None
                continue
            if current is None:
                continue
            role_match = _ROLE_RE.match(line)
            if role_match:
                role = role_match.group(1)
                continue
            if role == "user":
                current.user += line + "\n"
            elif role == "assistant":
                current.assistant += line + "\n"
        if current:
            turns.append(current)
        for t in turns:
            t.user = t.user.strip()
            t.assistant = t.assistant.strip()
        return turns

    def get_source(self, ref: str) -> Optional[Turn]:
        if "/" not in ref:
            return None
        date = ref.split("/", 1)[0]
        if not _DATE_RE.match(date):
            return None
        for turn in self.read_turns(date):
            if turn.ref == ref:
                return turn
        return None

    def source_days(self) -> List[str]:
        if not self.sources_dir.is_dir():
            return []
        days = [p.stem for p in self.sources_dir.glob("*.md") if _DATE_RE.match(p.stem)]
        return sorted(days)

    def count_user_turns(self, date: str) -> int:
        return sum(1 for t in self.read_turns(date) if t.user.strip())

    # -- runtime state (untracked) -------------------------------------------

    def _read_state(self) -> Dict[str, Any]:
        try:
            if self.state_path.is_file():
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception as exc:
            logger.warning("noctuary: could not read state.json: %s", exc)
        return {}

    def _write_state(self, state: Dict[str, Any]) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("noctuary: could not write state.json: %s", exc)

    def record_retrieval(self, node_ids: Iterable[str]) -> None:
        """Log passive/deliberate retrievals; folded into frontmatter nightly."""
        ids = [i for i in node_ids if i]
        if not ids:
            return
        stamp = now_iso()
        with self._state_lock:
            state = self._read_state()
            retrievals = state.setdefault("retrievals", {})
            for node_id in ids:
                entry = retrievals.setdefault(node_id, {"count": 0, "last": ""})
                entry["count"] = int(entry.get("count", 0)) + 1
                entry["last"] = stamp
            self._write_state(state)

    def pop_retrievals(self) -> Dict[str, Dict[str, Any]]:
        with self._state_lock:
            state = self._read_state()
            retrievals = state.pop("retrievals", {}) or {}
            self._write_state(state)
        return retrievals

    def consolidated_days(self) -> List[str]:
        with self._state_lock:
            return list(self._read_state().get("consolidated_days", []))

    def mark_consolidated(self, date: str) -> None:
        with self._state_lock:
            state = self._read_state()
            days = state.setdefault("consolidated_days", [])
            if date not in days:
                days.append(date)
                days.sort()
            state["last_consolidation"] = now_iso()
            self._write_state(state)

    def log_run(self, line: str) -> None:
        try:
            with open(self.log_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(f"{now_iso()} {line}\n")
        except Exception as exc:
            logger.warning("noctuary: could not write consolidation.log: %s", exc)

    # -- git -----------------------------------------------------------------

    def git_available(self) -> bool:
        try:
            subprocess.run(
                ["git", "--version"], capture_output=True, timeout=10, check=True
            )
            return True
        except Exception:
            return False

    def _git(self, *args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "-c", "user.name=Noctuary", "-c", "user.email=noctuary@local",
                 *args],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.warning(
                    "noctuary: git %s failed: %s", " ".join(args),
                    (result.stderr or result.stdout).strip()[:500],
                )
                return None
            return result.stdout
        except Exception as exc:
            logger.warning("noctuary: git %s failed: %s", " ".join(args), exc)
            return None

    def git_commit(self, message: str) -> bool:
        """Stage everything and commit. Returns True when a commit was made."""
        if not (self.root / ".git").exists():
            return False
        if self._git("add", "-A") is None:
            return False
        status = self._git("status", "--porcelain")
        if not status or not status.strip():
            return False
        return self._git("commit", "-m", message) is not None
