"""CLI commands: ``hermes noctuary <subcommand>``.

Registered by the memory-plugin CLI discovery when ``memory.provider`` is
``noctuary``. Kept import-light: heavy modules load inside handlers.
"""

from __future__ import annotations

import sys
from pathlib import Path


def register_cli(subparser) -> None:
    """Build the ``hermes noctuary`` argparse subcommand tree."""
    subs = subparser.add_subparsers(dest="noctuary_command")

    subs.add_parser("status", help="Show store, index, and consolidation status")

    ingest = subs.add_parser(
        "ingest", help="Bootstrap the graph from an existing session log"
    )
    ingest.add_argument("path", help="Session DB (.db/.sqlite), .json, .jsonl, .md, or .txt")
    ingest.add_argument("--format", default="auto",
                        choices=["auto", "hermes-db", "json", "jsonl", "text"])
    ingest.add_argument("--session", default=None,
                        help="Session id filter (hermes-db input)")
    ingest.add_argument("--date", default=None,
                        help="Fallback date YYYY-MM-DD for undated messages")
    ingest.add_argument("--no-consolidate", action="store_true",
                        help="Archive only; skip the librarian passes")
    ingest.add_argument("--max-days", type=int, default=None,
                        help="Consolidate at most N days this run")

    consolidate = subs.add_parser(
        "consolidate", help="Run the nightly librarian (all pending days)"
    )
    consolidate.add_argument("--date", default=None, help="Consolidate one specific day")
    consolidate.add_argument("--force", action="store_true",
                             help="Override the activity gate")

    recall = subs.add_parser("recall", help="Debug: print the passive recall packet")
    recall.add_argument("query", nargs="+")

    search = subs.add_parser("search", help="Debug: semantic search over the graph")
    search.add_argument("query", nargs="+")
    search.add_argument("--limit", type=int, default=8)

    show = subs.add_parser("show", help="Print one node")
    show.add_argument("node_id")

    verify = subs.add_parser("verify", help="Print the archived source for a ref")
    verify.add_argument("ref", help="Source ref YYYY-MM-DD/NNN")

    subs.add_parser("reindex", help="Rebuild the embedding index from the graph")


def noctuary_command(args) -> int:
    """Dispatch ``hermes noctuary ...``."""
    from .config import load_config
    from .store import NoctuaryStore

    cfg = load_config()
    store = NoctuaryStore(cfg.store_root)
    command = getattr(args, "noctuary_command", None) or "status"

    def echo(message: str) -> None:
        print(message)

    if command == "status":
        return _cmd_status(store, cfg)

    if command == "ingest":
        from .ingest import run_ingest
        path = Path(args.path).expanduser()
        if not path.is_file():
            print(f"error: no such file: {path}", file=sys.stderr)
            return 1
        run_ingest(
            store, cfg, path,
            fmt=args.format,
            session_id=args.session,
            default_date=args.date,
            consolidate_after=not args.no_consolidate,
            max_days=args.max_days,
            log=echo,
        )
        return 0

    if command == "consolidate":
        from .librarian import consolidate
        try:
            consolidate(store, cfg, date=args.date, force=args.force, log=echo)
        except Exception as exc:
            print(f"consolidation failed: {exc}", file=sys.stderr)
            print("no commit was made; the working tree in "
                  f"{store.root} holds the partial changes for inspection",
                  file=sys.stderr)
            return 1
        return 0

    if command in ("recall", "search"):
        from .recall import RecallEngine
        engine = RecallEngine(store, cfg)
        try:
            engine.reindex()
            query = " ".join(args.query)
            if command == "recall":
                packet = engine.build_packet(query)
                print(packet.text or "(no recognition — nothing surfaced)")
            else:
                matches = engine.search_nodes(query, limit=args.limit)
                if not matches:
                    print("(nothing surfaced)")
                for node, sim in matches:
                    print(f"{sim:.3f}  {node.id}  [{node.type}]  {node.title}")
        finally:
            engine.close()
        return 0

    if command == "show":
        node = store.load_node(args.node_id)
        if node is None:
            print(f"no node '{args.node_id}'", file=sys.stderr)
            return 1
        print(node.to_markdown())
        return 0

    if command == "verify":
        turn = store.get_source(args.ref)
        if turn is None:
            print(f"no archived turn '{args.ref}'", file=sys.stderr)
            return 1
        print(f"### turn {turn.ref} · {turn.time} · {turn.platform}")
        if turn.user:
            print(f"\nuser:\n{turn.user}")
        if turn.assistant:
            print(f"\nassistant:\n{turn.assistant}")
        return 0

    if command == "reindex":
        from .recall import RecallEngine
        engine = RecallEngine(store, cfg)
        try:
            updated, removed = engine.reindex()
            print(f"reindexed: {updated} updated, {removed} removed, "
                  f"{engine.index.count()} total")
        finally:
            engine.close()
        return 0

    print(f"unknown subcommand: {command}", file=sys.stderr)
    return 2


def _cmd_status(store, cfg) -> int:
    print(f"store:  {store.root}")
    print(f"git:    {'yes' if (store.root / '.git').exists() else 'no'}")
    days = store.source_days()
    print(f"source days: {len(days)}"
          + (f" ({days[0]} … {days[-1]})" if days else ""))
    consolidated = set(store.consolidated_days())
    pending = [d for d in days if d not in consolidated]
    print(f"pending consolidation: {len(pending)}"
          + (f" ({', '.join(pending[:5])}{'…' if len(pending) > 5 else ''})"
             if pending else ""))
    for node_type in ("episode", "concept", "pattern", "surface"):
        count = len(store.all_nodes(node_type))
        print(f"{node_type + ' nodes:':<16} {count}")
    index_path = store.root / "index.sqlite"
    print(f"index:  {'present' if index_path.exists() else 'missing'}")
    print(f"embedding model: {cfg.get_str('embeddingModel')}")
    librarian = cfg.get_str("librarianModel") or "(agent's model)"
    print(f"librarian model: {librarian}")
    return 0
