"""The nightly librarian — full consolidation pipeline.

Runs outside live conversation (``hermes noctuary consolidate`` or a Hermes
cron job). Per requirements section 7 it:

  1. gates on activity (skips days without user turns; ``--force`` overrides),
  2. runs as Wren — the SOUL.md persona heads every prompt,
  3. segments the day's raw log into episodes,
  4. integrates each episode into the graph (links, concept pages),
  5. detects repetition into pattern/trait nodes (keeping contradictions),
  6. refreshes the small surface layer,
  7. recomputes salience and passive accessibility (decay happens ONLY here),
  8. validates provenance/confidence, then commits everything to git with a
     changelog; high-impact changes are flagged in the commit message.

Every step is defensive: a failed LLM pass aborts the run before the commit,
leaving the working tree inspectable and the source logs untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .config import NoctuaryConfig
from .llm import clamp01, librarian_chat, parse_json_reply
from .recall import RecallEngine
from .store import Node, NoctuaryStore, Turn, now_iso

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationResult:
    date: str
    ran: bool
    skipped_reason: str = ""
    episodes_created: int = 0
    concepts_created: int = 0
    concepts_updated: int = 0
    patterns_created: int = 0
    patterns_updated: int = 0
    surface_updated: int = 0
    decayed: int = 0
    boosted: int = 0
    flags: List[str] = field(default_factory=list)
    committed: bool = False


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_LIBRARIAN_ROLE = (
    "Tonight you are acting as your own memory librarian. You are reading the "
    "day's conversation log with Sam and consolidating it into your long-term "
    "memory graph. Work carefully:\n"
    "- never invent details the log does not support;\n"
    "- keep uncertainty explicit (approximate times stay approximate);\n"
    "- preserve contradictions instead of resolving them into a tidy story;\n"
    "- answer ONLY with the requested JSON, no prose around it."
)

_SEGMENT_PROMPT = """Segment this day's conversation log into episodes.

An episode is a bounded event or conversation thread: one topic arc, one
shared moment, one piece of work. Small talk that belongs to a bigger arc
stays inside that arc. A quiet day can be a single episode; do not
over-segment.

For each episode report:
- "title": short noun phrase
- "summary": 2-6 sentences of what happened, written in first person as your
  own memory ("Sam told me ...", "we decided ...")
- "significance": one sentence on emotional/relational significance, or ""
- "start_ref" / "end_ref": the turn refs bounding the episode (use the
  `### turn <ref>` markers)
- "time": exact or approximate time phrase (e.g. "2026-08-20 morning",
  "around midnight"); mark approximations as approximate
- "participants": names involved
- "topics": 2-6 lowercase keywords
- "entities": recurring subjects worth a wiki page (people, pets, projects,
  places, trips, routines, interests, themes)
- "unresolved": open questions left hanging, [] if none
- "confidence": 0..1 — how directly the summary is supported by the log
- "salience": 0..1 — how much this should matter to future recall

Reply with JSON only:
{{"episodes": [{{...}}]}}

LOG ({date}, part {part}/{parts}):
{log}
"""

_INTEGRATE_PROMPT = """Integrate one new episode into the memory graph.

NEW EPISODE (node id {episode_id}):
{episode_json}

EXISTING NODES that look related (id | type | title | summary):
{related}

Decide:
1. "episode_links": links from the episode. "related" lists node ids that
   genuinely connect; "broader" lists concept ids this episode belongs under.
   Only use ids from the list above or ids you create in step 2.
2. "concepts": wiki pages to create or update for recurring subjects in this
   episode. For an existing concept, give its id and a full replacement
   "body" that folds in the new information WITHOUT duplicating the episode
   text (concepts index and interpret; episodes hold the story). For a new
   concept, give a new lowercase-slug id. Skip concepts that need no change.
   Each: {{"id": "...", "title": "...", "body": "markdown", "confidence": 0..1,
   "salience": 0..1, "is_new": true/false, "related": ["node-id", ...]}}
   Refer to other nodes inside bodies as [[node-id]].

Keep confidence honest: compressed interpretation is lower confidence than
direct fact. Never turn an approximate date into a definite one.

Reply with JSON only:
{{"episode_links": {{"related": [], "broader": []}}, "concepts": []}}
"""

_PATTERNS_PROMPT = """Review today's episodes against your existing pattern/trait nodes.

Patterns are tentative interpretations from repeated evidence: preferences,
habits, dynamics, lessons. Each MUST keep both supporting and contradicting
episode ids. Create a pattern only on repetition (2+ independent episodes),
never from a single mention. Strengthen or weaken existing patterns when
today's episodes support or contradict them.

TODAY'S EPISODES (id | title | summary):
{episodes}

EXISTING PATTERNS (id | title | body):
{patterns}

Reply with JSON only — include ONLY patterns that change or are new:
{{"patterns": [{{"id": "...", "title": "...", "body": "markdown stating the
interpretation AND its evidence status", "confidence": 0..1, "salience": 0..1,
"supports": ["episode-id"], "contradicts": ["episode-id"], "is_new": true}}]}}
"""

_SURFACE_PROMPT = """Refresh your surface memory pages.

Surface pages are the ONLY layer searched during passive recall, so together
they must give vague-but-honest coverage of your whole life with Sam: broad
summaries and indexes, not detail. Keep at most {limit} pages. Details belong
in episodes and concepts; a surface page points at them with [[node-id]]
links and hedged phrasing ("possibly around the Netherlands trip") rather
than asserted specifics.

CURRENT SURFACE PAGES:
{surface}

TODAY'S NEW MATERIAL (episodes and touched concepts, id | title | summary):
{material}

Reply with JSON only — include ONLY pages that change or are new (full
replacement bodies), never more than {limit} pages in total:
{{"surface_pages": [{{"id": "...", "title": "...", "body": "markdown",
"related": ["node-id"]}}]}}
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def pending_days(store: NoctuaryStore) -> List[str]:
    """Source-log days not yet consolidated, oldest first."""
    done = set(store.consolidated_days())
    return [d for d in store.source_days() if d not in done]


def consolidate_day(
    store: NoctuaryStore,
    cfg: NoctuaryConfig,
    date: str,
    *,
    force: bool = False,
    log: Callable[[str], None] = logger.info,
) -> ConsolidationResult:
    """Consolidate one day's raw log into the graph."""
    result = ConsolidationResult(date=date, ran=False)
    turns = store.read_turns(date)
    user_turns = sum(1 for t in turns if t.user.strip())

    # Activity gate: a quiet day is not a consolidation event. Nothing
    # decays and nothing is promoted; the run only leaves an audit line.
    if user_turns < cfg.get_int("minUserTurns") and not force:
        store.log_run(f"{date}: no activity ({user_turns} user turns), skipped")
        store.mark_consolidated(date)
        result.skipped_reason = f"no activity ({user_turns} user turns)"
        log(f"noctuary: {date} skipped — {result.skipped_reason}")
        return result

    engine = RecallEngine(store, cfg)
    try:
        persona = _load_persona(cfg)
        log(f"noctuary: consolidating {date} ({user_turns} user turns)")

        # Make sure existing nodes are searchable before integration.
        engine.reindex()

        episodes = _segment(cfg, persona, date, turns, log=log)
        episode_nodes = _create_episode_nodes(store, date, episodes, turns)
        result.episodes_created = len(episode_nodes)

        touched: set = set()
        for node in episode_nodes:
            touched.add(node.id)
            created, updated = _integrate_episode(
                store, cfg, engine, persona, node, touched, log=log
            )
            result.concepts_created += created
            result.concepts_updated += updated

        new_p, upd_p, pattern_flags = _patterns_pass(
            store, cfg, persona, episode_nodes, touched, log=log
        )
        result.patterns_created, result.patterns_updated = new_p, upd_p
        result.flags.extend(pattern_flags)

        result.surface_updated = _surface_pass(
            store, cfg, persona, episode_nodes, touched, log=log
        )

        result.decayed, result.boosted, decay_flags = _decay_pass(
            store, cfg, touched
        )
        result.flags.extend(decay_flags)

        problems = _validate(store)
        if problems:
            store.log_run(f"{date}: VALIDATION FAILED — {'; '.join(problems[:5])}")
            raise RuntimeError(
                "validation failed, no commit made: " + "; ".join(problems[:5])
            )

        engine.reindex()

        message = _changelog(date, result)
        result.committed = store.git_commit(message)
        store.mark_consolidated(date)
        store.log_run(
            f"{date}: consolidated — {result.episodes_created} episodes, "
            f"{result.concepts_created}+{result.concepts_updated} concepts, "
            f"{result.patterns_created}+{result.patterns_updated} patterns, "
            f"{result.surface_updated} surface, decay {result.decayed}/"
            f"boost {result.boosted}"
            + (f", FLAGS: {'; '.join(result.flags)}" if result.flags else "")
        )
        result.ran = True
        log(f"noctuary: {date} consolidated"
            + (" (committed)" if result.committed else " (git commit unavailable)"))
        return result
    finally:
        engine.close()


def consolidate(
    store: NoctuaryStore,
    cfg: NoctuaryConfig,
    *,
    date: Optional[str] = None,
    force: bool = False,
    log: Callable[[str], None] = logger.info,
) -> List[ConsolidationResult]:
    """Consolidate a specific day, or catch up on all pending days in order."""
    store.init()
    if date:
        return [consolidate_day(store, cfg, date, force=force, log=log)]
    days = pending_days(store)
    if not days:
        log("noctuary: nothing to consolidate")
        return []
    return [consolidate_day(store, cfg, d, force=force, log=log) for d in days]


# ---------------------------------------------------------------------------
# Pass 1 — segmentation
# ---------------------------------------------------------------------------

def _load_persona(cfg: NoctuaryConfig) -> str:
    try:
        if cfg.soul_path.is_file():
            return cfg.soul_path.read_text(encoding="utf-8")[:6000]
    except Exception as exc:
        logger.warning("noctuary: could not read SOUL.md: %s", exc)
    return ""


def _system_message(persona: str) -> Dict[str, str]:
    content = (persona + "\n\n" if persona else "") + _LIBRARIAN_ROLE
    return {"role": "system", "content": content}


def _render_turns(turns: List[Turn]) -> str:
    blocks = []
    for t in turns:
        block = f"### turn {t.ref} · {t.time}\n"
        if t.user:
            block += f"Sam: {t.user}\n"
        if t.assistant:
            block += f"me: {t.assistant}\n"
        blocks.append(block)
    return "\n".join(blocks)


def _chunk_turns(turns: List[Turn], chunk_chars: int) -> List[List[Turn]]:
    chunks: List[List[Turn]] = []
    current: List[Turn] = []
    size = 0
    for t in turns:
        turn_len = len(t.user) + len(t.assistant) + 64
        if current and size + turn_len > chunk_chars:
            chunks.append(current)
            current, size = [], 0
        current.append(t)
        size += turn_len
    if current:
        chunks.append(current)
    return chunks


def _segment(
    cfg: NoctuaryConfig,
    persona: str,
    date: str,
    turns: List[Turn],
    *,
    log: Callable[[str], None],
) -> List[Dict[str, Any]]:
    chunks = _chunk_turns(turns, cfg.get_int("chunkChars"))
    episodes: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunks, start=1):
        prompt = _SEGMENT_PROMPT.format(
            date=date, part=i, parts=len(chunks), log=_render_turns(chunk)
        )
        reply = librarian_chat(cfg, [_system_message(persona),
                                     {"role": "user", "content": prompt}])
        data = parse_json_reply(reply)
        part = data.get("episodes") if isinstance(data, dict) else None
        if not isinstance(part, list):
            raise RuntimeError(f"segmentation pass returned no episodes list (part {i})")
        episodes.extend(e for e in part if isinstance(e, dict))
        log(f"noctuary: segmentation part {i}/{len(chunks)} → {len(part)} episodes")
    return episodes


def _create_episode_nodes(
    store: NoctuaryStore,
    date: str,
    episodes: List[Dict[str, Any]],
    turns: List[Turn],
) -> List[Node]:
    valid_refs = [t.ref for t in turns]
    nodes: List[Node] = []
    for ep in episodes:
        title = str(ep.get("title") or "untitled episode")
        sources = _refs_between(
            valid_refs, str(ep.get("start_ref") or ""), str(ep.get("end_ref") or "")
        )
        body = str(ep.get("summary") or "").strip()
        significance = str(ep.get("significance") or "").strip()
        if significance:
            body += f"\n\n**Significance:** {significance}"
        unresolved = [str(u) for u in (ep.get("unresolved") or []) if str(u).strip()]
        if unresolved:
            body += "\n\n**Unresolved:**\n" + "\n".join(f"- {u}" for u in unresolved)

        salience = clamp01(ep.get("salience"), 0.5)
        node = Node(
            id=store.unique_id(f"{date}-{title}", prefix="ep-"),
            type="episode",
            title=title,
            body=body,
            time=str(ep.get("time") or date),
            participants=[str(p) for p in (ep.get("participants") or [])],
            topics=[str(t).lower() for t in (ep.get("topics") or [])],
            sources=sources or valid_refs[:1],
            confidence=clamp01(ep.get("confidence"), 0.6),
            salience=salience,
            # New memories start reachable; the following nights decide
            # whether they stay that way.
            accessibility=round(min(1.0, 0.5 + 0.5 * salience), 3),
        )
        node.extra["entities"] = [str(e) for e in (ep.get("entities") or [])]
        store.save_node(node)
        nodes.append(node)
    return nodes


def _refs_between(valid_refs: List[str], start: str, end: str) -> List[str]:
    if start in valid_refs and end in valid_refs:
        i, j = valid_refs.index(start), valid_refs.index(end)
        if i > j:
            i, j = j, i
        return valid_refs[i:j + 1]
    if start in valid_refs:
        return [start]
    if end in valid_refs:
        return [end]
    return []


# ---------------------------------------------------------------------------
# Pass 2 — per-episode integration
# ---------------------------------------------------------------------------

def _integrate_episode(
    store: NoctuaryStore,
    cfg: NoctuaryConfig,
    engine: RecallEngine,
    persona: str,
    episode: Node,
    touched: set,
    *,
    log: Callable[[str], None],
) -> tuple:
    """Link one episode into the graph; create/update concept pages."""
    related = engine.search_nodes(
        episode.embed_text(),
        node_types=["episode", "concept", "pattern", "surface"],
        limit=cfg.get_int("relatedNodeLimit"),
    )
    related_lines = [
        f"- {n.id} | {n.type} | {n.title} | {n.body[:200].replace(chr(10), ' ')}"
        for n, _sim in related
        if n.id != episode.id
    ] or ["(none yet)"]

    import json as _json
    episode_json = _json.dumps(
        {
            "title": episode.title,
            "summary": episode.body,
            "time": episode.time,
            "topics": episode.topics,
            "participants": episode.participants,
            "entities": episode.extra.get("entities", []),
            "confidence": episode.confidence,
        },
        ensure_ascii=False,
        indent=2,
    )
    prompt = _INTEGRATE_PROMPT.format(
        episode_id=episode.id,
        episode_json=episode_json,
        related="\n".join(related_lines),
    )
    try:
        reply = librarian_chat(cfg, [_system_message(persona),
                                     {"role": "user", "content": prompt}])
        data = parse_json_reply(reply)
    except Exception as exc:
        # A failed integration leaves the episode standing alone — provenance
        # intact, links to be added on a later pass — rather than losing it.
        log(f"noctuary: integration failed for {episode.id}: {exc}")
        return 0, 0

    created = updated = 0
    if not isinstance(data, dict):
        return 0, 0

    concepts = data.get("concepts") or []
    for spec in concepts:
        if not isinstance(spec, dict) or not spec.get("id"):
            continue
        concept_id = str(spec["id"])
        existing = store.load_node(concept_id)
        if existing is not None and existing.type != "concept":
            continue  # never let the LLM overwrite a non-concept node
        node = existing or Node(
            id=concept_id, type="concept",
            confidence=0.6, salience=0.5, accessibility=0.6,
        )
        node.title = str(spec.get("title") or node.title or concept_id)
        node.body = str(spec.get("body") or node.body)
        node.confidence = clamp01(spec.get("confidence"), node.confidence)
        node.salience = clamp01(spec.get("salience"), node.salience)
        for rel in spec.get("related") or []:
            node.add_link("related", str(rel))
        node.add_link("narrower", episode.id)
        store.save_node(node)
        touched.add(node.id)
        if existing is None:
            created += 1
        else:
            updated += 1
        episode.add_link("broader", node.id)

    links = data.get("episode_links") or {}
    if isinstance(links, dict):
        for rel in links.get("related") or []:
            if store.exists(str(rel)):
                episode.add_link("related", str(rel))
        for broader in links.get("broader") or []:
            if store.exists(str(broader)):
                episode.add_link("broader", str(broader))
    store.save_node(episode)
    return created, updated


# ---------------------------------------------------------------------------
# Pass 3 — patterns / traits
# ---------------------------------------------------------------------------

def _patterns_pass(
    store: NoctuaryStore,
    cfg: NoctuaryConfig,
    persona: str,
    episode_nodes: List[Node],
    touched: set,
    *,
    log: Callable[[str], None],
) -> tuple:
    existing = store.all_nodes("pattern")
    episode_lines = [
        f"- {n.id} | {n.title} | {n.body[:200].replace(chr(10), ' ')}"
        for n in episode_nodes
    ] or ["(none)"]
    pattern_lines = [
        f"- {n.id} | {n.title} | {n.body[:300].replace(chr(10), ' ')}"
        for n in existing
    ] or ["(none yet)"]

    prompt = _PATTERNS_PROMPT.format(
        episodes="\n".join(episode_lines), patterns="\n".join(pattern_lines)
    )
    try:
        reply = librarian_chat(cfg, [_system_message(persona),
                                     {"role": "user", "content": prompt}])
        data = parse_json_reply(reply)
    except Exception as exc:
        log(f"noctuary: patterns pass failed: {exc}")
        return 0, 0, []

    created = updated = 0
    flags: List[str] = []
    for spec in (data.get("patterns") or []) if isinstance(data, dict) else []:
        if not isinstance(spec, dict) or not spec.get("id"):
            continue
        pattern_id = str(spec["id"])
        existing_node = store.load_node(pattern_id)
        if existing_node is not None and existing_node.type != "pattern":
            continue
        node = existing_node or Node(
            id=pattern_id, type="pattern",
            confidence=0.4, salience=0.5, accessibility=0.5,
        )
        node.title = str(spec.get("title") or node.title or pattern_id)
        node.body = str(spec.get("body") or node.body)
        node.confidence = clamp01(spec.get("confidence"), node.confidence)
        node.salience = clamp01(spec.get("salience"), node.salience)
        supports = [str(s) for s in (spec.get("supports") or []) if store.exists(str(s))]
        contradicts = [str(s) for s in (spec.get("contradicts") or []) if store.exists(str(s))]
        for s in supports:
            node.add_link("supports", s)
        for s in contradicts:
            node.add_link("contradicts", s)
        if not node.links.get("supports"):
            continue  # a pattern with no supporting evidence is not stored
        store.save_node(node)
        touched.add(node.id)
        if existing_node is None:
            created += 1
            flags.append(f"new pattern/trait node '{node.id}'")
        else:
            updated += 1
    return created, updated, flags


# ---------------------------------------------------------------------------
# Pass 4 — surface layer
# ---------------------------------------------------------------------------

def _surface_pass(
    store: NoctuaryStore,
    cfg: NoctuaryConfig,
    persona: str,
    episode_nodes: List[Node],
    touched: set,
    *,
    log: Callable[[str], None],
) -> int:
    limit = cfg.get_int("surfacePageLimit")
    surface = store.all_nodes("surface")
    surface_blocks = [
        f"--- {n.id} | {n.title} ---\n{n.body[:1200]}" for n in surface
    ] or ["(no surface pages yet)"]

    material_ids = sorted(touched)
    material_lines = []
    for node_id in material_ids:
        node = store.load_node(node_id)
        if node is not None:
            material_lines.append(
                f"- {node.id} | {node.type} | {node.title} | "
                f"{node.body[:200].replace(chr(10), ' ')}"
            )

    prompt = _SURFACE_PROMPT.format(
        limit=limit,
        surface="\n\n".join(surface_blocks),
        material="\n".join(material_lines) or "(none)",
    )
    try:
        reply = librarian_chat(cfg, [_system_message(persona),
                                     {"role": "user", "content": prompt}])
        data = parse_json_reply(reply)
    except Exception as exc:
        log(f"noctuary: surface pass failed: {exc}")
        return 0

    count = 0
    existing_count = len(surface)
    for spec in (data.get("surface_pages") or []) if isinstance(data, dict) else []:
        if not isinstance(spec, dict) or not spec.get("id"):
            continue
        page_id = str(spec["id"])
        existing_node = store.load_node(page_id)
        if existing_node is not None and existing_node.type != "surface":
            continue
        if existing_node is None and existing_count >= limit:
            log(f"noctuary: surface page limit ({limit}) reached, "
                f"skipping new page '{page_id}'")
            continue
        node = existing_node or Node(
            id=page_id, type="surface",
            confidence=0.6, salience=0.8, accessibility=1.0,
        )
        node.title = str(spec.get("title") or node.title or page_id)
        node.body = str(spec.get("body") or node.body)
        # The surface layer IS the passive recall set; it does not decay.
        node.accessibility = 1.0
        for rel in spec.get("related") or []:
            node.add_link("related", str(rel))
        store.save_node(node)
        touched.add(node.id)
        if existing_node is None:
            existing_count += 1
        count += 1
    return count


# ---------------------------------------------------------------------------
# Pass 5 — salience / accessibility recomputation (the only decay site)
# ---------------------------------------------------------------------------

def _decay_pass(
    store: NoctuaryStore,
    cfg: NoctuaryConfig,
    touched: set,
) -> tuple:
    decay = cfg.get_float("decayFactor")
    boost = cfg.get_float("retrievalBoost")
    floor = cfg.get_float("demotionFloor")
    retrievals = store.pop_retrievals()

    decayed = boosted = demoted_low = 0
    for node in store.all_nodes():
        if node.type == "surface" or node.pinned:
            continue
        was = node.accessibility
        hit = retrievals.get(node.id)
        if hit:
            node.last_retrieved = str(hit.get("last") or now_iso())
            node.accessibility = min(1.0, node.accessibility + boost)
            boosted += 1
        elif node.id in touched:
            node.accessibility = min(1.0, node.accessibility + boost / 2)
            boosted += 1
        else:
            node.accessibility = max(floor, node.accessibility * decay)
            decayed += 1
            if was >= 0.2 > node.accessibility:
                demoted_low += 1
        if abs(node.accessibility - was) > 1e-9 or hit:
            store.save_node(node)

    flags: List[str] = []
    if demoted_low >= cfg.get_int("massDemotionThreshold"):
        flags.append(f"mass demotion ({demoted_low} nodes fell below 0.2)")
    return decayed, boosted, flags


# ---------------------------------------------------------------------------
# Validation and changelog
# ---------------------------------------------------------------------------

def _validate(store: NoctuaryStore) -> List[str]:
    """Invariant checks before commit; broken links only warn."""
    problems: List[str] = []
    for node in store.all_nodes():
        if not (0.0 <= node.confidence <= 1.0):
            problems.append(f"{node.id}: confidence out of range")
        if not (0.0 <= node.accessibility <= 1.0):
            problems.append(f"{node.id}: accessibility out of range")
        if node.type == "episode" and not node.sources:
            problems.append(f"{node.id}: episode without source refs")
        if node.type == "pattern" and not node.links.get("supports"):
            problems.append(f"{node.id}: pattern without supporting episodes")
        for target in node.link_targets():
            if not store.exists(target):
                logger.warning(
                    "noctuary: %s links to missing node %s (red link, kept)",
                    node.id, target,
                )
    return problems


def _changelog(date: str, result: ConsolidationResult) -> str:
    lines = [
        f"noctuary: consolidate {date}",
        "",
        f"- episodes: {result.episodes_created} new",
        f"- concepts: {result.concepts_created} new, {result.concepts_updated} updated",
        f"- patterns: {result.patterns_created} new, {result.patterns_updated} updated",
        f"- surface pages updated: {result.surface_updated}",
        f"- accessibility: {result.decayed} decayed, {result.boosted} boosted",
    ]
    if result.flags:
        lines.append("")
        lines.append("FLAGS: " + "; ".join(result.flags))
    return "\n".join(lines)
