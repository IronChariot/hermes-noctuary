# Noctuary

A human-like long-term memory system for Hermes Agent, built as a memory provider plugin.

_A noctuary is a night journal. The system records the day, and a nightly process turns the record into memory._

Sources: `wren-memory-requirements.txt`, `wren-memory-convo.txt`, and the Q&A session of 2026-08-20.

## 1. Purpose

Give Wren autobiographical memory that behaves like human recollection, not like "retrieve every relevant chunk" RAG.

Wren must experience graded recall — familiarity, gist, deliberate recollection, and exact source verification — while the system:

- never permanently loses the underlying evidence, and
- never invents missing details with false confidence.

The memory is an evolving, wiki-like graph, not a chronological hierarchy.

## 2. Scope and deployment decisions

These decisions are fixed for v1:

| Decision | Value |
|---|---|
| Form | Hermes **memory provider plugin** (implements the `MemoryProvider` ABC) |
| Installation | User-directory plugin: `$HERMES_HOME/plugins/noctuary/` |
| Coexistence | Runs alongside built-in MEMORY.md / USER.md; no mirroring of built-in writes |
| Ingestion source | The single ongoing Wren conversation on Discord, only |
| Users | Single user (Sam) |
| Instances | Each Hermes instance (Wren, Maomao, …) gets its own profile-scoped store via the `hermes_home` kwarg |
| Storage | Local disk on the Mini PC; no encryption at rest |
| Embeddings | Local embedding model |
| LLM calls | Default: the same model/provider the Hermes agent uses. Optional config: any other provider/model already configured in Hermes |
| Versioning | git repository inside the store |
| Review | Fully autonomous by default; retrospective inspection via git history; blocking review is not required |
| Decay | Accessibility changes only during nightly consolidation, and consolidation runs only on days with conversation |
| Wren-initiated promote/demote | Not in v1 (future extension) |
| Recall presentation | Technical: recall level + confidence scores, not experiential phrasing |
| v1 milestone | Stages 1–3 together (archive, passive recall, nightly librarian) |

Non-Wren execution contexts (subagents, cron jobs, other platforms) must not write to the store. The provider checks the `agent_context` / `platform` kwargs and skips ingestion outside the Discord conversation.

## 3. Core principles

1. **Storage ≠ accessibility.** Source records are immutable and permanent. Only *passive accessibility* fades, and deliberate search can always reach the archive.
2. **Multi-resolution.** Every memory exists at several levels: raw messages → episode summaries → thematic/period pages → surface pages → durable personal model. Each level links down to what it compressed.
3. **Graph, not timeline.** Nodes are organised by theme, entity, and relationship as well as date. One episode can belong to many pages without duplication. Broad pages are indexes and interpretations, not containers.
4. **Provenance and uncertainty.** Every derived node keeps source links, a confidence marking, and the distinction between exact fact and compressed interpretation. The system must never turn "possibly around the Netherlands trip" into a definite date.
5. **No confabulation.** Vague memories are labelled vague. "No recognition" means "nothing surfaced", never "this did not happen".
6. **Auditable evolution.** Every change to the graph is a git commit with an inspectable diff. The past is reorganised, never silently rewritten.

## 4. Information model

Stored as plain Markdown files with YAML frontmatter, in a git repository under the profile store. Wiki-links (`[[node-id]]`) connect nodes.

### Node types

- **Source records** — immutable archived transcript segments. The authority for exact recall. Never edited, never deleted.
- **Episode nodes** — summaries of bounded events/conversations: participants, time (exact or approximate), what happened, emotional/relational significance, topics, confidence, source references.
- **Concept (wiki) nodes** — pages for recurring subjects: people, pets, projects, places, trips, routines, interests, themes. They link to episodes; they do not duplicate them.
- **Pattern / trait nodes** — tentative interpretations from repeated evidence (preferences, habits, dynamics, lessons). Each one lists supporting *and* contradicting episodes.
- **Surface nodes** — a deliberately small set of broad summary pages. The passive recall layer searches only these plus key pinned nodes.

### Node metadata (frontmatter)

Stable ID; node type; confidence; salience; passive-accessibility score; created / modified timestamps; last-retrieved timestamp; pinned flag; typed links (broader, narrower, related, supports, contradicts); source references.

### Embedding index

A local index (e.g. SQLite + a vector extension, or an equivalent simple local store) maps node embeddings to node IDs. The index is derived data: it can always be rebuilt from the Markdown files.

## 5. Retrieval design

### Passive recall (every turn)

On each user message, the provider injects a **recall packet** via `prefetch()`:

- Search covers surface nodes and key nodes only, using the current message as cue.
- Ranking combines semantic similarity, graph links, salience, recency, and reactivation history.
- The packet contains, per hit: a snippet or short description, provenance (date range and/or theme), confidence, recall level, and the node ID for drill-down.
- Budget: **≤ 1000 tokens** by default (configurable). Latency target: seconds; hard ceiling 30 s. Use `queue_prefetch()` + cached results to stay off the critical path.
- Trivial prompts (greetings, acknowledgements) skip recall entirely (Hermes provides `is_trivial_prompt`).
- Retrieval may be pure vector RAG in v1; an agentic search pass with a cheap/fast model is an optional later refinement.

### Recall levels

The packet states one of:

1. **No recognition** — nothing surfaced ("does not ring a bell", not "never happened").
2. **Familiarity** — something related exists; details unavailable at this level.
3. **Gist** — enough context for a natural reply.

Levels 4–5 are reached only through tools:

4. **Deliberate recollection** — search episode nodes and graph neighbours.
5. **Source verification** — read the original archived messages.

### Tools (deliberate recall)

The provider exposes a small tool set, roughly:

- `memory_recall(node_id | query)` — expand a node; return its content plus neighbouring nodes (same theme, same period, linked episodes).
- `memory_verify(node_id | source_ref)` — return the exact archived source text.
- (v1 optional) `memory_search(query)` — free semantic search over episode nodes.

### Tool economy

The `system_prompt_block()` must instruct the agent: answer from the gist when the gist is sufficient; call memory tools only when specificity matters, when uncertain, or when Sam explicitly asks for accurate recall. Tool calls are expensive; minimise them.

## 6. Ingestion

- `sync_turn()` appends each completed Discord turn to a durable daily raw log (non-blocking, background thread).
- `on_pre_compress()` / `on_session_end()` flush any buffered turns so nothing is lost at context boundaries.
- Raw logs become source records; the nightly librarian does all interpretation. The live conversation path never edits the graph.

### Backlog bootstrap

A CLI command (`hermes noctuary ingest <path>`) ingests the existing long-running Hermes session log and builds the initial graph: segmentation into episodes, initial concept pages, initial surface pages. This runs in batches and may take multiple LLM passes.

## 7. The nightly librarian

A scheduled process (Hermes cron or provider CLI, e.g. `hermes noctuary consolidate`) that runs outside live conversation.

### Activity gate

The librarian runs only on days when Sam actually spoke to Wren on Discord. If the day's raw log holds no user turns, the run exits immediately: no LLM calls, no git commit, no accessibility recomputation.

Consequences of the gate:

- A quiet day is not a consolidation event. Nothing decays and nothing is promoted, so silence never erodes the graph.
- Decay therefore advances per *active* day, not per calendar day. A two-week gap costs a memory one decay step, not fourteen.
- The run logs a one-line "no activity, skipped" record so the schedule stays auditable.
- `hermes noctuary consolidate --force` overrides the gate for manual runs and testing.

### Identity and context

- The librarian runs **as Wren**: it loads the standard SOUL.md persona.
- Its context includes the last 24 hours of conversation (chunked/map-reduced when it exceeds the window) plus retrieved graph context.
- It can search the archived session when it needs older evidence.
- It uses the agent's configured model by default; a config key can select another already-configured Hermes model.

### Duties

1. Segment the day's log into candidate episodes.
2. Extract entities, themes, decisions, emotional moments, unresolved questions.
3. Retrieve related existing nodes; add typed links; update concept and surface pages.
4. Detect repetition; create or strengthen pattern/trait nodes (keeping contradicting evidence).
5. Recompute salience and passive-accessibility: promote newly significant memories, let stale detail recede. Decay happens only here.
6. Preserve contradictions instead of resolving them into a tidy narrative.
7. Commit the whole change set to git with a summary changelog.

### Governance

- The librarian applies changes autonomously. No blocking review step.
- Safety comes from invariants, not approval: source records are append-only; every change is a reversible git commit; a validation pass checks provenance links and confidence fields before commit.
- High-impact changes (personal-model claims, trait creation, mass demotion) are flagged in the changelog so Sam can inspect them in git when he wants to.
- Optional nicety: Wren may briefly mention notable consolidations in conversation the next day.

## 8. Configuration

Minimal `get_config_schema()`; everything else in `$HERMES_HOME/noctuary.json` with sane defaults:

- `recallTokenBudget` (default 1000)
- `librarianModel` (default: agent's model)
- `embeddingModel` (default: a small local model)
- `surfacePageLimit` (target size of the surface layer)
- `consolidateSchedule` (default nightly)
- `minUserTurns` (default 1 — the activity gate threshold; a day with fewer user turns is skipped)

## 9. Acceptance test — the mouse incident

1. Sam casually mentions the cats waking him with a mouse.
2. Passive recall surfaces only a vague packet: cats, live prey, bedroom, possibly the Netherlands trip — with confidence markings.
3. Wren replies naturally ("Again?!") without tool calls.
4. Asked what she remembers, Wren reports the gist and its uncertainty.
5. Asked for the exact event, Wren follows provenance links via tools down to the episode and the original messages.
6. Exact details appear only when the source supports them.
7. If nothing surfaces, Wren says it does not ring a bell — not that no such memory exists.

## 10. Evaluation criteria

- Passive recall feels natural and surfaces useful familiarity without excess context.
- Exact recall is reliably reconstructible via provenance links.
- Token and latency costs stay within budget.
- False-recognition and missed-recognition rates are tolerable.
- No unsupported certainty (confabulation).
- Consolidation preserves texture; Wren does not collapse into tidy biographical claims.

## 11. Implementation stages

1. **Archive + episodes** — raw-log capture, backlog ingestion, episode summaries, first concept pages, provenance links throughout.
2. **Passive recall + tools** — surface layer, embedding index, recall packet in `prefetch()`, drill-down tools, system-prompt guidance.
3. **Nightly librarian** — full consolidation pipeline with git commits and changelog.

Stages 1–3 together are the v1 milestone. Then:

4. **Dynamic accessibility tuning** — refine decay/promotion parameters against real use.
5. **Evaluation and refinement** — measure section 10; adjust budgets and surface size.

## 12. Future extensions (explicitly out of v1)

- Wren-initiated promote/pin/demote during conversation.
- Agentic (LLM-driven) passive retrieval instead of pure vector search.
- A more convenient review surface than raw git, if it ever proves necessary.
- Experiential phrasing of recall levels.
- Mirroring built-in MEMORY.md writes into the graph.
