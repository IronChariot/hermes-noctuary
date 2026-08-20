# hermes-noctuary

A human-like long-term memory plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

A noctuary is a night journal. The system records the day, and a nightly process turns the record into memory. The full design is in [noctuary-requirements.md](noctuary-requirements.md).

The plugin gives the agent graded recall instead of "retrieve every relevant chunk" RAG:

1. **No recognition** — nothing surfaced ("does not ring a bell", never "it did not happen").
2. **Familiarity** — something related exists; details are not available at this level.
3. **Gist** — enough context for a natural reply.
4. **Deliberate recollection** — the `memory_recall` / `memory_search` tools expand nodes and search the graph.
5. **Source verification** — the `memory_verify` tool reads the exact archived messages.

Source records are immutable and permanent. Only passive accessibility fades, and only during nightly consolidation. Every change to the graph is a git commit.

## Layout

The plugin is the `noctuary/` package in this repository. It implements the Hermes `MemoryProvider` ABC.

| Module | Purpose |
|---|---|
| `noctuary/__init__.py` | The provider: turn capture, passive recall packet, tools |
| `noctuary/store.py` | Markdown+YAML graph nodes, append-only source logs, git |
| `noctuary/embeddings.py` | Local embedder (fastembed / sentence-transformers / hash fallback) and the SQLite vector index |
| `noctuary/recall.py` | Ranking, recall levels, packet formatting, node expansion |
| `noctuary/librarian.py` | Nightly consolidation: segmentation, integration, patterns, surface refresh, decay, validation, commit |
| `noctuary/ingest.py` | Backlog bootstrap from a session DB, JSON/JSONL, or plain transcript |
| `noctuary/cli.py` | The `hermes noctuary` subcommands |

The store lives in `$HERMES_HOME/noctuary/`:

```
noctuary/
  graph/
    episodes/   concepts/   patterns/   surface/     # Markdown nodes
  sources/<YYYY-MM-DD>.md                            # append-only raw logs
  index.sqlite                                       # derived; rebuildable
  state.json                                         # retrieval log, run state
  consolidation.log                                  # one line per librarian run
  .git/                                              # every change set is a commit
```

## Installation

1. Copy or link the `noctuary/` package into the user plugin directory:

   ```powershell
   # Windows (junction keeps the repo as the source of truth)
   New-Item -ItemType Junction -Path "$env:USERPROFILE\.hermes\plugins\noctuary" -Target "D:\ironchariot\hermes-noctuary\noctuary"
   ```

   ```bash
   # Linux / macOS
   ln -s /path/to/hermes-noctuary/noctuary ~/.hermes/plugins/noctuary
   ```

   Use the active profile's HERMES_HOME if it is not the default.

2. Activate the provider in `$HERMES_HOME/config.yaml`:

   ```yaml
   memory:
     provider: noctuary
   ```

3. Optional: install a real embedding model. Without one, the plugin falls back to a built-in hashing embedder (works, lower quality):

   ```bash
   pip install fastembed        # or: pip install sentence-transformers
   ```

4. Optional: bootstrap the graph from the existing session history:

   ```bash
   hermes noctuary ingest ~/.hermes/state.db --session <session-id>
   ```

5. Schedule the nightly librarian (Hermes cron or the OS scheduler):

   ```bash
   hermes noctuary consolidate
   ```

   The command consolidates all pending days. Days without user turns are skipped, logged, and never decay the graph.

## Configuration

`$HERMES_HOME/noctuary.json`, all keys optional:

| Key | Default | Meaning |
|---|---|---|
| `recallTokenBudget` | `1000` | Token budget for the passive recall packet |
| `librarianModel` | `""` | Model for consolidation; empty = the agent's model |
| `librarianProvider` | `""` | Provider for the librarian model; empty = auto |
| `embeddingModel` | `"auto"` | `auto`, `hash`, or a fastembed / sentence-transformers model name |
| `surfacePageLimit` | `12` | Target size of the surface (passive recall) layer |
| `consolidateSchedule` | `"nightly"` | Informational; scheduling is external |
| `minUserTurns` | `1` | Activity gate: days with fewer user turns are skipped |
| `ingestPlatforms` | `["discord"]` | Platforms whose live turns are archived |
| `decayFactor` | `0.95` | Accessibility multiplier per active day for untouched nodes |
| `retrievalBoost` | `0.15` | Accessibility boost for retrieved nodes |
| `gistSimilarity` / `familiaritySimilarity` | `0.50` / `0.32` | Recall level floors |

Only the primary agent on a platform in `ingestPlatforms` writes to the archive. Subagents, cron jobs, and other platforms get read-only recall. Add `"cli"` to `ingestPlatforms` to capture CLI turns during testing.

## CLI

```
hermes noctuary status                     # store, index, pending days
hermes noctuary consolidate [--date D] [--force]
hermes noctuary ingest <path> [--format auto|hermes-db|json|jsonl|text]
                              [--session ID] [--date YYYY-MM-DD]
                              [--no-consolidate] [--max-days N]
hermes noctuary recall <query>             # debug: print the passive packet
hermes noctuary search <query>             # debug: semantic search, all layers
hermes noctuary show <node-id>             # print one node
hermes noctuary verify <YYYY-MM-DD/NNN>    # print the archived source turn
hermes noctuary reindex                    # rebuild the embedding index
```

## Review

The librarian applies changes autonomously. Inspect its work with git:

```bash
git -C ~/.hermes/noctuary log --stat
git -C ~/.hermes/noctuary show
```

High-impact changes (new pattern/trait nodes, mass demotion) are flagged in the commit message.

## Tests

The tests need a hermes-agent checkout next to this repository, or `HERMES_AGENT_ROOT` pointing at one. No network and no model downloads; the LLM is stubbed.

```bash
python -m pytest tests -q
```
