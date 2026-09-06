# Selective passive recall

This opt-in mode replaces prefetch with a session/profile-bound `pre_llm_call`
hook. It requires Hermes' current callback registrar, effective `api_content`
sidecars and disposal handles. An unsupported/cold route omits recall rather
than falling back to legacy selection. Main model, identity, session and
conversation rows are unchanged. Native main-chat tool loops remain unchanged.
Opaque native Responses compaction checkpoints are explicitly unsupported: when
one is present, passive injection is skipped because this hook cannot prove which
local history survives transport pruning. Standard Hermes in-place summaries
remain supported. This limitation is detected, not silently treated as full active
context. Manual recall still works.

## Policy

- Parse only recognized Noctuary packets in injected string `api_content`
  suffixes, not arbitrary visible user text, summaries or tool output.
- Suppress those exact node IDs while their packets remain active. Exclude
  before index ranking, shortlist limits, judge consumption and output budget.
- A vague compaction-summary mention alone does not suppress. This is node-ID
  packet suppression, not semantic deduplication of arbitrary conversation.
- Search concepts, episodes, patterns and surfaces; exact named concepts can
  enter a bounded shortlist even when multi-topic embeddings miss them.
- Permit zero. Normally at most two direct entries; a third requires distinct
  explicit domain cues. One optional association is separately labelled/gated.
- Prefer named concepts and prune linked or near-duplicate umbrella material.
- Relevance is an admission decision, not stored `confidence`. Do not display
  fixed confidence decimals. Manual source verification remains available.
- Automatic recall does not reinforce accessibility. Selective ranking ignores
  stored accessibility. `usageDecayEnabled: false` additionally freezes the
  librarian's usage-based accessibility decay/boost; it does not delete memories.
- Ordinary diagnostic logging includes only counts, duration and mode.

Example `$HERMES_HOME/noctuary.json` additions (merge, don't replace identity or
librarian settings):

```json
{
  "passiveRecallMode": "selective",
  "recallJudge": "openai-codex",
  "recallJudgeTimeoutSeconds": 15,
  "recallJudgeDailyCallLimit": null,
  "recallTokenBudget": 500,
  "maxDirectRecallEntries": 2,
  "maxMultiDomainRecallEntries": 3,
  "allowAssociativeRecall": true,
  "usageDecayEnabled": false
}
```

`recallTokenBudget` retains the existing approximate four-character token budget
and includes the header. It is not an exact model-token guarantee. Candidate
excerpts are at most 500 characters, twelve candidates, current query at most
1,500 characters and clean recent dialogue at most 2,000 characters. This is
purposefully not the full conversation or full memory archive.

## Judge routes and boundaries

Defaults are `passiveRecallMode: legacy`, `recallJudge: off`; upgrades do not
silently enable network judgement. `off` in selective mode uses conservative
local similarity/lexical admission. It can miss paraphrases and anaphora.

- `openai-codex`: pins `gpt-5.6-luna`, reasoning `none`, through Hermes' native
  subscription OAuth resolver and Responses adapter. Native auth remains owned
  by Hermes; no copied static token, third-party proxy, tools, automatic retry,
  model fallback or API-key billing route. Requests use `store: false` through
  the adapter. Daily request reservations persist in `recall-codex-calls.sqlite`
  when a positive cap is configured and are never refunded, even on failures.
  The default `null` (also `0`) means unlimited local requests and bypasses the
  ledger entirely, including any old exhausted cap. This consumes subscription allowance;
  it is not an unlimited/free API. Codex rejects output-token caps; this route
  does NOT claim a hard generated-token or dollar ceiling. The caller bounds
  input, foreground wall time and accepted response size; a daily request cap
  is optional. OpenAI subscription limits still apply.
- `openrouter`: pins `deepseek/deepseek-v4-flash`, disables fallback routing,
  tools and reasoning, bounds output at 256 tokens, input bytes, route prices
  and daily USD reservations. Optional `recallJudgeCredentialFile` is a local
  `.env` path, never a credential value. `recallJudgeDailyBudgetUsd` defaults to
  `0.05`; ledger `recall-budget.sqlite` retains unknown usage reservations.

The judge deadline defaults to 15 seconds. Both routes have one no-tools batch, strict known-ID/schema validation, a
foreground wall-clock deadline and a per-provider in-flight guard. A slow
transport can finish behind that deadline, but cannot inject late results or
spawn overlapping requests. When a quota/reservation is enabled it remains consumed. Judge
failure means no automatic packet, never unjudged fallback. Clean dialogue and
shortlisted private memory excerpts are sent to the chosen provider; enable only
with the account owner's approval.

## Verification and rollout

Use an isolated worktree and copied stores. Run the entire standalone suite with
credentials scrubbed and a temporary `HOME`/`HERMES_HOME`. The real Hermes test
uses an in-process localhost HTTP fixture and checks actual outgoing requests,
second-turn suppression, database restore, summary-only re-eligibility, and clean
source ingestion. It is not a paid/natural compressor run.

A bounded copied-data comparison found the native Luna route more reliable and
faster than the tested Flash route. Across 24 small Luna requests, all completed;
the final held-out batch correctly omitted ordinary unrelated and nonexistent
project cues. This is a smoke/acceptance sample, not a calibrated production
accuracy or tail-latency estimate. Private reports stay outside this repository.

Before live activation, verify a cold profile checkpoint and preserve the prior
source commit. Quiesce the owning gateway and consolidation timer/service;
upgrade the external source checkout and merge only these configuration keys.
Restart only that profile. Compare session ID, transcript counts/tail, identity
hash, memory-store hashes and database integrity. Verify profile-scoped plugin
loading and perform the next ordinary chat acceptance without resetting the
conversation. No Hermes-core patch is required, but future hook/API changes may
require compatibility work.

Rollback normally means quiescing Wren, restoring the previous code generation
and previous Noctuary config, then restarting with the SAME session and current
transcript. Do not overwrite the conversation from a checkpoint merely to roll
back a recall policy; preserve new conversation evidence.
