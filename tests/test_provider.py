"""Provider tests: gate, prefetch, tools, config."""

from __future__ import annotations

import json
import time

import pytest

from noctuary import NoctuaryProvider
from noctuary.store import Node, NoctuaryStore


def _write_config(hermes_home, values):
    (hermes_home / "noctuary.json").write_text(json.dumps(values), encoding="utf-8")


@pytest.fixture()
def provider(hermes_home):
    _write_config(hermes_home, {
        "embeddingModel": "hash",
        "familiaritySimilarity": 0.05,
        "gistSimilarity": 0.25,
    })
    p = NoctuaryProvider()
    yield p
    p.shutdown()


def _wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_discord_primary_turns_are_archived(provider, hermes_home):
    provider.initialize("session-1", hermes_home=str(hermes_home),
                        platform="discord", agent_context="primary")
    provider.sync_turn("the cats did it again", "Again?!")
    store = NoctuaryStore(hermes_home / "noctuary")
    assert _wait_for(lambda: len(store.source_days()) == 1)
    day = store.source_days()[0]
    turns = store.read_turns(day)
    assert turns[0].user == "the cats did it again"
    assert turns[0].platform == "discord"


@pytest.mark.parametrize("kwargs", [
    {"platform": "cli", "agent_context": "primary"},
    {"platform": "discord", "agent_context": "subagent"},
    {"platform": "cron", "agent_context": "cron"},
])
def test_non_discord_contexts_never_write(provider, hermes_home, kwargs):
    provider.initialize("session-1", hermes_home=str(hermes_home), **kwargs)
    provider.sync_turn("should not be stored", "ok")
    provider.on_session_end([])
    time.sleep(0.2)
    store = NoctuaryStore(hermes_home / "noctuary")
    assert store.source_days() == []


def test_prefetch_and_recall_status(provider, hermes_home):
    store = NoctuaryStore(hermes_home / "noctuary")
    store.init()
    store.save_node(Node(
        id="cats-and-prey", type="surface", title="Cats and prey",
        body="The cats sometimes bring live prey into the bedroom at night.",
        accessibility=1.0, salience=0.8, confidence=0.6,
        topics=["cats", "prey"],
    ))
    provider.initialize("session-1", hermes_home=str(hermes_home),
                        platform="discord", agent_context="primary")
    provider._engine.reindex()

    packet = provider.prefetch("the cats brought a mouse into the bedroom")
    assert "cats-and-prey" in packet
    status = provider.recall_status()
    assert status is not None
    assert status.count == 1
    assert status.glyph == "🌙"

    # Trivial prompts skip recall entirely.
    assert provider.prefetch("thanks!") == ""
    assert provider.recall_status() is None


def test_queue_prefetch_caches_for_next_turn(provider, hermes_home):
    store = NoctuaryStore(hermes_home / "noctuary")
    store.init()
    store.save_node(Node(
        id="cats-and-prey", type="surface", title="Cats and prey",
        body="The cats bring live prey inside.", accessibility=1.0,
    ))
    provider.initialize("session-1", hermes_home=str(hermes_home),
                        platform="discord", agent_context="primary")
    provider._engine.reindex()
    provider.queue_prefetch("mouse in the bedroom")
    assert _wait_for(lambda: provider._prefetch_cache is not None)
    packet = provider.prefetch("mouse in the bedroom")
    assert "cats-and-prey" in packet


def test_tools(provider, hermes_home):
    store = NoctuaryStore(hermes_home / "noctuary")
    store.init()
    ref = store.append_turn("a live mouse!", "Again?!", platform="discord",
                            ts=time.time())
    store.save_node(Node(
        id="ep-mouse", type="episode", title="Mouse incident",
        body="Cats woke Sam with a live mouse.", sources=[ref],
        confidence=0.85, time="last night",
    ))
    provider.initialize("session-1", hermes_home=str(hermes_home),
                        platform="discord", agent_context="primary")
    provider._engine.reindex()

    schemas = {s["name"] for s in provider.get_tool_schemas()}
    assert schemas == {"memory_recall", "memory_verify", "memory_search"}

    recall = json.loads(provider.handle_tool_call("memory_recall",
                                                  {"node_id": "ep-mouse"}))
    assert recall["found"] is True
    assert recall["sources"] == [ref]

    verify = json.loads(provider.handle_tool_call("memory_verify",
                                                  {"source_ref": ref}))
    assert verify["sources"][0]["found"] is True
    assert verify["sources"][0]["user"] == "a live mouse!"

    search = json.loads(provider.handle_tool_call("memory_search",
                                                  {"query": "mouse"}))
    assert search["results"][0]["id"] == "ep-mouse"

    missing = json.loads(provider.handle_tool_call("memory_recall",
                                                   {"node_id": "nope"}))
    assert missing["found"] is False
    assert "surfaced" in missing["note"]  # no-recognition, not nonexistence


def test_system_prompt_block_mentions_tool_economy(provider, hermes_home):
    provider.initialize("session-1", hermes_home=str(hermes_home),
                        platform="discord", agent_context="primary")
    block = provider.system_prompt_block()
    assert "gist" in block
    assert "minimise" in block


def test_save_and_load_config(provider, hermes_home):
    provider.save_config({"recallTokenBudget": 500}, str(hermes_home))
    from noctuary.config import load_config
    cfg = load_config(hermes_home)
    assert cfg.get_int("recallTokenBudget") == 500
    # Untouched keys keep defaults.
    assert cfg.get_int("surfacePageLimit") == 12


def test_register_entrypoint():
    import noctuary

    class Ctx:
        provider = None
        def register_memory_provider(self, p):
            self.provider = p

    ctx = Ctx()
    noctuary.register(ctx)
    assert isinstance(ctx.provider, NoctuaryProvider)
    assert ctx.provider.name == "noctuary"
    assert ctx.provider.is_available() is True
