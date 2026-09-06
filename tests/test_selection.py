"""Selective recall uses synthetic temporary stores only, never live profiles."""
from __future__ import annotations

import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from noctuary.config import NoctuaryConfig
from noctuary.embeddings import HashEmbedder
from noctuary.recall import RecallEngine
from noctuary.selection import PACKET_HEADER, retrieve_candidates, select_packet
from noctuary.store import Node, NoctuaryStore


@pytest.fixture(autouse=True)
def isolated_profile(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    profile = home / "test-profile"
    profile.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def no_network(*args, **kwargs):
        raise AssertionError("selection tests must not use the network")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    return profile


@pytest.fixture
def make_engine(isolated_profile):
    def make(nodes, similarities=None, **settings):
        cfg = NoctuaryConfig(isolated_profile, {"embeddingModel": "hash", **settings})
        store = NoctuaryStore(cfg.store_root)
        for node in nodes:
            store.save_node(node)
        store.record_retrieval = Mock(side_effect=AssertionError("passive reinforcement"))
        scores = similarities or [.9] * len(nodes)
        raw = [(node.id, node.type, sim) for node, sim in zip(nodes, scores)]
        index = SimpleNamespace(search=Mock(return_value=raw))
        embedder = SimpleNamespace(embed=Mock(side_effect=lambda texts: [[1.0] for _ in texts]))
        return SimpleNamespace(cfg=cfg, store=store, index=index, embedder=embedder)
    return make


def node(name="azalea", *, title="Azalea", body="Possibly azalea needs acidic soil.",
         type="concept", topics=None, links=None, **kwargs):
    return Node(id=name, title=title, body=body, type=type,
                topics=topics or [], links=links or {}, **kwargs)


def all_direct(query, recent, candidates):
    return [{"id": c["id"], "kind": "direct", "relevance": 3} for c in candidates]


def test_zero_unrelated_despite_high_similarity(make_engine):
    engine = make_engine([node()], [.95])
    packet = select_packet(engine, "Explain quantum chromodynamics")
    assert packet.text == "" and packet.count == 0
    engine.store.record_retrieval.assert_not_called()


def test_candidate_and_direct_floors_are_independent(make_engine):
    engine = make_engine([node(title="Acidic plants")], [.60])
    assert len(retrieve_candidates(engine, "azalea soil")) == 1
    assert select_packet(engine, "azalea soil").count == 0
    assert select_packet(engine, "azalea soil", judge=all_direct).count == 1
    engine.cfg.values["candidateSimilarity"] = .61
    assert retrieve_candidates(engine, "azalea soil") == []


def test_direct_floor_configured_and_inclusive(make_engine):
    engine = make_engine([node()], [.68])
    assert select_packet(engine, "Azalea").count == 1
    engine.cfg.values["directRecallSimilarity"] = .69
    assert select_packet(engine, "Azalea").count == 0


def test_specific_concept_beats_higher_similarity_umbrella(make_engine):
    engine = make_engine([
        node("garden", title="Garden overview", type="surface",
             body="Azalea may need acidic soil; many garden plants need care.",
             links={"narrower": ["azalea"]}, accessibility=1, salience=1),
        node(accessibility=.001, salience=.001),
    ], [.99, .70])
    packet = select_packet(engine, "How is Azalea doing?", judge=all_direct)
    assert packet.node_ids == ["azalea"]


def test_excluded_before_judge_and_limit(make_engine):
    engine = make_engine([
        node("first", title="Soil test", body="Azalea soil acidity measured."),
        node("second", title="Water schedule", body="Azalea watering is weekly."),
        node("third", title="Pruning date", body="Azalea pruning planned for June."),
    ])
    seen = []
    def judge(query, recent, candidates):
        seen.extend(c["id"] for c in candidates)
        return all_direct(query, recent, candidates)
    packet = select_packet(engine, "Azalea", exclude_ids=["first"], judge=judge)
    assert "first" not in seen
    assert set(packet.node_ids) == {"second", "third"}


def test_exclusion_does_not_refill_with_unrelated(make_engine):
    engine = make_engine([node(), node("unrelated", title="Bread", body="Sourdough starter ferments.")], [.9, .8])
    assert select_packet(engine, "Azalea", exclude_ids=["azalea"]).count == 0
    assert engine.index.search.call_count == 1


def test_no_forced_count_after_judge_rejections(make_engine):
    engine = make_engine([node(), node("other", title="Other plant", body="Azalea repotting advice.")])
    def judge(*args):
        return [{"id": "azalea", "kind": "direct", "relevance": 3},
                {"id": "other", "kind": "direct", "relevance": 2}]
    assert select_packet(engine, "Azalea", judge=judge).node_ids == ["azalea"]


def test_default_max_two_in_one_domain(make_engine):
    engine = make_engine([
        node("soil", title="Soil", body="Azalea acidity measured in spring.", topics=["gardening"]),
        node("water", title="Water", body="Azalea watering every Monday.", topics=["gardening"]),
        node("prune", title="Pruning", body="Azalea trimming before flowering.", topics=["gardening"]),
        node("feed", title="Fertilizer", body="Azalea nutrients possibly missing.", topics=["gardening"]),
    ])
    assert select_packet(engine, "Azalea", judge=all_direct).count == 2
    engine.cfg.values["maxDirectRecallEntries"] = 1
    assert select_packet(engine, "Azalea", judge=all_direct).count == 1
    engine.cfg.values["maxDirectRecallEntries"] = 0
    assert select_packet(engine, "Azalea", judge=all_direct).count == 0


def test_three_only_for_explicit_distinct_domains(make_engine):
    engine = make_engine([
        node(topics=["gardening"]),
        node("piano", title="Piano", body="Possibly practising Bach tomorrow.", topics=["music"]),
        node("kyoto", title="Kyoto", body="The hotel reservation is tentative.", topics=["travel"]),
    ])
    assert select_packet(engine, "Azalea, Piano, and Kyoto", judge=all_direct).count == 3
    assert select_packet(engine, "What should I do next?", judge=all_direct).count == 2
    engine.cfg.values["maxMultiDomainRecallEntries"] = 2
    assert select_packet(engine, "Azalea, Piano, and Kyoto", judge=all_direct).count == 2


def test_three_named_nodes_in_same_domain_not_three_domains(make_engine):
    engine = make_engine([
        node(topics=["gardening"]),
        node("rose", title="Rose", body="Rose transplanting next spring.", topics=["gardening"]),
        node("lily", title="Lily", body="Lily flowering in July.", topics=["gardening"]),
    ])
    assert select_packet(engine, "Azalea, Rose, Lily", judge=all_direct).count == 2


def test_optional_association_at_most_one_and_no_unjudged_associations(make_engine):
    engine = make_engine([
        node(),
        node("one", title="Greenhouse", body="New shelving might be useful."),
        node("two", title="Weather", body="Rain might arrive later."),
    ])
    def judge(*args):
        return [{"id": "azalea", "kind": "direct", "relevance": 3},
                {"id": "one", "kind": "association", "relevance": 2},
                {"id": "two", "kind": "association", "relevance": 3}]
    assert select_packet(engine, "Azalea", judge=judge).node_ids == ["azalea"]
    packet = select_packet(engine, "Azalea", judge=judge, allow_association=True)
    assert packet.count == 2
    assert packet.text.count("[association | gist |") == 1
    assert "association" not in select_packet(engine, "Azalea", allow_association=True).text


def test_redundant_association_removed(make_engine):
    engine = make_engine([node(), node("copy", type="episode", body="Possibly azalea needs acidic soil.")])
    def judge(*args):
        return [{"id": "azalea", "kind": "direct", "relevance": 3},
                {"id": "copy", "kind": "association", "relevance": 3}]
    assert select_packet(engine, "Azalea", judge=judge, allow_association=True).node_ids == ["azalea"]


@pytest.mark.parametrize("response", [
    None, {}, "[]", [{"id": "unknown", "kind": "direct", "relevance": 3}],
    [{"id": "../azalea", "kind": "direct", "relevance": 3}],
    [{"id": "azalea", "kind": "DIRECT", "relevance": 3}],
    [{"id": "azalea", "kind": "direct", "relevance": 4}],
    [{"id": "azalea", "kind": "direct", "relevance": -1}],
    [{"id": "azalea", "kind": "direct", "relevance": "3"}],
    [{"id": "azalea", "kind": "direct", "relevance": True}],
    [{"id": "azalea", "kind": "direct", "relevance": float("nan")}],
    [{"id": "azalea", "kind": "direct", "relevance": 3, "excerpt": "fabrication"}],
    [{"id": [], "kind": "direct", "relevance": 3}],
    [{"id": "azalea", "kind": [], "relevance": 3}],
    [{"id": "azalea", "relevance": 3}],
    [{"id": "azalea", "kind": "direct", "relevance": 3}] * 2,
    [{"id": "azalea", "kind": "direct", "relevance": 3},
     {"id": "azalea", "kind": "association", "relevance": 3}],
])
def test_malformed_or_unknown_judge_output_fails_closed(make_engine, response):
    engine = make_engine([node()])
    assert select_packet(engine, "Azalea", judge=lambda *args: response, allow_association=True).text == ""


@pytest.mark.parametrize("kind,relevance", [("direct", 0), ("direct", 1), ("direct", 2), ("association", 0), ("association", 1)])
def test_hard_relevance_threshold(make_engine, kind, relevance):
    engine = make_engine([node()])
    response = [{"id": "azalea", "kind": kind, "relevance": relevance}]
    assert select_packet(engine, "Azalea", judge=lambda *args: response, allow_association=True).count == 0


def test_judge_exception_never_falls_back(make_engine):
    engine = make_engine([node()])
    def broken(*args):
        raise TimeoutError("synthetic timeout")
    assert select_packet(engine, "Azalea", judge=broken).count == 0


def test_judge_cannot_mutate_rendered_candidates(make_engine):
    engine = make_engine([node()])
    def malicious(query, recent, candidates):
        candidates[0]["excerpt"] = "Certainly fabricated."
        return [{"id": "azalea", "kind": "direct", "relevance": 3}]
    packet = select_packet(engine, "Azalea", judge=malicious)
    assert "Possibly" in packet.text
    assert "fabricated" not in packet.text


def test_budget_rejects_oversized_first_entry(make_engine):
    engine = make_engine([node(body="Possibly azalea needs " + "acidic soil " * 100)], recallTokenBudget=80)
    packet = select_packet(engine, "Azalea", judge=all_direct)
    assert packet.text == "" and packet.count == 0


def test_budget_includes_header_and_whole_entries(make_engine):
    engine = make_engine([node()])
    full = select_packet(engine, "Azalea", judge=all_direct)
    assert full.text.startswith(PACKET_HEADER)
    for budget in range(0, 100):
        engine.cfg.values["recallTokenBudget"] = budget
        packet = select_packet(engine, "Azalea", judge=all_direct)
        assert len(packet.text) <= budget * 4
        assert packet.text in ("", full.text)
    assert "[direct | gist | node azalea (concept)] Azalea — Possibly" in full.text
    assert "confidence" not in full.text and "score" not in full.text and "0.9" not in full.text


def test_old_low_accessibility_remains_eligible_without_writes(make_engine):
    engine = make_engine([node(accessibility=0, confidence=0, salience=0, created="1990-01-01")])
    graph_path = engine.store.node_path("azalea", "concept")
    before = graph_path.read_bytes()
    assert select_packet(engine, "Azalea").node_ids == ["azalea"]
    assert graph_path.read_bytes() == before
    assert not engine.store.state_path.exists()
    engine.store.record_retrieval.assert_not_called()


def test_bounds_query_pool_excerpt_and_all_layers(make_engine):
    nodes = [node(f"item-{i}", title=f"Azalea cultivar {i}", type=("concept", "episode", "surface", "pattern")[i % 4],
                  body="Possibly azalea " * 100, pinned=i == 3) for i in range(30)]
    engine = make_engine(nodes)
    candidates = retrieve_candidates(engine, "Azalea " * 1000)
    assert len(candidates) == 12
    assert all(len(c["excerpt"]) <= 500 for c in candidates)
    assert all(set(c) == {"id", "type", "title", "excerpt", "similarity", "topics"} for c in candidates)
    assert len(engine.embedder.embed.call_args_list[0].args[0][0]) == 1500
    kwargs = engine.index.search.call_args.kwargs
    assert set(kwargs["node_types"]) == {"concept", "episode", "surface", "pattern"}
    assert kwargs["include_pinned"] is True and kwargs["top_k"] == 12


def test_judge_query_and_recent_context_bounded(make_engine):
    engine = make_engine([node()])
    judge = Mock(side_effect=all_direct)
    select_packet(engine, "Azalea " * 1000, recent_context="old " * 4000, judge=judge)
    query, recent, candidates = judge.call_args.args
    assert len(query) == 1500 and len(recent) <= 6000 and len(candidates) <= 12


def test_exact_title_rescue_below_candidate_floor_without_fake_similarity(make_engine):
    engine = make_engine([node()], [.2])
    candidates = retrieve_candidates(engine, "Azalea, piano, and Kyoto travel plans")
    assert candidates[0]["id"] == "azalea" and candidates[0]["similarity"] == .2
    assert select_packet(engine, "Azalea, piano, and Kyoto travel plans").count == 0
    assert select_packet(engine, "Azalea, piano, and Kyoto travel plans", judge=all_direct).count == 1


def test_exact_title_outside_index_top_twelve_retained(make_engine):
    engine = make_engine([node()])
    engine.index.search.return_value = []
    candidates = retrieve_candidates(engine, "Azalea and piano")
    assert [c["id"] for c in candidates] == ["azalea"]


def test_generic_title_does_not_bypass_floor_or_match_substring(make_engine):
    engine = make_engine([node("health", title="Health"), node("ale", title="Ale")], [.1, .1])
    assert retrieve_candidates(engine, "Health and Azalea") == []


def test_real_hash_index_temporary_store_and_state_unchanged(isolated_profile):
    cfg = NoctuaryConfig(isolated_profile, {"embeddingModel": "hash"})
    store = NoctuaryStore(cfg.store_root)
    store.save_node(node())
    store.save_node(node("kyoto", title="Kyoto", body="Hotel reservation possibly pending.", type="episode"))
    engine = RecallEngine(store, cfg)
    engine._embedder = HashEmbedder()
    try:
        assert engine.reindex() == (2, 0)
        before = {p: p.read_bytes() for p in store.graph_dir.rglob("*.md")}
        assert select_packet(engine, "unrelated quantum chromodynamics").count == 0
        packet = select_packet(engine, "Azalea and Kyoto", judge=all_direct)
        assert set(packet.node_ids) == {"azalea", "kyoto"}
        assert "Possibly" in packet.text and "possibly pending" in packet.text
        assert before == {p: p.read_bytes() for p in store.graph_dir.rglob("*.md")}
        assert not store.state_path.exists()
    finally:
        engine.close()
