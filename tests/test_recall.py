"""Recall engine tests: packet levels, budget, retrieval logging, expansion."""

from __future__ import annotations

from noctuary.recall import RecallEngine
from noctuary.store import Node


def _seed_graph(store):
    store.save_node(Node(
        id="cats-and-prey", type="surface", title="Cats and prey",
        body="The cats sometimes bring live prey into the bedroom at night — "
             "possibly starting around the [[netherlands-trip]]. Details in "
             "[[ep-2026-08-19-mouse-incident]].",
        confidence=0.6, salience=0.8, accessibility=1.0,
        topics=["cats", "prey", "bedroom"],
    ))
    store.save_node(Node(
        id="finances", type="surface", title="Finances",
        body="Broad summary of Sam's tax filings and budget spreadsheets.",
        confidence=0.7, salience=0.5, accessibility=1.0,
        topics=["taxes", "money"],
    ))
    store.save_node(Node(
        id="ep-2026-08-19-mouse-incident", type="episode",
        title="The mouse incident",
        body="Sam was woken by the cats with a live mouse in the bedroom.",
        confidence=0.85, salience=0.7, accessibility=0.7,
        time="2026-08-19 night", sources=["2026-08-19/001"],
        topics=["cats", "mouse"],
    ))
    store.save_node(Node(
        id="netherlands-trip", type="concept", title="Netherlands trip",
        body="A trip Sam took; timing of the first prey incident is vague.",
        confidence=0.6, salience=0.6, accessibility=0.6,
    ))


def test_packet_surfaces_only_surface_layer(store, cfg):
    _seed_graph(store)
    engine = RecallEngine(store, cfg)
    try:
        engine.reindex()
        packet = engine.build_packet("the cats woke me with a live mouse in the bedroom")
        assert packet.count >= 1
        assert "cats-and-prey" in packet.node_ids
        # Episodes are not in the passive layer.
        assert "ep-2026-08-19-mouse-incident" not in packet.node_ids
        assert "node cats-and-prey" in packet.text
        assert "conf" in packet.text
        # No-confabulation framing is part of the packet header.
        assert "nothing surfaced" in packet.text
    finally:
        engine.close()


def test_pinned_node_joins_passive_layer(store, cfg):
    _seed_graph(store)
    node = store.load_node("netherlands-trip")
    node.pinned = True
    store.save_node(node)
    engine = RecallEngine(store, cfg)
    try:
        engine.reindex()
        packet = engine.build_packet("remember the netherlands trip timing?")
        assert "netherlands-trip" in packet.node_ids
    finally:
        engine.close()


def test_no_recognition_returns_empty(store, cfg):
    _seed_graph(store)
    cfg.values["familiaritySimilarity"] = 0.99  # nothing can clear this bar
    engine = RecallEngine(store, cfg)
    try:
        engine.reindex()
        packet = engine.build_packet("completely unrelated quantum chromodynamics")
        assert packet.text == ""
        assert packet.count == 0
    finally:
        engine.close()


def test_budget_trims_entries(store, cfg):
    for i in range(10):
        store.save_node(Node(
            id=f"page-{i}", type="surface", title=f"Cats page {i}",
            body="cats mouse bedroom " * 40,
            accessibility=1.0, salience=0.5,
        ))
    cfg.values["recallTokenBudget"] = 120  # tiny budget
    cfg.values["maxRecallEntries"] = 10
    engine = RecallEngine(store, cfg)
    try:
        engine.reindex()
        packet = engine.build_packet("cats mouse bedroom")
        assert packet.count >= 1
        assert len(packet.text) <= 120 * 4 + 200  # header slack
        assert packet.count < 10
    finally:
        engine.close()


def test_retrievals_are_logged_not_written_to_graph(store, cfg):
    _seed_graph(store)
    before = store.load_node("cats-and-prey").modified
    engine = RecallEngine(store, cfg)
    try:
        engine.reindex()
        engine.build_packet("the cats brought a mouse again")
    finally:
        engine.close()
    # Graph file untouched by the live path…
    assert store.load_node("cats-and-prey").modified == before
    assert store.load_node("cats-and-prey").last_retrieved is None
    # …but the retrieval is logged for the librarian.
    assert "cats-and-prey" in store.pop_retrievals()


def test_expand_node_returns_neighbours_and_backlinks(store, cfg):
    _seed_graph(store)
    episode = store.load_node("ep-2026-08-19-mouse-incident")
    episode.add_link("related", "netherlands-trip")
    store.save_node(episode)
    engine = RecallEngine(store, cfg)
    try:
        expanded = engine.expand_node("ep-2026-08-19-mouse-incident")
    finally:
        engine.close()
    assert expanded["sources"] == ["2026-08-19/001"]
    neighbour_ids = {n["id"] for n in expanded["neighbours"]}
    assert "netherlands-trip" in neighbour_ids
    # cats-and-prey links to the episode in its body only, so no backlink;
    # netherlands-trip is linked forward. Missing node ids are flagged.
    assert expanded["confidence"] == 0.85


def test_search_nodes_covers_deep_layers(store, cfg):
    _seed_graph(store)
    engine = RecallEngine(store, cfg)
    try:
        engine.reindex()
        results = engine.search_nodes("live mouse in the bedroom at night")
        ids = [n.id for n, _ in results]
        assert "ep-2026-08-19-mouse-incident" in ids
    finally:
        engine.close()
