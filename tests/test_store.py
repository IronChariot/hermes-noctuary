"""Store tests: node roundtrip, source logs, state, git."""

from __future__ import annotations

from noctuary.store import Node, slugify


def test_node_roundtrip(store):
    node = Node(
        id="cats",
        type="concept",
        title="The cats",
        body="Sam's two cats.\n\nThey hunt [[mice]] at night.",
        confidence=0.8,
        salience=0.7,
        accessibility=0.9,
        pinned=True,
        topics=["cats", "pets"],
        participants=["Sam"],
        sources=["2026-08-19/001"],
    )
    node.add_link("related", "netherlands-trip")
    node.extra["entities"] = ["cats"]
    store.save_node(node)

    loaded = store.load_node("cats")
    assert loaded is not None
    assert loaded.title == "The cats"
    assert loaded.type == "concept"
    assert "[[mice]]" in loaded.body
    assert loaded.pinned is True
    assert loaded.confidence == 0.8
    assert loaded.links["related"] == ["netherlands-trip"]
    assert loaded.sources == ["2026-08-19/001"]
    assert loaded.topics == ["cats", "pets"]
    assert loaded.extra.get("entities") == ["cats"]
    assert loaded.created and loaded.modified


def test_add_link_dedupes_and_rejects_self(store):
    node = Node(id="a", type="concept", title="A")
    node.add_link("related", "b")
    node.add_link("related", "b")
    node.add_link("related", "a")
    node.add_link("bogus-kind", "c")
    assert node.links["related"] == ["b"]
    assert node.link_targets() == ["b"]


def test_unique_id_disambiguates(store):
    store.save_node(Node(id="cats", type="concept", title="Cats"))
    assert store.unique_id("cats") == "cats-2"


def test_slugify():
    assert slugify("The Mouse Incident!") == "the-mouse-incident"
    assert slugify("Ünïcode åccents") == "unicode-accents"
    assert slugify("") == "node"


def test_append_and_read_turns(store):
    ref1 = store.append_turn("the cats brought a mouse", "Again?!",
                             platform="discord", ts=1755600000)
    ref2 = store.append_turn("yes, again", "Poor Sam.",
                             platform="discord", ts=1755600100)
    date = ref1.split("/")[0]
    assert ref1.endswith("/001")
    assert ref2.endswith("/002")

    turns = store.read_turns(date)
    assert len(turns) == 2
    assert turns[0].user == "the cats brought a mouse"
    assert turns[0].assistant == "Again?!"
    assert turns[0].platform == "discord"
    assert store.count_user_turns(date) == 2

    found = store.get_source(ref2)
    assert found is not None
    assert found.user == "yes, again"
    assert store.get_source("2001-01-01/999") is None


def test_multiline_turn_content_survives(store):
    user = "line one\n\nline two with **bold**"
    assistant = "reply\nwith two lines"
    ref = store.append_turn(user, assistant, platform="discord", ts=1755600000)
    turn = store.get_source(ref)
    assert turn.user == user
    assert turn.assistant == assistant


def test_retrieval_state_roundtrip(store):
    store.record_retrieval(["cats", "cats", "mice"])
    popped = store.pop_retrievals()
    assert popped["cats"]["count"] == 2
    assert popped["mice"]["count"] == 1
    assert store.pop_retrievals() == {}


def test_consolidated_days(store):
    assert store.consolidated_days() == []
    store.mark_consolidated("2026-08-19")
    store.mark_consolidated("2026-08-19")
    assert store.consolidated_days() == ["2026-08-19"]


def test_git_commit(store):
    if not store.git_available():
        import pytest
        pytest.skip("git not installed")
    assert (store.root / ".git").exists()
    store.save_node(Node(id="cats", type="concept", title="Cats"))
    assert store.git_commit("noctuary: test commit") is True
    # Nothing changed — no empty commit.
    assert store.git_commit("noctuary: empty") is False


def test_state_files_gitignored(store):
    ignored = (store.root / ".gitignore").read_text(encoding="utf-8")
    for name in ("index.sqlite", "state.json", "consolidation.log"):
        assert name in ignored
