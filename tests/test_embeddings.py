"""Embedding and index tests (hash embedder — no model downloads)."""

from __future__ import annotations

import math

from noctuary.embeddings import EmbeddingIndex, HashEmbedder, get_embedder, reindex
from noctuary.store import Node


def test_hash_embedder_deterministic_and_normalized():
    embedder = HashEmbedder()
    a1, a2 = embedder.embed(["the cats caught a mouse", "the cats caught a mouse"])
    assert a1 == a2
    norm = math.sqrt(sum(v * v for v in a1))
    assert abs(norm - 1.0) < 1e-6


def test_hash_embedder_related_text_scores_higher():
    embedder = HashEmbedder()
    query, related, unrelated = embedder.embed([
        "cats brought a live mouse into the bedroom",
        "the cats hunt mice at night and bring them into the bedroom",
        "quarterly tax filing deadline spreadsheet",
    ])
    sim_related = sum(a * b for a, b in zip(query, related))
    sim_unrelated = sum(a * b for a, b in zip(query, unrelated))
    assert sim_related > sim_unrelated


def test_get_embedder_hash_and_fallback():
    assert get_embedder("hash").name == "hash"
    # A nonsense model name must fall back rather than raise.
    assert get_embedder("no-such-model-xyz").name == "hash"


def test_index_upsert_search_and_signature(tmp_path):
    index = EmbeddingIndex(tmp_path / "index.sqlite")
    embedder = HashEmbedder()
    index.reset_for_embedder(f"{embedder.name}:{embedder.dim}")

    texts = {
        "cats": "the cats hunt mice in the bedroom",
        "taxes": "tax filing spreadsheet",
    }
    for node_id, text in texts.items():
        index.upsert(node_id, "surface", False, "h-" + node_id,
                     embedder.embed([text])[0])
    assert index.count() == 2

    query_vec = embedder.embed(["cats and a mouse in the bedroom"])[0]
    results = index.search(query_vec, node_types=["surface"], top_k=2)
    assert results[0][0] == "cats"

    removed = index.remove_absent(["cats"])
    assert removed == 1
    assert index.count() == 1
    index.close()


def test_search_includes_pinned_of_other_types(tmp_path):
    index = EmbeddingIndex(tmp_path / "index.sqlite")
    embedder = HashEmbedder()
    vec = embedder.embed(["anything"])[0]
    index.upsert("surface-node", "surface", False, "h1", vec)
    index.upsert("pinned-concept", "concept", True, "h2", vec)
    index.upsert("plain-concept", "concept", False, "h3", vec)

    ids = [r[0] for r in index.search(vec, node_types=["surface"],
                                      include_pinned=True, top_k=10)]
    assert "surface-node" in ids
    assert "pinned-concept" in ids
    assert "plain-concept" not in ids
    index.close()


def test_reindex_from_store(store):
    embedder = HashEmbedder()
    index = EmbeddingIndex(store.root / "index.sqlite")
    store.save_node(Node(id="cats", type="concept", title="Cats",
                         body="the cats hunt mice"))
    store.save_node(Node(id="mice", type="surface", title="Mice",
                         body="mouse incidents summary"))

    updated, removed = reindex(store, index, embedder)
    assert (updated, removed) == (2, 0)
    # Second pass: nothing changed.
    assert reindex(store, index, embedder) == (0, 0)

    # Edit a node → exactly one re-embed.
    node = store.load_node("cats")
    node.body = "the cats hunt mice at night"
    store.save_node(node)
    assert reindex(store, index, embedder) == (1, 0)
    index.close()
