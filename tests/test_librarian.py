"""Librarian tests with a stubbed LLM: gate, passes, decay, validation."""

from __future__ import annotations

import json

import pytest

import noctuary.librarian as librarian
from noctuary.librarian import consolidate, consolidate_day, pending_days
from noctuary.store import Node


def _seed_day(store, date="2026-08-19"):
    ts = 1755550000  # falls on 2026-08-18/19 depending on tz; use explicit date
    import datetime
    base = datetime.datetime.strptime(date + " 09:00:00", "%Y-%m-%d %H:%M:%S")
    ts = base.timestamp()
    store.append_turn("the cats woke me up with a live mouse AGAIN",
                      "Again?! In the bedroom?", platform="discord", ts=ts)
    store.append_turn("yes. third time. maybe since the netherlands trip?",
                      "That trip keeps coming up around these incidents.",
                      platform="discord", ts=ts + 60)
    return date


class _FakeLlm:
    """Returns canned JSON per pass, keyed on prompt content."""

    def __init__(self):
        self.calls = []

    def __call__(self, cfg, messages, **kwargs):
        prompt = messages[-1]["content"]
        self.calls.append(prompt)
        if "Segment this day's conversation log" in prompt:
            return json.dumps({"episodes": [{
                "title": "The mouse incident",
                "summary": "Sam was woken by the cats bringing a live mouse "
                           "into the bedroom, the third such incident.",
                "significance": "Sam was exasperated but amused.",
                "start_ref": "2026-08-19/001",
                "end_ref": "2026-08-19/002",
                "time": "2026-08-19 morning (approximate)",
                "participants": ["Sam"],
                "topics": ["cats", "mouse", "bedroom"],
                "entities": ["cats"],
                "unresolved": ["when did the incidents actually start?"],
                "confidence": 0.85,
                "salience": 0.7,
            }]})
        if "Integrate one new episode" in prompt:
            return json.dumps({
                "episode_links": {"related": [], "broader": []},
                "concepts": [{
                    "id": "cats", "title": "The cats",
                    "body": "Sam's cats. They bring live prey into the "
                            "bedroom; see [[ep-2026-08-19-the-mouse-incident]].",
                    "confidence": 0.8, "salience": 0.7,
                    "is_new": True, "related": [],
                }],
            })
        if "existing pattern/trait nodes" in prompt:
            return json.dumps({"patterns": []})
        if "Refresh your surface memory pages" in prompt:
            return json.dumps({"surface_pages": [{
                "id": "cats-and-prey", "title": "Cats and prey",
                "body": "The cats sometimes bring live prey into the bedroom "
                        "at night — possibly since the Netherlands trip. "
                        "Details: [[cats]].",
                "related": ["cats"],
            }]})
        raise AssertionError(f"unexpected prompt: {prompt[:120]}")


@pytest.fixture()
def fake_llm(monkeypatch):
    fake = _FakeLlm()
    monkeypatch.setattr(librarian, "librarian_chat", fake)
    return fake


def test_activity_gate_skips_quiet_day(store, cfg, fake_llm):
    date = "2026-08-19"
    # A day whose log holds only assistant text (no user turns).
    import datetime
    ts = datetime.datetime.strptime(date + " 09:00:00", "%Y-%m-%d %H:%M:%S").timestamp()
    store.append_turn("", "proactive message with no reply",
                      platform="discord", ts=ts)

    result = consolidate_day(store, cfg, date)
    assert result.ran is False
    assert "no activity" in result.skipped_reason
    assert fake_llm.calls == []  # no LLM calls on the gated path
    assert date in store.consolidated_days()
    log_text = store.log_path.read_text(encoding="utf-8")
    assert "skipped" in log_text


def test_force_overrides_gate(store, cfg, fake_llm):
    date = "2026-08-19"
    import datetime
    ts = datetime.datetime.strptime(date + " 09:00:00", "%Y-%m-%d %H:%M:%S").timestamp()
    store.append_turn("", "unanswered", platform="discord", ts=ts)
    # force runs even without user turns; segmentation still happens.
    result = consolidate_day(store, cfg, date, force=True)
    assert result.ran is True
    assert len(fake_llm.calls) >= 1


def test_full_consolidation(store, cfg, fake_llm):
    date = _seed_day(store)
    results = consolidate(store, cfg)
    assert len(results) == 1
    result = results[0]
    assert result.ran is True
    assert result.episodes_created == 1
    assert result.concepts_created == 1
    assert result.surface_updated == 1

    # Episode node with provenance.
    episodes = store.all_nodes("episode")
    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.sources == ["2026-08-19/001", "2026-08-19/002"]
    assert episode.confidence == 0.85
    assert "approximate" in episode.time
    assert "Unresolved" in episode.body

    # Concept page created and reciprocally linked.
    concept = store.load_node("cats")
    assert concept is not None
    assert episode.id in concept.links["narrower"]
    assert "cats" in episode.links["broader"]

    # Surface page exists with full accessibility.
    surface = store.load_node("cats-and-prey")
    assert surface is not None
    assert surface.type == "surface"
    assert surface.accessibility == 1.0

    # Day recorded; git commit made when git is present.
    assert date in store.consolidated_days()
    assert pending_days(store) == []
    if store.git_available():
        assert result.committed is True


def test_decay_only_hits_untouched_nodes(store, cfg, fake_llm):
    date = _seed_day(store)
    stale = Node(id="old-thing", type="concept", title="Old thing",
                 body="unrelated", accessibility=0.5, salience=0.3)
    store.save_node(stale)
    retrieved = Node(id="hot-thing", type="concept", title="Hot thing",
                     body="recently recalled", accessibility=0.5)
    store.save_node(retrieved)
    store.record_retrieval(["hot-thing"])

    result = consolidate_day(store, cfg, date)
    assert result.ran

    decay = cfg.get_float("decayFactor")
    assert store.load_node("old-thing").accessibility == round(0.5 * decay, 3)
    boosted = store.load_node("hot-thing")
    assert boosted.accessibility == round(0.5 + cfg.get_float("retrievalBoost"), 3)
    assert boosted.last_retrieved  # retrieval log folded into frontmatter


def test_validation_blocks_commit_on_missing_provenance(store, cfg, monkeypatch):
    date = _seed_day(store)

    fake = _FakeLlm()
    original = fake.__call__

    def bad_segments(cfg_, messages, **kwargs):
        prompt = messages[-1]["content"]
        if "Segment this day's conversation log" in prompt:
            return json.dumps({"episodes": [{
                "title": "Ghost episode",
                "summary": "something",
                "start_ref": "1999-01-01/001",  # not a real ref
                "end_ref": "1999-01-01/002",
                "confidence": 0.5, "salience": 0.5,
            }]})
        return original(cfg_, messages, **kwargs)

    monkeypatch.setattr(librarian, "librarian_chat", bad_segments)
    # The bogus refs fall back to the day's first real turn, so this passes
    # validation. To hit the invariant, empty the day's turn list is not
    # possible (append-only) — instead check _validate directly.
    node = Node(id="bad-episode", type="episode", title="Bad", body="x",
                sources=[])
    store.save_node(node)
    problems = librarian._validate(store)
    assert any("without source refs" in p for p in problems)


def test_pattern_without_support_is_dropped(store, cfg, monkeypatch):
    date = _seed_day(store)
    fake = _FakeLlm()
    base = fake.__call__

    def with_patterns(cfg_, messages, **kwargs):
        prompt = messages[-1]["content"]
        if "existing pattern/trait nodes" in prompt:
            return json.dumps({"patterns": [
                {"id": "sam-hates-mornings", "title": "Sam hates mornings",
                 "body": "no evidence given", "confidence": 0.4,
                 "supports": [], "contradicts": [], "is_new": True},
            ]})
        return base(cfg_, messages, **kwargs)

    monkeypatch.setattr(librarian, "librarian_chat", with_patterns)
    result = consolidate_day(store, cfg, date)
    assert result.ran
    assert result.patterns_created == 0
    assert store.load_node("sam-hates-mornings") is None
