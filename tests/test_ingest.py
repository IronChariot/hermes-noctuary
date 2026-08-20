"""Backlog ingestion tests."""

from __future__ import annotations

import json
import sqlite3

import pytest

from noctuary.ingest import archive_messages, parse_input, run_ingest


def test_parse_text_transcript(tmp_path):
    transcript = tmp_path / "log.txt"
    transcript.write_text(
        "Sam: the cats brought a mouse\n"
        "Wren: Again?!\n"
        "Sam: yes\n"
        "continued on a second line\n"
        "Wren: Poor you.\n",
        encoding="utf-8",
    )
    messages = parse_input(transcript)
    assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"]
    assert messages[2].text == "yes\ncontinued on a second line"


def test_parse_json_and_jsonl(tmp_path):
    payload = [
        {"role": "user", "content": "hello", "timestamp": 1755550000},
        {"role": "assistant", "content": "hi"},
        {"role": "tool", "content": "ignored"},
        {"role": "user", "content": [{"type": "text", "text": "multi"},
                                     {"type": "image", "url": "x"}]},
    ]
    json_file = tmp_path / "log.json"
    json_file.write_text(json.dumps({"messages": payload}), encoding="utf-8")
    messages = parse_input(json_file)
    assert len(messages) == 3
    assert messages[0].ts == 1755550000
    assert messages[2].text == "multi"

    jsonl_file = tmp_path / "log.jsonl"
    jsonl_file.write_text("\n".join(json.dumps(m) for m in payload),
                          encoding="utf-8")
    assert len(parse_input(jsonl_file)) == 3


def test_parse_hermes_db(tmp_path):
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, "
                 "session_id TEXT, role TEXT, content TEXT, timestamp REAL)")
    rows = [
        ("s1", "user", "first question", 1755550000.0),
        ("s1", "assistant", "first answer", 1755550005.0),
        ("s2", "user", "other session", 1755550010.0),
        ("s1", "tool", "tool output", 1755550015.0),
    ]
    conn.executemany(
        "INSERT INTO messages (session_id, role, content, timestamp) "
        "VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()

    with pytest.raises(ValueError, match="multiple conversational sessions"):
        parse_input(db_path)
    s1_only = parse_input(db_path, session_id="s1")
    assert len(s1_only) == 2
    assert s1_only[0].text == "first question"


def test_parse_hermes_db_filters_scaffolds_and_replay_duplicates(tmp_path):
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, "
        "role TEXT, content TEXT, timestamp REAL, platform_message_id TEXT, "
        "display_kind TEXT, active INTEGER, compacted INTEGER, observed INTEGER)"
    )
    substantive = "A genuine Discord message that is long enough to be substantive " * 2
    second_platform = "A second genuine platform message"
    answer = "A genuine assistant response that was replayed during migration " * 2
    rows = [
        ("s1", "user", substantive, 10.0, "discord-1", None, 0, 1, 0),
        ("s1", "user", second_platform, 10.5, "discord-2", None, 0, 1, 0),
        # Replay copy lost its platform id; the authoritative Discord row wins.
        ("s1", "user", substantive, 11.0, None, None, 1, 0, 0),
        ("s1", "user", substantive + "\n\n" + second_platform,
         11.5, None, None, 1, 0, 0),
        ("s1", "assistant", answer, 12.0, None, None, 0, 1, 0),
        ("s1", "assistant", answer, 99.0, None, None, 1, 0, 0),
        ("s1", "user", "[Recent Summary (d0, node 1)] generated", 13.0, None, None, 1, 0, 0),
        ("s1", "user", "  [CONTEXT SUMMARY]: generated", 14.0, None, None, 1, 0, 0),
        ("s1", "assistant", " [CONTEXT SUMMARY]: generated", 14.5, None, None, 1, 0, 0),
        ("s1", "user", "[ASYNC DELEGATION BATCH COMPLETE — x] generated", 15.0, None, None, 1, 0, 0),
        ("s1", "assistant", "interrupted", 16.0, None, "hidden", 0, 1, 0),
        ("s1", "assistant", "timeline reaction", 17.0, None, "reaction", 1, 0, 0),
        ("s1", "user", "undone substantive row", 18.0, None, None, 0, 0, 0),
        ("s1", "user", "observed channel context", 19.0, None, None, 1, 0, 1),
        # Identical short messages at distinct times can be genuine repeats.
        ("s1", "user", "okay", 20.0, None, None, 1, 0, 0),
        ("s1", "user", "okay", 21.0, None, None, 1, 0, 0),
        # But an exact same-time replay is collapsed.
        ("s1", "user", "same-time", 30.0, None, None, 1, 0, 0),
        ("s1", "user", "same-time", 30.0, None, None, 1, 0, 0),
    ]
    conn.executemany(
        "INSERT INTO messages (session_id, role, content, timestamp, "
        "platform_message_id, display_kind, active, compacted, observed) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
    )
    conn.commit()
    conn.close()

    messages = parse_input(db_path, session_id="s1", dedupe_replays=True)
    assert [(m.role, m.text) for m in messages] == [
        ("user", substantive),
        ("user", second_platform),
        ("assistant", answer),
        ("user", "okay"),
        ("user", "okay"),
        ("user", "same-time"),
    ]


def test_parse_hermes_db_uses_id_order_not_regressing_timestamps(tmp_path):
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, "
                 "session_id TEXT, role TEXT, content TEXT, timestamp REAL)")
    conn.execute("INSERT INTO messages VALUES (1, 's1', 'user', 'question', 20)")
    conn.execute("INSERT INTO messages VALUES (2, 's1', 'assistant', 'answer', 10)")
    conn.commit()
    conn.close()

    messages = parse_input(db_path, session_id="s1")
    assert [(m.role, m.text) for m in messages] == [
        ("user", "question"), ("assistant", "answer"),
    ]


def test_replay_dedupe_is_explicit_for_distinct_timestamp_text(tmp_path):
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, "
                 "session_id TEXT, role TEXT, content TEXT, timestamp REAL)")
    text = "A deliberately repeated long message " * 4
    conn.execute("INSERT INTO messages VALUES (1, 's1', 'user', ?, 10)", (text,))
    conn.execute("INSERT INTO messages VALUES (2, 's1', 'user', ?, 20)", (text,))
    conn.commit()
    conn.close()

    assert len(parse_input(db_path, session_id="s1")) == 2
    assert len(parse_input(
        db_path, session_id="s1", dedupe_replays=True,
    )) == 1


def test_archive_messages_pairs_turns(store, tmp_path):
    from noctuary.ingest import RawMessage
    messages = [
        RawMessage("user", "q1", 1755550000.0),
        RawMessage("assistant", "a1", 1755550005.0),
        RawMessage("user", "part 1", None),
        RawMessage("user", "part 2", None),
        RawMessage("assistant", "a2", 1755550100.0),
    ]
    counts = archive_messages(store, messages, default_date="2026-08-10")
    assert sum(counts.values()) == 2
    day = list(counts)[0]
    turns = store.read_turns(day)
    assert turns[1].user == "part 1\n\npart 2"
    assert turns[1].assistant == "a2"


def test_run_ingest_archives_without_llm(store, cfg, tmp_path):
    transcript = tmp_path / "log.txt"
    transcript.write_text("Sam: hello\nWren: hi\n", encoding="utf-8")
    counts = run_ingest(store, cfg, transcript,
                        default_date="2026-08-01",
                        consolidate_after=False)
    assert counts == {"2026-08-01": 1}
    assert store.source_days() == ["2026-08-01"]
    # Idempotence is manual: a second run appends again (append-only store),
    # so the command is meant to run once per backlog file.
