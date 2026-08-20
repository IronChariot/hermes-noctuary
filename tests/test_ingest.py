"""Backlog ingestion tests."""

from __future__ import annotations

import json
import sqlite3

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

    all_messages = parse_input(db_path)
    assert len(all_messages) == 3  # tool row excluded
    s1_only = parse_input(db_path, session_id="s1")
    assert len(s1_only) == 2
    assert s1_only[0].text == "first question"


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
