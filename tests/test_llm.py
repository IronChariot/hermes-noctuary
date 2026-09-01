"""JSON extraction and diagnostic tests for librarian model replies."""

from __future__ import annotations

import pytest

from noctuary.llm import JsonReplyError, parse_json_reply


def test_parse_json_reply_tolerates_fences_and_prose():
    reply = "Here is the result:\n```json\n{\"items\": [1, 2]}\n```\nDone."
    assert parse_json_reply(reply) == {"items": [1, 2]}


def test_parse_json_reply_diagnoses_truncated_string():
    with pytest.raises(JsonReplyError) as caught:
        parse_json_reply('{"surface_pages": [{"body": "cut off')

    message = str(caught.value)
    assert "JSON parse error" in message
    assert "inside a JSON string" in message
    assert "possibly truncated" in message


def test_parse_json_reply_diagnoses_mismatched_delimiter():
    with pytest.raises(JsonReplyError) as caught:
        parse_json_reply('{"surface_pages": []]')

    message = str(caught.value)
    assert "mismatched closing" in message
    assert "expected '}'" in message


def test_parse_json_reply_reports_missing_json():
    with pytest.raises(JsonReplyError, match="no JSON object or array"):
        parse_json_reply("I forgot to return the requested structure.")
