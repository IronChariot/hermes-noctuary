"""LLM access for the nightly librarian.

Routed through ``agent.auxiliary_client.call_llm`` — the host-owned call path
that already handles every provider, auth, and fallback chain Hermes
supports. With no overrides, calls resolve to the agent's main configured
model; ``librarianProvider`` / ``librarianModel`` in noctuary.json select any
other already-configured Hermes model (requirements section 7).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .config import NoctuaryConfig

logger = logging.getLogger(__name__)


def librarian_chat(
    cfg: NoctuaryConfig,
    messages: List[Dict[str, str]],
    *,
    max_tokens: Optional[int] = None,
    temperature: float = 0.3,
) -> str:
    """One chat completion with the librarian's configured model."""
    from agent.auxiliary_client import call_llm

    kwargs: Dict[str, Any] = {}
    model = cfg.get_str("librarianModel").strip()
    provider = cfg.get_str("librarianProvider").strip()
    if model:
        kwargs["model"] = model
    if provider:
        kwargs["provider"] = provider

    response = call_llm(
        messages=messages,
        max_tokens=max_tokens or cfg.get_int("librarianMaxTokens"),
        temperature=temperature,
        timeout=cfg.get_float("llmTimeoutSeconds"),
        **kwargs,
    )
    text = extract_text(response)
    if not text.strip():
        raise RuntimeError("librarian LLM call returned empty text")
    return text


def extract_text(response: Any) -> str:
    """Pull the assistant text out of an OpenAI-shaped response (obj or dict)."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    try:
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                return str(message.get("content") or "")
            return ""
        choices = getattr(response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            return str(content or "")
    except Exception as exc:
        logger.warning("noctuary: could not extract LLM response text: %s", exc)
    return ""


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class JsonReplyError(ValueError):
    """A model reply contained JSON-looking text that could not be decoded."""


def _structure_hint(candidate: str) -> str:
    """Describe unmatched JSON delimiters without modifying the response."""
    pairs = {"{": "}", "[": "]"}
    stack: List[tuple[str, int]] = []
    in_string = False
    escaped = False
    for index, char in enumerate(candidate):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in pairs:
            stack.append((char, index))
        elif char in ("}", "]"):
            if not stack:
                return f"unexpected closing {char!r} at character {index}"
            opener, opener_index = stack[-1]
            expected = pairs[opener]
            if char != expected:
                return (
                    f"mismatched closing {char!r} at character {index}; "
                    f"expected {expected!r} for {opener!r} at character {opener_index}"
                )
            stack.pop()
    if in_string:
        return "response ends inside a JSON string (possibly truncated)"
    if stack:
        expected = "".join(pairs[opener] for opener, _ in reversed(stack))
        return f"unclosed JSON delimiter(s); expected {expected!r} before the end"
    return "JSON delimiters are balanced; check commas, colons, quoting, or escapes"


def _top_level_json_starts(candidate: str) -> List[int]:
    """Find possible JSON starts while excluding arrays/objects nested in one."""
    pairs = {"{": "}", "[": "]"}
    stack: List[str] = []
    starts: List[int] = []
    in_string = False
    escaped = False
    for index, char in enumerate(candidate):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"' and stack:
            in_string = True
        elif char in pairs:
            if not stack:
                starts.append(index)
            stack.append(char)
        elif char in ("}", "]") and stack:
            if char == pairs[stack[-1]]:
                stack.pop()
            else:
                # Preserve the containing opener so nested fragments cannot be
                # mistaken for an independent valid response after a mismatch.
                continue
    return starts


def parse_json_reply(text: str) -> Any:
    """Parse a JSON object/array out of a model reply.

    Tolerates code fences and prose around the JSON. Raises ``ValueError``
    when nothing parseable is found — the librarian treats that as a failed
    pass, never as an empty result.
    """
    candidates: List[str] = []
    fenced = _FENCE_RE.findall(text)
    candidates.extend(fenced)
    candidates.append(text)

    decoder = json.JSONDecoder()
    best_error: Optional[tuple[int, json.JSONDecodeError, str]] = None
    found_opener = False
    for candidate in candidates:
        candidate = candidate.strip()
        starts = _top_level_json_starts(candidate)
        for start in starts:
            found_opener = True
            fragment = candidate[start:]
            try:
                value, _ = decoder.raw_decode(fragment)
                return value
            except json.JSONDecodeError as exc:
                progress = start + exc.pos
                if best_error is None or progress > best_error[0]:
                    best_error = (progress, exc, fragment)

    if not found_opener or best_error is None:
        raise JsonReplyError(
            f"reply contains no JSON object or array; excerpt: {text[:200]!r}"
        )

    _progress, exc, fragment = best_error
    near_start = max(0, exc.pos - 60)
    near_end = min(len(fragment), exc.pos + 60)
    near = fragment[near_start:near_end]
    raise JsonReplyError(
        f"JSON parse error: {exc.msg} at line {exc.lineno}, column {exc.colno} "
        f"(character {exc.pos}); {_structure_hint(fragment)}; near {near!r}"
    )


def clamp01(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
