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
    for candidate in candidates:
        candidate = candidate.strip()
        for opener in ("{", "["):
            start = candidate.find(opener)
            if start < 0:
                continue
            try:
                value, _ = decoder.raw_decode(candidate[start:])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in model reply: {text[:200]!r}")


def clamp01(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default
