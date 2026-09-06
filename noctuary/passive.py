"""Active-sidecar-aware passive recall. No persistent seen set or transcript edits."""
from __future__ import annotations
import json
import logging
import re
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)
_BLOCK = re.compile(r'<memory-context>\n(.*?)\n</memory-context>', re.S)
_NODE = re.compile(r'^- \[(?:(?:gist|familiarity) \| conf [0-9.]+|(?:direct|association) \| (?:gist|familiarity)) \| node ([A-Za-z0-9_-]+) \(', re.M)


def active_recall_ids(history):
    """Trust only Noctuary framing in the *injected suffix* of active api_content.

    A user quoting a packet in visible content, a summary, raw node mention,
    or tool result is not a prior automatic recall. Legacy packets supported.
    """
    found = set()
    for msg in history:
        if not isinstance(msg, dict) or msg.get('role') != 'user':
            continue
        visible, api = msg.get('content'), msg.get('api_content')
        if not isinstance(visible, str) or not isinstance(api, str):
            continue
        # Restore sanitizes visible content and gateway timestamps are wire-only.
        # Recover the suffix boundary through those same canonical transforms;
        # never scan a block literally quoted in visible user content.
        from agent.memory_manager import sanitize_context
        from gateway.message_timestamps import strip_leading_message_timestamps
        def canonical(text):
            clean, _ = strip_leading_message_timestamps(text)
            return sanitize_context(clean).strip()
        visible_blocks = _BLOCK.findall(visible)
        for match in _BLOCK.finditer(api):
            block = match.group(1)
            if block in visible_blocks:
                continue
            if canonical(api[:match.start()]) != canonical(visible):
                continue
            if '\n\n## Noctuary passive recall\n' in block:
                found.update(_NODE.findall(block))
    return found


def recent_context(history, current_query):
    """Small clean conversational tail for anaphora, never old recall sidecars."""
    turns = []
    for msg in history:
        if not isinstance(msg, dict) or msg.get('role') not in ('user','assistant'):
            continue
        text = msg.get('content')
        if not isinstance(text, str) or msg.get('_compressed_summary') or '[CONTEXT COMPACTION' in text:
            continue
        if msg.get('role') == 'user' and text == current_query:
            continue
        if msg.get('tool_calls'):
            continue
        turns.append(f"{msg['role']}: {text[:800]}")
    return '\n'.join(turns[-3:])[-2000:]


def has_native_checkpoint(history):
    """Opaque wire pruning cannot be inferred from this hook's local history.

    Until Hermes exposes effective transport context, opt out for checkpoint
    carriers rather than falsely claiming local history equals active context.
    """
    for msg in history:
        if not isinstance(msg, dict):
            continue
        for key in ('codex_reasoning_items', 'codex_message_items'):
            items = msg.get(key)
            if isinstance(items, str):
                try:
                    items = json.loads(items)
                except ValueError:
                    continue
            if isinstance(items, list) and any(isinstance(x, dict) and x.get('type') == 'compaction' for x in items):
                return True
    return False


class PassiveRecallHook:
    def __init__(self, provider, session_id, home):
        self.provider = provider
        self.session_id = session_id
        self.home = Path(home).resolve()
        self.closed = False
        self.lock = threading.Lock()

    def close(self):
        self.closed = True

    def __call__(self, *, session_id='', user_message='', conversation_history=None, model='', **kwargs):
        from hermes_constants import get_hermes_home
        from agent.memory_provider import is_trivial_prompt
        from agent.memory_manager import build_memory_context_block
        p = self.provider
        if (self.closed or session_id != self.session_id or Path(get_hermes_home()).resolve() != self.home
                or not isinstance(user_message, str) or not isinstance(conversation_history, list)
                or is_trivial_prompt(user_message) or p._engine is None or not p._engine.is_warm):
            return None
        # The supported route persists string api_content. Explicitly reject known
        # opaque routes; multimodal current messages are rejected above.
        if (str(model).startswith(('moa/', 'codex_app_server', 'codex-app-server'))
                or has_native_checkpoint(conversation_history)):
            return None
        if not self.lock.acquire(blocking=False):
            return None
        started = time.monotonic()
        try:
            from .selection import select_packet
            excluded = active_recall_ids(conversation_history)
            cfg = p._cfg
            mode = cfg.values.get('recallJudge', 'off')
            judge = None
            if mode in ('openrouter', 'openai-codex'):
                if mode == 'openrouter':
                    from .judge import OpenRouterJudge as Judge
                else:
                    from .codex_judge import CodexJudge as Judge
                if getattr(p, '_recall_judge', None) is None:
                    p._recall_judge = Judge(cfg)
                judge = p._recall_judge
            elif mode != 'off':
                raise ValueError('unknown recall judge')
            # Only permit an optional association when recent active packets do
            # not already contain one. Deterministic, no random extra memories.
            recent_packets = [m.get('api_content','') for m in conversation_history[-16:] if isinstance(m,dict)]
            allow = cfg.values.get('allowAssociativeRecall', False) is True and not any(
                '[association |' in x for x in recent_packets if isinstance(x,str))
            packet = select_packet(p._engine, user_message, exclude_ids=excluded,
                                   recent_context=recent_context(conversation_history, user_message),
                                   judge=judge, allow_association=allow)
            elapsed = time.monotonic()-started
            if self.closed or session_id != self.session_id or elapsed > 9:
                return None
            # Diagnostic events are not retrieval/reinforcement events. No raw
            # query, memory excerpt, or credential is written to ordinary logs.
            logger.info('noctuary selective recall: excluded=%d selected=%d seconds=%.3f judge=%s',
                        len(excluded), packet.count, elapsed, mode)
            p._last_recall_count = packet.count
            if not packet.text:
                return None
            return {'context': build_memory_context_block(packet.text)}
        except Exception as exc:
            p._last_recall_count = 0
            logger.warning('noctuary selective recall skipped (%s)', type(exc).__name__)
            return None
        finally:
            self.lock.release()
