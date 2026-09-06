"""Opt-in Luna subscription judge via Hermes' native Codex adapter.

No copied OAuth token, custom endpoint, paid fallback, tool access, or retry.
The subscription API rejects output-token ceilings: bound input, wall time,
local response size and daily REQUEST COUNT, not a fictitious dollar budget.
"""
from __future__ import annotations
import json
import time
from urllib.parse import urlsplit
from .judge import DailyBudget, OpenRouterJudge, SYSTEM, validate_selection

MODEL = 'gpt-5.6-luna'


class CodexJudge(OpenRouterJudge):
    def _request(self, query, recent_context, candidates):
        if not candidates:
            return []
        if len(candidates) > 12:
            raise ValueError('too many candidates')
        data = {'current_message': str(query)[:1500], 'recent_context': str(recent_context)[-2000:],
                'candidates': [{k: c[k] for k in ('id', 'type', 'title', 'excerpt')} for c in candidates]}
        user = json.dumps(data, ensure_ascii=False)
        if len(user.encode('utf-8')) > 18000:
            raise ValueError('judge input too large')
        # Explicit provider resolver, not call_llm's automatic fallback chain.
        from agent.auxiliary_client import resolve_provider_client
        client, resolved = resolve_provider_client('openai-codex', model=MODEL)
        if client is None or resolved != MODEL:
            raise RuntimeError('requested Codex model unavailable')
        url = urlsplit(str(client.base_url))
        if (url.scheme != 'https' or url.hostname != 'chatgpt.com'
                or not url.path.startswith('/backend-api/codex')):
            raise RuntimeError('unexpected Codex route')
        limit = self.cfg.values.get('recallJudgeDailyCallLimit', 200)
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError('invalid daily request limit')
        budget = DailyBudget(self.cfg.store_root/'recall-codex-calls.sqlite')
        call_id = budget.reserve(1, limit)  # Never refunded, even on errors/restarts.
        start = time.monotonic()
        self.last_metrics = {'call_id': call_id, 'model': MODEL, 'provider': 'openai-codex'}
        try:
            # Bypass SDK retries; native client cache retains ownership of original.
            from agent.auxiliary_client import CodexAuxiliaryClient
            leaf = client._real_client.with_options(max_retries=0)
            adapter = CodexAuxiliaryClient(leaf, MODEL)
            response = adapter.chat.completions.create(
                model=MODEL, messages=[{'role':'system','content':SYSTEM}, {'role':'user','content':user}],
                timeout=float(self.cfg.values.get('recallJudgeTimeoutSeconds', 5)),
                extra_body={'reasoning': {'effort': 'none'}})
            choice = response.choices[0]
            text = choice.message.content
            if choice.finish_reason != 'stop' or not isinstance(text, str) or len(text.encode()) > 4096:
                raise ValueError('incomplete or oversized judge response')
            selections = validate_selection(json.loads(text), candidates)
            usage = response.usage
            self.last_metrics.update(status='ok', selected=len(selections),
                                     input_tokens=getattr(usage,'prompt_tokens',None),
                                     output_tokens=getattr(usage,'completion_tokens',None))
            return selections
        except Exception as exc:
            self.last_metrics.update(status='error', error_type=type(exc).__name__)
            raise
        finally:
            self.last_metrics['seconds'] = round(time.monotonic()-start, 3)
