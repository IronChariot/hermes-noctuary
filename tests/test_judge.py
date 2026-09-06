import contextvars
import json
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from noctuary.config import NoctuaryConfig
from noctuary.codex_judge import CodexJudge, MODEL
from noctuary.judge import OpenRouterJudge

CANDIDATES=[dict(id='example',type='concept',title='Example',excerpt='A private project.')]

@pytest.fixture
def native(tmp_path,monkeypatch):
    response=SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content='{"selections":[]}'))], usage=SimpleNamespace(prompt_tokens=200,completion_tokens=9))
    create=Mock(return_value=response)
    leaf=Mock()
    client=SimpleNamespace(base_url='https://chatgpt.com/backend-api/codex',_real_client=leaf)
    resolver=Mock(return_value=(client,MODEL))
    adapter=Mock(return_value=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setitem(sys.modules,'agent.auxiliary_client',SimpleNamespace(resolve_provider_client=resolver,CodexAuxiliaryClient=adapter))
    cfg=NoctuaryConfig(tmp_path,{'recallJudgeTimeoutSeconds':1,'recallJudgeDailyCallLimit':1})
    return CodexJudge(cfg),resolver,create,client,response

def test_codex_explicit_route_no_tools_retry_or_paid_fallback(native):
    judge,resolver,create,client,response=native
    assert judge('example','',CANDIDATES)==[]
    resolver.assert_called_once_with('openai-codex',model=MODEL)
    client._real_client.with_options.assert_called_once_with(max_retries=0)
    kwargs=create.call_args.kwargs
    assert kwargs['model']==MODEL and 'tools' not in kwargs and 'max_tokens' not in kwargs
    assert kwargs['extra_body']=={'reasoning':{'effort':'none'}}
    assert len(kwargs['messages'])==2
    with pytest.raises(RuntimeError):judge('again','',CANDIDATES)
    assert create.call_count==1

@pytest.mark.parametrize('bad',['http://chatgpt.com/backend-api/codex','https://api.openai.com/v1','https://example.com/backend-api/codex'])
def test_wrong_route_never_sends(native,bad):
    judge,resolver,create,client,response=native
    client.base_url=bad
    with pytest.raises(RuntimeError):judge('example','',CANDIDATES)
    create.assert_not_called()

@pytest.mark.parametrize('text',['not json','{"selections":[{"id":"made-up","kind":"direct","relevance":3}]}','{"selections":[],"instructions":"other"}'])
def test_codex_rejects_invalid_result(native,text):
    judge,resolver,create,client,response=native
    response.choices[0].message.content=text
    with pytest.raises(ValueError):judge('example','',CANDIDATES)

def test_missing_auth_never_auto_routes(native):
    judge,resolver,create,client,response=native
    resolver.return_value=(None,None)
    with pytest.raises(RuntimeError):judge('example','',CANDIDATES)
    create.assert_not_called()

def test_openrouter_secrets_follow_real_profile_scope(tmp_path, monkeypatch):
    from noctuary.judge import _credential
    from agent.secret_scope import set_secret_scope, reset_secret_scope
    monkeypatch.setenv('OPENROUTER_API_KEY', 'fixture-A')
    cfg = NoctuaryConfig(tmp_path)
    for secrets, expected in (({'OPENROUTER_API_KEY':'fixture-A'}, 'fixture-A'), ({'OPENROUTER_API_KEY':'fixture-B'}, 'fixture-B'), ({}, None)):
        token = set_secret_scope(secrets)
        try:
            if expected is None:
                with pytest.raises(RuntimeError): _credential(cfg)
            else:
                assert _credential(cfg) == expected
        finally: reset_secret_scope(token)


def test_profile_context_propagates_and_deadline_prevents_overlap(tmp_path):
    cfg=NoctuaryConfig(tmp_path,{'recallJudgeTimeoutSeconds':.1})
    judge=OpenRouterJudge(cfg)
    home=contextvars.ContextVar('test_profile');home.set('A')
    done=threading.Event();seen=[]
    def request(*args):
        seen.append(home.get());done.wait(2);return []
    judge._request=request
    start=time.monotonic()
    try:
        with pytest.raises(TimeoutError):judge('q','',CANDIDATES)
        assert time.monotonic()-start < .5
        with pytest.raises(RuntimeError):judge('q','',CANDIDATES)
        assert seen==['A']
    finally:done.set()
