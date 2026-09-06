import copy
import json
from types import SimpleNamespace
import pytest
from agent.memory_manager import build_memory_context_block
from noctuary.passive import active_recall_ids, recent_context, PassiveRecallHook
from noctuary.recall import RecallPacket
from noctuary.config import NoctuaryConfig
from noctuary.judge import DailyBudget, validate_selection


def carrier(node='cultivar', legacy=False):
    line = f'- [gist | conf 0.60 | node {node} (surface)] test' if legacy else f'- [direct | gist | node {node} (concept)] test'
    packet = '## Noctuary passive recall\n' + line
    return {'role':'user','content':'hello','api_content':'hello\n\n'+build_memory_context_block(packet)}


@pytest.mark.parametrize('legacy',[True,False])
def test_active_only_sidecar_packets(legacy):
    msg = carrier(legacy=legacy)
    before = copy.deepcopy(msg)
    assert active_recall_ids([msg]) == {'cultivar'}
    assert msg == before
    assert active_recall_ids([{'role':'user','content':msg['api_content']}]) == set()
    assert active_recall_ids([{'role':'user','content':'Summary: Cultivar was discussed.'}]) == set()
    quoted = {'role':'user','content':msg['api_content']}
    quoted['api_content'] = quoted['content'] + '\n\nother hook note'
    assert active_recall_ids([quoted]) == set()
    assert active_recall_ids([json.loads(json.dumps(msg))]) == {'cultivar'}
    assert active_recall_ids([]) == set()  # dropped by compaction/undo


@pytest.mark.parametrize('prefix', ['  hello  ', '[2026-09-06 22:00:00 UTC] hello', '<system_note>old harness</system_note>hello'])
def test_restored_visible_transform_keeps_packet(prefix):
    from agent.memory_manager import sanitize_context
    from gateway.message_timestamps import render_user_content_with_timestamp
    msg = carrier()
    if prefix.startswith('['):
        prefix = render_user_content_with_timestamp('hello', 1700000000)
    msg['api_content'] = prefix + msg['api_content'][len('hello'):]
    # Corresponds to canonical DB replay, with gateway persist override.
    if prefix.startswith('<'):
        msg['content'] = sanitize_context(prefix).strip()
    assert active_recall_ids([msg]) == {'cultivar'}


def test_native_checkpoint_is_explicitly_unsupported(monkeypatch, tmp_path):
    import hermes_constants
    from noctuary.passive import has_native_checkpoint
    monkeypatch.setattr(hermes_constants, 'get_hermes_home', lambda: tmp_path)
    p = SimpleNamespace(_engine=SimpleNamespace(is_warm=True), _cfg=NoctuaryConfig(tmp_path), _last_recall_count=0)
    hook = PassiveRecallHook(p, 's', tmp_path)
    for items in ([{'type':'compaction','encrypted_content':'fixture'}], json.dumps([{'type':'compaction'}])):
        history = [carrier(), {'role':'assistant','content':'','codex_reasoning_items':items}]
        assert has_native_checkpoint(history)
        assert hook(session_id='s', user_message='What about Cultivar?', conversation_history=history) is None


def test_recent_context_never_reads_sidecar():
    assert 'node cultivar' not in recent_context([carrier()], 'new question')


def test_hook_uses_local_history_and_guards(monkeypatch,tmp_path):
    import hermes_constants
    import noctuary.selection as selection
    monkeypatch.setattr(hermes_constants,'get_hermes_home',lambda:tmp_path)
    cfg = NoctuaryConfig(tmp_path, {'recallJudge':'off'})
    engine = SimpleNamespace(is_warm=True)
    p = SimpleNamespace(_engine=engine,_cfg=cfg,_last_recall_count=0)
    hook = PassiveRecallHook(p,'s',tmp_path)
    seen = []
    def select(engine,query,**kwargs):
        seen.append(kwargs['exclude_ids'])
        return RecallPacket('## Noctuary passive recall\n- [direct | gist | node other (concept)] other', ['other'])
    monkeypatch.setattr(selection,'select_packet',select)
    args = dict(session_id='s',user_message='What should I do with this project?',conversation_history=[carrier()])
    assert hook(**args)['context'].startswith('<memory-context>')
    assert seen == [{'cultivar'}]
    assert hook(**dict(args,session_id='other')) is None
    assert hook(**dict(args,conversation_history=None)) is None
    assert hook(**dict(args,user_message=[{'type':'text','text':'hello'}])) is None
    assert hook(**dict(args,model='moa/test')) is None
    hook.close()
    assert hook(**args) is None


def test_hook_timeout_error_no_fallback(monkeypatch,tmp_path):
    import hermes_constants
    import noctuary.selection as selection
    monkeypatch.setattr(hermes_constants,'get_hermes_home',lambda:tmp_path)
    p = SimpleNamespace(_engine=SimpleNamespace(is_warm=True),_cfg=NoctuaryConfig(tmp_path),_last_recall_count=6)
    hook = PassiveRecallHook(p,'s',tmp_path)
    def fail(*a,**kw):raise TimeoutError()
    monkeypatch.setattr(selection,'select_packet',fail)
    assert hook(session_id='s',user_message='A significant question',conversation_history=[]) is None
    assert p._last_recall_count == 0


def test_budget_unknown_failure_retains_reservation(tmp_path):
    b = DailyBudget(tmp_path/'budget.sqlite')
    token = b.reserve('0.04','0.05')
    with pytest.raises(RuntimeError):b.reserve('0.02','0.05')
    b.settle(token,'0.001')
    b.reserve('0.02','0.05')
    with pytest.raises(ValueError):b.settle(token,'0.001')
    with pytest.raises(ValueError):b.reserve('NaN','0.05')


@pytest.mark.parametrize('data',[
    {'selections':[{'id':'invented','kind':'direct','relevance':3}]},
    {'selections':[{'id':'a','kind':'direct','relevance':True}]},
    {'selections':[{'id':'a','kind':'direct','relevance':3,'reason':'x'}]},
    {'selections':[], 'execute':'bad'},
    {'selections':[{'id':'a','kind':'bad','relevance':3}]},
])
def test_judge_rejects_bad_output(data):
    with pytest.raises((ValueError,TypeError)):validate_selection(data,[{'id':'a'}])


def test_usage_decay_can_be_disabled(store,cfg):
    from noctuary.store import Node
    from noctuary.librarian import _decay_pass
    store.save_node(Node(id='old',type='concept',accessibility=.15))
    store.record_retrieval(['old'])
    before = store.load_node('old').to_markdown()
    cfg.values['usageDecayEnabled'] = False
    assert _decay_pass(store,cfg,set()) == (0,0,[])
    assert store.load_node('old').to_markdown() == before
