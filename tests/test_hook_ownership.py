"""Callback ownership, independent of active-memory suppression."""
import threading
from types import SimpleNamespace

from noctuary import NoctuaryProvider
from noctuary.config import NoctuaryConfig
from noctuary.passive import PassiveRecallHook
from noctuary.recall import RecallPacket


class Handle:
    def __init__(self, callbacks, callback):
        self.callbacks, self.callback = callbacks, callback
        self.disposals = 0

    def dispose(self):
        self.disposals += 1
        if self.callback in self.callbacks:
            self.callbacks.remove(self.callback)


def test_provider_replacement_switch_and_failed_registration(tmp_path):
    callbacks = []
    def register(name, callback):
        assert name == 'pre_llm_call'
        callbacks.append(callback)
        return Handle(callbacks, callback)
    def make(home, session):
        p = NoctuaryProvider(NoctuaryConfig(home, {'passiveRecallMode':'selective'}))
        p._hook_registrar = register
        p._install_passive_hook(session)
        return p
    first = make(tmp_path, 's')
    old_handle = first._hook_handle
    sibling = make(tmp_path/'sibling', 's')
    other_session = make(tmp_path, 'other')
    second = make(tmp_path, 's')
    try:
        assert first._passive_hook.closed and old_handle.disposals == 1
        assert callbacks == [sibling._passive_hook, other_session._passive_hook, second._passive_hook]
        first.shutdown()  # late cleanup cannot close the new owner
        assert old_handle.disposals == 1 and second._passive_hook._is_current()
        failed = NoctuaryProvider(NoctuaryConfig(tmp_path, {'passiveRecallMode':'selective'}))
        failed._hook_registrar = lambda *a: None
        failed._install_passive_hook('s')
        assert failed._passive_hook.closed and second._passive_hook._is_current()
        second.on_session_switch('other')
        assert other_session._passive_hook.closed
        assert callbacks == [sibling._passive_hook, second._passive_hook]
        second.on_session_switch('s')
        second._install_passive_hook('s')  # same instance reinstallation
        assert callbacks == [sibling._passive_hook, second._passive_hook]
    finally:
        for p in (first, sibling, other_session, second):
            p.shutdown()
    assert callbacks == []


def test_inflight_superseded_hook_discards_result(monkeypatch, tmp_path):
    import hermes_constants
    import noctuary.selection as selection
    monkeypatch.setattr(hermes_constants, 'get_hermes_home', lambda: tmp_path)
    callbacks, results, calls = [], [], []
    entered, release = threading.Event(), threading.Event()
    def make():
        p = SimpleNamespace(_engine=SimpleNamespace(is_warm=True),
                            _cfg=NoctuaryConfig(tmp_path, {'recallJudge':'off'}),
                            _last_recall_count=0)
        h = PassiveRecallHook(p, 's', tmp_path)
        callbacks.append(h)
        h.bind(Handle(callbacks, h))
        return h
    def select(*a, **kw):
        calls.append(True)
        entered.set()
        assert release.wait(10)
        return RecallPacket('## Noctuary passive recall\nfixture', ['fixture'])
    monkeypatch.setattr(selection, 'select_packet', select)
    old = make()
    args = dict(session_id='s', user_message='Tell me about the project', conversation_history=[])
    worker = threading.Thread(target=lambda: results.append(old(**args)))
    worker.start()
    new = None
    try:
        assert entered.wait(10)
        new = make()
        release.set()
        worker.join(10)
        assert not worker.is_alive() and results == [None]
        assert old(**args) is None and len(calls) == 1
        assert new(**args)['context'].startswith('<memory-context>')
        assert len(calls) == 2
    finally:
        release.set()
        worker.join(10)
        old.close()
        if new:
            new.close()
    assert callbacks == []
