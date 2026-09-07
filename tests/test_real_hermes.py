"""Real Hermes discovery, HTTP request/replay, DB restore. All data are fixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]


def test_real_hermes_recall_path():
    source = Path(os.environ.get('HERMES_AGENT_ROOT','/home/darten/.hermes/hermes-agent'))
    with tempfile.TemporaryDirectory(prefix='noctuary-real-') as tmp:
        root = Path(tmp)
        home = root/'profile'
        (home/'plugins').mkdir(parents=True)
        (root/'empty').mkdir()
        (home/'plugins/noctuary').symlink_to(REPO/'noctuary',target_is_directory=True)
        (home/'.env').write_text('OPENAI_API_KEY=local-fixture-only\n')
        (home/'config.yaml').write_text('''memory:
  provider: noctuary
  memory_enabled: false
  user_profile_enabled: false
model:
  streaming: false
  context_length: 128000
compression:
  enabled: false
agent:
  tool_use_enforcement: false
terminal:
  backend: local
curator:
  enabled: false
auxiliary:
  title_generation:
    enabled: false
fallback_models: []
''')
        (home/'noctuary.json').write_text(json.dumps({'embeddingModel':'hash','passiveRecallMode':'selective','recallJudge':'off','directRecallSimilarity':.25,'candidateSimilarity':.05,'ingestPlatforms':['discord']}))
        env={'PATH':'/usr/bin:/bin','HOME':tmp,'HERMES_HOME':str(home),'TMPDIR':tmp,
             'XDG_CACHE_HOME':str(root/'cache'),'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1',
             'PYTHONPATH':str(source),'LANG':'C.UTF-8','TZ':'UTC',
             'HERMES_BUNDLED_PLUGINS':str(root/'empty'),'HF_HUB_OFFLINE':'1'}
        run=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--probe'],
                           env=env,cwd=root,capture_output=True,text=True,timeout=90)
        assert run.returncode==0,run.stdout+'\n'+run.stderr
        assert 'REAL_NOCTUARY_OK' in run.stdout,run.stdout
        print(next(s for s in run.stdout.splitlines() if s.startswith('REAL_NOCTUARY_OK')))


def probe():
    import threading
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    root=Path(os.environ['HOME'])
    home=Path(os.environ['HERMES_HOME'])
    def audit(event,args):
        if event in ('socket.connect','socket.getaddrinfo'):
            address=args[1] if event=='socket.connect' else args
            host=address[0] if isinstance(address,tuple) else None
            if host not in ('127.0.0.1','localhost','::1'):raise PermissionError('external network forbidden')
        if event=='sqlite3.connect' and args[0]!=':memory:':
            if not Path(args[0]).absolute().is_relative_to(root):raise PermissionError('nonfixture database')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).absolute()
            mode,flags=args[1:3]
            writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR))
            sensitive=path.name in ('.env','auth.json','SOUL.md','MEMORY.md','USER.md')
            if not path.is_relative_to(root) and path!=Path('/dev/null') and (writing or sensitive):raise PermissionError('nonfixture state access')
    sys.addaudithook(audit)
    from hermes_cli import env_loader
    original=env_loader.load_hermes_dotenv
    def isolated_load_env(*,hermes_home=None,project_env=None,load_external_secrets=True):
        return original(hermes_home=hermes_home or home,project_env=None,load_external_secrets=False)
    env_loader.load_hermes_dotenv=isolated_load_env
    from run_agent import AIAgent
    from hermes_state import SessionDB
    from hermes_cli.plugins import get_plugin_manager
    wire=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            wire.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body={'id':'fixture','object':'chat.completion','model':'fixture','choices':[{'index':0,'message':{'role':'assistant','content':'I have considered the project.'},'finish_reason':'stop'}],'usage':{'prompt_tokens':10,'completion_tokens':8,'total_tokens':18}}
            data=json.dumps(body).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    db=SessionDB(home/'state.db')
    agent=None
    slow_calls=[]
    judge_calls=[]
    def make():
        a=AIAgent(base_url=f'http://127.0.0.1:{server.server_port}/v1',api_key='local-fixture-only',provider='custom',api_mode='chat_completions',model='fixture',platform='discord',session_id='selective-test',session_db=db,max_iterations=1,enabled_toolsets=[],skip_context_files=True,load_soul_identity=False,skip_memory=False,skip_background_review=True,save_trajectories=False,quiet_mode=True,fallback_model={},credential_pool=None,max_tokens=128,reasoning_config={'enabled':False})
        p=next(p for p in a._memory_manager._providers if p.name=='noctuary')
        assert p._hook_handle is not None
        assert p._passive_hook.provider is p  # actual memory-loader instance
        # Real host callback + selector + judge wrapper: the first request
        # must survive beyond both old 9-second and 10-second cutoffs.
        from importlib import import_module
        Judge=import_module(type(p).__module__+'.codex_judge').CodexJudge
        p._cfg.values.update(recallJudge='openai-codex',recallJudgeTimeoutSeconds=15,recallJudgeDailyCallLimit=None)
        p._recall_judge=Judge(p._cfg)
        def slow_request(query,recent,candidates):
            judge_calls.append(query)
            import time
            if not slow_calls:
                slow_calls.append(True)
                time.sleep(11)
            return [{'id':c['id'],'kind':'direct','relevance':3} for c in candidates]
        p._recall_judge._request=slow_request
        return a,p
    try:
        agent,p=make()
        # Import the node class from the actual dynamically loaded package.
        from importlib import import_module
        Node=import_module(type(p).__module__+'.store').Node
        p._store.save_node(Node(id='cultivar',type='concept',title='Cultivar',body='Cultivar selects generated writing using taste feedback.',topics=['cultivar','writing']))
        p._engine.reindex()
        # Soft gateway eviction can leave an old provider callback registered.
        # Recreate the same session before closing the previous agent.
        stale_agent = agent
        from gateway.run_agent_cache import GatewayAgentCacheMixin
        GatewayAgentCacheMixin()._release_evicted_agent_soft(stale_agent)
        agent,p=make();p._engine.warm()
        q='  How could Cultivar select generated writing using taste feedback?  '
        from gateway.message_timestamps import render_user_content_with_timestamp
        wire_q=render_user_content_with_timestamp(q,1700000000)
        r=agent.run_conversation(wire_q,persist_user_message=q,system_message='You are an offline test fixture.')
        assert r['completed'] and not r['failed'],r
        marker='| node cultivar ('
        users=lambda:[m for m in wire[-1]['messages'] if m['role']=='user']
        assert users()[-1]['content'].count(marker)==1, repr(users()[-1]['content'])
        assert len(judge_calls)==1, judge_calls
        # Closing the superseded owner must not unregister its replacement.
        stale_agent.close()
        first_api=users()[-1]['content']
        r2=agent.run_conversation(q,system_message='You are an offline test fixture.',conversation_history=r['messages'])
        assert r2['completed'] and not r2['failed']
        assert users()[-1]['content'].strip()==q.strip(), repr(users()[-1]['content'])
        assert sum(marker in m['content'] for m in users())==1
        assert users()[0]['content']==first_api
        assert p.prefetch(q)=='' and p._prefetch_cache is None
        # Real DB restore, new provider instance and hook ownership.
        agent.close();agent=None
        history=db.get_messages_as_conversation('selective-test')
        assert any(marker in (m.get('api_content') or '') for m in history)
        agent,p=make();p._engine.warm()
        r3=agent.run_conversation(q,system_message='You are an offline test fixture.',conversation_history=history)
        assert r3['completed'] and users()[-1]['content'].strip()==q.strip(), repr(users()[-1]['content'])
        assert sum(marker in m['content'] for m in users())==1
        # Replay the post-compaction shape: only vague summary remains. This is
        # an input-shape test, not a paid/natural compressor run.
        compacted=[{'role':'user','content':'Summary: Cultivar was discussed.'},{'role':'assistant','content':'Understood.'}]
        r4=agent.run_conversation(q,system_message='You are an offline test fixture.',conversation_history=compacted)
        assert r4['completed'] and marker in users()[-1]['content']
        p._turn_logger.flush()
        assert not any('Noctuary passive recall' in t.user for day in p._store.source_days() for t in p._store.read_turns(day))
        assert p._store._read_state().get('retrievals',{})=={}
        print('REAL_NOCTUARY_OK '+json.dumps({'requests':len(wire),'soft_eviction_single_packet_and_judge':True,'new_repeat_injections':0,'restart_suppressed':True,'summary_only_reeligible':True,'archive_clean':True,'same_session':'selective-test'}))
    finally:
        if agent:agent.close()
        db.close();get_plugin_manager().unload();server.shutdown();server.server_close()

if __name__=='__main__':probe()
