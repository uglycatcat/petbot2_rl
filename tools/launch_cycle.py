"""Launch an explicit, gated fixed-version experiment; no automatic design decisions."""
import argparse,fcntl,json,os,pathlib,shutil,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[3]
RUNS=ROOT/'demo_project/runs';SOURCE=ROOT/'demo_project/upright_rl_demo'
def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def usage():
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return {int(i):(int(t),int(u),int(v)) for i,t,u,v in (r.split(',') for r in text.splitlines())}
def main():
    p=argparse.ArgumentParser();p.add_argument('plan',type=pathlib.Path);a=p.parse_args();plan=json.loads(a.plan.read_text())
    lock=(RUNS/'upright_launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    out=RUNS/('_'+plan['name']);out.mkdir(exist_ok=True)
    if (out/'launch_status.json').exists():raise RuntimeError('Already launched; inspect and recover existing jobs')
    for f in RUNS.glob('_*/launch_status.json'):
        previous=json.loads(f.read_text())
        for job in previous.get('train',{}).values():
            proc=pathlib.Path('/proc')/str(job.get('pid',0))/'cmdline'
            if proc.exists() and b'run_guarded.py' in proc.read_bytes():raise RuntimeError('Existing training job: '+str(f))
    commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    if commit!=plan['commit'] or subprocess.check_output(['git','-C',str(SOURCE),'status','--porcelain'],text=True).strip():raise RuntimeError('Source must be clean fixed commit')
    gate=json.loads((ROOT/plan['gate']).read_text())
    if gate['commit']!=commit or not gate['passed']:raise RuntimeError('Gate does not match release')
    tested=gate.get('tested_commit',gate['commit'])
    if subprocess.check_output(['git','-C',str(SOURCE),'diff',tested,commit,'--','train_cycle.py','upright','checks'],text=True):raise RuntimeError('Task changed since gate')
    for evidence in gate['smokes']:
        folder=ROOT/evidence['folder']
        metadata=json.loads((folder/'config.json').read_text())
        metrics=json.loads((ROOT/evidence['scalars']).read_text())
        if metadata['working_tree_dirty'] or metadata['contract']['fingerprint']!=gate['fingerprint']:raise RuntimeError('Smoke source mismatch')
        if subprocess.check_output(['git','-C',str(SOURCE),'diff',metadata['commit'],commit,'--','train_cycle.py','upright','checks'],text=True):raise RuntimeError('Smoke task source changed')
        if not (folder/f"model_{evidence['iterations']-1}.pt").is_file():raise RuntimeError('Incomplete smoke')
        if not metrics or not all(tags and all(v['finite'] for v in tags.values()) for tags in metrics.values()):raise RuntimeError('Nonfinite smoke')
    for evidence in gate['evaluations']:
        f=ROOT/evidence['path'];data=json.loads(f.read_text())
        if data['contract']['fingerprint']!=gate['fingerprint']:raise RuntimeError('Gate fingerprint mismatch')
        if data['takeover']!=64 or data['unfinished']:raise RuntimeError('Gate coverage failure')
        if data['balance']<evidence.get('min_balance',1):raise RuntimeError('No balance reached in gate')
    variants=plan['variants']
    if {v['gpu'] for v in variants}!={0,1,2,3}:raise RuntimeError('Four-GPU plan required')
    if len({v['run'] for v in variants})!=len(variants):raise RuntimeError('Duplicate run names')
    mem=usage();reservations={i:0 for i in range(4)}
    for v in variants:
        if (RUNS/v['run']).exists():raise RuntimeError('Run exists: '+v['run'])
        if not (ROOT/v['initial']).is_file():raise RuntimeError('Missing initialization checkpoint')
        reservations[v['gpu']]+=v['reserve_mib']
    for gpu,reserve in reservations.items():
        total,used,util=mem[gpu]
        if used+reserve>=total*.78:raise RuntimeError(f'GPU{gpu} capacity insufficient: {used}+{reserve} >= {total*.78}')
    state=dict(name=plan['name'],commit=commit,container=plan['container'],thread=os.environ.get('CODEX_THREAD_ID'),iterations=plan['iterations'],checkpoints=plan['checkpoints'],heldout_seed=plan.get('heldout_seed'),train={},gate=gate,resource_plan=reservations,started_at=time.strftime('%FT%T%z'))
    # Materialize all fixed copies before launching any training.
    for v in variants:
        dst=RUNS/v['run'];subprocess.run(['git','clone','--no-local',str(SOURCE),str(dst)],env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1'},check=True,stdout=subprocess.DEVNULL)
        subprocess.run(['git','-C',str(dst),'checkout','--detach',commit],check=True,stdout=subprocess.DEVNULL)
        shutil.copytree(SOURCE/'artifacts/petbot_asset',dst/'artifacts/petbot_asset')
    save(out/'launch_status.json',state)
    starting=[]
    for v in variants:
        dst=RUNS/v['run'];envelope=plan['envelope']+['--wheel-slew',str(v['wheel_slew']),'--frame-slew',str(v['frame_slew'])]
        initial=pathlib.Path(v['initial']).relative_to('demo_project/runs')
        cmd=['python3',str(dst/'tools/run_guarded.py'),'--container',plan['container'],'--gpu',str(v['gpu']),'--reserve-mib',str(v['reserve_mib']),'--workdir','/workspace/runs/'+v['run'],'--entrypoint','train_cycle.py','--log',str(out/(v['run']+'.log')),'--headless','--device','cuda:0','--num_envs',str(v['envs']),'--iterations',str(plan['iterations']),'--save-interval',str(plan['save_interval']),'--learning-rate',str(plan['learning_rate']),'--seed',str(v['seed']),'--initialize-from','/workspace/runs/'+str(initial),'--log_dir','/workspace/runs/'+v['run']+'/logs/'+v['run'],*envelope]
        with (out/(v['run']+'_guard.log')).open('w') as f:proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        state['train'][v['run']]=dict(pid=proc.pid,gpu=v['gpu'],envelope=envelope,command=cmd,status='initializing',num_envs=v['envs']);save(out/'launch_status.json',state)
        starting.append((v['run'],proc,time.monotonic()+900))
        # Wait for this allocation/PPO before another launch.
        deadline=time.monotonic()+900
        while True:
            if proc.poll() is not None:raise RuntimeError('Startup exit: '+v['run'])
            log=out/(v['run']+'.log')
            if log.exists() and 'Learning iteration' in log.read_text(errors='replace'):break
            if time.monotonic()>deadline:raise RuntimeError('Startup timeout')
            time.sleep(2)
        state['train'][v['run']]['status']='ppo_started';save(out/'launch_status.json',state)
    while starting:
        for run,proc,deadline in list(starting):
            if proc.poll() is not None:raise RuntimeError(f'{run} exited before PPO verification; inspect partial launch')
            log=out/(run+'.log')
            if log.exists() and 'Learning iteration' in log.read_text(errors='replace'):
                state['train'][run]['status']='ppo_started';save(out/'launch_status.json',state)
                starting.remove((run,proc,deadline))
            elif time.monotonic()>deadline:raise RuntimeError(f'{run} startup timed out; inspect existing processes')
        if starting:time.sleep(2)
    # Use the queue from the same fixed version as the experiment.
    with (out/'eval_queue.log').open('w') as f:proc=subprocess.Popen(['python3',str(dst/'tools/evaluate_cycle.py'),'--root',str(ROOT),'--name',plan['name']],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    state.update(eval_pid=proc.pid,all_ppo_at=time.strftime('%FT%T%z'));save(out/'launch_status.json',state)
    print(json.dumps({'all_ppo_at':state['all_ppo_at'],'eval_pid':proc.pid}))
if __name__=='__main__':main()
