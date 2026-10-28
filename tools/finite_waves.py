"""Finite single-PPO-per-GPU queue. A failure freezes pending work for review.

This is an event-bound batch process, not a periodic task or a design agent.
"""
import json,pathlib,subprocess,time

def run_waves(out,state,root):
    from launch_cycle import save,usage
    running={};failed=False;deadline={}
    state['wave_pid']=__import__('os').getpid()
    while True:
        for name,proc in list(running.items()):
            j=state['train'][name];code=proc.poll();log=out/(name+'.log')
            if code is None:
                if j['status']=='initializing' and log.exists() and 'Learning iteration' in log.read_text(errors='replace'):
                    j['status']='ppo_started'
                if j['status']=='initializing' and time.monotonic()>deadline[name]:
                    # Never kill or reuse a hung allocation; the supervision event
                    # asks the current-round recovery path to inspect ownership.
                    state['wave_error']='Startup timeout '+name;save(out/'launch_status.json',state)
                    raise RuntimeError(state['wave_error'])
                continue
            end=out/(name+'.exit.json');record=json.loads(end.read_text()) if end.exists() else {}
            checkpoint=root/f"demo_project/runs/{name}/logs/{name}/model_{state['iterations']-1}.pt"
            valid=code==0 and record.get('exit_code')==0 and not record.get('guard_stopped') and not record.get('physics_error') and not record.get('runtime_error') and checkpoint.exists()
            j.update(status='training_complete' if valid else 'failed',exit_code=code,finished_at=time.strftime('%FT%T%z'))
            if not valid:failed=True;state['wave_error']='Failed training '+name
            del running[name]
        if failed:
            save(out/'launch_status.json',state)
            # Other token-owned jobs continue; queued jobs remain recoverable.
            raise RuntimeError(state['wave_error'])
        queued=[(k,v) for k,v in state['train'].items() if v['status']=='queued']
        if not queued and not running:break
        occupied={state['train'][k]['gpu'] for k in running};memory=usage()
        for name,j in queued:
            gpu=j['gpu'];total,used,_=memory[gpu]
            if gpu in occupied or used+j['start_reserve_mib']>=total*.78:continue
            with (out/(name+'_guard.log')).open('x') as log:
                proc=subprocess.Popen(j['command'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            j.update(pid=proc.pid,status='initializing',started_at=time.strftime('%FT%T%z'))
            running[name]=proc;deadline[name]=time.monotonic()+900;occupied.add(gpu)
        save(out/'launch_status.json',state);time.sleep(5)
    # All training is verified before final evaluation attaches. The last fixed
    # checkout supplies the immutable evaluation and watchdog implementations.
    name=next(reversed(state['train']));tools=root/'demo_project/runs'/name/'tools'
    with (out/'eval_queue.log').open('x') as log:
        proc=subprocess.Popen(['python3',str(tools/'evaluate_cycle.py'),'--root',str(root),'--name',state['name']],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    state.update(eval_pid=proc.pid,all_ppo_at=time.strftime('%FT%T%z'));save(out/'launch_status.json',state)
    with (out/'eval_watch.log').open('x') as log:
        subprocess.Popen(['python3',str(tools/'watch_cycle_evaluation.py'),str(out)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
