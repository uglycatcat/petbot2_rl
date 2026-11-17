"""One container worker: fixed-source training, independent evaluations, then review."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path,value):
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(value,indent=2)+'\n'); temp.replace(path)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--job',required=True); args=p.parse_args()
    job=json.loads(Path(args.job).read_text()); source=Path(__file__).resolve().parents[1]
    run=Path(args.job).resolve().parent/job['name']; run.mkdir(exist_ok=True)
    lock=(run/'worker.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status=dict(status='starting',worker_pid=os.getpid(),job=job,started_at=time.time())
    path=run/'worker_status.json'
    def memory_free():
        result=subprocess.run(['nvidia-smi','-i',job['gpu_uuid'],'--query-gpu=memory.free',
                               '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        return int(result.stdout.strip().splitlines()[0])
    def enabled():
        control=Path(args.job).resolve().parent/'control.json'
        return not control.exists() or json.loads(control.read_text()).get('enabled',False)
    def execute(phase,command):
        status.update(status='running',phase=phase,command=command)
        log_path=run/f'{phase}.log'; log_offset=log_path.stat().st_size if log_path.exists() else 0
        fatal_patterns=('PhysX error:', 'CUDA error:', 'GPU contact buffer overflow', 'Fatal Python error:', 'Traceback (most recent call last):')
        log_tail=''
        def physics_failure():
            nonlocal log_offset,log_tail
            if not log_path.exists(): return None
            with log_path.open(errors='replace') as reader:
                reader.seek(log_offset); new=reader.read(); log_offset=reader.tell()
            combined=log_tail+new; log_tail=combined[-128:]
            return next((term for term in fatal_patterns if term in combined),None)
        with (run/f'{phase}.log').open('a') as log:
            proc=subprocess.Popen(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            status['child_pid']=proc.pid; write(path,status)
            while proc.poll() is None:
                try:
                    free=memory_free(); status['card_free_mib']=free; status['heartbeat']=time.time()
                    write(path,status)
                    reason=None
                    if not enabled(): reason='control_disabled'
                    elif free<job['reserve_mib']: reason='card_memory_guard'
                    elif physics_failure(): reason='simulation_log_error'
                    if reason:
                        try: os.killpg(proc.pid,signal.SIGTERM)
                        except ProcessLookupError: pass
                        try: proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid,signal.SIGKILL); proc.wait()
                        status.update(status='interrupted',reason=reason,exit_code=proc.returncode)
                        write(path,status); return reason
                except subprocess.CalledProcessError:
                    # A transient telemetry error is not proof that the child stopped.
                    status['telemetry_error_at']=time.time(); write(path,status)
                time.sleep(5)
        status['exit_code']=proc.returncode; write(path,status)
        if physics_failure(): raise RuntimeError(f'{phase} emitted a physics/CUDA error even though its process exited')
        if proc.returncode: raise RuntimeError(f'{phase} exited {proc.returncode}; inspect its log')
        return None
    try:
        if not enabled(): raise RuntimeError('Batch control is disabled')
        while memory_free()<job['expected_peak_mib']+job['reserve_mib']:
            status.update(status='waiting_for_memory',heartbeat=time.time(),card_free_mib=memory_free())
            write(path,status)
            if not enabled(): raise RuntimeError('Batch control disabled while waiting')
            time.sleep(10)
        train=[sys.executable,'train.py','--headless','--num-envs',str(job['num_envs']),
               '--iterations',str(job['iterations']),'--steps',str(job['steps']),
               '--seed',str(job['seed']),'--entropy',str(job['entropy']),
               '--action-mode',job.get('action_mode','absolute'),
               '--delta-scale',str(job.get('delta_scale',.12)),
               '--noise-rho',str(job.get('noise_rho',0.)),
               '--upright-reward',str(job.get('upright_reward',0.)),
               '--save-every','100','--run-dir',str(run/'train')]
        interruption=execute('train',train)
        if interruption:
            # Keep the exact same run pending. No new trial or arbitrary lower budget is substituted.
            return
        final=json.loads((run/'train/status.json').read_text())
        if final['status']!='completed' or final['iteration']!=job['iterations']:
            raise RuntimeError('Training process exited without completing the requested budget')
        checkpoints=sorted((run/'train').glob('update_*.pt'))
        if not checkpoints: raise RuntimeError('Training did not produce checkpoints')
        for cp in checkpoints:
            interruption=execute(f'mujoco_{cp.stem}',[sys.executable,'evaluate_mujoco.py',
                '--checkpoint',str(cp),'--output',str(run/'evaluations'/cp.stem)])
            if interruption: return
            result=json.loads((run/'evaluations'/cp.stem).with_suffix('.json').read_text())
            if result['status']!='completed': raise RuntimeError('MuJoCo evaluation did not finish')
        interruption=execute('select',[sys.executable,'tools/select_candidates.py','--run-dir',str(run)])
        if interruption: return
        bundle=run/'best_per_pose.pt'
        interruption=execute('physx_best',[sys.executable,'evaluate.py','--headless','--checkpoint',str(bundle),
                                          '--output',str(run/'evaluations/physx_best')])
        if interruption: return
        interruption=execute('mujoco_best',[sys.executable,'evaluate_mujoco.py','--checkpoint',str(bundle),
                         '--output',str(run/'evaluations/mujoco_best'),'--video'])
        if interruption: return
        status.update(status='review_required',finished_at=time.time(),
                      result=json.loads((run/'evaluations/mujoco_best.json').read_text()))
        write(path,status)
    except Exception as error:
        status.update(status='failed',error=str(error),finished_at=time.time()); write(path,status)
        raise


if __name__=='__main__': main()
