"""Event-driven, resource-guarded scratch search for diversity, quality and height.

Only this project's containers are touched. Frozen source/assets and per-trial configs
are immutable. No old policy is loaded by training; resource recovery stays in one run.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import threading
import time
from campaign import PROJECT,container_path,free_mib,atomic
from research_logic import can_admit,next_config,descriptor_distance,quality_score


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def available_host_mib():
    line=next(l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    return int(line.split()[1])//1024


class Campaign:
    def __init__(self,plan,control,root):
        self.plan=plan;self.control=control;self.root=root
        self.source=root/'source';self.assets=root/'assets'
        self.lock=threading.RLock();self.running={};self.states={};self.archive=[]
        self.gpu_locks={g:threading.RLock() for g in plan['physical_gpus']}
        self.baseline=read(self.source/plan['baseline_metrics'])
        self.peak_observed={}

    def enabled(self):return read(self.control).get('enabled',False)

    def state(self,name,**data):
        with self.lock:
            self.states[name]={**self.states.get(name,{}),**data,'updated_at':time.time()}
            atomic(self.root/'status.json',{'pid':os.getpid(),'series':self.plan['series'],
                'lanes':self.states,'running':list(self.running),'updated_at':time.time()})
            atomic(self.control.with_name(self.control.stem.replace('_control','_status')+'.json'),
                {'pid':os.getpid(),'batch':str(self.root.relative_to(PROJECT)),
                 'lanes':self.states,'updated_at':time.time()})

    def sample(self,lane):
        gpu=lane['gpu'];name=lane['name'];container=lane['container']
        top=subprocess.check_output(['docker','top',container,'-eo','pid,args'],text=True)
        pids={l.split()[0] for l in top.splitlines()[1:]}
        apps=subprocess.check_output(['nvidia-smi','--id='+str(gpu),
            '--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
        own=sum(int(l.split(',')[1]) for l in apps.splitlines() if l.split(',')[0].strip() in pids)
        with self.lock:
            if name in self.running:
                entry=self.running[name];entry['used']=own
                entry['peak']=max(entry['peak'],own+512)
                self.peak_observed[name]=max(self.peak_observed.get(name,0),own)
        return {'time':time.time(),'card_free_mib':free_mib(gpu),'own_compute_mib':own}

    def acquire(self,lane,out):
        gpu=lane['gpu'];name=lane['name']
        peak=max(lane['estimated_peak_mib'],self.peak_observed.get(name,0)+512)
        while self.enabled():
            with self.gpu_locks[gpu],self.lock:
                peers={n:r for n,r in self.running.items() if r['gpu']==gpu}
                if available_host_mib()>12288 and shutil.disk_usage(PROJECT).free>8*2**30 and can_admit(free_mib(gpu),{n:r['peak'] for n,r in peers.items()},
                             {n:r['used'] for n,r in peers.items()},peak,self.plan['reserve_mib']):
                    self.running[name]={'gpu':gpu,'peak':peak,'used':0,'started':time.time(),
                        'pressure_stop':False,'container':lane['container']}
                    return True
            self.state(name,phase='waiting_for_memory',trial=str(out.relative_to(PROJECT)),reserved_peak_mib=peak)
            time.sleep(10)
        return False

    def release(self,lane):
        with self.lock:self.running.pop(lane['name'],None)

    def execute(self,lane,out,script,arguments,logfile,simulation=True):
        name=lane['name'];container=lane['container']
        launcher=['/workspace/isaaclab/isaaclab.sh','-p'] if simulation else ['/isaac-sim/python.sh']
        command=['docker','exec','-e','PYTHONUNBUFFERED=1','-e','OMP_NUM_THREADS=4',
            '-w',container_path(self.source),container,*launcher,script,*map(str,arguments)]
        stage=Path(logfile).stem
        self.state(name,phase=stage,trial=str(out.relative_to(PROJECT)))
        atomic(out/(stage+'_process.json'),{'command':command,'started_at':time.time()})
        reason=None;stop_at=None;started=time.time()
        with (out/logfile).open('w') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                sample=self.sample(lane)
                with (out/'resources.jsonl').open('a') as f:f.write(json.dumps(sample)+'\n')
                with self.lock:
                    peers=[(n,r) for n,r in self.running.items() if r['gpu']==lane['gpu']]
                    if (sample['card_free_mib']<self.plan['reserve_mib'] or available_host_mib()<4096) and not any(r['pressure_stop'] for _,r in peers):
                        # Only one victim per card; latest admitted job yields first.
                        victim=max(peers,key=lambda item:item[1]['started'])[0]
                        self.running[victim]['pressure_stop']=True
                    pressure=self.running.get(name,{}).get('pressure_stop',False)
                stop=read(self.control).get('stop_current',False)
                heartbeat=(out/'status.json').stat().st_mtime if script=='train.py' and (out/'status.json').exists() else started
                stalled=time.time()-heartbeat>(1200 if simulation else 300)
                if reason is None and (pressure or stop or stalled):
                    reason='memory_pressure' if pressure else ('engineering_timeout' if stalled else 'user_control')
                    stop_at=time.time()
                    # Pattern matches the exact invocation in our exclusive container only.
                    pattern='python3 '+re.escape(script)+'.*'+re.escape(str(arguments[arguments.index('--output')+1]))+'( |$)' if '--output' in arguments else None
                    if pattern:
                        signal='-USR1' if script=='train.py' else '-TERM'
                        subprocess.run(['docker','exec',container,'pkill',signal,'-f',pattern],check=False)
                    atomic(out/(stage+'_stop.json'),{'reason':reason,**sample})
                elif reason and stop_at and time.time()-stop_at>120 and pattern:
                    # A wedged simulator may not reach a checkpoint boundary. Kill only this command.
                    subprocess.run(['docker','exec',container,'pkill','-KILL','-f',pattern],check=False)
                if script=='train.py' and (out/'status.json').exists():
                    self.state(name,training=read(out/'status.json'),resource=sample)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
        if reason and script!='train.py':raise RuntimeError(stage+' stopped: '+reason)
        if proc.returncode:raise RuntimeError(f'{stage} exited{proc.returncode}; see{out/logfile}')
        return reason

    def play(self,lane,out,checkpoint,cfg,directory,scripted=False,video=False):
        args=['--policy',container_path(checkpoint),'--step-height',str(cfg['step_height']),
            '--control-dt',str(cfg['control_dt']),'--num-envs','1' if video else str(self.plan['evaluation_episodes']),
            '--asset-dir',container_path(self.assets),'--output',container_path(directory),'--headless','--device','cuda:0']
        if scripted:args+=['--scripted']
        if video:args+=['--video']
        self.execute(lane,out,'play.py',args,directory.name+'.log')
        return read(directory/'evaluation.json')

    def review(self,lane,out,directory,index):
        self.execute(lane,out,'tools/review.py',['--run',container_path(out),'--evaluation',container_path(directory),
            '--output',container_path(directory),'--max-step',str(index)],directory.name+'_review.log',False)
        self.execute(lane,out,'tools/motion_metrics.py',['--evaluation',container_path(directory),
            '--output',container_path(directory/'motion_metrics.json')],directory.name+'_metrics.log',False)
        return read(directory/'motion_metrics.json')

    def trial(self,lane,cfg,out):
        out.mkdir();atomic(out/'config.json',cfg)
        reason=None
        for attempt in range(4):
            if not self.acquire(lane,out):return {'status':'cancelled','step_height':cfg['step_height']}
            args=['--config',container_path(out/'config.json'),'--output',container_path(out),
                '--asset-dir',container_path(self.assets),'--headless','--device','cuda:0']
            if attempt:
                assert reason=='memory_pressure' and read(out/'status.json')['status']=='interrupted'
                args+=['--resume',container_path(out/'interrupted.pt')]
            reason=self.execute(lane,out,'train.py',args,f'train_attempt{attempt}.log')
            status=read(out/'status.json')
            if status['status']=='trained':break
            self.release(lane)
            if reason!='memory_pressure' or not self.enabled():
                return {'status':'stopped','reason':reason,'step_height':cfg['step_height']}
            time.sleep(15)
        else:
            return {'status':'resource_blocked','step_height':cfg['step_height']}
        # Training has ended, but keep its card reservation for sequential review/render/export.
        candidates=[]
        self.execute(lane,out,'tools/review.py',['--run',container_path(out),'--output',container_path(out),
            '--max-step',str(cfg['iterations']-1),'--training-only'],'training_curves.log',False)
        checkpoints=[p for p in [out/'tensorboard/model_200.pt',out/'tensorboard/model_500.pt',out/'final.pt'] if p.exists()]
        for checkpoint in checkpoints:
            index=int(checkpoint.stem.split('_')[-1]) if checkpoint.stem.startswith('model_') else cfg['iterations']-1
            directory=out/f'eval_u{index+1:04d}'
            result=self.play(lane,out,checkpoint,cfg,directory)
            metrics=self.review(lane,out,directory,index)
            candidates.append({'checkpoint':str(checkpoint),'index':index,'evaluation':str(directory),
                'success_rate':result['success_rate'],'metrics':metrics})
        if cfg['research_track']=='quality':
            best=max(candidates,key=lambda c:(c['success_rate']>=.9,c['success_rate'] if c['success_rate']<.9 else -quality_score(c['metrics'],self.baseline)))
        elif cfg['research_track']=='diverse':
            with self.lock:
                reference=[r['behavior_descriptor'] for r in self.archive if r.get('behavior_descriptor') and abs(r['step_height']-.07)<.001]
            reference.append(self.baseline['behavior_descriptor'])
            def rank(c):
                d=c['metrics']['behavior_descriptor']
                novelty=min(descriptor_distance(d,r) for r in reference) if d else 0.
                return (c['success_rate']>=.9,novelty if c['success_rate']>=.9 else c['success_rate'])
            best=max(candidates,key=rank)
        else:best=max(candidates,key=lambda c:(c['success_rate'],-c['index']))
        checkpoint=Path(best['checkpoint']);ev=Path(best['evaluation']);policy=checkpoint;scripted=False
        export_result=None
        if best['success_rate']>=self.plan['height_success_rate']:
            export=out/'export'
            self.execute(lane,out,'tools/export_policy.py',['--checkpoint',container_path(checkpoint),
                '--observations',container_path(ev/'policy_inputs.npz'),'--output',container_path(export)],'export.log',False)
            policy=export/'policy_raw.pt';scripted=True
            export_result=self.play(lane,out,policy,cfg,out/'export_eval64',scripted=True)
            self.review(lane,out,out/'export_eval64',best['index'])
        video=self.play(lane,out,policy,cfg,out/'video',scripted=scripted,video=True)
        summary={'status':'reviewed','track':cfg['research_track'],'step_height':cfg['step_height'],
            'success_rate':min(best['success_rate'],export_result['success_rate']) if export_result else best['success_rate'],
            'raw_checkpoint_success_rate':best['success_rate'],'selected_checkpoint':str(checkpoint.relative_to(PROJECT)),
            'selected_update':best['index']+1,'checkpoint_sha256':sha(checkpoint),'from_scratch':True,
            'source_commit':read(self.root/'series.json')['source_commit'],
            'evaluation':str(ev.relative_to(PROJECT)),'video_success_rate':video['success_rate'],
            'video':str((out/'video/execution.mp4').relative_to(PROJECT)),
            'export_result':export_result,'quality_ratio_to_baseline':quality_score(best['metrics'],self.baseline) if best['success_rate'] else None,
            'successful_motion':best['metrics']['successful_episodes'],'behavior_descriptor':best['metrics']['behavior_descriptor'],
            'raw_metrics':str((ev/'motion_metrics.json').relative_to(PROJECT))}
        descriptor=summary['behavior_descriptor']
        with self.lock:
            references=[r['behavior_descriptor'] for r in self.archive if r.get('behavior_descriptor') and abs(r['step_height']-cfg['step_height'])<.001]
            if abs(cfg['step_height']-.07)<.001:references.append(self.baseline['behavior_descriptor'])
            summary['nearest_behavior_distance']=min((descriptor_distance(descriptor,r) for r in references),default=None) if descriptor else None
            summary['different_behavior_candidate']=bool(summary['nearest_behavior_distance'] is not None and summary['nearest_behavior_distance']>.15 and best['success_rate']>=.9)
            summary['behavior_review_note']='Descriptor novelty is a candidate only; inspect video/contact order before calling it a new maneuver.'
            self.archive.append(summary);atomic(self.root/'catalog.json',self.archive)
            reliable=[r['step_height'] for r in self.archive if r['success_rate']>=.9 and (r.get('export_result') or {}).get('success_rate',0)>=.9]
            any_success=[r['step_height'] for r in self.archive if r['success_rate']>0]
            atomic(self.root/'frontier.json',{'highest_reliable_height_m':max(reliable,default=None),
                'highest_observed_success_height_m':max(any_success,default=None),
                'physical_limit_established':False,'reviewed_trials':len(self.archive)})
        atomic(out/'review.json',summary);atomic(out/'completion.json',{'status':'complete','resume_allowed':False})
        self.release(lane)
        return summary

    def lane(self,lane,number):
        lane=dict(lane);lane['container']=f"petbot-step-explore-gpu{lane['gpu']}"+('-b' if lane['slot']=='b' else '')
        directory=self.root/lane['name'];directory.mkdir()
        initial=read(self.source/lane['config']);history=[]
        for trial in range(self.plan['max_trials_per_lane']):
            if not self.enabled():break
            cfg=next_config(initial,history,number,self.plan['height_ceiling_m'])
            if 'iterations_override' in self.plan:cfg['iterations']=self.plan['iterations_override']
            out=directory/f'trial{trial+1:03d}'
            try:result=self.trial(lane,cfg,out)
            except Exception as error:
                import traceback
                atomic(directory/'error.json',{'error':repr(error),'traceback':traceback.format_exc(),'trial':str(out)})
                self.release(lane);self.state(lane['name'],phase='engineering_error',error=repr(error));return
            history.append(result);atomic(directory/'history.json',history)
            self.state(lane['name'],phase=result['status'],latest_result=result,completed_trials=len(history))
            if result['status']!='reviewed':return
        self.state(lane['name'],phase='search_window_complete' if self.enabled() else 'disabled',completed_trials=len(history))


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--control',type=Path,default=PROJECT/'control/research_control.json');a=p.parse_args()
    plan=read(a.plan);control=a.control.resolve();control.parent.mkdir(exist_ok=True)
    lock=control.with_suffix('.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not read(control)['enabled']:raise RuntimeError('Research control is disabled')
    if subprocess.check_output(['git','status','--porcelain'],cwd=PROJECT,text=True).strip():raise RuntimeError('Publish clean source before research')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip()
    published=subprocess.check_output(['git','ls-remote','ours','refs/heads/demoproj/step_explore'],cwd=PROJECT,text=True).split()[0]
    if published!=commit:raise RuntimeError('Source commit is not published')
    for lane in plan['lanes']:
        container=f"petbot-step-explore-gpu{lane['gpu']}"+('-b' if lane['slot']=='b' else '')
        info=read_json_command(['docker','inspect',container])[0]
        if info['HostConfig']['DeviceRequests'][0]['DeviceIDs']!=[str(lane['gpu'])]:raise RuntimeError('Incorrect GPU isolation')
        top=subprocess.check_output(['docker','top',container,'-eo','pid,args'],text=True)
        if re.search(r'python3 (train|play)\.py',top):raise RuntimeError(container+' is busy')
    root=PROJECT/'runs'/(plan['series']+'_'+time.strftime('%Y%m%d_%H%M%S'));root.mkdir()
    (root/'source').mkdir();(root/'assets').mkdir()
    with (root/'source.tar').open('wb') as f:subprocess.run(['git','archive',commit],cwd=PROJECT,stdout=f,check=True)
    with tarfile.open(root/'source.tar') as f:f.extractall(root/'source',filter='data')
    for name in ['robot.usda','model.json']:shutil.copy2(PROJECT/'assets'/name,root/'assets'/name)
    model=read(root/'assets/model.json')
    for name,digest in model['source_sha256'].items():
        if sha(PROJECT.parents[1]/'robot_description'/name)!=digest:raise RuntimeError('Source model hash mismatch: '+name)
    atomic(root/'series.json',{'source_commit':commit,'from_scratch':True,'plan':plan,
        'source_archive_sha256':sha(root/'source.tar'),'asset_sha256':sha(root/'assets/robot.usda'),'created_at':time.time()})
    atomic(control.with_name(control.stem+'_process.json'),{'pid':os.getpid(),'batch':str(root.relative_to(PROJECT))})
    campaign=Campaign(plan,control,root)
    with ThreadPoolExecutor(max_workers=len(plan['lanes'])) as pool:
        futures=[pool.submit(campaign.lane,lane,n) for n,lane in enumerate(plan['lanes'])]
        for future in as_completed(futures):future.result()
    atomic(root/'completion.json',{'status':'search_window_ended','goal_achieved':False,
        'lanes':campaign.states,'finished_at':time.time(),'note':'Budget window or engineering stop, not a proof of robot height limit.'})
    settings=read(control);settings.update(enabled=False,automatic_iteration=False,ended_batch=str(root.relative_to(PROJECT)))
    atomic(control,settings)


def read_json_command(command):return json.loads(subprocess.check_output(command,text=True))


if __name__=='__main__':main()
