#!/usr/bin/env python3
"""One-shot overnight train/evaluate pipeline. No restart of failed jobs.

Run from the workspace root, or pass --root. Uses only petbot-train and GPUs 2/3.
Status and logs live outside immutable run checkouts. A process lock prevents duplicates.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def root_dir():
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        for p in (base, *base.parents):
            if (p/'default.xml').exists():
                return p
    raise RuntimeError('Pass --root with the manifest workspace')


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=None)
parser.add_argument('--iterations', type=int, default=1500)
parser.add_argument('--prepare-only', action='store_true')
parser.add_argument('--status', action='store_true')
args = parser.parse_args()
ROOT = (args.root or root_dir()).resolve()
SOURCE = ROOT/'demo_project/recovery_agile_01'
RUNS = ROOT/'demo_project/runs'
OUT = RUNS/'_sweep4'
OUT.mkdir(parents=True, exist_ok=True)
STATE = OUT/'status.json'
PYTHON = '/isaac-sim/python.sh'
CONTAINER = 'petbot-train'
VARIANTS = [(f'sweep4_{kind.lower()}_s{seed}', kind, seed, gpu, [])
            for kind, seed, gpu in [('B',42,3),('C',42,3),('B',43,2),('C',43,2)]]


def git(*cmd, cwd=SOURCE):
    return subprocess.check_output(['git', '-C', str(cwd), *cmd], text=True).strip()


def save(data):
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    temp.replace(STATE)


def launch(name, gpu, argv, label):
    path = f'/workspace/runs/{name}'
    command = ['docker', 'exec', '-w', path, '-e', f'PYTHONPATH={path}',
               '-e', f'CUDA_VISIBLE_DEVICES={gpu}', '-e', 'OMP_NUM_THREADS=2', CONTAINER, PYTHON, *argv]
    log = OUT/f'{label}.log'
    handle = log.open('w')
    proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
    handle.close()
    return proc, command


def final_checkpoint(name):
    return list((RUNS/name).glob(f'logs/rsl_rl/recovery_agile_01/*/model_{args.iterations-1}.pt'))


def container_path(path):
    return '/workspace/runs/' + str(path.relative_to(RUNS))


def report():
    baseline_path = OUT/'default.json'
    if not baseline_path.exists():
        return
    baseline = json.loads(baseline_path.read_text())
    floor = {r['initial_index']: r['success'] for r in baseline['episodes']}
    rows = []
    for path in sorted(OUT.glob('eval_*.json')):
        data = json.loads(path.read_text())
        same = data['dataset_sha256'] == baseline['dataset_sha256']
        episodes = data['episodes']
        difficult = [r for r in episodes if not floor[r['initial_index']]] if same else []
        easy = [r for r in episodes if floor[r['initial_index']]] if same else []
        rows.append({'name': path.stem, 'paired': same,
                     'success': data['summary']['all']['success_rate'],
                     'stable_at_end': data['summary']['all']['stable_at_end_rate'],
                     'baseline_failed_n': len(difficult),
                     'new_recoveries': sum(r['success'] for r in difficult) if same else None,
                     'lost_easy_cases': sum(not r['success'] for r in easy) if same else None,
                     'invalid': sum(r['termination'].get('invalid', False) for r in episodes),
                     'boundary': sum(r['termination'].get('boundary', False) for r in episodes)})
    (OUT/'summary.json').write_text(json.dumps({'baseline': baseline['summary'], 'rows': rows}, indent=2)+'\n')
    lines = ['# 第四轮确定性评估', '', '仅 paired=true 的结果可逐样本比较。', '',
             '|实验|成功率|末态稳定|配对|救回开环失败|丢失开环成功|非法/越界|',
             '|---|---:|---:|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['name']}|{r['success']:.3f}|{r['stable_at_end']:.3f}|{r['paired']}|{r['new_recoveries']}|{r['lost_easy_cases']}|{r['invalid']}/{r['boundary']}|")
    (OUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n')


def main():
    if args.iterations != 1500:
        raise ValueError('This staged batch stops at 1500; extending requires trend review')
    if args.status:
        print(STATE.read_text() if STATE.exists() else 'Not started')
        report()
        return
    lock = (OUT/'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if STATE.exists() and not args.prepare_only:
        raise RuntimeError('Existing status: inspect it; this pipeline never silently restarts runs')
    if git('status', '--porcelain'):
        raise RuntimeError('Source checkout is dirty')
    commit = git('rev-parse', 'HEAD')
    gate_path = RUNS/'_sweep4_precheck/gate.json'
    gate = json.loads(gate_path.read_text())
    if not gate.get('passed'):
        raise RuntimeError('GPU precheck has not passed')
    if git('diff', gate['tested_commit'], commit, '--', 'agile', 'scripts', 'petbot_tests'):
        raise RuntimeError('Implementation differs from validated precheck; rerun acceptance')

    previous_state = RUNS/'_sweep3/status.json'
    if not args.prepare_only and not json.loads(previous_state.read_text()).get('finished_at'):
        raise RuntimeError('Sweep3 training/evaluation is not finished; preserve its resources')

    # Reuse the ongoing full-library preparation rather than classify in every run.
    preparation = RUNS/'_sweep4_precheck/prepare.json'
    deadline = time.monotonic() + 3600
    while not preparation.exists():
        if time.monotonic() >= deadline:
            raise RuntimeError('Full curriculum preparation did not finish; no training launched')
        print('WAIT full curriculum preparation', flush=True)
        time.sleep(30)
    prepared = json.loads(preparation.read_text())
    if prepared['demo_commit'] != gate['tested_commit'] or prepared['easy_curriculum']['count'] <= 0:
        raise RuntimeError('Full curriculum preparation provenance mismatch')

    for name, _, _, _, _ in VARIANTS:
        path = RUNS/name
        if not path.exists():
            subprocess.run(['git', 'clone', '--no-local', str(SOURCE), str(path)],
                           env={**os.environ, 'GIT_LFS_SKIP_SMUDGE':'1'}, check=True)
            subprocess.run(['git', '-C', str(path), 'checkout', '--detach', commit], check=True)
        if git('rev-parse', 'HEAD', cwd=path) != commit or git('status','--porcelain',cwd=path):
            raise RuntimeError(f'{name}: revision/cleanliness mismatch')
        dst = path/'artifacts/petbot_asset'
        if not dst.exists():
            shutil.copytree(SOURCE/'artifacts/petbot_asset', dst)
        # Reuse only fully written preparation caches; pre_learn validates every identity.
        cache_source = SOURCE/'artifacts/fallen_states'
        cache_target = path/'artifacts/fallen_states'
        cache_target.mkdir(parents=True, exist_ok=True)
        for metadata in cache_source.glob('train_*.json'):
            info = json.loads(metadata.read_text())
            if info.get('num_envs') == 2048 and info['collection']['num_states'] == 16384:
                data = metadata.with_suffix('.pt')
                if data.exists():
                    shutil.copy2(data, cache_target/data.name)
                    shutil.copy2(metadata, cache_target/metadata.name)
        for data in cache_source.glob('easy_*.pt'):
            shutil.copy2(data, cache_target/data.name)
    (OUT/'plan.json').write_text(json.dumps({'commit':commit, 'iterations':args.iterations,
                                           'variants':VARIANTS}, indent=2)+'\n')
    if args.prepare_only:
        return
    shutil.copy2(RUNS/'_sweep2/eval_states.pt', OUT/'eval_states.pt')
    state = {'gate':gate, 'commit':commit, 'started_at':time.strftime('%FT%T%z'), 'train':{}, 'eval':{}}
    active = {}
    pending = list(VARIANTS)
    warm = None
    # Start one run first; once its complete training cache exists reuse it for all variants.
    first_name = VARIANTS[0][0]
    evaluations = [('default', first_name, 3, 'default', None),
                   ('zero', first_name, 3, 'zero', None),
                   ('initial', first_name, 3, 'policy', None)]
    evaluated_checkpoints = set()
    eval_active = None
    while pending or active or evaluations or eval_active:
        if pending:
            name, kind, seed, gpu, extra = pending[0]
            gpu_rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            free_train = {int(line.split(',')[0]):int(line.split(',')[1]) for line in gpu_rows.splitlines()}
            allow = (not state['train'] or warm is not None) and free_train[gpu] >= 6500
            # Stagger initialization; reported GPU usage may lag a newly started Kit process.
            allow = allow and time.time() - state.get('last_launch_epoch', 0) >= 45
            if allow:
                pending.pop(0)
                if kind == 'C':
                    previous = RUNS/'sweep4_c_s42/artifacts/fallen_states'
                    target = RUNS/name/'artifacts/fallen_states'
                    target.mkdir(parents=True, exist_ok=True)
                    if name != 'sweep4_c_s42':
                        for cached in previous.glob('easy_*.pt'):
                            shutil.copy2(cached, target/cached.name)
                if warm is not None:
                    dst = RUNS/name/'artifacts/fallen_states'
                    dst.mkdir(parents=True, exist_ok=True)
                    for f in warm:
                        if not (dst/f.name).exists():
                            shutil.copy2(f, dst/f.name)
                argv = ['scripts/train.py','--task',f'PetBot-Recovery-Sweep4-{kind}-v0',
                        '--headless','--num_envs','2048','--max_iterations',str(args.iterations),
                        '--seed',str(seed),'--logger','tensorboard',*extra]
                proc, cmd = launch(name,gpu,argv,'train_'+name)
                active[name] = proc
                state['train'][name] = {'status':'running','pid':proc.pid,'gpu':gpu,'command':cmd}
                state['last_launch_epoch'] = time.time()
                print('START',name,'GPU',gpu,flush=True)
        if warm is None and first_name in state['train']:
            files = list((RUNS/first_name/'artifacts/fallen_states').glob('train_*.pt'))
            # Collection writes the pt then its JSON; both must exist before sharing.
            ready = [p for p in files if p.with_suffix('.json').exists()]
            if ready:
                warm = [item for p in ready for item in (p,p.with_suffix('.json'))]
                print('TRAIN CACHE READY',flush=True)
        for name, proc in list(active.items()):
            rc = proc.poll()
            if rc is not None:
                checkpoints = final_checkpoint(name)
                ok = rc == 0 and len(checkpoints) == 1
                state['train'][name].update(status='complete' if ok else 'failed', exit_code=rc)
                del active[name]
                if ok:
                    evaluations.append(('eval_'+name,name,state['train'][name]['gpu'],'policy',checkpoints[0]))
                print('TRAIN END',name,rc,'final checkpoint',bool(checkpoints),flush=True)
                if name == first_name and warm is None:
                    state['failure'] = 'First run failed before library preparation; no remaining runs launched'
                    save(state)
                    raise RuntimeError(state['failure'])
        # Intermediate snapshots for tomorrow's learning trend, no extra training restarts.
        for name in state['train']:
            for iteration in (250,750):
                key = (name,iteration)
                checkpoints = list((RUNS/name).glob(f'logs/rsl_rl/recovery_agile_01/*/model_{iteration}.pt'))
                if checkpoints and key not in evaluated_checkpoints:
                    # Allow the writer to finish saving before queuing the snapshot.
                    if time.time()-checkpoints[0].stat().st_mtime > 30:
                        evaluated_checkpoints.add(key)
                        evaluations.append((f'eval_{name}_i{iteration}',name,3,'policy',checkpoints[0]))
        if eval_active:
            proc,label = eval_active
            rc = proc.poll()
            if rc is not None:
                exists = (OUT/(label+'.json')).exists()
                state['eval'][label].update(status='complete' if rc==0 and exists else 'failed',exit_code=rc)
                eval_active = None
                report()
        # At most one evaluator, and only with >= 4.5 GiB actually free on the selected GPU.
        if evaluations and not eval_active and warm is not None and not pending:
            gpu_rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
            free = {int(line.split(',')[0]):int(line.split(',')[1]) for line in gpu_rows.splitlines()}
            label,name,gpu,mode,checkpoint = evaluations[0]
            available = [g for g in (3,2) if free[g] >= 4600]
            if available:
                gpu = max(available,key=lambda g:free[g])
                evaluations.pop(0)
                argv = ['scripts/petbot_sweep2_eval.py','--headless','--episodes','512','--mode',mode,
                        '--task', 'PetBot-Recovery-Sweep4-B-v0',
                        '--allow-legacy-xyzw-dataset',
                        '--output',f'/workspace/runs/_sweep4/{label}.json']
                if label == 'initial':
                    argv += ['--untrained-policy']
                shared = OUT/'eval_states.pt'
                if shared.exists():
                    argv += ['--dataset','/workspace/runs/_sweep4/eval_states.pt']
                else:
                    argv += ['--save-dataset','/workspace/runs/_sweep4/eval_states.pt']
                if checkpoint:
                    argv += ['--checkpoint',container_path(checkpoint)]
                proc,cmd = launch(name,gpu,argv,label)
                eval_active = (proc,label)
                state['eval'][label] = {'status':'running','pid':proc.pid,'gpu':gpu,'command':cmd}
                print('EVAL START',label,'GPU',gpu,flush=True)
        state['updated_at'] = time.strftime('%FT%T%z')
        save(state)
        time.sleep(10)
    state['finished_at'] = time.strftime('%FT%T%z')
    save(state)
    report()
    print('PIPELINE FINISHED',flush=True)


if __name__ == '__main__':
    main()
