"""Re-run failed evaluations of a batch with the arguments they should have had.

Native scope uses the run's own trained envelope, common scope the wide one; both
take the checkpoint of the recorded iteration and never an initialization source.
Existing results are preserved under previous_attempts, and the queue's lock keeps
two retries from overlapping.
"""
import argparse, fcntl, json, pathlib, subprocess, time

p = argparse.ArgumentParser()
p.add_argument('--root', type=pathlib.Path, required=True)
p.add_argument('--name', required=True)
p.add_argument('--gpus', default='2,3')
p.add_argument('--parallel', type=int, default=2)
a = p.parse_args()

root = a.root.resolve()
out = root / 'demo_project/runs' / ('_' + a.name)
lock = (out / 'eval.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
status_path = out / 'eval_status.json'
state = json.loads(status_path.read_text())
launch = json.loads((out / 'launch_status.json').read_text())
gpus = [int(x) for x in a.gpus.split(',')]

WIDE = ['--speed-limit', '0.3', '--yaw-limit', '1.0', '--frame-min=-1.0', '--frame-max', '1.2']
NARROW = ['--speed-limit', '0.1', '--yaw-limit', '0.3', '--frame-min=-0.3', '--frame-max', '0.3']

todo = []
for key, entry in state.items():
    if entry.get('status') in ('complete', 'retry_running'):
        continue
    if '_i' not in key:
        continue
    run = key.rsplit('_i', 1)[0]
    iteration, scope = key.rsplit('_i', 1)[1].split('_')
    job = launch['train'].get(run)
    if not job:
        print('unknown run in state:', key)
        continue
    envelope = list(job.get('envelope') or [])
    # The stored training envelope must never carry an initialization source into
    # an evaluation: train_moving refuses to start with both that and a checkpoint.
    cleaned, skip = [], False
    for token in envelope:
        if token == '--initialize-from':
            skip = True
            continue
        if skip:
            skip = False
            continue
        cleaned.append(token)
    carry = [cleaned[i:i + 2] for i in range(0, len(cleaned), 2)
             if cleaned[i] in ('--wheel-mode', '--wheel-slew', '--feedback-mode')]
    carries = [x for pair in carry for x in pair]
    scope_args = NARROW if scope == 'native' else WIDE
    seed = '20261019' if scope == 'native' else '20261020'
    todo.append((key, run, iteration, scope, job, scope_args, carries, seed))

print(f'{len(todo)} evaluations to retry')
running = []
while todo or running:
    while todo and len(running) < a.parallel:
        key, run, iteration, scope, job, scope_args, carries, seed = todo.pop(0)
        target = out / ('eval_' + key)
        gpu = gpus[len(running) % len(gpus)]
        cmd = ['python3', str(root / f'demo_project/runs/{run}/tools/run_guarded.py'),
               '--container', launch['container'], '--gpu', str(gpu), '--reserve-mib', '3500',
               '--entrypoint', 'train_moving.py', '--workdir', f'/workspace/runs/{run}',
               '--log', str(out / f'retry_{key}.log'), '--headless', '--device', 'cuda:0',
               '--num_envs', '64', '--evaluate', '--seed', seed, *scope_args,
               '--checkpoint', f'/workspace/runs/{run}/logs/{run}/model_{iteration}.pt',
               '--log_dir', f'/workspace/runs/_{a.name}/eval_{key}', *carries]
        entry = state[key]
        entry['previous_attempts'] = entry.get('previous_attempts', []) + [
            {k: v for k, v in entry.items() if k != 'previous_attempts'}]
        entry.update(status='retry_running', retry_started_at=time.strftime('%FT%T%z'))
        status_path.write_text(json.dumps(state, indent=2))
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                                start_new_session=True)
        running.append((key, proc, target, time.time()))
        print('started', key, 'on gpu', gpu, flush=True)
    time.sleep(5)
    for item in list(running):
        key, proc, target, started = item
        if proc.poll() is None:
            continue
        running.remove(item)
        result = target / 'evaluation.json'
        state[key].update(status='complete' if proc.returncode == 0 and result.exists() else 'failed',
                          exit_code=proc.returncode, retry_finished_at=time.strftime('%FT%T%z'),
                          elapsed_s=round(time.time() - started, 1))
        if result.exists():
            state[key]['result'] = json.loads(result.read_text())
        status_path.write_text(json.dumps(state, indent=2))
        print('finished', key, state[key]['status'], flush=True)

old = out / 'eval_finished.json'
if old.exists() and not (out / 'eval_finished_after_retry.json').exists():
    (out / 'eval_finished_after_retry.json').write_text(json.dumps(dict(
        finished_at=time.strftime('%FT%T%z'),
        complete=sum(v['status'] == 'complete' for v in state.values()),
        expected=len(state), retry=True)))
(old if True else old).write_text(json.dumps(dict(
    finished_at=time.strftime('%FT%T%z'),
    complete=sum(v['status'] == 'complete' for v in state.values()),
    expected=len(state), retry=True)))
print('complete', sum(v['status'] == 'complete' for v in state.values()), 'of', len(state))
