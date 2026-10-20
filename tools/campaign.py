#!/usr/bin/env python3
"""Autonomous round supervisor for the upright_rl_demo moving task.

Deterministic by design: this process never invents a reward. The loop is

    wait for a batch's evaluations -> attribute failures per criterion
    -> apply the campaign's pre-registered decision rules -> launch the next
    *declared* round, or park with needs_review so a code change can be
    written, gated and committed by the agent.

Hard rules mirrored from the project docs:
  * one batch at a time; a run never starts while another launch_status exists
  * every run is a clean detached checkout of a fixed commit
  * 78% whole-card GPU memory guard through tools/run_guarded.py
  * source model (robot_description/) is read-only, success criteria are not
    relaxed, no auto push of docs
"""
import argparse, fcntl, json, os, pathlib, re, shutil, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[3]
DEMO = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / 'demo_project/runs'
CONTAINER = 'petbot-upright-night'
PY = '/isaac-sim/python.sh'


# ---------------------------------------------------------------- utilities
def stamp():
    return time.strftime('%FT%T%z')


def load(path, default=None):
    p = pathlib.Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text())


def save(path, data):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def gpu_usage():
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.total,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True)
    out = {}
    for row in rows.splitlines():
        i, total, used, util = (int(x) for x in row.split(','))
        out[i] = dict(total=total, used=used, util=util)
    return out


def batch_dir(name):
    return RUNS / ('_' + name)


def git(*args, cwd=DEMO):
    return subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()


def container_running():
    try:
        return subprocess.check_output(
            ['docker', 'inspect', CONTAINER, '--format', '{{.State.Running}}'],
            text=True).strip() == 'true'
    except subprocess.CalledProcessError:
        return False


# ------------------------------------------------------------------- status
def batch_status(name):
    """Where the batch actually is, from artefacts and live pids (never from intent)."""
    out = batch_dir(name)
    launch = load(out / 'launch_status.json')
    finished = load(out / 'eval_finished.json')
    status = load(out / 'eval_status.json', {})
    info = dict(batch=name, dir=str(out), launched=launch is not None,
                evaluations=len(status), expected=16,
                eval_finished=finished is not None, train={}, eval_pid=None)
    if launch:
        info['eval_pid'] = launch.get('eval_pid')
        info['eval_alive'] = alive(launch.get('eval_pid'))
        info['iterations'] = launch.get('iterations')
        info['commit'] = launch.get('commit')
        for run, job in launch.get('train', {}).items():
            log = out / (run + '.log')
            it = None
            if log.exists():
                hits = re.findall(r'iteration (\d+)', log.read_text(errors='replace')[-200000:])
                it = int(hits[-1]) if hits else None
            exit_file = out / (run + '.exit.json')
            ckpt = RUNS / run / 'logs' / run / f"model_{launch.get('iterations') - 1}.pt"
            info['train'][run] = dict(pid=job.get('pid'), gpu=job.get('gpu'),
                                      alive=alive(job.get('pid')), iteration=it,
                                      exited=load(exit_file), final_checkpoint=ckpt.exists())
    return info


def cmd_status(args):
    plan = load(args.plan)
    view = dict(plan=plan['name'], rounds=[], gpus=gpu_usage())
    for rnd in plan['rounds']:
        if rnd.get('status') == 'in_flight':
            view['rounds'].append(dict(id=rnd['id'], status=rnd['status'],
                                       batch=batch_status(rnd['batch'])))
        else:
            view['rounds'].append(dict(id=rnd['id'], status=rnd.get('status'),
                                       commit=rnd.get('commit')))
    print(json.dumps(view, indent=2))
    return 0


# ------------------------------------------------------------------ analyze
CRITERIA = ['support', 'forward', 'yaw_rate', 'path', 'heading', 'fixed', 'roll', 'tilt_rate']


def weighted_tail(result):
    """Per-criterion pass rate weighted by how many tail samples each env contributed."""
    samples = result.get('tail_samples') or []
    frac = result.get('tail_criterion_fraction_per_env') or []
    out = {}
    for idx, crit in enumerate(result.get('criterion_names') or CRITERIA):
        num = den = 0.0
        for w, row in zip(samples, frac):
            if not w or idx >= len(row):
                continue
            num += w * row[idx]
            den += w
        out[crit] = round(num / den, 4) if den else None
    return out


def mean(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def col_mean(values, idx):
    """Mean of one column when the per-env array holds a vector (e.g. [dv, dw])."""
    vals = [v[idx] for v in (values or []) if isinstance(v, (list, tuple)) and len(v) > idx]
    return round(sum(vals) / len(vals), 4) if vals else None


def spans(takeovers, key):
    vals = [t.get(key) for t in (takeovers or []) if isinstance(t.get(key), (int, float))]
    if not vals:
        return None
    return [round(min(vals), 4), round(max(vals), 4)]


def summarize_variant(result):
    return dict(taken=result.get('taken_over'), ever=result.get('success'),
                hold=result.get('maintained_final_2s'), final=result.get('final_success'),
                premature=result.get('premature'), warmup_failed=result.get('warmup_failed'),
                unfinished=result.get('unfinished'),
                path_mae=mean(result.get('raw_path_mae_per_env')),
                vel_mae=col_mean(result.get('raw_velocity_mae_per_env'), 0),
                yaw_mae=col_mean(result.get('raw_velocity_mae_per_env'), 1),
                heading_rate_mae=mean(result.get('heading_rate_mae_per_env')),
                rear_missing=mean(result.get('rear_missing_fraction_per_env')),
                support_fraction=mean(result.get('support_fraction_per_env')),
                first_lift=mean(result.get('first_lift_seconds')),
                front_height=mean(result.get('max_front_height_per_env')),
                accel_max=result.get('max_forward_acceleration'),
                stalled=mean(result.get('nonzero_command_low_speed_fraction')),
                ended_physical=sum(bool(x) for x in result.get('physical_failure_per_env') or []),
                ended_path=sum(bool(x) for x in result.get('path_failure_per_env') or []),
                takeovers=len(result.get('takeovers') or []),
                coverage=dict(forward=spans(result.get('takeovers'), 'forward'),
                              yaw_rate=spans(result.get('takeovers'), 'yaw_rate'),
                              frame_left=[round(min(t['frame'][0] for t in result['takeovers']), 4),
                                          round(max(t['frame'][0] for t in result['takeovers']), 4)]
                              if result.get('takeovers') else None),
                tail=weighted_tail(result))


def analyze(name, plan):
    """Group every finished evaluation of a batch by variant, seed and scope."""
    out = batch_dir(name)
    status = load(out / 'eval_status.json', {})
    round_spec = next(r for r in plan['rounds'] if r.get('batch') == name or r['id'] == name)
    runs = {v['run']: v for v in round_spec['variants']}
    variants, groups = {}, {}
    for key, entry in status.items():
        m = re.match(r'(.+)_i(\d+)_(native|common)$', key)
        if not m or entry.get('status') != 'complete' or not entry.get('result'):
            continue
        run, iteration, scope = m.group(1), int(m.group(2)), m.group(3)
        spec = runs.get(run)
        if spec is None:
            continue
        rec = summarize_variant(entry['result'])
        rec.update(iteration=iteration, scope=scope, group=spec['group'], seed=spec['seed'])
        variants.setdefault(run, {})[scope] = rec
        agg = groups.setdefault((spec['group'], scope, iteration),
                                dict(hold=0, ever=0, taken=0, premature=0,
                                     path_mae=[], rear_missing=[], accel=[],
                                     support=[], stalled=[], ended_physical=0,
                                     ended_path=0, coverage={}, seeds=[],
                                     tail={c: [] for c in CRITERIA}))
        agg['seeds'].append(spec['seed'])
        agg['hold'] += rec['hold'] or 0
        agg['ever'] += rec['ever'] or 0
        agg['taken'] += rec['taken'] or 0
        agg['premature'] += rec['premature'] or 0
        agg['ended_physical'] += rec['ended_physical']
        agg['ended_path'] += rec['ended_path']
        agg['path_mae'].append(rec['path_mae'])
        agg['rear_missing'].append(rec['rear_missing'])
        agg['accel'].append(rec['accel_max'])
        agg['support'].append(rec['support_fraction'])
        agg['stalled'].append(rec['stalled'])
        for name, span in (rec['coverage'] or {}).items():
            if not span:
                continue
            cur = agg['coverage'].get(name)
            agg['coverage'][name] = span if cur is None else [min(cur[0], span[0]), max(cur[1], span[1])]
        for crit, value in rec['tail'].items():
            agg['tail'].setdefault(crit, []).append(value)
    summary = {}
    for (group, scope, iteration), agg in sorted(groups.items()):
        summary[f'{group}_{scope}_i{iteration}'] = dict(
            iteration=iteration, scope=scope, seeds=sorted(agg['seeds']),
            hold=agg['hold'], ever=agg['ever'], taken=agg['taken'],
            premature=agg['premature'],
            ended_physical=agg['ended_physical'], ended_path=agg['ended_path'],
            path_mae=mean(agg['path_mae']),
            rear_missing=mean(agg['rear_missing']),
            support_fraction=mean(agg['support']),
            stalled=mean(agg['stalled']),
            accel_max=round(max([a for a in agg['accel'] if a] or [0]), 2),
            coverage=agg['coverage'],
            tail={c: mean(v) for c, v in agg['tail'].items()})
    terminal = {}
    for scope in ('native', 'common'):
        its = [v['iteration'] for v in summary.values() if v['scope'] == scope]
        terminal[scope] = max(its) if its else None
    report = dict(batch=name, analysed_at=stamp(), evaluations=len(status),
                  expected=16, variants=variants, groups=summary, terminal=terminal,
                  source_commit=round_spec.get('commit'))
    save(out / 'analysis.json', report)
    return report


def terminal_groups(report, scope):
    """Only the last checkpoint, which is the one a verdict may be based on."""
    it = report.get('terminal', {}).get(scope)
    return {k: v for k, v in report['groups'].items()
            if v['scope'] == scope and v['iteration'] == it}


def failing_criteria(report, scope='native'):
    tails = [v['tail'] for v in terminal_groups(report, scope).values()]
    out = {}
    for crit in CRITERIA:
        vals = [t[crit] for t in tails if t.get(crit) is not None]
        if vals:
            out[crit] = round(sum(vals) / len(vals), 4)
    ranked = sorted(out.items(), key=lambda kv: kv[1])
    return ranked


def decide(report, plan):
    """Apply the pre-registered rules. Returns a dict that never relaxes a criterion."""
    rules = plan.get('decision_rules', {})
    need = rules.get('min_hold_gain', 8)
    a_native = terminal_groups(report, 'native')
    a = next((v for k, v in a_native.items() if k.startswith('a_')), None)
    b = next((v for k, v in a_native.items() if k.startswith('b_')), None)
    complete = report['evaluations'] >= report['expected']
    ranked = failing_criteria(report)
    decision: dict = dict(complete=complete, at=stamp(), failing=ranked)
    if not complete:
        decision.update(verdict='incomplete', action='wait',
                        reason=f"{report['evaluations']}/{report['expected']} evaluations finished")
        return decision
    if not a or not b:
        decision.update(verdict='incomplete', action='review', reason='missing a/b native group')
        return decision
    gain = b['hold'] - a['hold']
    worse_path = b['path_mae'] is not None and a['path_mae'] is not None and b['path_mae'] > a['path_mae']
    decision.update(hold_a=a['hold'], hold_b=b['hold'], gain=gain,
                    path_a=a['path_mae'], path_b=b['path_mae'], path_worse=worse_path)
    if gain >= need and not (rules.get('require_path_mae_gain', True) and worse_path):
        decision.update(verdict='keep_new', action='launch_declared',
                        reason=f'+{gain} maintained samples and no tracking regression')
    elif gain > 0:
        decision.update(verdict='no_advantage', action='needs_review',
                        reason=f'gain {gain:+d} below the pre-registered {need}; tracking '
                               f'{"worse" if worse_path else "not better"}')
    else:
        decision.update(verdict='regress', action='needs_review',
                        reason=f'new reward loses {-gain} maintained samples')
    if decision['action'] == 'needs_review':
        decision['diagnosis'] = diagnose_layers(report, ranked, plan)
    save(batch_dir(report['batch']) / 'decision.json', decision)
    return decision


def diagnose_layers(report, ranked, plan):
    """Attribute the failure to a layer of the task, not just to a reward term.

    Every entry names the layer, the number behind it, the change it implies and
    what that change risks. Ordered by how directly the evidence supports it.
    """
    native = terminal_groups(report, 'native')
    wide = terminal_groups(report, 'common')
    env = plan['envelope']

    def tail(crit):
        return mean([v['tail'].get(crit) for v in native.values()])

    def field(key):
        return mean([v.get(key) for v in native.values()])

    rear, support = field('rear_missing'), tail('support')
    heading, path = tail('heading'), tail('path')
    accel, stalled = max([v.get('accel_max') or 0 for v in native.values()] or [0]), field('stalled')
    targets = []

    if (rear is not None and rear >= 0.01) or (support is not None and support < 0.98):
        targets.append(dict(
            layer='instrumentation', evidence=f'tail support pass {support}, rear-contact missing fraction {rear}',
            change='log rear contact, front-wheel height and frame angle at physics rate (history_length>1) '
                   'before choosing a fix: the current single 0.06 s sample cannot separate a real '
                   'bounce from a one-step dropout',
            cost='code only, no retraining', risk='none to the task; needed to make the next decision falsifiable'))
        targets.append(dict(
            layer='action/transition', evidence='contact loss appears after full lift, so it is a load-transfer '
                                               f'effect; peak forward acceleration {accel} m/s2',
            change='re-examine the wheel-slew and frame rate limits jointly with the transition shaping instead of '
                   'reward weights alone',
            cost='code + retraining', risk='slower lift; must keep 64/64 lifting in the gate'))

    if heading is not None and heading < 0.85:
        targets.append(dict(
            layer='observation', evidence=f'tail heading pass {heading} while path pass {path}',
            change='accumulated heading error is not a state: the policy sees sin/cos of the reference yaw and a '
                   'position offset, but no integral of heading error; add the integrated error (bounded) and '
                   'credit it explicitly',
            cost='code + retraining, changes the 205-dim contract (new task version)', risk='observation shape change '
                   'invalidates old checkpoints, so it has to start from a clean contract'))

    cov = next((v.get('coverage') or {} for v in native.values()), {})
    fwd = cov.get('forward')
    if fwd and env.get('speed_limit'):
        if max(abs(fwd[0]), abs(fwd[1])) < 0.4 * env['speed_limit']:
            targets.append(dict(
                layer='initialization/curriculum', evidence=f'actual takeover forward speed spans {fwd} against a '
                                                            f'+/-{env["speed_limit"]} m/s envelope',
                change='the warmup drive hands over near zero speed; either widen the produced initial states or '
                       'narrow the requested envelope so the trained regime is the one that is evaluated',
                cost='code only', risk='a narrower envelope would look better without being better; prefer widening '
                                       'real initial states'))

    if accel and accel > 8:
        targets.append(dict(
            layer='reward/impact', evidence=f'peak forward acceleration {accel} m/s2 is far from smooth',
            change='credit a bounded acceleration term and penalise impact during the first support second',
            cost='code + retraining', risk='may trade success for smoothness; both must be reported'))

    if stalled is not None and stalled > 0.1:
        targets.append(dict(
            layer='reward', evidence=f'{stalled} of samples with a non-zero command are below 0.025 m/s',
            change='the policy still buys stability by standing still; make the raw-command credit dominate during '
                   'the transition',
            cost='code + retraining', risk='can reduce support stability'))

    if wide and native:
        wh = mean([v['hold'] for v in wide.values()])
        nh = mean([v['hold'] for v in native.values()])
        if wh is not None and nh is not None and wh < 0.25 * max(nh, 1):
            targets.append(dict(
                layer='generalization', evidence=f'wide-envelope hold {wh} against native {nh}',
                change='not the same failure: keep the wide envelope as an evaluation only, do not widen training '
                       'until the trained envelope holds',
                cost='none', risk='none'))
    if not targets:
        targets.append(dict(layer='review', evidence='no criterion is dominant', change='read the per-environment '
                                                     'trajectories before changing code', cost='none', risk='none'))
    return dict(ranked_failures=ranked, targets=targets, primary=targets[0]['layer'])


def cmd_analyze(args):
    plan = load(args.plan)
    report = analyze(args.batch, plan)
    print(f"# {args.batch}: {report['evaluations']}/{report['expected']} evaluations")
    header = f"{'group/scope':<12}{'hold':>6}{'ever':>6}{'prem':>6}{'pathMAE':>9}{'rearMiss':>10}{'accel':>8}"
    print(header)
    for key, g in sorted(report['groups'].items()):
        print(f"{key:<12}{g['hold']:>6}{g['ever']:>6}{g['premature']:>6}"
              f"{str(g['path_mae']):>9}{str(g['rear_missing']):>10}{str(g['accel_max']):>8}")
    print('\ntail criterion pass rate (sample-weighted, native scope):')
    for crit, rate in failing_criteria(report):
        print(f'  {crit:<10}{rate}')
    return 0


def cmd_decide(args):
    plan = load(args.plan)
    report = load(batch_dir(args.batch) / 'analysis.json')
    if report is None:
        report = analyze(args.batch, plan)
    print(json.dumps(decide(report, plan), indent=2))
    return 0


# --------------------------------------------------------------------- gate
def gate_run(cmd, log, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})})
    return p.returncode


def cmd_gate(args):
    """Contract check + short PPO + 64-env reload on the round's fixed commit."""
    plan = load(args.plan)
    rnd = next(r for r in plan['rounds'] if r['id'] == args.round)
    if not rnd.get('commit'):
        print(f"round {args.round} has no fixed commit; gate cannot run")
        return 2
    out = batch_dir(rnd.get('gate_dir', rnd['id']))
    out.mkdir(parents=True, exist_ok=True)
    commit = rnd['commit']
    if git('rev-parse', 'HEAD') != commit or git('status', '--porcelain'):
        print(f'source checkout is not the clean fixed commit {commit[:7]}')
        return 2
    if not container_running():
        print(f'container {CONTAINER} is not running')
        return 2
    gpus = rnd.get('gate', {}).get('gpus', [2, 3])
    env = dict(CUDA_VISIBLE_DEVICES=str(gpus[0]))
    contract = out / 'contract.log'
    code = gate_run(['docker', 'exec', '-w', '/workspace/demo_project/upright_rl_demo',
                     '-e', f'CUDA_VISIBLE_DEVICES={gpus[0]}', CONTAINER, PY,
                     'checks/moving_contract.py'], contract)
    if code != 0:
        print(f'contract check failed ({code}): {contract}')
        return 2
    gate = dict(engineering_passed=False, tested_commit=commit, contract='contract.log',
                short_iterations=rnd.get('gate', {}).get('smoke_iterations', 100),
                created_at=stamp(), smoke={}, reload={})
    procs = []
    for i, var in enumerate(rnd.get('gate', {}).get('variants', [])):
        gpu = gpus[i % len(gpus)]
        name = f"smoke_{var['label']}"
        log = out / (name + '.log')
        if log.exists():
            print(f'{log} exists; inspect instead of rerunning')
            return 2
        cmd = ['python3', str(DEMO / 'tools/run_guarded.py'), '--container', CONTAINER,
               '--gpu', str(gpu), '--reserve-mib', '3500', '--entrypoint', 'train_moving.py',
               '--workdir', '/workspace/demo_project/upright_rl_demo', '--log', str(log),
               '--headless', '--device', 'cuda:0', '--num_envs', '128',
               '--iterations', str(gate['short_iterations']), '--seed', '20261019',
               '--log_dir', f'/workspace/runs/{rnd.get("gate_dir", rnd["id"])}/{name}',
               *envelope_args(plan, var)]
        if args.dry_run:
            print('dry-run:', ' '.join(cmd))
            continue
        with (out / (name + '_guard.log')).open('w') as f:
            procs.append((name, subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                                 start_new_session=True)))
    if args.dry_run:
        print('gate dry-run finished; nothing executed')
        return 0
    for name, proc in procs:
        code = proc.wait()
        log = out / (name + '.log')
        text = log.read_text(errors='replace') if log.exists() else ''
        finite = 'nan' not in text.lower() and 'Learning iteration' in text
        gate['smoke'][name] = dict(exit_code=code, all_finite=finite,
                                   iterations=len(re.findall(r'Learning iteration', text)))
        if code != 0 or not finite:
            save(out / 'gate.json', gate)
            print(f'{name} failed the short-training gate; see {log}')
            return 2
    for i, var in enumerate(rnd.get('gate', {}).get('variants', [])):
        label = var['label']
        name = f'reload_{label}'
        ckpt = f'/workspace/runs/{rnd.get("gate_dir", rnd["id"])}/smoke_{label}/model_{gate["short_iterations"] - 1}.pt'
        cmd = ['python3', str(DEMO / 'tools/run_guarded.py'), '--container', CONTAINER,
               '--gpu', str(gpus[i % len(gpus)]), '--reserve-mib', '3500',
               '--entrypoint', 'train_moving.py', '--workdir', '/workspace/demo_project/upright_rl_demo',
               '--log', str(out / (name + '.log')), '--headless', '--device', 'cuda:0',
               '--num_envs', '64', '--evaluate', '--seed', '20261019', '--checkpoint', ckpt,
               '--log_dir', f'/workspace/runs/{rnd.get("gate_dir", rnd["id"])}/{name}',
               *envelope_args(plan, var)]
        code = gate_run(cmd, out / (name + '_guard.log'))
        result = load(out / name / 'evaluation.json')
        if code != 0 or not result:
            gate['reload'][label] = dict(exit_code=code, error='no evaluation.json')
            save(out / 'gate.json', gate)
            print(f'{name} reload failed; see {out / name}')
            return 2
        gate['reload'][label] = {k: result[k] for k in
                                 ('taken_over', 'lifted', 'success', 'maintained_final_2s',
                                  'premature', 'warmup_failed', 'unfinished')}
    gate['engineering_passed'] = True
    save(out / 'gate.json', gate)
    print(json.dumps(gate, indent=2))
    return 0


def envelope_args(plan, var):
    env = plan['envelope']
    args = ['--speed-limit', str(env['speed_limit']), '--yaw-limit', str(env['yaw_limit']),
            f"--frame-min={env['frame_min']}", '--frame-max', str(env['frame_max']),
            '--wheel-mode', env['wheel_mode'], '--wheel-slew', str(env['wheel_slew']),
            '--learning-rate', str(env['learning_rate'])]
    if var.get('feedback_mode'):
        args += ['--feedback-mode', var['feedback_mode']]
    if var.get('initialize_from'):
        args += ['--initialize-from', var['initialize_from']]
    return args


# ------------------------------------------------------------------- launch
def cmd_launch(args):
    """Same contract as tools/launch_moving.py, driven by the plan instead of literals."""
    plan = load(args.plan)
    rnd = next(r for r in plan['rounds'] if r['id'] == args.round)
    if rnd.get('status') not in ('planned',):
        print(f'round {args.round} is {rnd.get("status")}; refusing to launch')
        return 2
    if not rnd.get('commit'):
        print('round has no fixed commit; register the code first')
        return 2
    out = batch_dir(rnd['id'])
    out.mkdir(exist_ok=True)
    if (out / 'launch_status.json').exists():
        print('launch_status.json exists; inspect logs/pids instead of restarting')
        return 2
    gate_file = batch_dir(rnd.get('gate_dir', rnd['id'])) / 'gate.json'
    gate = load(gate_file)
    if not gate or not gate.get('engineering_passed'):
        print(f'no passed gate at {gate_file}')
        return 2
    if git('rev-parse', 'HEAD') != rnd['commit'] or git('status', '--porcelain'):
        print('source checkout is not the clean fixed commit; fix before launching')
        return 2
    if git('diff', gate['tested_commit'], rnd['commit'], '--', 'upright', 'train_moving.py'):
        print('code changed after the gate was passed; re-gate')
        return 2
    if not container_running():
        print('container is not running')
        return 2
    mem = gpu_usage()
    for gpu in sorted({v['gpu'] for v in rnd['variants']}):
        if mem[gpu]['used'] + 4000 > mem[gpu]['total'] * 0.78:
            print(f'GPU{gpu} has no headroom: {mem[gpu]}')
            return 2
    state = dict(name=rnd['id'], commit=rnd['commit'], container=CONTAINER,
                 thread=os.environ.get('CODEX_THREAD_ID'), started_at=stamp(),
                 iterations=plan['iterations'], checkpoints=plan['checkpoints'],
                 gate=gate, train={})
    if args.dry_run:
        print('dry-run launch:', json.dumps(dict(round=rnd['id'], commit=rnd['commit'][:7],
                                                 variants=[v['run'] for v in rnd['variants']]), indent=2))
        return 0
    for var in rnd['variants']:
        dst = RUNS / var['run']
        if dst.exists():
            print(f'{dst} exists; refusing to overwrite a run copy')
            return 2
        subprocess.run(['git', 'clone', '--no-local', str(DEMO), str(dst)],
                       env={**os.environ, 'GIT_LFS_SKIP_SMUDGE': '1'}, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        subprocess.run(['git', '-C', str(dst), 'checkout', '--detach', rnd['commit']], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        shutil.copytree(DEMO / 'artifacts/petbot_asset', dst / 'artifacts/petbot_asset', dirs_exist_ok=True)
        envelope = envelope_args(plan, var)
        cmd = ['python3', str(dst / 'tools/run_guarded.py'), '--container', CONTAINER,
               '--gpu', str(var['gpu']), '--reserve-mib', '4000', '--entrypoint', 'train_moving.py',
               '--workdir', f"/workspace/runs/{var['run']}", '--log', str(out / (var['run'] + '.log')),
               '--headless', '--device', 'cuda:0', '--num_envs', str(plan.get('num_envs', 2048)),
               '--iterations', str(plan['iterations']), '--seed', str(var['seed']), *envelope,
               '--log_dir', f"/workspace/runs/{var['run']}/logs/{var['run']}"]
        with (out / (var['run'] + '_guard.log')).open('w') as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        state['train'][var['run']] = dict(pid=proc.pid, gpu=var['gpu'], envelope=envelope,
                                          status='initializing', command=cmd)
        save(out / 'launch_status.json', state)
        deadline = time.monotonic() + 600
        log = out / (var['run'] + '.log')
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                print(f"{var['run']} exited {proc.returncode}; inspect before continuing")
                return 2
            if log.exists() and 'Learning iteration' in log.read_text(errors='replace'):
                break
            time.sleep(5)
        else:
            print('PPO startup timeout; inspect the live process')
            return 2
        state['train'][var['run']]['status'] = 'ppo_started'
        save(out / 'launch_status.json', state)
    queue = subprocess.Popen(['python3', str(DEMO / 'tools/evaluate_moving.py'), '--root', str(ROOT),
                              '--name', rnd['id']], stdout=(out / 'eval_queue.log').open('w'),
                             stderr=subprocess.STDOUT, start_new_session=True)
    state.update(eval_pid=queue.pid, all_ppo_at=stamp())
    save(out / 'launch_status.json', state)
    rnd['status'] = 'in_flight'
    plan_path = pathlib.Path(args.plan)
    save(plan_path, plan)
    print(json.dumps(dict(round=rnd['id'], eval_pid=queue.pid, all_ppo_at=state['all_ppo_at']), indent=2))
    return 0


# -------------------------------------------------------------------- drive
def wait_evals(name, timeout_h):
    out = batch_dir(name)
    deadline = time.monotonic() + timeout_h * 3600
    while time.monotonic() < deadline:
        status = batch_status(name)
        if status['eval_finished']:
            return True
        time.sleep(120)
    return False


def cmd_drive(args):
    """One autonomous cycle per invocation; run it from cron for a campaign."""
    plan_path = pathlib.Path(args.plan)
    lock = (batch_dir('campaign') / 'drive.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    handle = lock.open('w')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('another driver holds the lock')
        return 0
    plan = load(plan_path)
    stop = plan.get('stop', {})
    if stop.get('deadline') and time.strftime('%FT%T%z') > stop['deadline']:
        print(f"campaign deadline {stop['deadline']} passed; no action")
        return 0
    active = [r for r in plan['rounds'] if r.get('status') == 'in_flight']
    if active:
        rnd = active[0]
        name = rnd.get('batch', rnd['id'])
        if not wait_evals(name, args.timeout_h):
            print(f"{name}: evaluations not finished yet")
            return 0
        report = analyze(name, plan)
        decision = decide(report, plan)
        rnd['status'] = 'done'
        rnd['verdict'] = decision['verdict']
        save(plan_path, plan)
        print(json.dumps(decision, indent=2))
        if decision['action'] != 'launch_declared':
            park(plan, rnd, decision)
            return 0
    nxt = next((r for r in plan['rounds'] if r.get('status') == 'planned'), None)
    if nxt is None:
        print('no planned round left; campaign idle')
        return 0
    if not nxt.get('commit'):
        park(plan, nxt, dict(action='needs_review',
                             reason='round is registered without a fixed commit; write and gate the code'))
        return 0
    if not (batch_dir(nxt.get('gate_dir', nxt['id'])) / 'gate.json').exists():
        print(f"gate missing for {nxt['id']}; run: campaign.py gate --round {nxt['id']}")
        return 0
    if args.dry_run:
        print('dry-run: would gate-check and launch', nxt['id'])
        return 0
    if cmd_launch(argparse.Namespace(plan=str(plan_path), round=nxt['id'], dry_run=False)) != 0:
        return 2
    return 0


def park(plan, rnd, decision):
    """Stop cleanly with a review request instead of inventing a reward."""
    out = batch_dir(rnd.get('id', 'campaign'))
    out.mkdir(parents=True, exist_ok=True)
    body = [f"# 需要设计变更的轮次：{rnd.get('id')}", '',
            f"时间：{stamp()}", f"判定：{decision.get('verdict')}",
            f"原因：{decision.get('reason')}", '']
    hyp = decision.get('diagnosis') or {}
    for target in hyp.get('targets', []):
        body.append(f"- **{target.get('layer')}**：{target.get('evidence')}")
        body.append(f"  - 可改：{target.get('change')}")
        body.append(f"  - 代价：{target.get('cost')}；风险：{target.get('risk')}")
    if hyp.get('ranked_failures'):
        body += ['', '| 判据 | 末段通过率 |', '|---|---:|']
        body += [f'| {c} | {r} |' for c, r in hyp['ranked_failures']]
    body += ['', '下一步：按上面的分层归因选一层实现（观测 / 动作 / 奖励 / 终止 / 课程 / 采样 / 初始化均可改）→ '
                 '`campaign.py gate --round <next>` → 在 plan 里登记 fixed commit 与 variants → `campaign.py drive`。',
             '禁止：放宽成功判据、改源模型、改正在跑的批次、改任务范围。']
    (out / 'review_request.md').write_text('\n'.join(body) + '\n')
    if rnd.get('status') != 'needs_review':
        rnd['status'] = 'needs_review'
    print('\n'.join(body))
    return 0


# ------------------------------------------------------------------- report
def cmd_report(args):
    plan = load(args.plan)
    report = load(batch_dir(args.batch) / 'analysis.json') or analyze(args.batch, plan)
    decision = load(batch_dir(args.batch) / 'decision.json')
    lines = [f"## {args.batch} 自动分析（{report['analysed_at']}）", '',
             '| 组/范围 | 末2秒保持 | 曾达标 | 提前退出 | 轨迹MAE | 缺后轮接触 | 峰值加速度 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for key, g in sorted(report['groups'].items()):
        lines.append(f"| {key} | {g['hold']} | {g['ever']} | {g['premature']} | "
                     f"{g['path_mae']} | {g['rear_missing']} | {g['accel_max']} |")
    ranked = failing_criteria(report)
    lines += ['', '末段判据通过率（按样本加权）：' + '、'.join(f'{c} {r}' for c, r in ranked), '']
    if decision:
        lines += [f"判定：**{decision['verdict']}** —— {decision['reason']}"]
    text = '\n'.join(lines) + '\n'
    print(text)
    if args.apply:
        target = pathlib.Path(args.doc)
        existing = target.read_text() if target.exists() else ''
        marker = f"<!-- auto:{args.batch} -->"
        if marker not in existing:
            target.write_text(existing + ('\n' if existing else '') + marker + '\n' + text + '\n')
            print(f'appended to {target}')
    return 0


# --------------------------------------------------------------------- cli
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', default=str(DEMO / 'campaigns/moving.json'))
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('status')
    a = sub.add_parser('analyze'); a.add_argument('batch')
    a = sub.add_parser('decide'); a.add_argument('batch')
    a = sub.add_parser('gate'); a.add_argument('--round', required=True); a.add_argument('--dry-run', action='store_true')
    a = sub.add_parser('launch'); a.add_argument('--round', required=True); a.add_argument('--dry-run', action='store_true')
    a = sub.add_parser('drive'); a.add_argument('--timeout-h', type=float, default=6.); a.add_argument('--dry-run', action='store_true')
    a = sub.add_parser('report'); a.add_argument('batch'); a.add_argument('--doc'); a.add_argument('--apply', action='store_true')
    args = p.parse_args(argv)
    return {'status': cmd_status, 'analyze': cmd_analyze, 'decide': cmd_decide, 'gate': cmd_gate,
            'launch': cmd_launch, 'drive': cmd_drive, 'report': cmd_report}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
