"""Finite physical action sequences to diagnose the scratch rise barrier.

No neural network, parent checkpoint, policy fitting or training dataset. The
source controller, admission, failure and success conditions remain active.
Only the first independent call per scene is recorded.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
try:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg
    from upright.feedback_task import FeedbackEnvCfg, ensure
    from upright.continuous_task import reward

    cases = [dict(first_hip=h1, next_hip=h2, change_at=t,
                  first_wheel=w[0], next_wheel=w[1])
             for h1, h2, t, w in itertools.product(
                 [.3, .8, 1.2], [-1., 0., .6, 1.2], [.4, .8, 1.2],
                 [(0., 25.), (0., -25.), (-25., 25.),
                  (25., -25.), (25., 25.), (-25., -25.)])]
    cfg = FeedbackEnvCfg()
    cfg.scene.num_envs = len(cases)
    cfg.sim.device = args.device
    cfg.seed = 20261030
    cfg.from_scratch = True
    cfg.decimation = cfg.sim.render_interval = 10
    cfg.speed_limit, cfg.yaw_limit = .75, 15.
    cfg.frame_range, cfg.hold_range = (0., 0.), (4., 4.)
    cfg.wheel_slew, cfg.frame_slew = 480., 3.
    cfg.wheel_mode, cfg.feedback_mode = 'common_diff', 'pose_feedback_tight'
    cfg.tracking_objective, cfg.observation_frame = 'stability_first', 'balance_local'
    cfg.handoff_alignment, cfg.transition_contract = 'after', 'linear_heading'
    cfg.task_profile, cfg.training_mixture = 'full_range', 'off'
    cfg.capability_version = 'v25'
    cfg.mirror_training = cfg.adaptive_curriculum = False
    cfg.evaluating, cfg.evaluation_pattern = True, 'mixed'
    cfg.rise_speed_reference_mode = 'capture'
    cfg.yaw_feedback_design, cfg.yaw_feedback_gain = 'fast2', 16.
    cfg.active_seconds, cfg.episode_length_s = 16., 44.
    cfg.rewards.rise_progress = RewardTermCfg(func=reward, weight=4., params={'kind': 'rise_progress_reward'})
    cfg.rewards.enter_balance.weight = 5.
    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset()
    progress = ensure(env)
    progress.set_command([0., 0.])
    progress.request_mode(1)
    params = {k: torch.tensor([c[k] for c in cases], device=env.device) for k in cases[0]}
    alive = torch.ones(len(cases), device=env.device, dtype=torch.bool)
    zero = torch.zeros(len(cases), device=env.device)
    gap, height, support, credit = [zero.clone() for _ in range(4)]
    entered = alive & False
    failures = zero.long()
    takeover = alive & False
    trace = []
    peak_target = peak_actual = peak_effort = 0.
    with torch.inference_mode():
        for tick in range(900):
            second = progress.phase_time >= params['change_at']
            hip = torch.where(second, params['next_hip'], params['first_hip'])
            wheel = torch.where(second, params['next_wheel'], params['first_wheel'])
            action = torch.stack((2 * (hip + 1.) / 2.2 - 1., wheel / 25., zero), -1)
            obs, rew, term, timeout, _ = env.step(action)
            s, m, requested, command, applied, since = env.moving_snapshot
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(rew).all()
            assert m['wheel_target'].abs().max() <= 25.0001
            assert m['effort'][:, -2:].abs().max() <= .38001
            assert torch.allclose(m['position_target'][:, 3], m['position_target'][:, 4])
            assert m['position_target'][:, 3].min() >= -1.0001
            assert m['position_target'][:, 3].max() <= 1.2001
            peak_target = max(peak_target, float(m['wheel_target'][alive].abs().max()))
            peak_actual = max(peak_actual, float(s['wheel'][alive].abs().max()))
            peak_effort = max(peak_effort, float(m['effort'][alive, -2:].abs().max()))
            gap = torch.maximum(gap, torch.where(alive, m['wheel_gap'].amin(-1), 0.))
            height = torch.maximum(height, torch.where(alive, s['front_z'], 0.))
            support = torch.maximum(support, torch.where(alive, progress.support_run, 0.))
            credit += alive * m['rise_progress_reward'] * env.step_dt
            takeover |= alive & m['switched']
            entered |= alive & m['entered_balance']
            failures = torch.where(alive & m['failure'], m['failure_code'], failures)
            # Include every terminal/transition tick; sample normal motion at40ms.
            if tick % 2 == 0 or (alive & (m['failure'] | m['entered_balance'] | m['switched'])).any():
                values = dict(t=(tick + 1) * env.step_dt, valid=alive, phase=m['phase'],
                    phase_time=m['phase_time'], requested=requested, command=command,
                    action=action, applied=applied, wheel_target=m['wheel_target'],
                    wheel_actual=s['wheel'], joint_position=s['q'], position_target=m['position_target'],
                    gravity=s['g'], front_z=s['front_z'], gap=m['wheel_gap'],
                    rear_fraction=s['rear_fraction'], motion=torch.stack((s['forward'], s['heading_rate']), -1),
                    failure_code=m['failure_code'], progress=m['rise_progress_reward'])
                trace.append({k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in values.items()})
            # Explicit failure mask prevents reset buffer aliasing from recording a second call.
            alive &= ~(m['failure'] | m['completed'] | term | timeout)
            if not alive.any():
                break
    report = dict(policy=False, uses_network=False, uses_training_dataset=False,
        count=len(cases), takeover=int(takeover.sum()), balance_entries=int(entered.sum()),
        unfinished=int(alive.sum()), max_front_gap=float(gap.max()),
        max_actual_front_height=float(height.max()), max_support_duration=float(support.max()),
        wheel_target_peak=peak_target, wheel_actual_peak=peak_actual, rear_effort_peak=peak_effort,
        cases=[dict(**c, env=e, gap=float(gap[e]), height=float(height[e]),
                    support=float(support[e]), progress_credit=float(credit[e]),
                    balance=bool(entered[e]), failure=int(failures[e]), unfinished=bool(alive[e]))
               for e, c in enumerate(cases)])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.stem + '_trace.json').write_text(json.dumps(trace) + '\n')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}))
    env.close()
except BaseException:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
else:
    app.close()
