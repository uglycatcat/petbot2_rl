"""Finite first-episode evaluation; development probes are explicitly labelled."""
from pathlib import Path
import json
import torch
from .controller import Phase, Reason
from .experiment import atomic_json, rng_state, restore_rng


@torch.inference_mode()
def evaluate(env, policy, level, count, duration=None, moving=True, output=None, kind='independent_validation'):
    if kind not in ('independent_validation', 'development_curriculum_probe'):
        raise ValueError('Evaluation kind must be explicit')
    p = env.upright_runtime
    if level not in range(p.cfg.max_course_level+1):
        raise ValueError('Evaluation course level outside configured stages')
    old = (p.probe_level, p.probe_duration, p.probe_moving, env.upright_record)
    saved_rng, scenario_rng = rng_state(), p.generator.get_state()
    p.probe_level, p.probe_duration, p.probe_moving = level, duration, moving
    env.upright_record = True
    observed = min(count, env.num_envs)
    if observed != count:
        raise ValueError('More evaluation cases requested than environments')
    live = torch.ones(count, dtype=torch.bool, device=env.device)
    goal = torch.zeros_like(live); complete = goal.clone(); entered = goal.clone()
    failures = torch.zeros(count, dtype=torch.long, device=env.device)
    longest = torch.zeros(count, device=env.device)
    max_wheel = 0.; max_target = 0.; max_effort = torch.zeros(7, device=env.device)
    sums = torch.zeros(count, 2, device=env.device); samples = torch.zeros(count, device=env.device)
    traces = []; finished = False
    try:
        env.reset()
        for step in range(env.max_episode_length):
            with torch.inference_mode():
                from tensordict import TensorDict
                action = policy(TensorDict(env.obs_buf, batch_size=[env.num_envs]))
                _, _, term, timeout, _ = env.step(action)
            snap = env.upright_snapshot
            active = live.clone()
            goal |= active & snap['goal'][:count]
            complete |= active & snap['complete'][:count]
            entered |= active & snap['support_truth'][:count] & (snap['phase_before'][:count] == Phase.BALANCE)
            longest = torch.maximum(longest, torch.where(active, snap['longest_hold'][:count], 0.))
            failed = active & snap['failure'][:count]
            failures[failed] = snap['reason'][:count][failed]
            # Preserve unmatched perception/timeout failures even if their controller reason was NONE.
            failures[failed & (failures == 0)] = 8
            eligible = active & (snap['phase_before'][:count] == Phase.BALANCE) & (snap['stage_time'][:count] > .5)
            error = (snap['motion'][:count]-snap['requested_command'][:count]).abs()
            sums += error*eligible[:, None]; samples += eligible
            if active.any():
                max_wheel = max(max_wheel, float(snap['wheel'][:count][active].abs().max()))
                max_target = max(max_target, float(snap['executed_targets'][:count, 1:][active].abs().max()))
                max_effort = torch.maximum(max_effort, snap['effort'][:count][active].abs().amax(0))
            if output is not None or level >= 2:
                # All evaluated episodes, including failures, remain reviewable.
                traces.append(dict(t=(step+1)*env.step_dt, active=active.cpu().tolist(),
                    **{k: value[:count].detach().cpu().tolist() for k, value in snap.items()}))
            ended = (term | timeout)[:count]
            timed_out = active & timeout[:count] & ~snap['done'][:count]
            failures[timed_out] = 9
            live &= ~ended
            if not live.any():
                finished = True
                break
        failures[live] = 9
        effective_duration = p.cfg.hold_s if level < 2 else (duration if duration is not None else p.cfg.balance_train_s)
        from .model import load_model, POSITION_JOINTS, REAR_WHEELS
        source = load_model()
        effort_limits = max_effort.new_tensor([source['actuators'][name]['effort'] for name in POSITION_JOINTS+REAR_WHEELS])
        result = dict(kind=kind, count=count, level=level, requested_balance_s=effective_duration,
                      command_profile=p.stage_ranges[level].cpu().tolist(), command_dwell_s=float(p.stage_dwell[level]),
                      scoring_command='raw_requested_command',
                      sequence='full_signed_straight_and_mirrored_turns' if level >= 3 else 'baseline_v1',
                      all_episodes_finished=finished, stage_goal=int(goal.sum()), complete=int(complete.sum()),
                      entered_two_wheel=int(entered.sum()), success=int((goal if level == 0 else complete).sum()),
                      longest_support_s=longest.cpu().tolist(), failure_codes=failures.cpu().tolist(),
                      mean_abs_motion_error=(sums/samples.clamp_min(1)[:, None]).cpu().tolist(),
                      balance_samples=samples.cpu().tolist(), max_actual_wheel_rad_s=max_wheel,
                      max_target_wheel_rad_s=max_target, max_effort_nm=max_effort.cpu().tolist(),
                      target_limit_pass=max_target <= 25.00001,
                      effort_limit_pass=bool((max_effort <= effort_limits+1e-3).all()),
                      policy_pass_not_inferred_from_exit=True)
        if level >= 2:
            from .response import basic_response
            result.update(basic_response(traces, count, complete.cpu().tolist(), env.step_dt))
        if output is not None:
            output = Path(output); output.mkdir(parents=True, exist_ok=False)
            atomic_json(output/'result.json', result)
            with (output/'trace.jsonl').open('w') as handle:
                for frame in traces:
                    handle.write(json.dumps(frame)+'\n')
        return result
    finally:
        p.probe_level, p.probe_duration, p.probe_moving, env.upright_record = old
        p.generator.set_state(scenario_rng)
        restore_rng(saved_rng)
        # Probe episodes never enter PPO rollouts or count as independent validation.
        env.reset()
