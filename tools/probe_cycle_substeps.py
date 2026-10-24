"""Cycle-v8 evaluation-only substep recording and target-ramp interventions.

Never changes the fixed checkpoint, policy cadence, physical model or acceptance.
Run in a dedicated clean copy; diagnostic outputs are separate from batch scores.
"""
import argparse
import hashlib
import json
import pathlib
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--substep-mode', choices=['baseline', 'common_ramp', 'wheel_ramp', 'common_frame_ramp'], default='baseline')
options, remaining = parser.parse_known_args()
assert '--evaluate' in remaining
sys.argv = [sys.argv[0], *remaining]
sys.path.insert(0, str(pathlib.Path.cwd()))


def install_substep_probe(env, folder):
    import numpy as np
    import torch
    from upright.continuous_task import ensure
    from upright.cycle_task import tensor
    from upright.moving import projected_heading_rate

    action = env.action_manager.get_term('moving')
    original_process = action.process_actions
    original_apply = action.apply_actions
    rear_ids = action.robot.find_bodies(["left_rear_wheel", "right_rear_wheel"], preserve_order=True)[0]
    probe = dict(samples=[], intervals=[], selected=[], old_speed=action.speed.clone(),
                 old_frame=action.target.clone(), index=0, blend=torch.zeros(env.num_envs, device=env.device))

    def process(actions):
        probe['old_speed'].copy_(action.speed)
        probe['old_frame'].copy_(action.target)
        p = ensure(env)
        probe['blend'] = torch.where(p.phase == 2, ((p.phase_time-.6)/.6).clamp(0, 1), 0.)
        probe['index'] = 0
        probe['samples'].clear()
        original_process(actions)

    def apply():
        original_apply()
        factor = 1.-probe['blend']*(1.-(probe['index']+1)/env.cfg.decimation)
        mask = probe['blend'] > 0
        speed = action.speed.clone()
        frame = action.target.clone()
        if options.substep_mode != 'baseline':
            if options.substep_mode == 'wheel_ramp':
                ramp = probe['old_speed']+(action.speed-probe['old_speed'])*factor[:, None]
            else:
                offset = (1.-factor)*(probe['old_speed'].mean(-1)-action.speed.mean(-1))
                ramp = action.speed+offset[:, None]
            speed[mask] = ramp[mask]
            action.robot.set_joint_velocity_target(speed, joint_ids=action.wheel_ids)
            if options.substep_mode == 'common_frame_ramp':
                ramp_frame = probe['old_frame']+(action.target-probe['old_frame'])*factor[:, None]
                frame[mask] = ramp_frame[mask]
                action.robot.set_joint_position_target(frame, joint_ids=action.pos_ids)
        # Do not read ContactSensor.data here: a read can advance its lazy history
        # at a different time from the uninstrumented task.
        data = action.robot.data;root = tensor(data.root_state_w)
        x,y,z,w = root[:,3:7].unbind(-1)
        longitudinal = torch.stack((1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)),-1)
        lateral = torch.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),-1)
        heading = longitudinal[:,:2]/longitudinal[:,:2].norm(dim=-1,keepdim=True).clamp_min(1e-6)
        forward = (root[:,7:9]*heading).sum(-1)
        yaw_rate = projected_heading_rate(longitudinal,root[:,10:13])
        lateral_rate = projected_heading_rate(lateral,root[:,10:13])
        lateral_heading = torch.atan2(-lateral[:,0],lateral[:,1])
        centers = tensor(data.body_pos_w)[:,rear_ids]
        axis = centers[:,0]-centers[:,1]
        center_heading = torch.atan2(-axis[:,0],axis[:,1])
        torque = tensor(data.applied_torque)[:,action.wheel_ids]
        # Physical samples precede each 2ms step, including interval start.
        sample = torch.cat((forward[:,None],yaw_rate[:,None],
                            tensor(data.joint_vel)[:,action.wheel_ids],speed,frame[:,3:4],
                            tensor(data.joint_pos)[:,action.pos_ids[3:4]],root[:,10:13],torque,
                            lateral_heading[:,None],lateral_rate[:,None],center_heading[:,None]),-1)
        probe['samples'].append(sample)
        probe['index'] += 1

    def finish_interval():
        samples = torch.stack(probe['samples'])
        assert samples.shape[0] == env.cfg.decimation
        probe['intervals'].append(torch.stack((samples.mean(0), samples.std(0, correction=0)), 0).cpu().numpy())
        probe['selected'].append(samples[:, :4].cpu().numpy())

    def save():
        np.savez_compressed(folder/'substeps.npz', interval=np.stack(probe['intervals']), selected=np.stack(probe['selected']))

    action.process_actions = process
    action.apply_actions = apply
    probe['finish_interval'] = finish_interval
    probe['save'] = save
    (folder/'substep_probe.json').write_text(json.dumps(dict(diagnostic_only=True,
        mode=options.substep_mode, intervention='phase2 after0.6s, ramp blend over0.6s; targets reach original endpoint each control interval',
        physics_dt=env.physics_dt, control_dt=env.step_dt, selected_envs=[0, 1, 2, 3],
        fields=['forward', 'heading_rate', 'wheel_left', 'wheel_right', 'target_left', 'target_right', 'frame_target', 'frame_measured', 'wx', 'wy', 'wz', 'torque_left', 'torque_right', 'lateral_heading', 'lateral_rate', 'rear_center_heading'],
        sample_semantics='before each physics step; torque is implicit actuator estimate',
        script_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()), indent=2))
    return probe


source = pathlib.Path('train_cycle.py').read_text()
def replace(old, new):
    global source
    assert source.count(old) == 1, 'Fixed evaluator changed: inspect injection'
    source = source.replace(old, new)

replace('        alive=torch.ones(n,device=args.device,dtype=torch.bool)',
        '        substep_probe=install_substep_probe(env,folder)\n        alive=torch.ones(n,device=args.device,dtype=torch.bool)')
replace('                obs,r,done,extra=wrapped.step(action)',
        '                obs,r,done,extra=wrapped.step(action)\n                substep_probe["finish_interval"]()')
replace("        result=dict(count=n,", '        substep_probe["save"]()\n        result=dict(count=n,')
replace("folder/'evaluation.json'", "folder/'diagnostic_evaluation.json'")
replace("folder/'trace.json'", "folder/'diagnostic_trace.json'")
exec(compile(source, 'train_cycle.py', 'exec'), globals())
