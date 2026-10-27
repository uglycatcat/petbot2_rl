"""Fixed-v22 inference intervention: retain, remove, or refresh balance feedback.

The saved checkpoint contract is checked before adding diagnostic metadata.
Only phase2 after1.8s changes; training, task clocks,60ms history,2ms steering,
source physics, termination and physical target/slew limits remain unchanged.
"""
import argparse
import hashlib
import json
import pathlib
import sys

sys.path.insert(0,str(pathlib.Path.cwd()))
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--balance-refresh',choices=['identity','off','current20'],required=True)
probe,remaining=p.parse_known_args()
assert '--evaluate' in remaining and '--checkpoint' in remaining
assert '--probe' not in remaining and not any(x.startswith('--initialize') for x in remaining)
sys.argv=[sys.argv[0],*remaining]


def install(env,actor,contract,metadata):
    import torch
    from tensordict import TensorDict
    from upright.balance_reflex import attach_balance_reflex
    from upright.feedback_task import observations,ensure
    assert contract['task']=='upright_cycle_v22' and contract['balance_reflex']=='frozen20'
    assert not actor.training and env.cfg.training_mixture=='off'
    intervention=dict(kind='fixed_balance_feedback_refresh_v1',mode=probe.balance_refresh,
        active='phase2 after1.8s',outer_period=.06,inner_period=.02,
        older_history='four unchanged60ms frames',steering='unchanged2ms',
        checkpoint_weights_unchanged=True,physical_limits_unchanged=True,
        task_and_failure_conditions_unchanged=True,
        tool_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
    contract['diagnostic_intervention']=intervention
    metadata['diagnostic_intervention']=intervention
    a=env.action_manager.get_term('moving')
    assert env.step_dt==.06 and env.physics_dt==.002 and a.fast_stride==1
    if probe.balance_refresh=='identity':attach_balance_reflex(env,actor,True)
    process=a.process_actions;apply=a.apply_actions;reset=a.reset
    cache={};stats=dict(active_samples=0,refresh_calls=0,max_frame_interval=0.,max_wheel_interval=0.,max_refresh_frame=0.,max_refresh_common=0.)

    def process_actions(actions):
        cache['raw']=a.policy_observation.clone()
        cache['frame']=a.target[:,3].clone()
        cache['wheel']=a.speed.clone()
        process(actions)

    def apply_actions():
        progress=ensure(env)
        mask=(progress.phase==2)&(progress.phase_time>=1.8)
        if probe.balance_refresh=='current20' and a.inner_index in (10,20) and mask.any():
            phase=progress.phase.clone();clock=progress.phase_time.clone();command=progress.raw.clone()
            raw=cache['raw'].clone();raw[:,-58:]=observations(env)
            assert torch.equal(phase,progress.phase) and torch.equal(clock,progress.phase_time) and torch.equal(command,progress.raw)
            assert torch.equal(raw[:,:-58],cache['raw'][:,:-58])
            with torch.no_grad():action=actor(TensorDict({'policy':raw},batch_size=[env.num_envs]))
            previous=a.inner_policy.mean(-1);desired=action[:,1]*a.max_speed
            common=previous+(desired-previous).clamp(-env.cfg.wheel_slew*.02,env.cfg.wheel_slew*.02)
            bound=env.cfg.wheel_slew*env.step_dt;anchor=a.inner_anchor
            lower=(anchor-bound).clamp_min(-a.max_speed).mean(-1)
            upper=(anchor+bound).clamp_max(a.max_speed).mean(-1)
            common=common.clamp(min=lower,max=upper)
            stats['max_refresh_common']=max(stats['max_refresh_common'],float((common[mask]-previous[mask]).abs().max()))
            a.inner_policy[mask]=common[mask,None]
            desired=a.lo+(action[:,0]+1)*.5*(a.hi-a.lo)
            previous=a.target[:,3];limit=env.cfg.frame_slew*.02
            frame=previous+(desired-previous).clamp(-limit,limit)
            bound=env.cfg.frame_slew*env.step_dt
            frame=frame.clamp(min=cache['frame']-bound,max=cache['frame']+bound).clamp(a.lo,a.hi)
            stats['max_refresh_frame']=max(stats['max_refresh_frame'],float((frame[mask]-previous[mask]).abs().max()))
            a.target[mask,3:]=frame[mask,None]
            a.applied[:,0]=2*(a.target[:,3]-a.lo)/(a.hi-a.lo)-1
            stats['refresh_calls']+=1;stats['active_samples']+=int(mask.sum())
        apply()
        f=float((a.target[:,3]-cache['frame']).abs().max());w=float((a.speed-cache['wheel']).abs().max())
        assert f<=env.cfg.frame_slew*env.step_dt+1e-5 and w<=env.cfg.wheel_slew*env.step_dt+1e-4
        assert a.speed.abs().max()<=a.max_speed+1e-5 and a.target[:,3].min()>=a.lo-1e-5 and a.target[:,3].max()<=a.hi+1e-5
        stats['max_frame_interval']=max(stats['max_frame_interval'],f);stats['max_wheel_interval']=max(stats['max_wheel_interval'],w)

    def reset_actions(env_ids=None):
        reset(env_ids)
        ids=slice(None) if env_ids is None else env_ids
        for value in cache.values():value[ids]=0.

    a.process_actions=process_actions;a.apply_actions=apply_actions;a.reset=reset_actions
    return stats


source=pathlib.Path('train_cycle.py').read_text()
anchor="    attach_balance_reflex(env,runner.alg.get_policy(),args.balance_reflex=='frozen20')"
assert source.count(anchor)==1
# get_inference_policy below switches to eval; do that before diagnostic setup.
source=source.replace(anchor,"    runner.alg.get_policy().eval()\n    probe_stats=install(env,runner.alg.get_policy(),contract,metadata)")
anchor='        result=dict(max_wheel_axis_alignment_error='
assert source.count(anchor)==1
source=source.replace(anchor,"        (folder/'balance_refresh_review.json').write_text(json.dumps(dict(diagnostic_only=True,mode=probe.balance_refresh,**probe_stats),indent=2))\n"+anchor)
exec(compile(source,'train_cycle.py','exec'),globals())
