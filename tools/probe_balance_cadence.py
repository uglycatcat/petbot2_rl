"""Fixed-v14 deterministic balance-only refresh; never a PPO training entry.

Outer policy/history/task clocks remain60ms. At20/40ms, optionally refresh
frame/common targets from current physical features and the same older frames.
No observation-manager history or task clock is advanced by the inner reads.
"""
import argparse
import hashlib
import json
import pathlib
import sys

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--balance-refresh',choices=['baseline','common','full','adapter'],default='baseline')
options,remaining=p.parse_known_args()
assert '--evaluate' in remaining and '--checkpoint' in remaining
sys.argv=[sys.argv[0],*remaining];sys.path.insert(0,str(pathlib.Path.cwd()))


def install_probe(runner,env,folder,policy):
    import torch
    from tensordict import TensorDict
    from upright.feedback_task import observations,ensure
    from upright.distribution import BoundedBetaMean
    from upright.balance_adapter import balance_gate
    a=env.action_manager.get_term('moving');actor=runner.alg.get_policy()
    assert env.step_dt==.06 and env.physics_dt==.002 and a.fast_stride==1
    original_process=a.process_actions;original_apply=a.apply_actions
    cache={};stats=dict(inner_calls=0,active_samples=0,max_frame_interval=0.,max_wheel_interval=0.)

    def outer(obs):
        cache['raw']=obs['policy'].clone()
        if options.balance_refresh=='adapter':
            cache['base']=BoundedBetaMean(127.)(actor.mlp(actor.obs_normalizer(cache['raw'])))
        return policy(obs)

    def process(actions):
        cache['frame_anchor']=a.target[:,3].clone()
        original_process(actions)

    def apply():
        progress=ensure(env);mask=(progress.phase==2)&(progress.phase_time>=1.8)
        if options.balance_refresh!='baseline' and a.inner_index in (10,20) and mask.any():
            command=progress.raw.clone();phase=progress.phase.clone();clock=progress.phase_time.clone()
            raw=cache['raw'].clone();raw[:,-58:]=observations(env)
            assert torch.equal(command,progress.raw) and torch.equal(phase,progress.phase) and torch.equal(clock,progress.phase_time)
            assert torch.equal(raw[:,:-58],cache['raw'][:,:-58])
            if options.balance_refresh=='adapter':
                features=(raw-actor.adapter_mean)/(actor.adapter_std+.01)
                delta=torch.tanh(actor.adapter(features))*actor.adapter_scale*balance_gate(raw)[:,None]
                action=cache['base'].clone();action[:,:2]=(action[:,:2]+delta).clamp(-.99,.99)
            else:
                action=policy(TensorDict({'policy':raw},batch_size=[env.num_envs]))
            # Keep the original differential controller/residual. Refresh only
            # common wheel speed, with both20ms and whole60ms slew bounds.
            previous=a.inner_policy.mean(-1);desired=action[:,1]*a.max_speed
            common=previous+(desired-previous).clamp(-env.cfg.wheel_slew*.02,env.cfg.wheel_slew*.02)
            bound=env.cfg.wheel_slew*env.step_dt;anchor=a.inner_anchor
            lower=(anchor-bound).clamp_min(-a.max_speed).mean(-1)
            upper=(anchor+bound).clamp_max(a.max_speed).mean(-1)
            common=common.clamp(min=lower,max=upper)
            a.inner_policy[mask]=common[mask,None]
            if options.balance_refresh in ('full','adapter'):
                desired=a.lo+(action[:,0]+1)*.5*(a.hi-a.lo)
                previous=a.target[:,3];limit=env.cfg.frame_slew*.02
                frame=previous+(desired-previous).clamp(-limit,limit)
                bound=env.cfg.frame_slew*env.step_dt
                frame=frame.clamp(min=cache['frame_anchor']-bound,max=cache['frame_anchor']+bound).clamp(a.lo,a.hi)
                a.target[mask,3:]=frame[mask,None]
                a.applied[:,0]=2*(a.target[:,3]-a.lo)/(a.hi-a.lo)-1
            stats['inner_calls']+=1;stats['active_samples']+=int(mask.sum())
        original_apply()
        if options.balance_refresh!='baseline':
            f=(a.target[:,3]-cache['frame_anchor']).abs().max()
            w=(a.speed-a.inner_anchor).abs().max()
            assert f<=env.cfg.frame_slew*env.step_dt+1e-5
            assert w<=env.cfg.wheel_slew*env.step_dt+1e-4
            stats['max_frame_interval']=max(stats['max_frame_interval'],float(f))
            stats['max_wheel_interval']=max(stats['max_wheel_interval'],float(w))

    a.process_actions=process;a.apply_actions=apply
    def finish():
        (folder/'balance_refresh_probe.json').write_text(json.dumps(dict(diagnostic_only=True,
            mode=options.balance_refresh,outer_period=.06,inner_period=.02,active='phase2 clock>=1.8',
            observation='current physical frame with unchanged older four60ms frames and cached task progress',
            **stats,source_hash=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()),indent=2))
    return outer,finish


source=pathlib.Path('train_cycle.py').read_text()
def replace(old,new):
    global source
    assert source.count(old)==1,('Evaluator changed',old)
    source=source.replace(old,new)
replace('policy=runner.get_inference_policy(device=args.device);obs=',
        'policy,finish_probe=install_probe(runner,env,folder,runner.get_inference_policy(device=args.device));obs=')
replace('        result=dict(max_wheel_axis_alignment_error=',
        '        finish_probe()\n        result=dict(max_wheel_axis_alignment_error=')
replace("folder/'evaluation.json'","folder/'diagnostic_evaluation.json'")
replace("folder/'trace.json'","folder/'diagnostic_trace.json'")
exec(compile(source,'train_cycle.py','exec'),globals())
