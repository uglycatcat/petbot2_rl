"""PhysX acceptance: cached history/clocks and physical substep action limits."""
import json,pathlib,sys
sys.path.insert(0,str(pathlib.Path.cwd()))
assert '--probe' in sys.argv

def install_validation(env,folder):
    import torch
    from upright.feedback_task import ensure
    a=env.action_manager.get_term('moving');process=a.process_actions;apply=a.apply_actions
    cache={};stats=dict(steps=0,active_steps=0,max_frame_interval=0.,max_wheel_interval=0.)
    def process_actions(action):
        cache['frame']=a.target[:,3].clone();cache['wheel']=a.speed.clone()
        process(action)
    def apply_actions():
        p=ensure(env);phase=p.phase.clone();clock=p.phase_time.clone();command=p.raw.clone()
        history=a.policy_observation.clone()
        apply()
        assert torch.equal(phase,p.phase) and torch.equal(clock,p.phase_time) and torch.equal(command,p.raw)
        assert torch.equal(history,a.policy_observation)
        f=float((a.target[:,3]-cache['frame']).abs().max());w=float((a.speed-cache['wheel']).abs().max())
        assert f<=env.cfg.frame_slew*env.step_dt+1e-5 and w<=env.cfg.wheel_slew*env.step_dt+1e-4
        assert a.speed.abs().max()<=a.max_speed+1e-5 and a.target[:,3].min()>=a.lo-1e-5 and a.target[:,3].max()<=a.hi+1e-5
        stats['steps']+=1;stats['active_steps']+=int(((p.phase==2)&(p.phase_time>=1.8)).any())
        stats['max_frame_interval']=max(stats['max_frame_interval'],f);stats['max_wheel_interval']=max(stats['max_wheel_interval'],w)
    a.process_actions=process_actions;a.apply_actions=apply_actions
    def finish():
        assert stats['active_steps']>0
        (folder/'reflex_physical_contract.json').write_text(json.dumps(dict(passed=True,**stats),indent=2))
    return finish

source=pathlib.Path('train_cycle.py').read_text()
old="    attach_balance_reflex(env,runner.alg.get_policy(),args.balance_reflex=='frozen20')"
assert source.count(old)==1;source=source.replace(old,old+'\n    finish_validation=install_validation(env,folder)')
old='        result=dict(max_wheel_axis_alignment_error='
assert source.count(old)==1;source=source.replace(old,'        finish_validation()\n'+old)
exec(compile(source,'train_cycle.py','exec'),globals())
