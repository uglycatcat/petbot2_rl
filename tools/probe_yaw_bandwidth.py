"""Fixed-policy yaw bandwidth diagnostic; results are not formal batch scores."""
import argparse
import hashlib
import inspect
import json
import pathlib
import sys
import textwrap
import types

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--yaw-probe-period',type=float,choices=[.02,.006,.002],default=.02)
p.add_argument('--yaw-probe-kp',type=float,choices=[8.,16.],default=8.)
options,remaining=p.parse_known_args()
assert '--evaluate' in remaining and '--checkpoint' in remaining
sys.argv=[sys.argv[0],*remaining];sys.path.insert(0,str(pathlib.Path.cwd()))


def install_probe(env,folder):
    from upright import fast_feedback,feedback_task
    a=env.action_manager.get_term('moving')
    assert env.cfg.yaw_feedback_design=='fast20' and env.cfg.decimation==30
    period=options.yaw_probe_period;cadence=round(period/env.physics_dt)
    assert abs(period-cadence*env.physics_dt)<1e-9
    # Keep every existing bound, common target, integration gain and phase gate.
    # The no-op setting calls the original methods, permitting exact replay.
    if period!=.02 or options.yaw_probe_kp!=8.:
        source=textwrap.dedent(inspect.getsource(fast_feedback.FastYawFeedback.process))
        for old,new in [('.02*4.',f'{period!r}*4.'),('8.*error',f'{options.yaw_probe_kp!r}*error'),
                        ('-12.*.02,12.*.02',f'-12.*{period!r},12.*{period!r}')]:
            assert source.count(old)==1,old
            source=source.replace(old,new)
        namespace=vars(fast_feedback).copy();exec(compile(source,__file__,'exec'),namespace)
        a.fast_feedback.process=types.MethodType(namespace['process'],a.fast_feedback)
        source=textwrap.dedent(inspect.getsource(feedback_task.FeedbackAction.apply_actions))
        assert source.count('self.inner_index%10==0')==1
        source=source.replace('self.inner_index%10==0',f'self.inner_index%{cadence}==0')
        # Explicit super avoids a missing __class__ closure in the diagnostic.
        source=source.replace('super().apply_actions()', 'super(FeedbackAction,self).apply_actions()')
        namespace=vars(feedback_task).copy();namespace['__name__']=feedback_task.__name__
        exec(compile(source,__file__,'exec'),namespace)
        a.apply_actions=types.MethodType(namespace['apply_actions'],a)
    (folder/'yaw_bandwidth_probe.json').write_text(json.dumps(dict(diagnostic_only=True,
        period=period,kp=options.yaw_probe_kp,ki=4.,difference_rate=12.,half_limit=2.,
        policy_period=.06,physics_period=env.physics_dt,active='phase2 clock>=1.2',
        source_hash=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()),indent=2))


source=pathlib.Path('train_cycle.py').read_text()
def replace(old,new):
    global source
    assert source.count(old)==1,('Evaluator changed',old)
    source=source.replace(old,new)
replace('obs=wrapped.get_observations();n=args.num_envs',
        'install_probe(env,folder);obs=wrapped.get_observations();n=args.num_envs')
replace("folder/'evaluation.json'","folder/'diagnostic_evaluation.json'")
replace("folder/'trace.json'","folder/'diagnostic_trace.json'")
exec(compile(source,'train_cycle.py','exec'),globals())
