"""Evaluate frozen policies with batch22's original scenario generator.

This overrides only v25's evaluation RNG and command sampler. It preserves
each policy's controller, timeout, action mapping, and landing destination.
The old six-class benchmark is a regression check, not the new eight-class
capability acceptance. Matching draws do not imply identical physical states
or command switch times: those still depend on the native closed loop.
"""
import hashlib
import pathlib
import sys


def install():
    from upright.capability import CapabilityProgress
    from upright.full_range import FullRangeProgress
    from upright.moving import MovingProgress

    CapabilityProgress.uniform = MovingProgress.uniform
    CapabilityProgress.sample_commands = FullRangeProgress.sample_commands
    return dict(
        name='batch22_six_class_retention_v1',
        reference_commit='62add951c8b12773bae5d0242fc07518a6ff2c51',
        matching='seed_environment_generation_command_count',
        command_timing='native_ramp_and_phase_clocks_not_forced_state_replay',
        independent_holdout=False,
        network_and_controller_unchanged=True,
        landing_destination='native_policy_contract',
        physical_parameters_unchanged=True,
        source_sha256={name:hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
                       for name in ('upright/moving.py', 'upright/continuous.py',
                                    'upright/full_range.py')},
        wrapper_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())


def prepare_source(source):
    anchor = '    cfg=CycleEnvCfg();cfg.scene.num_envs=args.num_envs;'
    assert source.count(anchor) == 1
    source = source.replace(anchor, "    retention_spec=install()\n"+anchor)
    anchor = "    metadata['effective_training_mixture']=cfg.training_mixture"
    assert source.count(anchor) == 1
    source = source.replace(anchor, anchor+'''
    assert args.evaluate and cfg.training_mixture=='off'
    assert args.evaluation_pattern=='mixed'
    contract['scenario_override']=retention_spec
    contract['command_sampling']='feasible_six_sequence_v4'
    metadata['scenario_override']=retention_spec
''')
    return source


def run():
    assert '--evaluate' in sys.argv and '--checkpoint' in sys.argv
    assert '--probe' not in sys.argv
    assert not any(x.startswith(('--initialize', '--allow-dirty')) for x in sys.argv)
    sys.path.insert(0, str(pathlib.Path.cwd()))
    source = prepare_source(pathlib.Path('train_cycle.py').read_text())
    exec(compile(source, 'train_cycle.py', 'exec'), dict(globals(), __name__='__main__'))


if __name__ == '__main__':
    run()
