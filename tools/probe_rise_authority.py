"""Fixed-checkpoint rise-authority diagnostics; never trains or edits a checkpoint.

Gain 1/1 is an identity control. Larger gains change only the in-memory rise
scale buffer. Record this explicitly in the contract and retain physical action
limits, slew, phase clocks, failure conditions and the frozen balance controller.
"""
import argparse
import hashlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.cwd()))
p = argparse.ArgumentParser(add_help=False)
p.add_argument('--rise-frame-gain', type=float, choices=(1., 2.), default=1.)
p.add_argument('--rise-wheel-gain', type=float, choices=(1., 4.), default=1.)
probe, remaining = p.parse_known_args()
sys.argv = [sys.argv[0], *remaining]
assert '--evaluate' in sys.argv, 'Diagnostics require a saved checkpoint, never PPO'


def configure(actor, contract, metadata):
    import torch
    from upright.rise_adapter import RiseAdapterModel
    assert type(actor) is RiseAdapterModel
    old = actor.rise_scale.clone()
    assert torch.equal(old, old.new_tensor([.15, .25]))
    gains = old.new_tensor([probe.rise_frame_gain, probe.rise_wheel_gain])
    actor.rise_scale.copy_(old * gains)
    intervention = dict(kind='fixed_rise_scale_diagnostic_v1',
                        frame_gain=probe.rise_frame_gain, wheel_gain=probe.rise_wheel_gain,
                        original_limits=old.tolist(), limits=actor.rise_scale.tolist(),
                        checkpoint_weights_unchanged=True, state_buffer_scaled_in_memory=True,
                        physical_limits_unchanged=True,
                        tool_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
    contract['diagnostic_intervention'] = intervention
    metadata['diagnostic_intervention'] = intervention


source = pathlib.Path('train_cycle.py').read_text()
anchor = "    from upright.balance_reflex import attach_balance_reflex"
assert source.count(anchor) == 1
source = source.replace(anchor, "    configure(runner.alg.get_policy(),contract,metadata)\n" + anchor)
anchor = "                    lift_gate=rise_gate(obs['policy']);lift_correction=action-parent_action"
assert source.count(anchor) == 1
source = source.replace(anchor, anchor + '''
                    raw_lift=obs['policy']
                    requested_delta=torch.tanh(actor.rise_adapter((raw_lift-actor.rise_mean)/(actor.rise_std+.01)))*actor.rise_scale*lift_gate[:,None]
                    requested_delta=torch.cat((requested_delta,torch.zeros_like(requested_delta[:,:1])),-1)
                    before_clamp=parent_action+requested_delta
                    assert torch.equal(action,before_clamp.clamp(-.99,.99))
                    clip_mask=before_clamp.abs()>.99
''')
anchor = "lift_correction.new_tensor([.15,.25,0.])"
assert source.count(anchor) == 1
source = source.replace(anchor, "torch.cat((actor.rise_scale,actor.rise_scale.new_zeros(1)))")
anchor = '                trace.append(dict(rise_adapter_gate='
assert source.count(anchor) == 1
source = source.replace(anchor, '''                trace.append(dict(
                    rise_requested_correction=requested_delta.tolist(),
                    rise_parent_action=parent_action.tolist(),
                    rise_preclip_action=before_clamp.tolist(),
                    rise_clipped=clip_mask.tolist(),
                    rise_adapter_gate=''')
exec(compile(source, 'train_cycle.py', 'exec'), globals())
