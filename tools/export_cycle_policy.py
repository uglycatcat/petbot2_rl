"""Export both deterministic networks from a completed, traced cycle evaluation.

This is a neural-network bundle, not a standalone simulator controller. The
manifest preserves the task contract and names the required control plumbing.
Run inside the training container; CUDA checks exported CPU-portable graphs.
"""
import argparse
import copy
import hashlib
import json
import pathlib
import subprocess
import sys

import torch
from tensordict import TensorDict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from upright.balance_reflex import ReflexAdapterModel
from upright.model import load_model, POSITION_JOINTS, POSITION_KP, POSITION_KD, REAR_WHEELS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(evaluation, output):
    assert torch.cuda.is_available(), 'Export parity must be checked on CUDA'
    meta_path = evaluation / 'config.json'
    meta = json.loads(meta_path.read_text())
    result_path = evaluation / 'evaluation.json'
    result = json.loads(result_path.read_text())
    end = json.loads(evaluation.with_name(evaluation.name + '.exit.json').read_text())
    assert end['exit_code'] == 0 and not end['guard_stopped'] and not end.get('physics_error')
    assert result['count'] == 64 and result['unfinished'] == 0
    contract = meta['contract']
    assert contract['task'] == 'upright_cycle_v18'
    assert contract['balance_reflex'] == 'frozen20' and contract['yaw_feedback_design'] == 'fast2'
    assert not meta['working_tree_dirty']
    model = load_model()
    assert model['source_sha256'] == contract['source_sha256']
    # Exporter-only changes may follow the accepted task; task source must match.
    assert not subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip()
    assert not subprocess.check_output(['git', '-C', str(ROOT), 'diff', meta['commit'], 'HEAD',
                                       '--', 'train_cycle.py', 'upright'], text=True)
    args = meta['args']
    if args['probe'] and args['initialize_cycle_from']:
        checkpoint = pathlib.Path(args['initialize_cycle_from'])
        assert digest(checkpoint) == meta['initialization']['checkpoint_sha256']
    else:
        assert args['checkpoint'] and not args['initialize_cycle_from']
        checkpoint = pathlib.Path(args['checkpoint'])
        parent_meta = json.loads((checkpoint.parent / 'config.json').read_text())
        assert parent_meta['contract'] == contract
    saved = torch.load(checkpoint, map_location='cuda:0', weights_only=False)
    torch.manual_seed(18)
    stats = saved['critic_state_dict']
    raw = stats['obs_normalizer._mean'] + torch.randn(1024, 290, device='cuda:0') * (
        stats['obs_normalizer._std'] + .01)
    # Cover all stages and the adapter's transition, not just a single balance row.
    raw[:, -58+41:-58+46] = 0.
    raw[torch.arange(len(raw)), -58+41+torch.arange(len(raw)) % 5] = 1.
    raw[:, -58+48] = torch.linspace(0., 1., len(raw), device=raw.device)
    obs = TensorDict({'policy': raw}, batch_size=[len(raw)])
    cfg = copy.deepcopy(meta['runner']['actor']); cfg.pop('class_name')
    actor = ReflexAdapterModel(obs, obs_groups={'actor': ['policy']}, obs_set='actor',
                              output_dim=3, **cfg).to('cuda:0').eval()
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    output.mkdir(parents=True, exist_ok=False)
    outer = torch.jit.script(copy.deepcopy(actor.as_jit()).cpu().eval())
    inner = torch.jit.script(copy.deepcopy(actor.balance_reflex).cpu().eval())
    outer.save(str(output / 'actor.pt')); inner.save(str(output / 'balance_reflex.pt'))
    loaded_outer = torch.jit.load(str(output / 'actor.pt'), map_location='cuda:0')
    loaded_inner = torch.jit.load(str(output / 'balance_reflex.pt'), map_location='cuda:0')
    errors = {}
    with torch.no_grad():
        for name, expected, actual in (
            ('outer', actor(obs), loaded_outer(raw)),
            ('inner', actor.balance_reflex(raw), loaded_inner(raw)),
            ('inner_base', actor.balance_reflex.base(raw), loaded_inner.base(raw)),
        ):
            assert torch.isfinite(actual).all()
            assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6), name
            errors[name] = float((expected-actual).abs().max())
        assert loaded_outer(raw).abs().max() <= .99 + 1e-6
    sources = ['upright/balance_reflex.py', 'upright/feedback_task.py',
               'upright/continuous_task.py', 'upright/continuous.py', 'upright/cycle.py',
               'upright/cycle_task.py', 'upright/local_frame.py', 'upright/fast_feedback.py',
               'upright/feedback.py', 'upright/ground_motion.py', 'upright/model.py',
               'upright/task.py', 'upright/asset_cfg.py', 'upright/moving.py']
    manifest = dict(
        format='petbot_cycle_network_bundle_v1', task_commit=meta['commit'],
        exporter_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        checkpoint_name=checkpoint.name, checkpoint_sha256=digest(checkpoint),
        evaluation_name=evaluation.name, evaluation_sha256=digest(result_path),
        configuration_sha256=digest(meta_path), contract=contract,
        tested_range={k: args[k] for k in ('speed_limit', 'yaw_limit', 'frame_min', 'frame_max', 'hold_min', 'hold_max')},
        evaluation_outcome={k: result[k] for k in ('count', 'complete', 'failure', 'unfinished')},
        networks={name: digest(output/name) for name in ('actor.pt', 'balance_reflex.pt')},
        actuation=dict(position_joints=POSITION_JOINTS, velocity_joints=REAR_WHEELS,
                       position_kp=POSITION_KP, position_kd=POSITION_KD,
                       joints=model['joints'], drives=model['actuators']),
        interfaces=dict(input='float32 [N,290], five encoded 58-feature samples, oldest first',
                        outer='[N,3]: normalized frame position, common rear-wheel speed, yaw residual',
                        inner='[N,2]: bounded frame/common corrections; .base returns base frame/common',
                        timing='outer 60ms; frozen feedback at +20/+40ms; yaw feedback 2ms'),
        controller_sources={name: digest(ROOT/name) for name in sources},
        limitations=['Graphs do not include observations, history, state machine, ordinary control, '
                     'coordinate conversion, inner scheduling, action limits or position/velocity servos.',
                     'Use the pinned controller sources and contract together. This is not MuJoCo validation.'],
        parity=dict(passed=True, cuda=True, samples=1024, max_abs_error=errors))
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(output=str(output), parity=manifest['parity']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('evaluation', type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()
    export(args.evaluation, args.output)
