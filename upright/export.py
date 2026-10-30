"""Single-actor ONNX export with numerical parity; not a simulator acceptance test."""
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from .learning import runner_config
from .runtime import OBSERVATION_DIM, OBSERVATION_FIELDS
from .model import load_model
from .experiment import atomic_json


def make_actor(cfg):
    obs = TensorDict({'policy': torch.zeros(1, OBSERVATION_DIM*cfg.history)}, batch_size=[1])
    actor_cfg = runner_config(cfg, load_model(), 'cpu')['actor'].copy()
    actor_cfg.pop('class_name')
    return MLPModel(obs, {'actor': ['policy']}, 'actor', 3, **actor_cfg)


def export_actor(actor, cfg, destination, metadata):
    import onnx
    import onnxruntime as ort
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    actor = actor.cpu().eval()
    module = actor.as_onnx(verbose=False).cpu().eval()
    path = destination/'policy.onnx'
    torch.onnx.export(module, module.get_dummy_inputs(), path, input_names=['obs'], output_names=['actions'],
                      opset_version=17, dynamic_axes={'obs': {0: 'batch'}, 'actions': {0: 'batch'}}, dynamo=False)
    onnx.checker.check_model(str(path))
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    probe = torch.randn(32, OBSERVATION_DIM*cfg.history, generator=torch.Generator().manual_seed(179))
    with torch.no_grad():
        reference = actor(TensorDict({'policy': probe}, batch_size=[len(probe)])).numpy()
    actual = session.run(None, {'obs': probe.numpy()})[0]
    error = float(np.abs(actual-reference).max())
    if not np.isfinite(actual).all() or error > 1e-5 or np.abs(actual).max() > 1.00001:
        raise RuntimeError(f'ONNX numerical contract failed: maximum difference {error}')
    spec = dict(schema='upright_restart_onnx_v1', model='policy.onnx', metadata=metadata,
                observation_fields=OBSERVATION_FIELDS, frame_dim=OBSERVATION_DIM, history=cfg.history,
                history_order='Isaac observation group: oldest frame first, current frame last',
                quaternion_order='xyzw', control_period_s=cfg.decimation*load_model()['physics_dt'],
                action_order=['common_hip_normalized', 'common_wheel_normalized', 'differential_wheel_normalized'],
                action_range=[-1, 1], control_config=cfg.to_dict(),
                numeric_max_abs_error=error, simulation_validated=False,
                compatible_with_legacy_C58_runtime=False)
    atomic_json(destination/'manifest.json', spec)
    return spec
