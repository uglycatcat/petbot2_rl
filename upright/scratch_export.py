"""Export the single scratch actor; validate ONNX on random and live inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def export(actor, observation, folder, metadata, checkpoint):
    import onnx
    import onnxruntime as ort
    folder=Path(folder);folder.mkdir(exist_ok=False)
    model=actor.as_onnx(False).cpu().eval()
    path=folder/'policy.onnx'
    torch.onnx.export(model,(torch.zeros(1,290),),str(path),opset_version=17,
        input_names=['observation'],output_names=['action'],
        dynamic_axes={'observation':{0:'batch'},'action':{0:'batch'}},dynamo=False)
    onnx.checker.check_model(str(path))
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    generator=torch.Generator(device='cpu').manual_seed(20261029)
    samples=torch.cat((observation.detach().cpu(),torch.randn(256,290,generator=generator)))
    with torch.no_grad():expected=model(samples).numpy()
    actual=session.run(None,{'observation':samples.numpy()})[0]
    error=float(np.max(np.abs(expected-actual)))
    if not np.isfinite(actual).all() or error>1e-5:raise RuntimeError(f'ONNX parity failed: {error}')
    manifest=dict(schema='petbot_scratch_onnx_v1',artifact='policy.onnx',
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        source_commit=metadata['commit'],origin=metadata['scratch_origin'],
        control_contract=metadata['contract'],input_shape=['batch',290],output_shape=['batch',3],
        onnx_max_abs_error=error,parity_samples=len(samples),sim2sim_passed=False,
        deployment_requires='20ms policy/history, matching task state machine and explicit 2ms controls')
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
