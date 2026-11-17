"""Export an own-project checkpoint and verify portable numerical inference on CPU.
Numerical parity is not a task-success certificate; the physical evaluation remains required.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import torch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step_explore.policy import load_actor,RawPolicy,TargetPolicy
from tensordict import TensorDict

p=argparse.ArgumentParser()
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--observations',type=Path,help='NPZ with actual policy inputs under observations')
p.add_argument('--onnx',action='store_true')
p.add_argument('--verify-existing',action='store_true',help='Verify existing export files without overwriting them')
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
ac,saved=load_actor(a.checkpoint)
raw=RawPolicy(ac).eval();target=TargetPolicy(raw).eval()
if not a.verify_existing:
    raw_script=torch.jit.script(raw);target_script=torch.jit.script(target)
    raw_script.save(str(a.output/'policy_raw.pt'))
    target_script.save(str(a.output/'policy_targets.pt'))
elif (a.output/'export.json').exists():
    previous=json.loads((a.output/'export.json').read_text())
    assert previous['checkpoint_sha256']==hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    for name,digest in previous['files_sha256'].items():
        assert hashlib.sha256((a.output/name).read_bytes()).hexdigest()==digest,name
# An incomplete export may contain artifacts but no report. In that case verify
# those existing bytes against the checkpoint below before creating their report.
loaded=torch.jit.load(str(a.output/'policy_raw.pt'),map_location='cpu').eval()
loaded_targets=torch.jit.load(str(a.output/'policy_targets.pt'),map_location='cpu').eval()
torch.manual_seed(1901)
obs=torch.cat((torch.zeros(1,40),torch.randn(1024,40),torch.randn(128,40)*20.))
source='synthetic CPU inference coverage'
if a.observations:
    data=np.load(a.observations)['observations'].reshape(-1,40)
    obs=torch.from_numpy(data.astype(np.float32))
    source=str(a.observations)
with torch.inference_mode():
    ref=ac.act_inference(TensorDict({'policy':obs.clamp(-20.,20.)},batch_size=[len(obs)]))
    actual=loaded(obs);targets=loaded_targets(obs)
    torch.testing.assert_close(actual,ref,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(targets,target(obs),atol=1e-6,rtol=1e-6)
    assert torch.isfinite(targets).all()
    assert (targets[:,0].abs()<=1.3+1e-6).all()
    assert ((targets[:,1:3]>=-1.000001)&(targets[:,1:3]<=1.200001)).all()
    assert (targets[:,3:].abs()<=25.000001).all()
report={'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        'completed_updates':saved['infos']['completed_updates'],'input_shape':['batch',40],
        'raw_output_shape':['batch',5],'target_order':['head','left_frame','right_frame','left_rear_wheel','right_rear_wheel'],
        'target_units':['rad','rad','rad','rad/s','rad/s'],'control_dt':.02,
        'normalization_included':True,'parity_samples':len(obs),'parity_input_source':source,
        'raw_max_abs_error':float((actual-ref).abs().max()),'torchscript_parity_passed':True,
        'task_success_verified_by_export':False}
if a.onnx:
    if not a.verify_existing:
        torch.onnx.export(raw,obs[:1],str(a.output/'policy_raw.onnx'),opset_version=17,
            input_names=['observations'],output_names=['raw_actions'],
            dynamic_axes={'observations':{0:'batch'},'raw_actions':{0:'batch'}},dynamo=False)
    import onnx
    from onnx.reference import ReferenceEvaluator
    model=onnx.load(str(a.output/'policy_raw.onnx'));onnx.checker.check_model(model)
    output=ReferenceEvaluator(model).run(None,{'observations':obs.numpy()})[0]
    np.testing.assert_allclose(output,ref.numpy(),atol=1e-4,rtol=1e-5)
    report['onnx_max_abs_error']=float(np.abs(output-ref.numpy()).max())
    report['onnx_reference_parity_passed']=True
report['files_sha256']={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in a.output.iterdir() if x.suffix in {'.pt','.onnx'}}
report['verified_existing_artifacts']=a.verify_existing
report_name='parity_actual.json' if a.verify_existing and (a.output/'export.json').exists() else 'export.json'
(a.output/report_name).write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
