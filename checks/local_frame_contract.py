"""CUDA metamorphic test: global yaw must not change local physical features."""
import argparse,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from upright.local_frame import ground_frame_features
p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
assert torch.cuda.is_available();device='cuda:0';torch.manual_seed(17)
x=torch.randn(1024,49,device=device);yaw=torch.randn(1024,device=device)*3
reference=torch.randn_like(yaw);x[:,31]=reference.sin();x[:,32]=reference.cos()
amplitude=torch.rand_like(yaw);amplitude[:10]=0.
x[:,3]=amplitude*yaw.cos();x[:,4]=amplitude*yaw.sin()
rotation=torch.randn_like(yaw)*6;rotated=x.clone();c,s=rotation.cos(),rotation.sin()
for first in (3,5,8,27,29):
    a,b=x[:,first],x[:,first+1]
    rotated[:,first]=c*a-s*b;rotated[:,first+1]=s*a+c*b
rotated[:,31]=(reference+rotation).sin();rotated[:,32]=(reference+rotation).cos()
local=ground_frame_features(x,yaw,'local')
other=ground_frame_features(rotated,yaw+rotation,'local')
assert torch.allclose(local,other,atol=6e-6,rtol=3e-6)
assert torch.equal(ground_frame_features(x,yaw,'world'),x)
assert torch.equal(ground_frame_features(x,torch.zeros_like(yaw),'local'),x)
changed={3,4,5,6,8,9,27,28,29,30,31,32}
unchanged=[i for i in range(49) if i not in changed]
assert torch.equal(local[:,unchanged],x[:,unchanged])
assert torch.allclose(local[:,3],amplitude,atol=1e-6)
assert local[:,4].abs().max()<1e-6
assert torch.allclose(local[:,31],(reference-yaw).sin(),atol=2e-6)
assert torch.allclose(local[:,32],(reference-yaw).cos(),atol=2e-6)
script=torch.jit.script(ground_frame_features)
assert torch.equal(script(x,yaw,'local'),local)
result=dict(passed=True,global_yaw_invariance=True,zero_heading_identity=True,
    preserved_body_and_task_features=True,zero_heading_amplitude_finite=True,
    scripted=True,max_invariance_error=float((local-other).abs().max()))
pathlib.Path(args.output).write_text(json.dumps(result,indent=2));print(result)
