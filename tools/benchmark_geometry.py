"""Container-only equivalence and timing probe; never policy training data."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    if not Path('/.dockerenv').exists():raise RuntimeError('Use the STEP container')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=2048)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    import torch
    from step_demo.terrain import convex_edges,wheel_terrain_clearance,compiled_clearance
    from step_demo.tensor_math import rotate
    torch.manual_seed(914)
    asset=json.loads((a.asset_dir/'asset.json').read_text())
    fused=compiled_clearance();reports={}
    n=a.num_envs;device='cuda:0'
    q=torch.randn(n,4,device=device);q/=q.norm(dim=-1,keepdim=True)
    axes=rotate(q[:,None,:].expand(-1,3,-1),torch.eye(3,device=device)[None].expand(n,-1,-1))
    translation=torch.rand(n,1,3,device=device)*torch.tensor([2.3,2.2,.24],device=device)+torch.tensor([-.15,-1.1,-.06],device=device)
    for name,vertices in asset['wheel_vertices'].items():
        v=torch.tensor(vertices,device=device);edges=torch.tensor(convex_edges(vertices),device=device)
        points=torch.matmul(v,axes)+translation
        args=(points,edges,torch.zeros(n,device=device),torch.zeros(n,device=device),.02,2.,1.)
        with torch.inference_mode():
            expected=wheel_terrain_clearance(*args);actual=fused(*args)
            torch.testing.assert_close(actual,expected,atol=2e-7,rtol=2e-6)
            assert torch.equal(actual>.004,expected>.004), 'Airborne decisions differ'
            times={}
            for label,fn in [('script',wheel_terrain_clearance),('compiled',fused)]:
                fn(*args);torch.cuda.synchronize()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10):fn(*args)
                end.record();torch.cuda.synchronize();times[label]=start.elapsed_time(end)/10.
            reports[name]=dict(vertices=len(vertices),edges=len(edges),
                max_error_m=float((expected-actual).abs().max()),threshold_disagreements=0,milliseconds=times)
    (a.output/'result.json').write_text(json.dumps(dict(status='passed',num_envs=n,reports=reports,
        used_for_training=False,source_mesh_modified=False),indent=2)+'\n')


if __name__=='__main__':main()
