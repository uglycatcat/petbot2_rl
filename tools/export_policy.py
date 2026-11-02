"""CPU-only export in the Step training container; no training or simulator startup."""
import argparse,hashlib,json,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();batch=a.batch.resolve();run=batch/f'seed{a.seed}/train'
    sys.path.insert(0,str(batch/'source'))
    import numpy as np
    import torch
    from rsl_rl.models import MLPModel
    from step_demo.ppo import runner_config,state_digest
    from step_demo.terrain import convex_edges
    torch.set_num_threads(1)
    origin=json.loads((run/'origin.json').read_text());latest=json.loads((run/'latest.json').read_text())
    analysis=json.loads((batch/f'seed{a.seed}/analysis.json').read_text())
    if not analysis['valid'] or analysis['completed_updates']!=origin['training']['iterations']:
        raise ValueError('Only a completed independently evaluated run can be published')
    checkpoint=run/latest['checkpoint'];ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['origin']==origin and ck['completed_updates']==analysis['completed_updates']
    layout=json.loads((run/'layout.json').read_text());cfg=runner_config(origin['training'])
    ac=cfg['actor'];ac.pop('class_name');actor=MLPModel({'policy':torch.zeros(1,26)},cfg['obs_groups'],'actor',2,**ac)
    actor.load_state_dict(ck['actor_state_dict'],strict=True);actor.eval()
    assert state_digest(actor.state_dict())==latest['actor_sha256']
    class Export(torch.nn.Module):
        def __init__(self):super().__init__();self.actor=actor
        def forward(self,observation):return self.actor({'policy':observation.clamp(-100.,100.)})
    wrapper=Export().eval();a.output.mkdir(parents=True,exist_ok=True);onnx=a.output/'policy.onnx'
    torch.onnx.export(wrapper,torch.zeros(1,26),str(onnx),input_names=['observation'],output_names=['action'],
        dynamic_axes={'observation':{0:'batch'},'action':{0:'batch'}},opset_version=17,dynamo=False)
    import onnx as ox
    ox.checker.check_model(ox.load(onnx))
    rng=np.random.default_rng(772);obs=rng.normal(size=(2048,26)).astype(np.float32)*3
    obs=np.vstack((np.zeros((1,26),np.float32),obs,np.full((1,26),150,np.float32),np.full((1,26),-150,np.float32)))
    with torch.no_grad():expected=wrapper(torch.from_numpy(obs)).numpy()
    np.savez(a.output/'export_reference.npz',observation=obs,action=expected)
    asset=json.loads((batch/'asset/asset.json').read_text())
    asset={k:asset[k] for k in ['source_bodies','initial_com','initial_root_pos','wheel_vertices']}
    asset['rear_edges']={name:convex_edges(asset['wheel_vertices'][name]) for name in ['left_rear_wheel','right_rear_wheel']}
    metadata=dict(schema='petbot_step_onnx_v1',policy_id=f'{batch.name}_seed{a.seed}_u{ck["completed_updates"]}',
        source_batch=batch.name,source_seed=a.seed,source_commit=json.loads((batch/'revision.json').read_text())['commit'],
        source_run_id=origin['run_id'],source_code_sha256=origin['code_sha256'],source_model_sha256=origin['source_sha256'],
        source_checkpoint=checkpoint.name,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        actor_sha256=state_digest(actor.state_dict()),onnx_sha256=hashlib.sha256(onnx.read_bytes()).hexdigest(),
        completed_updates=ck['completed_updates'],training=origin['training'],task=origin['task'],
        joint_names=[n.removesuffix('_1') for n in layout['joints']],physics_dt=.002,policy_dt=layout['step_dt'],
        observation_dim=26,action_dim=2,observation_normalization='included_in_onnx',
        deterministic_output='native_Beta_mean',asset=asset,isaac_evaluation=analysis['evaluation'],
        limitation='Preview candidate only; no complete task success in the published Isaac evaluation. No rule-based parking.')
    (a.output/'policy.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print({k:metadata[k] for k in ['policy_id','actor_sha256','onnx_sha256']})


if __name__=='__main__':main()
