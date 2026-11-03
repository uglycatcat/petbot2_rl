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
    ac=cfg['actor'];class_name=ac.pop('class_name')
    dim=29 if origin['training'].get('task_memory')=='sensor_stop_v16' else 26
    if class_name=='step_demo.models:SensorMemoryMLP':
        from step_demo.models import SensorMemoryMLP
        actor_type=SensorMemoryMLP
    elif class_name=='MLPModel':actor_type=MLPModel
    else:raise ValueError(f'Unsupported actor class: {class_name}')
    actor=actor_type({'policy':torch.zeros(1,dim)},cfg['obs_groups'],'actor',2,**ac)
    actor.load_state_dict(ck['actor_state_dict'],strict=True);actor.eval()
    assert state_digest(actor.state_dict())==latest['actor_sha256']
    class Export(torch.nn.Module):
        def __init__(self):super().__init__();self.actor=actor
        def forward(self,observation):return self.actor({'policy':observation.clamp(-100.,100.)})
    wrapper=Export().eval();a.output.mkdir(parents=True,exist_ok=True);onnx=a.output/'policy.onnx'
    torch.onnx.export(wrapper,torch.zeros(1,dim),str(onnx),input_names=['observation'],output_names=['action'],
        dynamic_axes={'observation':{0:'batch'},'action':{0:'batch'}},opset_version=17,dynamo=False)
    import onnx as ox
    ox.checker.check_model(ox.load(onnx))
    rng=np.random.default_rng(772);obs=rng.normal(size=(2048,dim)).astype(np.float32)*3
    if dim==29:
        obs[:,26]=rng.integers(0,2,len(obs));obs[:,27]=rng.random(len(obs));obs[:,28]=rng.integers(0,2,len(obs))
    obs=np.vstack((np.zeros((1,dim),np.float32),obs,np.full((1,dim),150,np.float32),np.full((1,dim),-150,np.float32)))
    # Reconstruct observations from recorded estimator/encoder data, not truth.
    from step_demo.estimator import InertialEstimator
    asset=json.loads((batch/'asset/asset.json').read_text())
    names=[n.removesuffix('_1') for n in layout['joints']]
    recorded=[]
    with (batch/f'seed{a.seed}/evaluation/trace.jsonl').open() as f:
        for line in f:
            row=json.loads(line);n=len(row['episode']);est=InertialEstimator(n,'cpu',asset['source_bodies'],names)
            for dest,key in [('q','estimated_quaternion'),('com_position','estimated_com'),('com_velocity','estimated_velocity')]:
                setattr(est,dest,torch.tensor(row[key],dtype=torch.float32))
            x=est.observation(torch.tensor([[.02,.30,np.pi/2]]).repeat(n,1),torch.tensor(row['joint_position']),torch.tensor(row['joint_velocity']),
                [names.index(j) for j in ['left_frame','right_frame']],[names.index(j) for j in ['left_rear_wheel','right_rear_wheel']],
                torch.tensor(row['imu_gyro']),torch.tensor(row['executed']))
            if dim==29:x=torch.cat((x,torch.tensor(row['sensor_stop_memory'])),dim=-1)
            recorded.append(x.numpy())
    synthetic_count=len(obs);obs=np.vstack((obs,*recorded))
    with torch.no_grad():expected=wrapper(torch.from_numpy(obs)).numpy()
    np.savez(a.output/'export_reference.npz',observation=obs,action=expected,synthetic_count=synthetic_count)
    asset={k:asset[k] for k in ['source_bodies','initial_com','initial_root_pos','wheel_vertices']}
    asset['rear_edges']={name:convex_edges(asset['wheel_vertices'][name]) for name in ['left_rear_wheel','right_rear_wheel']}
    metadata=dict(schema='petbot_step_onnx_v1',policy_id=f'{batch.name}_seed{a.seed}_u{ck["completed_updates"]}',
        source_batch=batch.name,source_seed=a.seed,source_commit=json.loads((batch/'revision.json').read_text())['commit'],
        source_run_id=origin['run_id'],source_code_sha256=origin['code_sha256'],source_model_sha256=origin['source_sha256'],
        source_checkpoint=checkpoint.name,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        actor_sha256=state_digest(actor.state_dict()),onnx_sha256=hashlib.sha256(onnx.read_bytes()).hexdigest(),
        completed_updates=ck['completed_updates'],training=origin['training'],task=origin['task'],
        joint_names=[n.removesuffix('_1') for n in layout['joints']],physics_dt=.002,policy_dt=layout['step_dt'],
        observation_dim=dim,action_dim=2,observation_normalization='included_in_onnx',
        actor_class=class_name,memory_normalization='raw_0_to_1' if actor_type is not MLPModel else 'empirical',
        deterministic_output='native_Beta_mean',asset=asset,isaac_evaluation=analysis['evaluation'],
        limitation='Preview candidate only; no complete task success in the published Isaac evaluation. No rule-based parking.')
    (a.output/'policy.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print({k:metadata[k] for k in ['policy_id','actor_sha256','onnx_sha256']})


if __name__=='__main__':main()
