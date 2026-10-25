"""Verify the frozen base and continued/new adapter learning in GPU checkpoints."""
import argparse,json,pathlib
import torch
p=argparse.ArgumentParser();p.add_argument('parent');p.add_argument('folders',nargs='+');p.add_argument('--output',required=True);args=p.parse_args()
parent=torch.load(args.parent,map_location='cpu',weights_only=False);checks={}
for name in args.folders:
    folder=pathlib.Path(name);meta=json.loads((folder/'config.json').read_text());iteration=meta['args']['iterations']-1
    trained=torch.load(folder/f'model_{iteration}.pt',map_location='cpu',weights_only=False)
    actor=trained['actor_state_dict'];critic=trained['critic_state_dict'];prior=parent['actor_state_dict']
    assert all(torch.equal(actor[k],v) for k,v in prior.items() if not k.startswith(('adapter.','yaw_adapter.')))
    assert any(not torch.equal(actor[k],v) for k,v in prior.items() if k.startswith('adapter.'))
    assert actor['yaw_adapter.4.weight'].abs().sum()>0
    if 'yaw_adapter.4.weight' in prior:
        assert any(not torch.equal(actor[k],v) for k,v in prior.items() if k.startswith('yaw_adapter.'))
    assert all(torch.isfinite(v).all() for v in actor.values())
    samples=meta['args']['num_envs']*meta['runner']['num_steps_per_env']*meta['args']['iterations']
    assert int(critic['obs_normalizer.count'])==int(parent['critic_state_dict']['obs_normalizer.count'])+samples
    checks[folder.name]=dict(frozen_base_and_coordinates_exact=True,old_adapter_updated=True,yaw_adapter_learned=True,
        max_yaw_last_weight=float(actor['yaw_adapter.4.weight'].abs().max()),critic_added_samples=samples,iteration=iteration)
pathlib.Path(args.output).write_text(json.dumps(dict(passed=True,runs=checks),indent=2));print(checks)
