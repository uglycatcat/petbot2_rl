"""Verify frozen bases and actual refinement learning in completed GPU smokes."""
import argparse,json,pathlib
import torch
p=argparse.ArgumentParser();p.add_argument('parent');p.add_argument('folders',nargs='+');p.add_argument('--output',required=True);args=p.parse_args()
parent=torch.load(args.parent,map_location='cpu',weights_only=False);checks={}
for name in args.folders:
    folder=pathlib.Path(name);meta=json.loads((folder/'config.json').read_text())
    iteration=meta['args']['iterations']-1
    trained=torch.load(folder/f'model_{iteration}.pt',map_location='cpu',weights_only=False)
    actor=trained['actor_state_dict'];critic=trained['critic_state_dict'];original=parent['actor_state_dict']
    assert all(torch.equal(actor[k],v) for k,v in original.items()), 'Frozen actor changed'
    assert torch.equal(actor['adapter_mean'],parent['critic_state_dict']['obs_normalizer._mean'])
    assert torch.equal(actor['adapter_std'],parent['critic_state_dict']['obs_normalizer._std'])
    assert actor['adapter.4.weight'].abs().sum()>0 and torch.isfinite(actor['adapter.4.weight']).all()
    samples=meta['args']['num_envs']*meta['runner']['num_steps_per_env']*meta['args']['iterations']
    assert int(critic['obs_normalizer.count'])==int(parent['critic_state_dict']['obs_normalizer.count'])+samples
    assert all(torch.isfinite(v).all() for v in actor.values())
    checks[folder.name]=dict(frozen_actor_exact=True,adapter_coordinates_frozen=True,adapter_learned=True,
        max_adapter_last_weight=float(actor['adapter.4.weight'].abs().max()),critic_added_samples=samples,iteration=iteration)
pathlib.Path(args.output).write_text(json.dumps(dict(passed=True,runs=checks),indent=2));print(checks)
