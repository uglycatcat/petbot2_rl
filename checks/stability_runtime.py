"""Validate actual GPU short-training updates and frozen inherited controllers."""
import argparse,json,pathlib
import torch
p=argparse.ArgumentParser();p.add_argument('folders',nargs='+');p.add_argument('--output',required=True);args=p.parse_args()
runs={}
for name in args.folders:
    folder=pathlib.Path(name);meta=json.loads((folder/'config.json').read_text())
    assert not meta['working_tree_dirty'] and meta['contract']['task']=='upright_cycle_v16'
    initial=pathlib.Path(meta['args']['initialize_cycle_from'])
    parent=torch.load(initial,map_location='cpu',weights_only=False)
    iteration=meta['args']['iterations']-1
    trained=torch.load(folder/f'model_{iteration}.pt',map_location='cpu',weights_only=False)
    actor=trained['actor_state_dict'];before=parent['actor_state_dict']
    assert set(actor)==set(before)
    frozen=[k for k in actor if not k.startswith(('adapter.','yaw_adapter.'))]
    assert all(torch.equal(actor[k],before[k]) for k in frozen),'Inherited controller/statistics changed'
    changed={prefix:any(not torch.equal(actor[k],before[k]) for k in actor if k.startswith(prefix)) for prefix in ('adapter.','yaw_adapter.')}
    assert all(changed.values()),'No refinement update'
    assert all(torch.isfinite(v).all() for v in actor.values())
    samples=meta['args']['num_envs']*meta['runner']['num_steps_per_env']*meta['args']['iterations']
    assert int(trained['critic_state_dict']['obs_normalizer.count'])==int(parent['critic_state_dict']['obs_normalizer.count'])+samples
    runs[folder.name]=dict(frozen_actor_and_reflex_exact=True,refinement_updates=changed,critic_added_samples=samples,iteration=iteration)
pathlib.Path(args.output).write_text(json.dumps(dict(passed=True,runs=runs),indent=2));print(runs)
