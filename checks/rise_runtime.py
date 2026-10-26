"""Verify actual short PPO changed only the new lift head, plus critic updates."""
import argparse,json,pathlib,torch
p=argparse.ArgumentParser();p.add_argument('folders',nargs='+');p.add_argument('--output',required=True);args=p.parse_args()
runs={}
for name in args.folders:
    folder=pathlib.Path(name);meta=json.loads((folder/'config.json').read_text())
    assert not meta['working_tree_dirty'] and meta['contract']['task']=='upright_cycle_v19'
    assert meta['contract']['refinement_target']=='rise'
    parent=torch.load(meta['args']['initialize_cycle_from'],map_location='cpu',weights_only=False)
    iteration=meta['args']['iterations']-1
    trained=torch.load(folder/f'model_{iteration}.pt',map_location='cpu',weights_only=False)
    actor=trained['actor_state_dict'];before=parent['actor_state_dict']
    assert set(actor)-set(before)=={k for k in actor if k.startswith('rise_')}
    assert all(torch.equal(actor[k],v) for k,v in before.items())
    assert torch.equal(actor['rise_mean'],parent['critic_state_dict']['obs_normalizer._mean'])
    assert torch.equal(actor['rise_std'],parent['critic_state_dict']['obs_normalizer._std'])
    assert actor['rise_adapter.4.weight'].abs().sum()>0
    assert all(torch.isfinite(v).all() for v in actor.values())
    samples=meta['args']['num_envs']*meta['runner']['num_steps_per_env']*meta['args']['iterations']
    assert int(trained['critic_state_dict']['obs_normalizer.count'])==int(parent['critic_state_dict']['obs_normalizer.count'])+samples
    runs[folder.name]=dict(frozen_inherited_actor_exact=True,rise_head_updated=True,critic_added_samples=samples,iteration=iteration)
pathlib.Path(args.output).write_text(json.dumps(dict(passed=True,runs=runs),indent=2)+'\n');print(runs)
