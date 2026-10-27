"""Inspect actual PPO updates and frozen inner feedback after a short run."""
import argparse,json,pathlib,torch
p=argparse.ArgumentParser();p.add_argument('folder',type=pathlib.Path);p.add_argument('--final',default='model_7.pt');a=p.parse_args()
c=json.loads((a.folder/'config.json').read_text())
assert c['contract']['task_profile']=='full_range' and not c['working_tree_dirty']
assert c['initialization']['normalization_review']['passed']
first=torch.load(a.folder/'model_0.pt',map_location='cpu',weights_only=False)
last=torch.load(a.folder/a.final,map_location='cpu',weights_only=False)
x,y=first['actor_state_dict'],last['actor_state_dict']
updated={prefix:sum(not torch.equal(v,y[k]) for k,v in x.items() if k.startswith(prefix)) for prefix in ('mlp.','adapter.','yaw_adapter.','rise_adapter.')}
assert all(updated.values()),updated
frozen=[k for k in x if k.startswith(('balance_reflex.','parent_balance.','obs_normalizer.')) or k in ('adapter_mean','adapter_std','rise_mean','rise_std')]
assert all(torch.equal(x[k],y[k]) for k in frozen)
assert all(torch.isfinite(v).all() for v in y.values())
assert int(last['critic_state_dict']['obs_normalizer.count'])>int(first['critic_state_dict']['obs_normalizer.count'])
result=dict(passed=True,updated_tensors=updated,frozen_tensors=len(frozen),normalization_review=c['initialization']['normalization_review'],actor_normalization_static=True,critic_statistics_updated=True)
(a.folder/'runtime_review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
