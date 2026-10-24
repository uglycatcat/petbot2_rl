"""CUDA function-preservation check on actual checkpoint statistics and weights."""
import argparse
import json
import pathlib
import sys
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from upright.normalization import rebase_actor_normalization
from upright.distribution import BoundedBetaMean

p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--output',required=True);args=p.parse_args()
assert torch.cuda.is_available();torch.manual_seed(31)
saved=torch.load(args.checkpoint,map_location='cuda:0',weights_only=False)
old=saved['actor_state_dict'];recent=saved['critic_state_dict']
new=rebase_actor_normalization(old,recent)
assert int(new['obs_normalizer.count'])==10_000_000
assert int(old['obs_normalizer.count'])>int(new['obs_normalizer.count'])
assert all(torch.equal(old[k],new[k]) for k in old if not k.startswith('obs_normalizer.') and k not in ['mlp.0.weight','mlp.0.bias'])

def output(state,raw):
    x=(raw-state['obs_normalizer._mean'])/(state['obs_normalizer._std']+.01)
    for i in [0,2,4]:
        x=torch.nn.functional.linear(x,state[f'mlp.{i}.weight'],state[f'mlp.{i}.bias'])
        if i<4:x=torch.nn.functional.elu(x)
    return x

errors=[];mean_errors=[]
for stats in [old,recent]:
    raw=stats['obs_normalizer._mean']+torch.randn(4096,290,device='cuda:0')*(stats['obs_normalizer._std']+.01)
    a=output(old,raw);b=output(new,raw)
    errors.append(float((a-b).abs().max()))
    mean_errors.append(float((BoundedBetaMean(127.)(a.reshape(-1,2,3))-BoundedBetaMean(127.)(b.reshape(-1,2,3))).abs().max()))
    assert torch.allclose(a,b,atol=2e-4,rtol=2e-5)
    assert mean_errors[-1]<2e-6
assert all(torch.isfinite(v).all() for v in new.values())
result=dict(passed=True,source_actor_count=int(old['obs_normalizer.count']),recent_count=int(recent['obs_normalizer.count']),
            rebased_count=int(new['obs_normalizer.count']),max_network_delta=max(errors),max_action_delta=max(mean_errors),
            limitation='Function preservation at transfer; subsequent learning/normalizer updates intentionally differ.')
pathlib.Path(args.output).write_text(json.dumps(result,indent=2));print(result)
