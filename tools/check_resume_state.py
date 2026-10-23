"""Verify full optimizer continuation from checkpoint step counters after a short run."""
import argparse,json,pathlib,torch
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('target');p.add_argument('--iterations',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
s=torch.load(a.source,map_location='cpu',weights_only=False);t=torch.load(a.target,map_location='cpu',weights_only=False)
cfg=json.loads((pathlib.Path(a.target).parent/'config.json').read_text());updates=cfg['runner']['algorithm']['num_learning_epochs']*cfg['runner']['algorithm']['num_mini_batches']*a.iterations
assert t['iter']==s['iter']+a.iterations-1
ss=s['optimizer_state_dict']['state'];ts=t['optimizer_state_dict']['state'];assert ss and set(ss)==set(ts)
for k in ss:
    assert float(ts[k]['step'])==float(ss[k]['step'])+updates
    for v in ts[k].values():
        if torch.is_tensor(v):assert torch.isfinite(v).all()
for name in ['actor_state_dict','critic_state_dict']:
    assert set(s[name])==set(t[name])
    for k,v in t[name].items():
        if torch.is_tensor(v):assert torch.isfinite(v).all()
result=dict(passed=True,source_iteration=s['iter'],target_iteration=t['iter'],additional_updates=updates,optimizer_parameter_states=len(ss),limitation='Confirms optimizer step continuity and finite actor/critic; simulator and RNG state are not restored, so not bit-exact continuation.')
pathlib.Path(a.output).write_text(json.dumps(result,indent=2));print(result)
