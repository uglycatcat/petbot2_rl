"""Read real TensorBoard/PhysX evidence before allowing the six-run batch."""
import argparse
import json
import math
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runs-root',type=Path,required=True)
p.add_argument('--write-gate',action='store_true')
a=p.parse_args();root=a.runs_root;out=root/'_sweep3_precheck'
results={};commits=set();passed=True
for kind in ('b','c'):
    logs=list((root/f'sweep3_precheck_v2_{kind}/logs/rsl_rl/recovery_agile_01').glob('*/params/petbot.json'))
    if len(logs)!=1:
        results[kind]={'ready':False};passed=False;continue
    provenance=json.loads(logs[0].read_text());commits.add(provenance['demo_commit'])
    acc=EventAccumulator(str(logs[0].parent.parent),size_guidance={'scalars':0});acc.Reload()
    tags=acc.Tags()['scalars'];values={}
    required=['Episode_Reward/recovery_region','Episode_Reward/low_speed','Episode_Reward/success','Recovery/success_rate']
    for suffix in required+['Actions/clipped_fraction','Loss/surrogate','Loss/value_function','Curriculum/easy_fraction']:
        keys=[k for k in tags if k.endswith(suffix)]
        if len(keys)==1:
            events=acc.Scalars(keys[0]);v=[e.value for e in events]
            values[suffix]={'last_step':events[-1].step,'max':max(v),'last20_mean':sum(v[-20:])/len(v[-20:]),'finite':all(math.isfinite(x) for x in v)}
    rcpath=out/f'train_v2_{kind}.exit'
    complete=rcpath.exists() and rcpath.read_text().strip()=='0'
    checkpoints=list(logs[0].parent.parent.glob('model_99.pt'))
    ok=complete and bool(checkpoints) and all(k in values and values[k]['last20_mean']>0 and values[k]['last_step']>=99 for k in required)
    all_finite=all(math.isfinite(e.value) for tag in tags for e in acc.Scalars(tag))
    ok=ok and all_finite and all(v['finite'] for v in values.values())
    ok=ok and values.get('Actions/clipped_fraction',{}).get('max',1)==0
    if kind=='c':ok=ok and provenance.get('easy_curriculum',{}).get('count',0)>0
    results[kind]={'passed':ok,'complete':complete,'all_scalars_finite':all_finite,'metrics':values,'easy_count':provenance.get('easy_curriculum',{}).get('count')}
    passed &= ok
runtime=out/'runtime_v2.json'
physics=runtime.exists() and json.loads(runtime.read_text()).get('passed',False)
probability=(out/'probability_v2.exit').read_text().strip()=='0'
reload_path=out/'checkpoint_v2.exit'
reload_ok=reload_path.exists() and reload_path.read_text().strip()=='0' and (out/'checkpoint_v2.json').exists()
passed=bool(passed and physics and probability and reload_ok and len(commits)==1)
report={'passed':passed,'tested_commit':next(iter(commits)) if len(commits)==1 else None,
        'runtime_passed':physics,'probability_passed':probability,'checkpoint_reload_passed':reload_ok,'training':results}
print(json.dumps(report,indent=2))
if a.write_gate:
    if not passed:raise RuntimeError('Precheck not passed; no gate written')
    (out/'gate.json').write_text(json.dumps(report,indent=2)+'\n')
