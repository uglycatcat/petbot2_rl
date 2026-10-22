"""Offline contact-sampling phase sensitivity; does not alter policy observations."""
import argparse,pathlib,json,numpy as np,math
p=argparse.ArgumentParser();p.add_argument('directory',type=pathlib.Path);p.add_argument('diagnostic',type=pathlib.Path);a=p.parse_args()
x=np.load(a.directory/'physics_trace.npz');rows=json.loads(a.diagnostic.read_text());r=json.loads((a.directory/'evaluation.json').read_text());meta=json.loads((a.directory/'replay_meta.json').read_text())
assert len(rows)==len(x['valid']);arr=lambda k:np.array([v[k] for v in rows])
assert np.allclose(x['front_z'],arr('front_z'),atol=1e-7)
active=arr('active');since=arr('since');rear=arr('rear');raw=arr('raw_error')
other=(active & ~arr('other') & (arr('front_z')>.12) & (arr('gravity')[:,:,2]<-.25) & (np.abs(raw[:,:,0])<.12) & (np.abs(raw[:,:,1])<.3) & (arr('raw_distance')<.15) & (np.abs(arr('heading_error'))<.25) & (arr('fixed_error')<.05) & (np.abs(arr('roll'))<.15) & (arr('tilt_rate')<.8))
window=math.ceil(1/meta['control_dt']);tail=active&(since>=10)
def score(contact):
 good=other&contact;held=np.zeros_like(good)
 for t in range(window-1,len(good)):held[t]=good[t-window+1:t+1].all(0)&(since[t]>=1)
 return held.any(0)&(held|~tail).all(0)&tail.any(0)&~np.array(r['premature_per_env'])
baseline=score(rear);assert baseline.tolist()==r['maintained_per_env'], 'Offline reconstruction differs from official scoring'
counts=[]
for phase in range(x['force'].shape[2]):
 contact=np.where(x['valid'],(x['force'][:,:,phase,:]>.05).all(-1),rear)
 counts.append(int(score(contact).sum()))
assert counts[-1]==r['maintained_final_2s']
result=dict(official=counts[-1],phase_counts=counts,min=min(counts),max=max(counts),note='Offline counterfactual score only: current noncontact predicates and original closed-loop trajectory held fixed. Not new success criterion or policy validation.')
(a.directory/'phase_sensitivity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
