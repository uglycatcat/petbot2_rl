"""CUDA contract checks; synthetic controls validate detection, not physical reachability."""
import torch
from upright.progress import Progress,GAMMA
assert torch.cuda.is_available()
device='cuda:0';n=4
s=dict(pos=torch.zeros(n,3,device=device),g=torch.tensor([[0.,0.,-1.]],device=device).repeat(n,1),
    heading=torch.tensor([[1.,0.]],device=device).repeat(n,1),vel=torch.zeros(n,6,device=device),q=torch.zeros(n,5,device=device),front_z=torch.full((n,),.19,device=device),
    rear=torch.ones(n,device=device,dtype=torch.bool),other=torch.zeros(n,device=device,dtype=torch.bool))
fail=torch.zeros(n,device=device,dtype=torch.bool);p=Progress(n,device);p.reset(torch.arange(n,device=device),s)
# Valid at home / stable away / ground stance / wrong-body support.
s['pos'][1,0]=.3;s['front_z'][2]=.045;s['other'][3]=True
fresh=0
for _ in range(80):m=p.update(s,fail,s['q'].new_zeros(3));fresh+=int(m['fresh'].sum())
assert m['held'].tolist()==[True,False,False,False]
assert m['balanced'].tolist()==[True,True,False,False] and fresh==1
assert m['balance'][0]>m['balance'][1]
# Losing support invalidates held; reentering cannot collect another lift event.
s['rear'][0]=False
for _ in range(8):m=p.update(s,fail,s['q'].new_zeros(3))
assert p.phase[0]==0 and not m['held'][0]
s['rear'][0]=True
for _ in range(10):m=p.update(s,fail,s['q'].new_zeros(3));assert not m['lift_event'][0]
# Partial reset must erase only that environment's histories and flags.
keep=p.age[1].clone();s['pos'][0,0]=.1;p.reset(torch.tensor([0],device=device),s)
assert p.age[0]==0 and p.age[1]==keep and not p.ever[0]
assert p.origin[0,0]==s['pos'][0,0]
# One passing instant near home must not pass a full-window success check.
s['pos'][1,0]=0;m=p.update(s,fail,s['q'].new_zeros(3));assert not m['held'][1]
# Discounted reward ranking over 12 s for representative synthetic trajectories.
def total(kind):
 q=Progress(1,device);st={k:v[:1].clone() for k,v in s.items()};st['pos'].zero_();st['rear'][:]=True;st['other'][:]=False
 q.reset(torch.tensor([0],device=device),st);ret=0.
 for t in range(600):
  st['front_z'][:]=.045 if t<100 or kind=='ground' else .19
  offset=0. if t<100 else (.3 if kind=='away' else max(0.,.3-(t-100)*.002))
  st['pos'][:,0]=offset;st['vel'].zero_()
  if kind=='return' and 100<=t<250:st['vel'][:,0]=-.1
  m=q.update(st,torch.zeros(1,device=device,dtype=torch.bool),st['q'].new_zeros(3))
  reward=.02*(3*m['progress']+m['prepare_reward']+2*m['support_reward']+2*m['balance']+5*m['home_hold'])+3*m['lift_event']+30*m['fresh']
  ret+=GAMMA**t*float(reward)
 return ret
scores={k:total(k) for k in ['ground','away','return']}
assert scores['return']>scores['away']>scores['ground'],scores
print('PASS CUDA progress/reset/anti-repeat/window/reward ranking',scores)

# The stage boundary uses integer ticks and a physical-time contact window.
for dt in (.02,.06):
 p=Progress(1,device,dt);st={k:v[:1].clone() for k,v in s.items()}
 st['front_z'][:]=.19;st['rear'][:]=True;st['other'][:]=False
 p.reset(torch.tensor([0],device=device),st)
 for tick in range(p.enter_steps+p.contacts.shape[0]//2):
  m=p.update(st,torch.zeros(1,device=device,dtype=torch.bool),st['q'].new_zeros(3))
  assert bool(m['lift_event'][0])==(tick==p.enter_steps+p.contacts.shape[0]//2-1)
 print('PASS exact phase boundary',dt,p.enter_steps,p.contacts.shape[0])
# Peak progress cannot be earned twice by oscillating between the same states.
p=Progress(1,device);st={k:v[:1].clone() for k,v in s.items()};st['rear'][:]=True;st['other'][:]=False;st['pos'].zero_()
p.reset(torch.tensor([0],device=device),st);earned=0.
for t in range(150):
 st['front_z'][:]=.19 if t%10<5 else .045
 m=p.update(st,torch.zeros(1,device=device,dtype=torch.bool),st['q'].new_zeros(3));earned+=float(m['progress'])*p.dt
assert abs(earned-float(p.maximum_phi))<1e-5 and earned<=2.25
print('PASS peak reward bounded across repeated lifts',earned)

# Smooth pose credit distinguishes quiet support from repeated frame oscillation.
p=Progress(2,device,.06);st={k:v[:2].clone() for k,v in s.items()};st['pos'].zero_();st['front_z'][:]=.19;st['rear'][:]=True;st['other'][:]=False;st['q'].zero_()
p.reset(torch.arange(2,device=device),st)
for i in range(40):
 st['q'][1,3:]=.04 if i%2 else -.04
 m=p.update(st,torch.zeros(2,device=device,dtype=torch.bool),st['q'].new_zeros(3))
assert m['settle'][0]>m['settle'][1]*5
assert m['held'].tolist()==[True,False]
print('PASS continuous stability credit',m['settle'].tolist())
