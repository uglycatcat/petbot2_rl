"""Compare deployed 60/20/2ms targets with the pinned training feedback hook."""
import argparse,ast,copy,json,pathlib,sys
from types import SimpleNamespace
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from petbot_s2s.simulation import Simulation
from petbot_s2s.core.action_terms import FeedbackAction
p=argparse.ArgumentParser();p.add_argument('--source',type=pathlib.Path,required=True);p.add_argument('--policy',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args()
text=(args.source/'upright/balance_reflex.py').read_text();node=next(n for n in ast.parse(text).body if getattr(n,'name',None)=='attach_balance_reflex')
code=ast.get_source_segment(text,node).replace('    from .feedback_task import observations,ensure\n','')
namespace=dict(__package__='petbot_s2s.core',torch=torch,observations=lambda env:env.features(),ensure=lambda env:env.progress);exec(compile(code,'pinned_balance_reflex.py','exec'),namespace);attach=namespace['attach_balance_reflex']
torch.manual_seed(20261028);maximum=0.;cases=[]
for phase,clock in [(0,0.),(1,.6),(2,0.),(2,1.74),(2,1.8),(2,4.),(3,.6),(4,.6)]:
 actual=Simulation(args.policy).controller;reference=Simulation(args.policy).controller
 assert actual.full_parent_feedback and reference.full_parent_feedback
 # Reference wiring is built explicitly from the training hook, independently
 # of Controller's mode dispatch and its already-installed action closure.
 reference.action=FeedbackAction(None,reference);reference.action_manager=SimpleNamespace(get_term=lambda name:reference.action)
 attach(reference,SimpleNamespace(parent_balance=reference.parent),True,full_parent=True);reference.action.reset()
 for c in (actual,reference):
  c.progress.phase[:]=phase;c.progress.phase_time[:]=clock;c.progress.active[:]=phase>0;c.progress.since[:]=clock
  c.armed=phase>0;c.set_command(.15,.5);c.progress.request_mode(1,0.)
 state={k:v.clone() for k,v in actual.s.items()};actual.begin(state);reference.begin(state)
 old_parent=[x.detach().clone() for x in actual.parent.parameters()]
 for tick in range(30):
  changed={k:v.clone() for k,v in state.items()}
  for key in ('pos','vel','q','qd','wheel','forward','heading_rate'):
   changed[key]=changed[key]+torch.randn_like(changed[key])*.03
  a=actual.physics_targets(changed);b=reference.physics_targets(changed)
  for x,y in zip(a,b):
   maximum=max(maximum,float(np.max(np.abs(x-y))));np.testing.assert_allclose(x,y,rtol=0,atol=1e-7)
  assert np.max(np.abs(a[1]))<=25.00001
  assert min(a[0][3:])>=-1.00001 and max(a[0][3:])<=1.20001
 assert all(torch.equal(x,y) for x,y in zip(old_parent,actual.parent.parameters()))
 cases.append(dict(phase=phase,phase_time=clock,substeps=30))
result=dict(passed=True,source='pinned training attach_balance_reflex with full_parent=True',cases=cases,physics_substeps=240,max_target_error=maximum,parent_unchanged=True,note='Synthetic physical observations validate controller wiring and clocks; separate real MuJoCo scenarios validate closed-loop behavior.')
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
