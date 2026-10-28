"""Outcome-based curriculum tests: two good windows, weak side and holdout isolation."""
import argparse,json,pathlib,sys,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from upright.capability import CapabilityProgress
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
n=256;device='cuda:0';ids=torch.arange(n,device=device)
def make(evaluating=False):
 obj=CapabilityProgress(n,device,.06,.75,15.,(-1.,1.2),58,'pose_feedback_tight',hold_range=(12.,30.),tracking_objective='stability_first',handoff_alignment='after',transition_contract='linear_heading',mirror_training=True,adaptive_curriculum=True,evaluating=evaluating)
 obj.reset(ids,torch.zeros(n,3,device=device),torch.zeros(n,device=device))
 return obj
def window(obj,entry_good=True,motion_good=True,weak_side=None):
 obj.course_at_reset[:]=torch.tensor(obj.levels,device=device);obj.captured[:,0]=torch.where(ids%2==0,.1,-.1)
 obj.rise_settled[:]=entry_good;obj.motion_count[:]=100;obj.motion_good[:]=100 if motion_good else 0
 if weak_side=='entry':obj.rise_settled[ids%2==1]=False
 if weak_side=='motion':obj.motion_good[:,1]=0
 obj.record_outcomes(torch.ones(n,device=device,dtype=torch.bool),{'completed':torch.ones(n,device=device,dtype=torch.bool)})
obj=make();window(obj);assert obj.levels==[0,0];window(obj);assert obj.levels==[1,1]
window(obj,entry_good=False,motion_good=False);assert obj.levels[0]==0
weak=make();window(weak,weak_side='entry');window(weak,weak_side='entry');assert weak.levels[0]==0
weak=make();window(weak,weak_side='motion');window(weak,weak_side='motion');assert weak.levels[1]==0
holdout=make(True);assert not holdout.adaptive_curriculum and holdout.levels==[0,0]
result=dict(passed=True,cuda=True,two_windows_required=True,entry_weaker_side_blocks=True,motion_weaker_side_blocks=True,regression_supported=True,independent_evaluation_disabled=True)
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
