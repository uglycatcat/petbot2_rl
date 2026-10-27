"""CUDA sampling invariants; physics/reload validation is required separately."""
import argparse,importlib.util,json,pathlib,sys
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from upright.full_range import FullRangeProgress,TRAINING_MIXTURE
p=argparse.ArgumentParser();p.add_argument('--previous',type=pathlib.Path,required=True);a=p.parse_args()
assert torch.cuda.is_available()
spec=importlib.util.spec_from_file_location('upright.previous_full_range',a.previous)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
n=4096;device='cuda:0';ids=torch.arange(n,device=device)
kwargs=dict(hold_range=(12.,30.),tracking_objective='stability_first',handoff_alignment='after',transition_contract='linear_heading')
def make(cls,**options):return cls(n,device,.06,.75,15.,(-1.,1.2),42,'pose_feedback_tight',**kwargs,**options)
prior=make(old.FullRangeProgress);control=make(FullRangeProgress);mixed=make(FullRangeProgress,training_mixture='stratified_v1')
pos=torch.zeros(n,3,device=device);heading=torch.zeros(n,device=device)
for q in (prior,control,mixed):q.reset(ids,pos,heading)
counts=torch.bincount(mixed.training_level,minlength=5)
assert int(counts.max()-counts.min())<=1
level=mixed.training_level.clone()
limits=mixed.sampling_limits(ids)
seen=torch.zeros(5,6,device=device,dtype=torch.bool)
reached=torch.zeros(5,2,device=device)
for tick in range(120):
    for key in ('raw','frame','destination','mode','next_command','switch_at'):
        assert torch.equal(getattr(prior,key),getattr(control,key)),('disabled changed baseline',key,tick)
    assert (mixed.raw.abs()<=limits+1e-6).all()
    assert (mixed.raw[:,0].abs()/.75+mixed.raw[:,1].abs()/15.<=1.00001).all()
    for i,tier in enumerate(TRAINING_MIXTURE['levels']):
        selected=level==i
        seen[i,mixed.mode[selected]]=True
        reached[i]=torch.maximum(reached[i],mixed.raw[selected].abs().amax(0))
        for value in (mixed.frame,mixed.destination):
            assert ((value[selected]>=tier['frame'][0]-1e-6)&(value[selected]<=tier['frame'][1]+1e-6)).all()
    if tick%11==0:
        subset=ids[ids%7==tick%7]
        untouched=level!=int(tick%5)
        before=mixed.raw.clone();reset_ids=ids[~untouched]
        mixed.reset(reset_ids,pos,heading)
        assert torch.equal(mixed.raw[untouched],before[untouched])
        for q in (prior,control):q.reset(subset,pos,heading)
    for q in (prior,control,mixed):q.sample_commands(ids)
assert seen.all()
assert torch.equal(level,mixed.training_level)
assert (reached[-1]>torch.tensor([.70,14.],device=device)).all(),'Full-domain samples missing'
# Training sampling never narrows externally supplied controls.
mixed.set_command([0.,15.]);assert (mixed.raw[:,1]==15.).all()
print(json.dumps(dict(passed=True,cuda=True,environment_counts=counts.tolist(),maximum_sampled_motion=reached.tolist(),checks=['disabled_bitwise_parent_sampling','equal_step_strata','independent_resets','persistent_episode_level','frame_and_command_bounds','all_six_classes_each_level','full_domain_retained','external_request_unrestricted'])))
