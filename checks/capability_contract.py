"""v25 CUDA contracts: symmetry, physical coverage, preparation and curriculum."""
import argparse,json,pathlib,sys,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from upright.capability import CapabilityProgress,balance_common,speed_score
from upright.full_range import FullRangeProgress,project_command
from upright.symmetry import mirror_observation,mirror_action,augment
from upright.ordinary import OrdinaryDrive
p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
device='cuda:0';n=64;ids=torch.arange(n,device=device)
args=(n,device,.06,.75,15.,(-1.,1.2),57,'pose_feedback_tight')
kwargs=dict(hold_range=(12.,30.),tracking_objective='stability_first',handoff_alignment='after',transition_contract='linear_heading')
pos=torch.zeros(n,3,device=device);pos[:,2]=.1;yaw=torch.zeros(n,device=device)
checks=[]
raw=torch.randn(128,290,device=device);actions=torch.rand(128,3,device=device)*2-1
assert torch.equal(mirror_observation(mirror_observation(raw)),raw)
assert torch.equal(mirror_action(mirror_action(actions)),actions)
raw.requires_grad_();mirror_observation(raw).sum().backward();assert torch.isfinite(raw.grad).all()
command=torch.rand(128,2,device=device)*2-1;command*=command.new_tensor([.75,15.]);mirror=command*command.new_tensor([1.,-1.])
assert torch.equal(project_command(mirror),project_command(command)*command.new_tensor([1.,-1.]))
checks.append('reflection_involution_gradient_and_joint_domain')
assert torch.equal(balance_common(torch.tensor([-.99,.99],device=device)),torch.tensor([-25.,25.],device=device))
requested=torch.tensor([[.75,0.],[-.75,0.]],device=device)
scores=[speed_score(requested*x,requested) for x in (.5,.8,.95,1.)]
assert all((x<y).all() for x,y in zip(scores,scores[1:]));checks.append('full_common_target_and_no_half_speed_plateau')
fresh=CapabilityProgress(*args,mirror_training=True,adaptive_curriculum=True,training_mixture='stratified_v1',**kwargs)
fresh.reset(ids,pos,yaw)
assert not fresh.destination.any() and torch.equal(fresh.frame[::2],fresh.frame[1::2])
for _ in range(20):
    assert torch.equal(fresh.raw[::2,0],fresh.raw[1::2,0]);assert torch.equal(fresh.raw[::2,1],-fresh.raw[1::2,1]);fresh.sample_commands(ids)
checks.append('physical_initial_commands_paired')
def state(lift=False):
    return dict(pos=pos.clone(),yaw=yaw.clone(),forward=torch.zeros(n,device=device),vel=torch.zeros(n,6,device=device),q=torch.zeros(n,5,device=device),
        g=torch.tensor([-.8,0.,-.6] if lift else [0.,0.,-1.],device=device).expand(n,3).clone(),front_z=torch.full((n,),.18 if lift else .045,device=device),rear=torch.ones(n,dtype=torch.bool,device=device),other=torch.zeros(n,dtype=torch.bool,device=device),four=torch.full((n,),not lift,device=device),rear_fraction=torch.ones(n,device=device),four_fraction=torch.full((n,),0. if lift else 1.,device=device),other_history=torch.zeros(n,dtype=torch.bool,device=device),body_history=torch.zeros(n,dtype=torch.bool,device=device),front_vertical_speed=torch.zeros(n,device=device),front_force_peak=torch.zeros(n,device=device),front_contact=torch.full((n,),not lift,device=device),heading_rate=torch.zeros(n,device=device))
fixed=torch.zeros(3,device=device)
fresh.reset(ids,pos,yaw);fresh.active[:]=True;fresh.phase[:]=1;fresh.external_mode[:]=True
fresh.transition_speed[:]=.3
for _ in range(4):m=fresh.update(state(),fixed);fresh.before_step()
assert m['preparing'].all() and not m['failure'].any() and (fresh.transition_speed==.3).all()
assert (m['preparation_cost']>=.03).all() and not fresh.rise_settled.any()
for _ in range(6):m=fresh.update(state(True),fixed)
assert (fresh.phase==2).all() and not m['settled_rise_reward'].any()
for _ in range(51):m=fresh.update(state(True),fixed)
assert fresh.rise_settled.all();checks.append('preparation_optional_no_forced_brake_and_settled_credit')
# Same physical landing, references and controller switch as v24; only the
# required destination is explicitly matched to zero for comparison.
old=FullRangeProgress(*args,**kwargs);fresh=CapabilityProgress(*args,**kwargs)
for obj in (old,fresh):
    obj.reset(ids,pos,yaw);obj.active[:]=True;obj.phase[:]=2;obj.external_mode[:]=True
    obj.requested_mode[:]=0.;obj.destination[:]=0.;obj.raw[:]=0.;obj.lift_seen[:]=True
s=state(True);s['forward'][:]=.24
old.update(s,fixed);fresh.update(s,fixed)
for _ in range(24):
    for obj in (old,fresh):obj.before_step();obj.update(state(),fixed)
    for key in ('phase','transition_speed','transition_heading','ground_run','handoff_run','handoff','command'):
        assert torch.equal(getattr(old,key),getattr(fresh,key)),key
assert fresh.handoff.all();checks.append('landing_reference_and_handoff_exact_v24')
# Single call exit still armed by real lift, including abnormal returns.
fresh.reset(ids,pos,yaw);fresh.active[:]=True;fresh.phase[:]=1;fresh.external_mode[:]=True
fresh.update(state(True),fixed)
for _ in range(2):m=fresh.update(state(),fixed)
assert m['interrupted'].all() and not m['completed'].any();checks.append('accidental_return_exit')
fresh.levels=[2,3];fresh.streak=[1,0];fresh.course_trials[:]=11;fresh.course_success[:]=9
saved=fresh.curriculum_state_dict();clone=CapabilityProgress(*args,**kwargs);clone.load_curriculum_state_dict(saved)
assert clone.levels==[2,3] and torch.equal(clone.course_trials,fresh.course_trials) and torch.equal(clone.generation,fresh.generation)
checks.append('curriculum_state_restore')
drive=OrdinaryDrive(n,device);command=torch.tensor([.75,0.],device=device).expand(n,2);motion=command.clone()
assert (drive.wheels(command,motion,.002,torch.ones(n,device=device,dtype=torch.bool))==25.).all();checks.append('ordinary_full_source_target')
result=dict(passed=True,cuda=True,checks=checks);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
