"""CUDA transfer, frozen dynamics, PPO gradients and strict reload checks."""
import argparse,copy,json,pathlib,sys,torch
from tensordict import TensorDict
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from upright.parent_feedback import ParentFeedbackModel
from upright.normalization import rebase_all_branches

p=argparse.ArgumentParser();p.add_argument('checkpoint',type=pathlib.Path)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--baseline',type=pathlib.Path,required=True);a=p.parse_args()
def check_reference():
    import importlib.util
    from upright.full_range import FullRangeProgress,rise_speed_reference
    from tools.probe_transition_speed import brake_reference
    baseline=a.baseline
    spec=importlib.util.spec_from_file_location('upright._v23_reference',baseline)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    phase=torch.arange(5,device='cuda').repeat_interleave(7)
    speed=torch.tensor([-1.1,-.75,-.03,0.,.03,.75,1.1],device='cuda').repeat(5)
    assert torch.equal(rise_speed_reference(speed,phase,.06,'capture'),speed)
    args=(35,'cuda',.06,.75,15.,(-1.,1.2),49,'pose_feedback_tight')
    kwargs=dict(hold_range=(12.,30.),tracking_objective='stability_first',handoff_alignment='after',transition_contract='linear_heading')
    for mode in ('capture','brake'):
        fresh=FullRangeProgress(*args,rise_speed_reference_mode=mode,**kwargs)
        prior=old.FullRangeProgress(*args,**kwargs)
        for obj in (fresh,prior):
            obj.reset(torch.arange(35,device='cuda'),torch.zeros(35,3,device='cuda'),torch.zeros(35,device='cuda'))
            obj.phase.copy_(phase);obj.transition_speed.copy_(speed);obj.active[:]=True
        for _ in range(40):
            if mode=='brake': prior.transition_speed.copy_(brake_reference(prior.transition_speed,prior.phase,.06,'rise_brake'))
            prior.before_step();fresh.before_step()
            for name,value in vars(prior).items():
                if isinstance(value,torch.Tensor):assert torch.equal(value,getattr(fresh,name)),(mode,name)
            assert torch.equal(fresh.transition_speed[phase!=1],speed[phase!=1])
        if mode=='brake':assert not fresh.transition_speed[phase==1].any()
    return dict(capture_v23_exact=True,brake_prior_probe_exact=True,all_five_phases=True,nonrise_unchanged=True,forty_steps=True)

assert torch.cuda.is_available()
c=json.loads((a.checkpoint.parent/'config.json').read_text())
assert c['contract']['task']=='upright_cycle_v23'
saved=torch.load(a.checkpoint,map_location='cuda:0',weights_only=False)
torch.manual_seed(237)
raw=saved['critic_state_dict']['obs_normalizer._mean']+torch.randn(1024,290,device='cuda:0')*(saved['critic_state_dict']['obs_normalizer._std']+.01)
raw[:,-58+41:-58+46]=0.;phase=torch.arange(len(raw),device=raw.device)%5
raw[torch.arange(len(raw),device=raw.device),-58+41+phase]=1.
raw[:,-58+48]=.6
obs=TensorDict({'policy':raw},batch_size=[len(raw)])
cfg=copy.deepcopy(c['runner']['actor']);cfg.pop('class_name')
def make():
    return ParentFeedbackModel(obs,obs_groups={'actor':['policy']},obs_set='actor',output_dim=3,**copy.deepcopy(cfg)).to('cuda:0')
actor=make();actor.load_state_dict(saved['actor_state_dict'],strict=True)
frozen_source={k:v.clone() for k,v in actor.state_dict().items() if k.startswith(('balance_reflex.','parent_balance.'))}
with torch.no_grad(): source_outer=actor(obs).clone(); source_parent=actor.parent_balance(raw).clone()
normalization=rebase_all_branches(actor,saved['critic_state_dict'])
assert all(torch.equal(actor.state_dict()[k],v) for k,v in frozen_source.items())
for name,value in actor.named_parameters():value.requires_grad_(not name.startswith(('balance_reflex.','parent_balance.')))
with torch.no_grad():
    outer=actor(obs);parent=actor.parent_balance(raw)
    error=float((outer-source_outer).abs().max());assert torch.allclose(outer,source_outer,atol=2e-5,rtol=2e-5),error
    assert torch.equal(parent,source_parent)
    delta=torch.randn_like(outer[:,:2])*.1
    sampled=(outer[:,:2]+delta).clamp(-.99,.99)
    reconstructed=(parent[:,:2]+sampled-parent[:,:2]).clamp(-.99,.99)
    assert torch.allclose(reconstructed,sampled,atol=2e-7)
    scripted=torch.jit.script(actor.parent_balance)
    assert torch.allclose(scripted(raw),parent,atol=2e-6,rtol=2e-6)
before={k:v.clone() for k,v in actor.state_dict().items()}
optimizer=torch.optim.Adam(actor.parameters(),lr=.0001)
actor.train();actor(obs,stochastic_output=True)
distribution=actor.distribution._distribution
assert torch.isfinite(distribution.entropy()).all()
loss=-distribution.log_prob(torch.full((len(raw),3),.4,device=raw.device)).mean()
optimizer.zero_grad();loss.backward()
assert all(p.grad is None for p in actor.parent_balance.parameters())
assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
optimizer.step()
after=actor.state_dict();frozen=[k for k in before if k.startswith(('balance_reflex.','parent_balance.'))]
assert all(torch.equal(before[k],after[k]) for k in frozen)
assert any(not torch.equal(v,after[k]) for k,v in before.items() if k.startswith('mlp.'))
reloaded=make();reloaded.load_state_dict(after,strict=True)
with torch.no_grad():
    assert torch.equal(reloaded.parent_balance(raw),actor.parent_balance(raw))
    assert torch.equal(reloaded(obs),actor(obs))
reference_check=check_reference()
result=dict(reference_check=reference_check,passed=True,cuda=True,samples=len(raw),phases=[0,1,2,3,4],continuation_outer_max_error=error,parent_preserved_exact=True,
    normalization=normalization,sampled_outer_action_preserved=True,frozen_tensors=len(frozen),
    parent_gradients_absent=True,outer_updated=True,strict_reload_equal=True,scripted_parent_equal=True)
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
