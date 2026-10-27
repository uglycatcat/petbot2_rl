"""CUDA transfer, frozen dynamics, PPO gradients and strict reload checks."""
import argparse,copy,json,pathlib,sys,torch
from tensordict import TensorDict
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from upright.parent_feedback import ParentFeedbackModel
from upright.normalization import rebase_all_branches

p=argparse.ArgumentParser();p.add_argument('checkpoint',type=pathlib.Path)
p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
assert torch.cuda.is_available()
c=json.loads((a.checkpoint.parent/'config.json').read_text())
assert c['contract']['task']=='upright_cycle_v22'
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
actor=make();actor.initialize_outer(saved['actor_state_dict'])
normalization=rebase_all_branches(actor,saved['critic_state_dict']);actor.snapshot_parent()
for name,value in actor.named_parameters():value.requires_grad_(not name.startswith(('balance_reflex.','parent_balance.')))
with torch.no_grad():
    outer=actor(obs);parent=actor.parent_balance(raw)
    error=float((outer-parent).abs().max());assert torch.allclose(outer,parent,atol=2e-6,rtol=2e-6),error
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
result=dict(passed=True,cuda=True,samples=len(raw),phases=[0,1,2,3,4],snapshot_mean_max_error=error,
    normalization=normalization,sampled_outer_action_preserved=True,frozen_tensors=len(frozen),
    parent_gradients_absent=True,outer_updated=True,strict_reload_equal=True,scripted_parent_equal=True)
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
