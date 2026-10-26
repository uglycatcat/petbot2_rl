"""CUDA exact full v14 actor transfer, frozen state, gradients and JIT checks."""
import argparse,copy,importlib.util,json,pathlib,sys
import torch
from tensordict import TensorDict
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from upright.balance_reflex import ReflexAdapterModel as BalanceAdapterModel
from upright.distribution import BoundedBetaMean
from upright.continuous import instantaneous_tolerance_score

p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--output',required=True);args=p.parse_args()
assert torch.cuda.is_available();device='cuda:0';torch.manual_seed(13)
checkpoint=pathlib.Path(args.checkpoint);saved=torch.load(checkpoint,map_location=device,weights_only=False)
metadata=json.loads((checkpoint.parent/'config.json').read_text())
assert metadata['contract']['task']=='upright_cycle_v14' and metadata['contract']['balance_noise_factor']==16.
assert metadata['commit']=='272aaea822c0df0f7460ae9b0e372228e32718b5'
spec=importlib.util.spec_from_file_location('upright.balance_adapter_v14_reference',checkpoint.parents[2]/'upright/balance_adapter.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
parent=saved['actor_state_dict'];recent=saved['critic_state_dict']
raw=recent['obs_normalizer._mean']+torch.randn(1024,290,device=device)*(recent['obs_normalizer._std']+.01)
raw[:,-58+41:-58+46]=0.;raw[:,-58+43]=1.;raw[:,-58+48]=.5
obs=TensorDict({'policy':raw},batch_size=[len(raw)])
config=copy.deepcopy(metadata['runner']['actor']);config.pop('class_name')
other=dict(obs_groups={'actor':['policy']},obs_set='actor',output_dim=3)
old=reference.BalanceAdapterModel(obs,**other,**copy.deepcopy(config)).to(device);old.load_state_dict(parent)
actor=BalanceAdapterModel(obs,**other,**copy.deepcopy(config)).to(device);actor.initialize_parent(parent)
assert all(not p.requires_grad for p in actor.balance_reflex.parameters())
assert torch.equal(actor(obs),old(obs)), 'Full transfer must retain the learned parent policy'
actor(obs,stochastic_output=True);old(obs,stochastic_output=True)
assert all(torch.equal(a,b) for a,b in zip(actor.output_distribution_params,old.output_distribution_params))
frozen={k:v.clone() for k,v in actor.state_dict().items() if not k.startswith(('adapter.','yaw_adapter.'))}
opt=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=1e-4)
sample=actor(obs,stochastic_output=True);lp=actor.get_output_log_prob(sample)
aa,bb=actor.output_distribution_params
manual=(torch.distributions.Beta(aa,bb).log_prob(((sample+1)/2).clamp(1e-6,1-1e-6))-torch.log(raw.new_tensor(2.))).sum(-1)
assert torch.allclose(lp,manual,atol=1e-5)
loss=-(lp*torch.linspace(-1,1,len(raw),device=device)).mean()-.001*actor.output_entropy.mean()
opt.zero_grad();loss.backward()
assert all(p.grad is None for p in actor.mlp.parameters())
assert actor.yaw_adapter[-1].weight.grad.abs().sum()>0 and actor.adapter[-1].weight.grad.abs().sum()>0
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())
opt.step();actor.update_normalization(obs)
assert all(torch.equal(actor.state_dict()[k],v) for k,v in frozen.items())
base=BoundedBetaMean(127.)(actor.mlp(actor.obs_normalizer(raw)))
assert ((actor(obs)-base).abs()<=raw.new_tensor([.03,.08,.8])+1e-6).all()
reflex_script=torch.jit.script(actor.balance_reflex)
assert torch.equal(reflex_script(raw),actor.balance_reflex(raw)) and torch.equal(reflex_script.base(raw),actor.balance_reflex.base(raw))
scripted=torch.jit.script(actor.as_jit());assert torch.allclose(scripted(raw),actor(obs),atol=1e-6,rtol=1e-6)
clone=BalanceAdapterModel(obs,**other,**copy.deepcopy(config)).to(device);clone.load_state_dict(actor.state_dict(),strict=True);assert torch.equal(clone(obs),actor(obs))
for phase in [0,1,3,4]:
    x=raw.clone();x[:,-58+41:-58+46]=0.;x[:,-58+41+phase]=1.;td=TensorDict({'policy':x},batch_size=[len(x)])
    assert torch.equal(actor(td),old(td))
    actor(td,stochastic_output=True);old(td,stochastic_output=True)
    assert all(torch.equal(a,b) for a,b in zip(actor.output_distribution_params,old.output_distribution_params))
tol=raw.new_tensor([.005,.015]);grid=torch.arange(0.,2.1,.5,device=device)[:,None]*tol
score=instantaneous_tolerance_score(grid,tol)
assert torch.allclose(score[0],score.new_tensor(1.)) and (score[1:]<score[:-1]).all()
assert torch.equal(score,instantaneous_tolerance_score(-grid,tol)) and torch.isfinite(score).all()
result=dict(passed=True,initial_parent_mean_exact=True,initial_distribution_exact=True,frozen_state_exact=True,frozen_inner_controller=True,reflex_jit=True,
    phase_isolation=True,finite_frame_common_yaw_gradients=True,jit=True,reload=True,reward_score_grid=score.tolist())
pathlib.Path(args.output).write_text(json.dumps(result,indent=2));print(result)
