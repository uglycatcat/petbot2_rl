"""CUDA transfer, phase isolation, PPO distribution and export acceptance."""
import argparse,copy,json,pathlib,sys
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from upright.balance_adapter import BalanceAdapterModel,initialize_adapter,balance_gate

p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--output',required=True);args=p.parse_args()
assert torch.cuda.is_available();torch.manual_seed(12);device='cuda:0'
parent=torch.load(args.checkpoint,map_location=device,weights_only=False)
state=parent['actor_state_dict'];recent=parent['critic_state_dict']
raw=recent['obs_normalizer._mean']+torch.randn(1024,290,device=device)*(recent['obs_normalizer._std']+.01)
raw[:,-58+41:-58+46]=0.;raw[:,-58+43]=1.;raw[:,-58+48]=.5
obs=TensorDict({'policy':raw},batch_size=[len(raw)])
config=dict(obs_groups={'actor':['policy']},obs_set='actor',output_dim=3,hidden_dims=[128,128],activation='elu',obs_normalization=True,
    distribution_cfg=dict(class_name='upright.distribution:UprightBetaDistribution',initial_mean=[0.,0.,0.],initial_std=[.35,.45,.45],shape_cap=128.,concentration_scale=[4.,16.,16.]))
old=MLPModel(obs,**copy.deepcopy(config)).to(device);old.load_state_dict(state);old.eval()
results=[]
for factor in [1.,16.]:
    actor=BalanceAdapterModel(obs,balance_noise_factor=factor,**copy.deepcopy(config)).to(device)
    initialize_adapter(actor,state,recent);actor.eval()
    assert torch.equal(old(obs),actor(obs)), 'Zero adapter must preserve deterministic actions'
    old(obs,stochastic_output=True);actor(obs,stochastic_output=True)
    if factor==1.:assert all(torch.equal(a,b) for a,b in zip(old.output_distribution_params,actor.output_distribution_params))
    ratio=actor.output_std/old.output_std
    assert torch.equal(ratio[:,2],torch.ones_like(ratio[:,2]))
    if factor==16.:
        concentration=sum(old.output_distribution_params)
        expected=torch.sqrt((concentration+1)/(16.*concentration+1))
        assert torch.allclose(ratio[:,:2],expected[:,:2],atol=1e-6,rtol=1e-6)
    untouched={k:v.clone() for k,v in actor.state_dict().items() if not k.startswith('adapter.')}
    optimizer=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=1e-4)
    sample=actor(obs,stochastic_output=True);lp=actor.get_output_log_prob(sample)
    aa,bb=actor.output_distribution_params
    manual=(torch.distributions.Beta(aa,bb).log_prob(((sample+1)/2).clamp(1e-6,1-1e-6))-torch.log(raw.new_tensor(2.))).sum(-1)
    assert torch.allclose(lp,manual,atol=1e-5) and torch.isfinite(lp).all()
    assert torch.isfinite(actor.get_kl_divergence((aa,bb),(aa,bb))).all()
    loss=-(lp*torch.linspace(-1,1,len(raw),device=device)).mean()-.001*actor.output_entropy.mean()
    optimizer.zero_grad();loss.backward()
    assert all(p.grad is None for p in actor.mlp.parameters())
    assert actor.adapter[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())
    optimizer.step();actor.update_normalization(obs)
    assert all(torch.equal(actor.state_dict()[k],v) for k,v in untouched.items())
    change=(actor(obs)-old(obs)).abs();assert (change[:,:2]<=raw.new_tensor([.03,.08])+1e-7).all() and not change[:,2].any()
    scripted=torch.jit.script(actor.as_jit());assert torch.allclose(scripted(raw),actor(obs),atol=1e-6,rtol=1e-6)
    clone=BalanceAdapterModel(obs,balance_noise_factor=factor,**copy.deepcopy(config)).to(device)
    clone.load_state_dict(actor.state_dict(),strict=True);assert torch.equal(clone(obs),actor(obs))
    # Learned corrections and reduced exploration cannot leak into rise/landing.
    for phase in [0,1,3,4]:
        inactive=raw.clone();inactive[:,-58+41:-58+46]=0.;inactive[:,-58+41+phase]=1.
        x=TensorDict({'policy':inactive},batch_size=[len(raw)])
        assert not balance_gate(inactive).any() and torch.equal(actor(x),old(x))
        actor(x,stochastic_output=True);old(x,stochastic_output=True)
        assert all(torch.equal(a,b) for a,b in zip(old.output_distribution_params,actor.output_distribution_params))
    clock=raw.clone();clock[:,-58+48]=.3;assert not balance_gate(clock).any()
    clock[:,-58+48]=.375;assert torch.allclose(balance_gate(clock),torch.full((len(raw),),.5,device=device),atol=1e-6)
    results.append(dict(factor=factor,initial_mean_exact=True,frozen_base_unchanged=True,phase_isolation=True,
        finite_gradients=True,jit=True,reload=True,std_ratio_median=ratio.median(0).values.tolist()))
pathlib.Path(args.output).write_text(json.dumps(dict(passed=True,checkpoint=pathlib.Path(args.checkpoint).name,results=results),indent=2));print('PASS',results)
