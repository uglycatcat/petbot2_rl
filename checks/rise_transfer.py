"""CUDA transfer, distribution, gradient, stage isolation and JIT contracts."""
import argparse,copy,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from tensordict import TensorDict
from upright.balance_reflex import ReflexAdapterModel
from upright.rise_adapter import RiseAdapterModel,rise_gate

p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--output',required=True);args=p.parse_args()
assert torch.cuda.is_available();device='cuda:0';torch.manual_seed(19)
checkpoint=pathlib.Path(args.checkpoint);saved=torch.load(checkpoint,map_location=device,weights_only=False)
meta=json.loads((checkpoint.parent/'config.json').read_text())
assert meta['commit']=='03deaca07644e244040319df07ebfc1347fb5f1e' and not meta['working_tree_dirty']
critic=saved['critic_state_dict'];parent=saved['actor_state_dict']
raw=critic['obs_normalizer._mean']+torch.randn(1024,290,device=device)*(critic['obs_normalizer._std']+.01)
ids=torch.arange(len(raw),device=device);phase=ids%5;seconds=(ids%11).float()*.12
raw[:,-58+41:-58+46]=0.;raw[ids,-58+41+phase]=1.;raw[:,-58+48]=seconds/4.
obs=TensorDict({'policy':raw},batch_size=[len(raw)])
cfg=copy.deepcopy(meta['runner']['actor']);cfg.pop('class_name')
other=dict(obs_groups={'actor':['policy']},obs_set='actor',output_dim=3)
old=ReflexAdapterModel(obs,**other,**copy.deepcopy(cfg)).to(device);old.load_state_dict(parent)
actor=RiseAdapterModel(obs,**other,**copy.deepcopy(cfg)).to(device);actor.initialize_rise(parent,critic)
assert torch.equal(actor(obs),old(obs))
assert all(torch.equal(actor.state_dict()[k],v) for k,v in parent.items())
assert [n for n,p in actor.named_parameters() if p.requires_grad]==[n for n,p in actor.named_parameters() if n.startswith('rise_adapter.')]
gate=rise_gate(raw);inactive=gate==0
old(obs,stochastic_output=True);old_params=[x.clone() for x in old.output_distribution_params]
sample=actor(obs,stochastic_output=True);new_params=actor.output_distribution_params
assert all(torch.equal(x[inactive],y[inactive]) for x,y in zip(old_params,new_params))
ratio=1.-.75*gate
assert torch.equal(new_params[0][:,2],old_params[0][:,2]) and torch.equal(new_params[1][:,2],old_params[1][:,2])
assert all(torch.allclose(x[:,:2],y[:,:2]*ratio[:,None]) for x,y in zip(new_params,old_params))
lp=actor.get_output_log_prob(sample)
aa,bb=new_params
manual=(torch.distributions.Beta(aa,bb).log_prob(((sample+1)/2).clamp(1e-6,1-1e-6))-torch.log(raw.new_tensor(2.))).sum(-1)
assert torch.allclose(lp,manual,atol=1e-5)
opt=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=1e-4)
loss=-(lp*torch.linspace(-1,1,len(raw),device=device)*gate).mean()-.001*actor.output_entropy.mean()
opt.zero_grad();loss.backward()
assert actor.rise_adapter[-1].weight.grad.abs().sum()>0
assert all(p.grad is None for n,p in actor.named_parameters() if not n.startswith('rise_adapter.'))
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in actor.parameters())
opt.step();actor.update_normalization(obs)
assert all(torch.equal(actor.state_dict()[k],v) for k,v in parent.items())
assert torch.equal(actor(obs)[inactive],old(obs)[inactive])
delta=actor(obs)-old(obs);assert (delta.abs()<=raw.new_tensor([.15,.25,0.])+1e-6).all()
scripted=torch.jit.script(actor.as_jit());assert torch.allclose(scripted(raw),actor(obs),atol=1e-6,rtol=1e-6)
inner=torch.jit.script(actor.balance_reflex);assert torch.equal(inner(raw),old.balance_reflex(raw))
clone=RiseAdapterModel(obs,**other,**copy.deepcopy(cfg)).to(device);clone.load_state_dict(actor.state_dict(),strict=True)
assert torch.equal(clone(obs),actor(obs))
for phase_id,timepoint,expected in [(0,1.,0.),(1,0.,0.),(1,.06,.5),(1,.12,1.),(2,0.,1.),(2,.3,.5),(2,.6,0.),(3,0.,0.),(4,0.,0.)]:
    x=raw[:1].clone();x[:,-58+41:-58+46]=0.;x[:,-58+41+phase_id]=1.;x[:,-58+48]=timepoint/4.
    assert torch.allclose(rise_gate(x),x.new_tensor([expected]),atol=1e-6)
result=dict(passed=True,initial_mean_exact=True,inherited_state_frozen=True,distribution_outside_rise_exact=True,
            exploration_factor_verified=True,finite_gradients=True,only_rise_head_trainable=True,
            stage_isolation=True,yaw_unchanged=True,jit=True,inner_exact=True,reload=True)
pathlib.Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(result)
