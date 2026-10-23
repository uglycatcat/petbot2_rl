"""CUDA checks for physical-target continuity, feedback bounds and fine exploration."""
import pathlib
import sys
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from upright.feedback import YawFeedback
from upright.distribution import UprightBetaDistribution
from upright.transfer import expand_actor

assert torch.cuda.is_available();device='cuda:0';n=64;dt=.06
torch.manual_seed(24)
feedback=YawFeedback(n,device,dt,25.)
previous=torch.zeros(n,2,device=device);policy=torch.randn_like(previous)
phase=torch.zeros(n,device=device,dtype=torch.long);clock=torch.zeros(n,device=device)
command=torch.randn(n,2,device=device)*.15;motion=torch.zeros_like(command);residual=torch.zeros(n,device=device)
for code in [0,1,4]:
    phase[:]=code
    assert torch.equal(feedback.process(previous,policy,phase,clock,command,motion,residual),policy)
phase[:]=2;clock[:]=2.
for _ in range(50):
    result=feedback.process(previous,policy,phase,clock,command,motion,residual)
    assert torch.allclose(result.mean(-1),policy.mean(-1),atol=1e-6)
    half=(result[:,1]-result[:,0])*.5;old=(previous[:,1]-previous[:,0])*.5
    assert ((half-old).abs()<=4*dt+1e-6).all()
    assert result.abs().max()<=25 and feedback.integral.abs().max()<=1.5
    previous=result
feedback.reset(torch.arange(32,device=device));assert not feedback.integral[:32].any()
result=feedback.process(torch.full_like(previous,-25.),torch.full_like(policy,25.),phase,clock,command,motion,residual)
assert (result+25.).abs().max()<=480*dt+1e-6

raw=torch.randn(n,2,3,device=device,requires_grad=True)*3+20
distributions=[]
for scales in [[4,16,16],[16,64,64]]:
    d=UprightBetaDistribution(3,[0,0,0],[.35,.45,.45],128.,scales).to(device);d.update(raw)
    sample=d.sample();logp=d.log_prob(sample)
    manual=torch.distributions.Beta(*d.params).log_prob((sample+1)/2).sum(-1)-3*torch.log(torch.tensor(2.,device=device))
    assert torch.allclose(logp,manual,atol=1e-4)
    assert torch.isfinite(logp).all() and (d.std>0).all()
    gradient=torch.autograd.grad(logp.mean()+.001*d.entropy.mean(),raw,retain_graph=True)[0]
    assert torch.isfinite(gradient).all()
    distributions.append(d)
assert torch.equal(distributions[0].deterministic_output(raw),distributions[1].deterministic_output(raw))
ratio=distributions[1].std/distributions[0].std
assert ((ratio>.49)&(ratio<.51)).all()

old={'mlp.0.weight':torch.randn(8,285,device=device),'obs_normalizer._mean':torch.randn(1,285,device=device),'obs_normalizer._std':torch.ones(1,285,device=device),'obs_normalizer._var':torch.ones(1,285,device=device)}
new={'mlp.0.weight':torch.zeros(8,290,device=device),'obs_normalizer._mean':torch.zeros(1,290,device=device),'obs_normalizer._std':torch.ones(1,290,device=device),'obs_normalizer._var':torch.ones(1,290,device=device)}
expanded=expand_actor(old,new,old_width=57,new_width=58)
indices=torch.cat([torch.arange(57,device=device)+58*i for i in range(5)])
x=torch.randn(n,290,device=device)
assert torch.allclose((x[:,indices]-old['obs_normalizer._mean'])@old['mlp.0.weight'].T,(x-expanded['obs_normalizer._mean'])@expanded['mlp.0.weight'].T,atol=4e-5,rtol=4e-5)
print('PASS exact pre-intervention targets, bounded feedback, reset,57-to58 transfer, finite half-std precision sampling')
