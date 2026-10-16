"""CUDA checks for bounded concentration, consistent PPO likelihood and export."""
import io, torch
from upright.distribution import UprightBetaDistribution
assert torch.cuda.is_available()
d=UprightBetaDistribution(2,[0.,0.],[.25,.15]).cuda()
x=d.initial_bias.reshape(1,2,2).repeat(4096,1,1).clone().requires_grad_(True)
d.update(x)
assert torch.allclose(d.std[0],x.new_tensor([.25,.15]),atol=1e-6)
y=d.sample();lp=d.log_prob(y);old=tuple(v.detach().clone() for v in d.params)
assert torch.isfinite(lp).all() and (y.abs()<=1).all()
d.update(x);assert torch.allclose(lp,d.log_prob(y))
assert d.kl_divergence(old,d.params).abs().max()<1e-5
(lp.mean()+.001*d.entropy.mean()).backward();assert torch.isfinite(x.grad).all()
z=torch.randn(4096,2,2,device='cuda')*1000;d.update(z)
assert d.std.min()>.058 and d.std.max()<.58
assert torch.allclose(d.mean,d.deterministic_output(z))
export=torch.jit.script(d.as_deterministic_output_module());b=io.BytesIO();torch.jit.save(export,b);b.seek(0);export=torch.jit.load(b)
assert torch.allclose(d.mean,export(z),atol=1e-6)
broad=UprightBetaDistribution(2,[0.,0.],[.35,.45]).cuda();broad.update(broad.initial_bias.reshape(1,2,2))
assert torch.allclose(broad.std[0],x.new_tensor([.35,.45]),atol=1e-6)
print('PASS bounded Beta initial std, sample/likelihood/recompute/KL/gradient/export',float(d.std.min()))
