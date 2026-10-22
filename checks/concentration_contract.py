"""CUDA mean/export preservation, sample variance, likelihood and gradient contract."""
import pathlib,sys,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from upright.distribution import UprightBetaDistribution
assert torch.cuda.is_available();device='cuda:0';torch.manual_seed(42)
raw=torch.full((20000,2,3),50.,device=device,requires_grad=True)
raw.data[:,0,:]+=3.
std=[];mean=[]
for scale in ([1,1,1],[1,4,4],[4,16,16]):
 d=UprightBetaDistribution(3,[0,0,0],[.4,.4,.4],128.,scale).to(device);d.update(raw)
 action=d.sample();assert action.abs().max()<1
 expected=d.mean;export=d.as_deterministic_output_module()(raw)
 assert torch.allclose(expected,export,atol=2e-6)
 manual=torch.distributions.Beta(*d.params).log_prob((action+1)/2).sum(-1)-3*torch.log(torch.tensor(2.,device=device))
 assert torch.allclose(d.log_prob(action),manual,atol=2e-5)
 assert torch.allclose(action.std(0),d.std.mean(0),rtol=.025,atol=.001)
 assert (d.kl_divergence(d.params,d.params).abs()<1e-5).all()
 loss=-d.log_prob(action.detach()).mean();loss.backward();assert torch.isfinite(raw.grad).all();raw.grad.zero_()
 std.append(d.std[0].detach());mean.append(expected[0].detach())
assert all(torch.allclose(mean[0],m,atol=2e-6) for m in mean)
assert (std[2]<std[1]).all() and (std[1][1:]<std[0][1:]).all()
assert std[2].min()>.001
print('PASS deterministic/export mean unchanged, sampled std, log_prob, KL, gradients; std=',torch.stack(std).tolist())
