"""Affine gate: shuffled likelihood, nonzero update, KL conditioning and JIT parity."""
import torch
from tensordict import TensorDict
from agile.petbot.distribution import PetBotResidualActor

def test_residual_gate_update_contract():
    assert torch.cuda.is_available(), 'This acceptance test requires the training GPU container'
    device='cuda'
    data=torch.zeros(64,270,device=device);data[:32,-22]=-1;data[32:,-22]=1
    obs=TensorDict({'policy':data},batch_size=[64])
    actor=PetBotResidualActor(obs,{'actor':['policy']},'actor',7,hidden_dims=[32],obs_normalization=True,
        distribution_cfg={'class_name':'agile.petbot.distribution:PetBotResidualBetaDistribution',
                          'initial_mean':[0,-.5,-.5,-1/11,-1/11,0,0],'initial_std':[.2]*5+[.1]*2}).to(device)
    actions=actor(obs,stochastic_output=True).detach();dist=actor.distribution
    old=tuple(p.detach().clone() for p in dist.params);logp=dist.log_prob(actions).detach()
    assert (dist.std[:32,:5]<.000201).all() and (dist.std[:32,5:]<.0000501).all()
    perm=torch.randperm(64,device=device);actor(obs[perm],stochastic_output=True)
    newlogp=dist.log_prob(actions[perm]);torch.testing.assert_close(newlogp,logp[perm],atol=1e-5,rtol=1e-5)
    loss=-(torch.exp(newlogp-logp[perm])*torch.linspace(-1,1,64,device=device)).mean()-.0002*dist.entropy.mean()
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
    with torch.no_grad():
        last=[m for m in actor.mlp.modules() if isinstance(m,torch.nn.Linear)][-1]
        last.bias[:7]+=.01
        actor(obs,stochastic_output=True)
        kl=dist.kl_divergence(old,dist.params)
        assert torch.isfinite(kl).all() and kl.max()<.001
        torch.testing.assert_close(kl[:32],kl[32:])
        eager=actor(obs);scripted=torch.jit.script(actor.as_jit())
        torch.testing.assert_close(scripted(data),eager)
        assert (eager[:32]-dist.home).abs().max()<2e-6
        assert (eager[32:]-dist.home).abs().max()>1e-4
