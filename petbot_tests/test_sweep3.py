"""PPO probability contract for concentration gating at rollout and shuffled update time."""
import torch
from tensordict import TensorDict
from agile.petbot.distribution import PetBotRecoveryActor


def test_state_gate_probability_and_shuffled_ppo_batch():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    data=torch.zeros(64,270,device=device);data[:32,-22]=-1;data[32:,-22]=1
    obs=TensorDict({'policy':data},batch_size=[64])
    actor=PetBotRecoveryActor(obs,{'actor':['policy']},'actor',7,hidden_dims=[32],obs_normalization=True,
        distribution_cfg={'class_name':'agile.petbot.distribution:PetBotRecoveryBetaDistribution',
                          'initial_mean':[0,-.5,-.5,-1/11,-1/11,0,0],'initial_std':[.2]*5+[.1]*2}).to(device)
    actions=actor(obs,stochastic_output=True).detach()
    dist=actor.distribution;old=tuple(p.detach().clone() for p in dist.params);logp=dist.log_prob(actions).detach()
    assert (dist.std[:32,:5]<.00201).all() and (dist.std[:32,5:]<.000501).all()
    assert (dist.std[32:,:5]>.1999).all() and (dist.std[32:,5:]>.0999).all()
    permutation=torch.randperm(64,device=device)
    actor(obs[permutation],stochastic_output=True)
    newlogp=dist.log_prob(actions[permutation]);torch.testing.assert_close(newlogp,logp[permutation],atol=1e-5,rtol=1e-5)
    kl=dist.kl_divergence(tuple(p[permutation] for p in old),dist.params)
    assert torch.isfinite(kl).all() and kl.abs().max()<1e-5
    loss=-(torch.exp(newlogp-logp[permutation])*torch.linspace(-1,1,64,device=device)).mean()-.0002*dist.entropy.mean()
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
    with torch.no_grad():
        eager=actor(obs)
        exported=actor.as_jit();scripted=torch.jit.script(exported)
        torch.testing.assert_close(scripted(data),eager)
