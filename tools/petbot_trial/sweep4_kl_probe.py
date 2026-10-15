import json,torch
from agile.petbot.distribution import PetBotRecoveryBetaDistribution
home=[0,-.5,-.5,-1/11,-1/11,0,0]
d=PetBotRecoveryBetaDistribution(7,home,[.2]*5+[.1]*2).to('cuda')
z=d.initial_bias.reshape(1,2,7)
report={}
for name,scale in [('fallen',[1.]*7),('terminal',[.001]*5+[.0005]*2)]:
 d.recovery_scale=torch.tensor([scale],device='cuda');d.update(z);old=tuple(x.clone() for x in d.params)
 before=d.mean.clone();d.update(z+torch.tensor([[[.01]*7,[0.]*7]],device='cuda'))
 report[name]={'kl':d.kl_divergence(old,d.params).item(),'mean_change_max':(d.mean-before).abs().max().item(),'std':d.std.tolist()}
print(json.dumps(report,indent=2))
