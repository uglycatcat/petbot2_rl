"""Check that the proposed exploration contrast changes variance, not action means."""
import argparse,copy,json,pathlib,sys,torch
from tensordict import TensorDict
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from upright.rise_adapter import RiseAdapterModel
p=argparse.ArgumentParser();p.add_argument('checkpoint',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
assert torch.cuda.is_available()
c=json.loads((a.checkpoint.parent/'config.json').read_text());saved=torch.load(a.checkpoint,map_location='cuda:0',weights_only=False)
raw=saved['critic_state_dict']['obs_normalizer._mean'].expand(64,-1).clone();raw[:,-58+41:-58+46]=0.;raw[:,-58+43]=1.;raw[:,-58+48]=.6
obs=TensorDict({'policy':raw},batch_size=[64]);outputs=[];variances=[]
for factor in (1.,4.):
 cfg=copy.deepcopy(c['runner']['actor']);cfg.pop('class_name');cfg['balance_noise_factor']=factor
 actor=RiseAdapterModel(obs,obs_groups={'actor':['policy']},obs_set='actor',output_dim=3,**cfg).to('cuda:0').eval();actor.load_state_dict(saved['actor_state_dict'],strict=True)
 with torch.no_grad():
  outputs.append(actor(obs,stochastic_output=False));actor(obs,stochastic_output=True)
  d=actor.distribution._distribution;assert torch.isfinite(d.entropy()).all();variances.append(d.variance.clone())
assert torch.equal(outputs[0],outputs[1]);assert (variances[0][:,:2]>variances[1][:,:2]).all();assert torch.equal(variances[0][:,2],variances[1][:,2])
result=dict(passed=True,cuda=True,factors=[1,4],mean_difference=0.,variance_ratio_frame_common=(variances[0][:,:2]/variances[1][:,:2]).mean(0).tolist(),yaw_variance_unchanged=True)
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
