"""Read-only actual-policy activation probe; never samples or changes actions."""
import hashlib
import json
import pathlib
import sys

sys.path.insert(0,str(pathlib.Path.cwd()))
assert '--evaluate' in sys.argv
captured=[]


def install_distribution_probe(runner,env):
    from upright.continuous_task import ensure
    def capture(module,inputs,output):
        p=ensure(env)
        mask=alive & (p.phase==2) & (p.phase_time>1.2)
        if mask.any():
            captured.append((inputs[0][mask].detach().cpu(),output[mask].detach().cpu()))
    runner.alg.get_policy().mlp.register_forward_hook(capture)


def finish_distribution_probe(folder):
    import torch
    from upright.distribution import parameters
    latent=torch.cat([x[0] for x in captured]);logits=torch.cat([x[1] for x in captured])
    a,b=logits.unbind(-2);alpha,beta=parameters(a,127.),parameters(b,127.)
    scale=alpha.new_tensor([4.,16.,16.]);aa=alpha*scale;bb=beta*scale
    std=2.*torch.sqrt(aa*bb/((aa+bb).square()*(aa+bb+1)))
    def derivative(x):
        return 127.**2*torch.sigmoid(x)/(127.+torch.nn.functional.softplus(x)).square()
    da=2*beta/(alpha+beta).square()*derivative(a)
    db=-2*alpha/(alpha+beta).square()*derivative(b)
    gain=torch.sqrt(da.square()+db.square())
    quant=lambda x:torch.quantile(x,torch.tensor([.01,.1,.5,.9,.99]),dim=0).tolist()
    report=dict(diagnostic_only=True,samples=len(logits),quantiles=[.01,.1,.5,.9,.99],
        channel_order=['frame','common_wheel','yaw_residual'],logits_quantiles=quant(logits),
        std_quantiles=quant(std),mean_output_logit_gain_quantiles=quant(gain),
        alpha_quantiles=quant(alpha),beta_quantiles=quant(beta),
        physical_std_quantiles=quant(std*std.new_tensor([1.1,25.,.25])),
        source_hash=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
    (folder/'distribution_probe.json').write_text(json.dumps(report,indent=2))
    torch.save(dict(latent=latent,logits=logits),folder/'distribution_samples.pt')


source=pathlib.Path('train_cycle.py').read_text()
def replace(old,new):
    global source
    assert source.count(old)==1,'Fixed evaluator changed'
    source=source.replace(old,new)
replace('policy=runner.get_inference_policy(device=args.device);obs=',
        'install_distribution_probe(runner,env);policy=runner.get_inference_policy(device=args.device);obs=')
replace("folder/'evaluation.json'","folder/'diagnostic_evaluation.json'")
replace("folder/'trace.json'","folder/'diagnostic_trace.json'")
replace('finally:', '    if captured:finish_distribution_probe(folder)\nfinally:')
exec(compile(source,'train_cycle.py','exec'),globals())
