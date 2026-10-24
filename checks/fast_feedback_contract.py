"""CUDA acceptance for shared common target, saturation, reset and interval bounds."""
import pathlib
import sys
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from upright.fast_feedback import FastYawFeedback

assert torch.cuda.is_available()
torch.manual_seed(30);device='cuda:0';n=4096
f=FastYawFeedback(n,device)
anchor=torch.rand(n,2,device=device)*50-25
policy=(anchor+(torch.rand_like(anchor)*2-1)*28.8).clamp(-25,25)
previous=anchor.clone();command=torch.randn_like(anchor);rate=torch.randn(n,device=device)
residual=torch.rand(n,device=device)*2-1;mask=torch.arange(n,device=device)%2==0
for _ in range(3):
    out=f.process(previous,policy,anchor,command,rate,residual,mask)
    assert torch.allclose(out.mean(-1),policy.mean(-1),atol=3e-6)
    assert out.abs().max()<=25.00001
    assert (out-anchor).abs().max()<=28.80001
    assert torch.equal(out[~mask],policy[~mask])
    assert not f.integral[~mask].any()
    previous=out
assert f.integral.abs().max()<=1.5
f.reset(torch.arange(n//2,device=device));assert not f.integral[:n//2].any()
# In the normal unsaturated regime, inner half-difference changes <=0.24 each20ms.
f.reset();previous=torch.zeros_like(anchor);policy.zero_();anchor.zero_();mask[:]=True
for _ in range(100):
    out=f.process(previous,policy,anchor,command,rate,residual,mask)
    assert (out-previous).abs().max()<=.240001 and out.abs().max()<=2.00001
    previous=out
print('PASS fast yaw: common preserved, bounded wheels/control slew, inactive exact, reset,20ms slew and integral')
