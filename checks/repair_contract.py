"""Regression checks for discarded changes; run inside the GPU container."""
import pathlib,sys,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from upright.moving import MovingProgress
from tools.evaluation_contract import validate
assert torch.cuda.is_available()
p=MovingProgress(64,'cuda:0',.06)
assert torch.equal(p.mode,torch.arange(64,device='cuda:0')%6)
assert not hasattr(p,'support_hist'), 'Temporal tolerance must not silently return'
r=dict(count=64,taken_over=64,success=0,maintained_final_2s=0,final_success=0,premature=0,warmup_failed=0,unfinished=0,lifted=0)
for k in ('success_per_env','maintained_per_env','premature_per_env','lifted_per_env'):r[k]=[False]*64
validate(r)
for bad in (dict(r,unfinished=1),dict(r,success=1),dict(r,metric=float('inf'))):
 try:validate(bad)
 except ValueError:pass
 else:raise AssertionError('Invalid result accepted')
print('PASS CUDA six-mode baseline / no support filter / result completeness and finiteness')
