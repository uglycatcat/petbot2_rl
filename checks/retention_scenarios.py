"""CUDA sampler equivalence without overriding the native policy controller."""
import pathlib
import sys

import torch

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from tools.evaluate_retention import install, prepare_source
from upright.capability import CapabilityProgress
from upright.full_range import FullRangeProgress

assert torch.cuda.is_available(), 'Run this check in the training container'
device='cuda';n=64;ids=torch.arange(n,device=device)
native_methods={name:getattr(CapabilityProgress,name) for name in
                ('before_step','update','reset','request_mode')}
install()
assert all(getattr(CapabilityProgress,k) is v for k,v in native_methods.items())
for speed,yaw,frame in ((.05,.15,(-.2,.2)),(.15,.5,(-.2,.2)),
                        (.3,1.5,(-.2,.2)),(.05,.15,(-1.,1.2))):
    kwargs=dict(n=n,device=device,dt=.06,speed_limit=speed,yaw_limit=yaw,
                frame_range=frame,seed=20261020,training_mixture='off',
                hold_range=(30.,30.),transition_contract='linear_heading',
                handoff_alignment='after',tracking_objective='stability_first')
    old=FullRangeProgress(**kwargs)
    new=CapabilityProgress(**kwargs,mirror_training=True,adaptive_curriculum=True,
                           evaluating=True)
    assert not new.adaptive_curriculum
    for generation in range(3):
        selected=ids if generation==0 else ids[generation::3]
        for obj in (old,new):obj.reset(selected,torch.zeros(n,3,device=device),torch.zeros(n,device=device))
        for name in ('generation','command_count','frame','switch_at','raw','next_command','hold_duration'):
            assert torch.equal(getattr(old,name),getattr(new,name)), name
        assert not new.destination.any()
        for step in range(40):
            selected=ids[(ids+step)%3!=0]
            # Shared state isolates the external scenario generator. Native
            # closed-loop timing/state differences are audited in simulation.
            for obj in (old,new):
                obj.command.copy_(old.raw);obj.since[:]=step*.18
            for obj in (old,new):obj.sample_commands(selected)
            for name in ('command_count','mode','raw','next_command'):
                assert torch.equal(getattr(old,name),getattr(new,name)), (step,name)
source=prepare_source((root/'train_cycle.py').read_text())
compile(source,'train_cycle.py','exec')
assert source.count('retention_spec=install()')==1
print('CUDA: four retention envelopes, asynchronous resets, 40 command draws, native methods and evaluation-only wrapper passed')
