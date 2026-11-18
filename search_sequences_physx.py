"""Run the same scratch episode-search distribution in parallel PhysX environments."""
import argparse
import json
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(add_help=False)
AppLauncher.add_app_launcher_args(parser)
launcher_args,search_args=parser.parse_known_args()
app=AppLauncher(launcher_args).app

try:
    import numpy as np
    import torch
    import search_sequences as search
    from recover.env import RecoverEnv,EnvCfg
    from recover.fidelity import check_initial_body_poses
    from recover.model import task_implementation_fingerprint

    class PhysXSequences:
        backend='physx'
        physics_dt=.002

        def __init__(self,threads=4,seed=0):
            self.env=None;self.seed=seed;self.kernel_sha256=task_implementation_fingerprint()

        def run(self,initial,actions,record=False):
            if record:raise ValueError('Use the independent PhysX target replayer to record full trajectories')
            if actions.ndim!=3 or actions.shape[2]!=7 or actions.shape[1]!=800 or len(actions)%4:
                raise ValueError('Expected four equal groups of 800-step sequences')
            if not np.isfinite(actions).all() or np.abs(actions).max()>1:raise ValueError('Invalid actions')
            if self.env is None:
                cfg=EnvCfg();cfg.scene.num_envs=len(actions);cfg.seed=self.seed
                self.env=RecoverEnv(cfg);self.env.render_enabled=False;self.env.auto_reset=False
            env=self.env
            if env.num_envs!=len(actions):raise ValueError('Population size changed')
            # The shared optimizer must supply precisely the protected initial poses.
            poses,_,_=search.load_poses()
            if not np.array_equal(initial,np.repeat(poses,len(actions)//4,axis=0)):
                raise ValueError('Initial poses differ from the protected recordings')
            env.reset();fidelity=check_initial_body_poses(env)
            if search.ACTIVE_OUTPUT is not None:
                (search.ACTIVE_OUTPUT/'initial_fidelity.json').write_text(json.dumps(fidelity,indent=2)+'\n')
            sequence=torch.from_numpy(np.ascontiguousarray(actions)).to(env.device)
            summary=torch.zeros(env.num_envs,10,device=env.device);summary[:,1]=-1;summary[:,3]=-2;summary[:,5]=-1
            eligible=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
            bad=torch.zeros_like(eligible);tail=torch.zeros(env.num_envs,device=env.device)
            with torch.no_grad():
                for step in range(sequence.shape[1]):
                    if step%50==0 and torch.cuda.mem_get_info(env.device)[0]<2*1024**3:
                        raise search.ResourceInterruption('GPU free memory below 2 GiB')
                    env.step(sequence[:,step]);s=env.last_step
                    bad |= ~(torch.isfinite(s['root']).all(-1)&torch.isfinite(s['q']).all(-1)&torch.isfinite(s['qd']).all(-1)&torch.isfinite(s['wheel']).all(-1))
                    invalid=s['invalid']|bad
                    summary[:,5]=torch.where(invalid&eligible,(step+1)*env.step_dt,summary[:,5])
                    eligible &= ~invalid
                    hit=s['success']&eligible&(summary[:,1]<0)
                    summary[:,1]=torch.where(hit,(step+1)*env.step_dt,summary[:,1])
                    summary[:,2]=torch.maximum(summary[:,2],torch.where(eligible,s['hold'],0.))
                    summary[:,3]=torch.maximum(summary[:,3],torch.where(eligible,s['up'],-2.))
                    summary[:,4]+=(s['four']&eligible).float()
                    quality=2*(s['up']+1)+.5*s['contacts'].sum(-1)
                    summary[:,9]=torch.maximum(summary[:,9],torch.where(eligible,quality,0.))
                    if step>=700:tail+=torch.where(eligible,quality,0.)/100
                summary[:,0]=tail+.25*summary[:,9]+10*summary[:,2].clamp_max(.5)+100*(summary[:,1]>=0)-10*(summary[:,5]>=0)
                summary[:,6]=s['up'];summary[:,7]=s['contacts'].sum(-1);summary[:,8]=bad
            result=summary.cpu().numpy().astype(np.float64)
            if not np.isfinite(result).all() or result[:,8].any():raise RuntimeError('Nonfinite PhysX trajectory')
            return result,None

        def __enter__(self):return self
        def __exit__(self,*_):
            if self.env is not None:self.env.close()

    sys.argv=[sys.argv[0],*search_args]
    search.main(engine_factory=PhysXSequences)
except BaseException as error:
    import traceback
    traceback.print_exc()
    if 'search' in globals() and search.ACTIVE_OUTPUT is not None:
        failure=dict(status='failed',error=repr(error))
        (search.ACTIVE_OUTPUT/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
        (search.ACTIVE_OUTPUT/'status.json').write_text(json.dumps(failure,indent=2)+'\n')
    sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:
    app.close()
