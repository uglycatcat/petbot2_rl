"""Deterministic independent-rollout evaluation; one episode per selected environment."""
import json
from pathlib import Path
import numpy as np
import torch

def evaluate(env,wrapped,policy,output,episodes):
    output=Path(output)
    episodes=min(episodes,env.num_envs)
    obs,_=wrapped.reset()
    alive=torch.ones(episodes,dtype=torch.bool,device=env.device)
    succeeded=torch.zeros_like(alive)
    max_top=torch.zeros(episodes,device=env.device,dtype=torch.long)
    frames=[]
    with torch.inference_mode():
        for step in range(env.max_episode_length+1):
            action=policy(obs)
            obs,reward,done,extras=wrapped.step(action)
            tr=env.last_transition
            top=tr['top'][:episodes].sum(-1)
            max_top=torch.where(alive,torch.maximum(max_top,top),max_top)
            succeeded|=alive&tr['success'][:episodes]
            # Trace each policy step, with per-frame validity across resets.
            frames.append({**{k:v[:episodes].cpu().numpy() for k,v in tr.items()},
                           'alive':alive.cpu().numpy(),'reward':reward[:episodes].cpu().numpy(),
                           'done':done[:episodes].cpu().numpy()})
            alive &= ~done[:episodes].bool()
            if not alive.any():break
    arrays={k:np.stack([f[k] for f in frames]) for k in frames[0]}
    np.savez_compressed(output/'evaluation_traces.npz',**arrays)
    # Review from recorded physical geometry and contact, rather than the online success flag.
    pos,bot,force=arrays['wheel_pos'],arrays['bottom'],arrays['forces']
    physical_top=(pos[:,:,:,0]>=env.cfg.edge+.003)&(pos[:,:,:,0]<=env.cfg.edge+env.cfg.platform_length-.003)
    physical_top&=(np.abs(pos[:,:,:,1])<=env.cfg.platform_width/2-.003)
    physical_top&=(np.abs(bot-env.cfg.height)<=.004)&(force[:,:,:,2]>.03)
    independently_successful=(physical_top.all(-1)&arrays['alive']).any(0)
    reported=succeeded.cpu().numpy()
    if not np.array_equal(reported,independently_successful):
        raise RuntimeError('Recorded physics does not agree with online success')
    result={'episodes':episodes,'successes':int(reported.sum()),'success_rate':float(reported.mean()),
        'max_simultaneous_wheels_on_top':max_top.cpu().tolist(),'steps':len(frames),
        'initial_state':'default_on_flat','step_height':env.cfg.height,'step_edge':env.cfg.edge,
        'independent_geometry_check':True,'joint_names':env.robot.joint_names,
        'body_contact_allowed':True,'hold_required':False,'trace_dt':env.step_dt}
    (output/'evaluation.json').write_text(json.dumps(result,indent=2))
    print('EVALUATION',json.dumps(result),flush=True)
    return result
