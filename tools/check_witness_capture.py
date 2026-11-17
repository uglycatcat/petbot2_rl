"""PhysX check that successful episode targets survive auto-reset without an off-by-one error."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();app=AppLauncher(args).app
try:
    import numpy as np
    import torch
    from recover.env import RecoverEnv,EnvCfg
    from recover.fidelity import check_initial_body_poses
    cfg=EnvCfg();cfg.scene.num_envs=64;cfg.capture_witnesses=True
    env=RecoverEnv(cfg);env.render_enabled=False;env.reset()
    fidelity=check_initial_body_poses(env)
    # This is an engineering test of the recorder, not a recovery candidate or
    # a training curriculum. Standing is used to exercise actual success/reset.
    env.standing_probe=True;env.reset()
    action=torch.zeros(64,7,device=env.device)
    action[:,:5]=2*(0-env.low)/(env.high-env.low)-1
    expected=np.zeros((env.max_episode_length,64,7),dtype=np.float32)
    checked=set();records=[]
    for step in range(400):
        action[:,0]=.002*(1+step%5)
        before=env.episode_length_buf.detach().cpu().numpy().copy()
        expected[before,np.arange(64)]=action.detach().cpu().numpy()
        env.step(action)
        counts=env.witness_count.detach().cpu().tolist()
        for pose,count in enumerate(counts):
            for slot in range(count):
                if (pose,slot) in checked:continue
                length=int(env.witness_lengths[pose,slot]);index=int(env.witness_env_ids[pose,slot])
                actual=env.witness_actions[pose,slot,:length].detach().cpu().numpy()
                assert length==int(before[index])+1,(length,before[index])
                assert np.array_equal(actual,expected[:length,index]),(pose,slot,length)
                assert int(env.episode_length_buf[index])==0,'The test must exercise auto-reset'
                records.append(dict(pose=pose,slot=slot,environment=index,steps=length,
                                    final_head_target=float(actual[-1,0]),exact_target_sequence=True))
                checked.add((pose,slot))
        if len(checked)==8:break
    assert len(checked)==8,records
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='passed',captured_episodes=records,initial_fidelity=fidelity,
                simulated_control_steps=step+1,
                action_history_bytes=env.action_history.numel()*env.action_history.element_size(),
                note='Synthetic standing success/reset checks recorder integrity only; not fallen-pose recovery.')
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    env.close()
except BaseException:
    import traceback,os
    traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
else:
    app.close()
