"""Pinned source parity and real MuJoCo operator ownership for restart ONNX."""
import argparse, json, sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from petbot_s2s.restart import RestartSimulation
from petbot_s2s.restart_core.controller import Phase


def run(source, output, policy):
    sys.path.insert(0,str(source))
    from upright.runtime import Runtime
    from upright.config import Config
    from upright.model import load_model
    import mujoco
    torch.set_num_threads(1)
    manifest=json.loads((policy/'manifest.json').read_text())
    result={'source_commit':manifest['source_commit'],'policy':policy.name,'cases':{}}
    for name in ('controller.py','geometry.py'):
        local='controller_v26.py' if name=='controller.py' and manifest.get('controller_contract')=='restart_v26' else name
        assert (source/'upright'/name).read_bytes()==(ROOT/'petbot_s2s/restart_core'/local).read_bytes()
    with patch('torch.load',side_effect=AssertionError('PT runtime forbidden')),patch('torch.jit.load',side_effect=AssertionError('PT runtime forbidden')):
        sim=RestartSimulation(policy);c=sim.controller
        original=Runtime(1,'cpu',c.dt,Config(**c.manifest['control_config']),load_model())
        original.probe_level=1
        gen=torch.Generator().manual_seed(231)
        max_obs=max_target=0.
        for i in range(180):
            s=sim.state();phase=i%6
            for ctl in (c.control,original.control):
                ctl.phase[:]=phase;ctl.age[:]=(i%17)*.3
            for name in ('command','motion_reference','transition_motion','executed'):
                value=torch.randn(getattr(c.control,name).shape,generator=gen)*.2
                getattr(c.control,name).copy_(value);getattr(original.control,name).copy_(value)
            c.control.yaw_reference[:]=.13;original.control.yaw_reference[:]=.13
            a=c.observation.observation(s);b=original.observation(s)
            max_obs=max(max_obs,float((a-b).abs().max()));assert torch.equal(a,b)
            action=torch.rand(1,3,generator=gen)*2-1
            a=c.control.targets(action,s).clone();b=original.control.targets(action,s)
            max_target=max(max_target,float((a-b).abs().max()));assert torch.equal(a,b)
        result['cases']['source_parity']={'cases':180,'observation_max_error':max_obs,'target_max_error':max_target,'source_files_exact':True}
        max_gap=0.
        for pitch in (-.5,0.,.8):
            for hip in (-.2,0.,.2):
                for yaw in (-1.,0.,1.):
                    q=np.zeros(4);mujoco.mju_euler2Quat(q,np.array([0.,pitch,yaw]),'xyz')
                    sim.data.qpos[3:7]=q;sim.data.qpos[sim.qpos_ids[3:]]=hip;mujoco.mj_forward(sim.model,sim.data)
                    geometry=c.manifest['wheel_geometry']
                    nominal=np.array([geometry[f'{side}_hip'][2]+geometry[f'{side}_front'][2] for side in ('left','right')])
                    nominal-=np.mean([geometry[f'{side}_hip'][2]+geometry[f'{side}_rear'][2] for side in ('left','right')])
                    s=sim.state();height=sim.data.xpos[sim.front,2]-sim.data.xpos[sim.rear,2].mean()-nominal
                    max_gap=max(max_gap,float(np.abs(s['gap'][0].numpy()-height).max()))
        assert max_gap<1e-6,max_gap
        result['cases']['mjcf_geometry']={'poses':27,'max_gap_error_m':max_gap}
        sim.reset();c=sim.controller
        c.set_policy_limits(.05,.15);c.set_command(.75,0.)
        for _ in range(100):sim.step()
        forward=sim.record();assert forward['motion'][0]>.6
        c.set_command(-.75,0.)
        for _ in range(100):sim.step()
        reverse=sim.record();assert reverse['motion'][0]<-.6
        assert max(abs(x) for x in reverse['wheel_target'])<=25
        c.set_command(0.,0.)
        for _ in range(100):sim.step()
        c.set_frame_input(1.)
        for _ in range(150):sim.physics_step()
        c.set_frame_input(0.);held=c.frame
        for _ in range(100):sim.step()
        up=sim.record();assert abs(held-.15)<1e-6 and c.frame==held
        assert up['four_fraction']>.9,up
        before=(sim.data.qpos.copy(),sim.data.qvel.copy(),c.control.executed.clone())
        c.toggle();c.set_frame_input(1.)
        assert not c.manual_frame_allowed and c.frame_input==0
        assert np.array_equal(before[0],sim.data.qpos) and np.array_equal(before[1],sim.data.qvel)
        assert torch.equal(before[2],c.control.executed)
        sim.step();assert sim.record()['phase']==1
        assert c.last_observation.shape==(1,195)
        assert np.allclose(c.last_observation.reshape(5,39).numpy(),c.last_observation[0,:39].numpy())
        result['cases']['operator_ownership']={'forward_measured':forward['motion'][0],'reverse_measured':reverse['motion'][0],'qe_target':held,'qe_four_fraction':up['four_fraction'],'takeover_keeps_state':True,'history_reset_at_space':True,'onnx_only':True}
        s=sim.state();s['g'][:]=torch.tensor([[0.,1.,0.]])
        c.end(s,float(sim.data.time),False,.15)
        assert int(c.control.phase[0])==Phase.EXIT and c.interruptions==1 and not c.armed
        c.begin(sim.state());assert int(c.control.phase[0])==Phase.EXIT
        c.toggle();sim.step();assert int(c.control.phase[0])==Phase.RISE
        result['cases']['failure_exit']={'ordinary_after_failure':True,'new_space_required':True}
    result['passed']=True
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy',type=Path,default=ROOT/'policies/restart20_s102')
    a=p.parse_args();run(a.source.resolve(),a.output,a.policy.resolve())
