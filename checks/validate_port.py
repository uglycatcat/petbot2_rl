"""Verify the portable glue against pinned source; run from this checkout."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import glfw
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from petbot_s2s.simulation import Simulation
from petbot_s2s.core.local_frame import ground_frame_features
from petbot_s2s.core.continuous import bound_cycle_features
from petbot_s2s.viewer import Keyboard
from petbot_s2s.core.full_range import effective_request


def definition(path,name):
    text=path.read_text()
    return ast.get_source_segment(text,next(n for n in ast.parse(text).body if getattr(n,'name',None)==name))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--policy',type=Path,default=ROOT/'policies/cycle18_long_r43')
    parser.add_argument('--transition-contract',choices=['legacy','linear_heading'],default=None)
    args=parser.parse_args()
    provenance=json.loads((ROOT/'provenance.json').read_text())
    for name,sha in provenance['source_sha256'].items():
        assert hashlib.sha256((args.source/name).read_bytes()).hexdigest()==sha,name
    for name in ('moving','cycle','continuous','stability','feedback','fast_feedback','ground_motion','local_frame','functional','kinematics','full_range'):
        assert (ROOT/'petbot_s2s/core'/f'{name}.py').read_bytes()==(args.source/'upright'/f'{name}.py').read_bytes()
    for filename,name in [('cycle_task.py','MovingAction'),('continuous_task.py','ContinuousAction'),
                          ('feedback_task.py','FeedbackAction'),('balance_reflex.py','attach_balance_reflex')]:
        expected=definition(args.source/'upright'/filename,name)
        expected=expected.replace('base.MovingAction','MovingAction').replace('base.ContinuousAction','ContinuousAction')
        expected=expected.replace('            from .cycle_task import tensor\n','').replace('    from .feedback_task import observations,ensure\n','')
        expected=expected.replace("SPEC['joints']['left_frame']['range']","env.spec['joints']['left_frame']['range']")
        expected=expected.replace("SPEC['actuators']['left_rear_wheel']['control_range'][1]","env.spec['drives']['left_rear_wheel']['control_range'][1]")
        expected=expected.replace("SPEC['joints']['left_frame']['default']","self._env.spec['joints']['left_frame']['default']")
        actual=definition(ROOT/'petbot_s2s/core/action_terms.py',name)
        assert ast.dump(ast.parse(expected))==ast.dump(ast.parse(actual)),name

    sim=Simulation(args.policy,transition_contract=args.transition_contract)
    c,p=sim.controller,sim.controller.progress
    c.cfg.observation_frame='balance_local'
    base=None
    for filename in ('cycle_task.py','continuous_task.py','feedback_task.py'):
        code=definition(args.source/'upright'/filename,'observations')
        code=code.replace('    from .local_frame import ground_frame_features\n','').replace('        from .full_range import effective_request\n','')
        ns=dict(effective_request=effective_request,torch=torch,state=lambda env:env.s,ensure=lambda env:env.progress,
                ground_frame_features=ground_frame_features,bound_cycle_features=bound_cycle_features,
                prepare_commands=lambda env:None,base=SimpleNamespace(observations=base))
        exec(compile(code,filename,'exec'),ns)
        base=ns['observations']
    maximum=0.
    torch.manual_seed(20261027)
    for phase in range(5):
        for clock in (0.,.60,1.2,1.5,1.8,5.,180.):
            c.s=sim.state()
            for key in ('pos','vel','q','qd','g','heading','wheel','yaw','forward','heading_rate'):
                c.s[key]=torch.randn_like(c.s[key])
            for key in ('raw','command','reference','raw_reference','raw_yaw','motion','error_integral'):
                getattr(p,key).copy_(torch.randn_like(getattr(p,key)))
            p.phase[:]=phase;p.phase_time[:]=clock;p.since[:]=clock
            p.active[:]=phase>0;p.support_run[:]=.3
            c.action.applied[:]=torch.rand_like(c.action.applied)
            c.action.yaw_feedback.integral[:]=.47
            expected=base(c);actual=c.features()
            maximum=max(maximum,float((expected-actual).abs().max()))
            torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-7)

    # Check world angular/linear translation with an independently integrated yaw.
    m,d=sim.model,sim.data
    mujoco.mj_resetData(m,d)
    q=np.array([.91,.18,-.32,.13]);q/=np.linalg.norm(q)
    d.qpos[3:7]=q;d.qvel[:6]=[.05,-.03,.02,.2,-.1,.4]
    mujoco.mj_forward(m,d)
    s=sim.state();start=float(s['yaw'][0]);rate=float(s['heading_rate'][0])
    eps=1e-4
    mujoco.mj_integratePos(m,d.qpos,d.qvel,eps);mujoco.mj_forward(m,d)
    measured=(float(sim.state()['yaw'][0])-start)/eps
    assert abs(measured-rate)<.002,(rate,measured)

    sim.reset()
    c=sim.controller
    forces=[];max_torque=0.
    for i in range(40):
        sim.step()
        if i>20:forces.append(np.array(sim.contact_history)[:,1:,2].sum(1).mean())
        assert np.all(np.abs(sim.data.actuator_force[:5])<=sim.effort+1e-6)
        assert np.max(np.abs(sim.data.actuator_force[5:]))<=.38+1e-6
        max_torque=max(max_torque,float(np.abs(sim.data.actuator_force).max()))
    weight=float(m.body_mass.sum()*9.81)
    assert abs(np.mean(forces)-weight)/weight<.02,(forces,weight)
    assert c.failure_code==0

    keys=Keyboard();keys.event(glfw.KEY_W,glfw.PRESS);keys.event(glfw.KEY_A,glfw.PRESS)
    assert keys.command(.05,.15)==(.05,.15)
    keys.event(glfw.KEY_W,glfw.REPEAT);assert keys.command(.05,.15)==(.05,.15)
    keys.event(glfw.KEY_W,glfw.RELEASE);keys.event(glfw.KEY_A,glfw.RELEASE)
    assert keys.command(.05,.15)==(0.,0.)
    keys.event(glfw.KEY_SPACE,glfw.PRESS);assert keys.toggle
    keys.toggle=False;keys.event(glfw.KEY_SPACE,glfw.REPEAT);assert not keys.toggle
    result=dict(passed=True,transition_contract=args.transition_contract,source_files=len(provenance['source_sha256']),
                action_definitions_identical=4,observation_cases=35,max_observation_error=maximum,
                heading_rate_error=abs(measured-rate),mean_contact_support=float(np.mean(forces)),
                weight=weight,actuator_limits_passed=True,keyboard_press_release_passed=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
