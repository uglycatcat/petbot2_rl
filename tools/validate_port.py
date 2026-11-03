"""Compare NumPy port against the selected frozen training implementation on CPU."""
import argparse,json,sys
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--project',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.project.resolve()));sys.path.insert(0,str(a.source.resolve()))
    import torch
    from step_demo.estimator import InertialEstimator
    from step_demo.terrain import wheel_terrain_clearance
    from step_s2s.estimator import Estimator
    from step_s2s.control import Controller
    from step_s2s.monitor import terrain_clearance
    torch.set_num_threads(1);m=json.loads(a.metadata.read_text());rng=np.random.default_rng(14)
    ref=InertialEstimator(1,'cpu',m['asset']['source_bodies'],m['joint_names']);ref.reset(torch.tensor([0]))
    port=Estimator(m['asset']['source_bodies'],m['joint_names']);packet=np.array([.02,.3,np.pi/2],np.float32)
    obs_error=0.;state_error=0.;kinematics_error=0.;geometry_error=0.
    for k in range(1200):
        q=(rng.normal(size=9)*.12).astype(np.float32);v=(rng.normal(size=9)*.2).astype(np.float32)
        gyro=(rng.normal(size=3)*.1).astype(np.float32);acc=(rng.normal(size=3)*.2+np.array([0,0,9.81])).astype(np.float32)
        if k%300<30:q[:]=0;v[:]=0;gyro[:]=0;acc[:]=[0,0,9.81]
        if k in [300,600,900]:ref.reset(torch.tensor([0]));port.reset()
        ref.update(torch.from_numpy(acc[None]),torch.from_numpy(gyro[None]),torch.from_numpy(q[None]),.002,torch.from_numpy(v[None]))
        port.update(acc,gyro,q,.002,v)
        ex=(rng.random(2)*2-1).astype(np.float32)
        x=ref.observation(torch.from_numpy(packet[None]),torch.from_numpy(q[None]),torch.from_numpy(v[None]),[1,2],[6,8],torch.from_numpy(gyro[None]),torch.from_numpy(ex[None])).numpy()[0]
        y=port.observation(packet,q,v,gyro,ex);obs_error=max(obs_error,float(np.max(abs(x-y))))
        for name in ['q','com_position','com_velocity']:state_error=max(state_error,float(np.max(abs(getattr(ref,name).numpy()[0]-getattr(port,name)))))
        kinematics_error=max(kinematics_error,float(np.max(abs(ref.com_offset(torch.from_numpy(q[None])).numpy()[0]-port.com_offset(q)))))
    for _ in range(60):
        name='left_rear_wheel';points=np.array(m['asset']['wheel_vertices'][name],np.float32)
        angle=rng.uniform(-2,2);rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]],np.float32)
        points=points@rotation.T+np.array([rng.uniform(.25,2.4),rng.uniform(-1.2,1.2),rng.uniform(.03,.12)],np.float32)
        edges=np.array(m['asset']['rear_edges'][name],np.int64)
        x=wheel_terrain_clearance(torch.from_numpy(points[None]),torch.from_numpy(edges),torch.tensor([.3]),torch.tensor([0.]),.02,2.,1.).item()
        y=terrain_clearance(points,edges,.3,.02,2.,1.);geometry_error=max(geometry_error,abs(x-y))
    controls={};target_error=0.;count=0
    with a.trace.open() as f:
        for line in f:
            r=json.loads(line)
            for i,e in enumerate(r['episode']):
                c=controls.setdefault((i,e),Controller(m['training']));c.request(r['action'][i])
                for _ in range(10):c.advance(.002)
                target_error=max(target_error,abs(c.hip-r['hip_target'][i]),abs(c.wheel-r['wheel_target'][i]));count+=1
    d=dict(sensor_steps=1200,including_reset_and_stationary_correction=True,observation_maxabs=obs_error,state_maxabs=state_error,kinematic_com_maxabs=kinematics_error,
        terrain_cases=60,terrain_maxabs=float(geometry_error),recorded_control_cases=count,target_maxabs=float(target_error))
    a.output.write_text(json.dumps(d,indent=2)+'\n');print(d)
    assert obs_error<1e-4 and state_error<1e-4 and kinematics_error<1e-6 and geometry_error<1e-6 and target_error<2e-5,d


if __name__=='__main__':main()
