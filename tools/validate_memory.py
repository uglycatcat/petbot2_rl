"""Check the deployed memory and single-flight detector against frozen training."""
import argparse,json,sys
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--project',type=Path,required=True)
    p.add_argument('--metadata',type=Path,required=True);p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--substeps',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.source.resolve()));sys.path.insert(0,str(a.project.resolve()))
    import torch
    from step_demo.stop_memory import SensorStopMemory as ReferenceMemory,wheel_request
    from step_s2s.stop_memory import SensorStopMemory
    from step_s2s.front_motion import FrontMotion
    from step_s2s.estimator import Estimator,rotate
    from step_s2s.control import Controller
    torch.set_num_threads(1);m=json.loads(a.metadata.read_text());packet=np.array([.02,.3,np.pi/2],np.float32)
    ref=ReferenceMemory(1,'cpu');port=SensorStopMemory();maximum=0.;states=set()
    # Include pass/home debounce, progress reversal, latched stop and reset.
    for k in range(400):
        if k==200:ref.reset(torch.tensor([0]));port.reset()
        j=k%200;com=np.array([j*.005,0,0],np.float32);rear=com-np.array([.08,0,0],np.float32)
        if j>160:com[0]-=.2
        hips=np.full(2,.1 if j%31<5 else 0.,np.float32)
        ref.update(torch.from_numpy(packet[None]),torch.from_numpy(com[None]),torch.from_numpy(rear[None]),torch.from_numpy(hips[None]),.02)
        port.update(packet,com,rear,hips,.02);maximum=max(maximum,float(np.max(abs(ref.observation().numpy()[0]-port.observation()))))
        states.add(tuple(port.observation()[[0,2]]))
    assert states=={(0.,0.),(1.,0.),(1.,1.)}
    # Raw requests and executed feedback use different mappings; compare both.
    action_error=0.
    for mapping in ('affine','zero_centered','zero_margin'):
        c=Controller(dict(m['training'],wheel_mapping=mapping,hip_target_rate_rad_s=10000,wheel_target_accel_rad_s2=100000))
        for value in np.r_[np.linspace(-1,1,1001,dtype=np.float32),[-.900001,-.9,-.899999]]:
            c.request([0,value]);_,actual=c.advance(.002)
            expected=wheel_request(torch.tensor(float(value),dtype=torch.float32),mapping).item()
            action_error=max(action_error,abs(float(actual)-expected))
            np.testing.assert_allclose(c.executed[1],actual/12.5-1,atol=2e-7)
    estimator=Estimator(m['asset']['source_bodies'],m['joint_names']);hips=[m['joint_names'].index(x) for x in ('left_frame','right_frame')]
    memories={};count=0;trace_error=0.;memory_states=set()
    with a.trace.open() as f:
        for line in f:
            row=json.loads(line)
            for i,episode in enumerate(row['episode']):
                memory=memories.setdefault((i,episode),SensorStopMemory())
                q=np.array(row['joint_position'][i],np.float32);orientation=np.array(row['estimated_quaternion'][i],np.float32)
                com=np.array(row['estimated_com'][i],np.float32);offset=estimator.com_offset(q)
                rear=com-rotate(orientation,offset)+rotate(orientation,estimator.rear_offset)
                memory.update(packet,com,rear,q[hips],.02)
                trace_error=max(trace_error,float(np.max(abs(memory.observation()-np.array(row['sensor_stop_memory'][i])))))
                memory_states.add(tuple(memory.observation()[[0,2]]));count+=1
    flights={};physical_count=0
    with a.substeps.open() as f:
        for line in f:
            row=json.loads(line)
            for i,episode in enumerate(row['episode']):
                flight=flights.setdefault((i,episode),FrontMotion())
                minimum=np.zeros((2,3),np.float32);minimum[:,0]=row['wheel_min_x'][i][:2];minimum[:,2]=row['wheel_z'][i][:2]
                # Replay wheel_min_x is already relative to the platform edge.
                flight.update(minimum,np.array(row['net_force'][i][:2],np.float32),np.array(row['platform_normal'][i][:2],np.float32),
                    0.,.02,.002)
                for key,field in [('count','front_liftoff_count'),('air','front_in_air'),('returned_floor','front_returned_floor'),('landed_top','front_top_confirmed')]:
                    np.testing.assert_array_equal(getattr(flight,key),row[field][i])
                assert flight.invalid==row['front_motion_invalid'][i] and flight.complete==row['front_motion_complete'][i]
                physical_count+=1
    report=dict(synthetic_memory_steps=400,memory_maxabs=maximum,reset_and_all_memory_states=True,
        action_mapping_cases=3012,action_mapping_maxabs=action_error,recorded_memory_steps=count,recorded_memory_maxabs=trace_error,
        recorded_memory_states=[[float(v) for v in x] for x in sorted(memory_states)],physical_flight_steps=physical_count,flight_count_and_flags_exact=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(report)
    assert maximum<1e-6 and trace_error<2e-5 and action_error<2e-6,report


if __name__=='__main__':main()
