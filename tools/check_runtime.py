"""Real MuJoCo checks: immutable physics, idle/start/reset and repeatable inference."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from step_s2s.runtime import Preview,PROJECT
from step_s2s.scene import source_fingerprint


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    model_dir=PROJECT.parents[1]/'robot_description';before=source_fingerprint(model_dir);s=Preview(continue_on_failure=False)
    original=mujoco.MjModel.from_xml_path(str(model_dir/'mjcf/scene.xml'))
    fields=['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat','jnt_type','jnt_pos','jnt_axis','jnt_range',
        'dof_damping','dof_armature','dof_frictionloss','actuator_ctrlrange','actuator_forcerange','actuator_gainprm','actuator_biasprm','actuator_gear']
    for field in fields:np.testing.assert_array_equal(getattr(original,field),getattr(s.model,field))
    for name in ['lower_body','head','left_ear','right_ear','left_frame','right_frame',*['left_front_wheel','right_front_wheel','left_rear_wheel','right_rear_wheel']]:
        a=original.geom(name).id;b=s.model.geom(name).id
        for field in ['geom_pos','geom_quat','geom_friction','geom_contype','geom_conaffinity','geom_solref','geom_solimp']:
            np.testing.assert_array_equal(getattr(original,field)[a],getattr(s.model,field)[b])
    q0=s.data.qpos.copy()
    for _ in range(100):s.step()
    assert s.state=='IDLE' and s.policy_calls==0 and s.data.time==0 and s.packet_count==0
    np.testing.assert_array_equal(q0,s.data.qpos)
    assert abs(s.edge-(s.data.subtree_com[s.root,0]+.30))<1e-8
    sequences=[];finals=[]
    for _ in range(2):
        assert s.key(32);assert not s.key(32);seq=[]
        for i in range(750):
            row=s.step();seq.append(np.r_[s.data.qpos,s.data.qvel,s.last_observation,s.controller.executed])
            if s.state!='RUNNING':break
        sequences.append(np.array(seq));finals.append(s.snapshot());assert s.policy_calls>0 and s.packet_count==1
        s.key(ord('R'));assert s.state=='IDLE' and s.policy_calls==0 and s.packet_count==0 and s.substeps==0
        np.testing.assert_array_equal(q0,s.data.qpos);np.testing.assert_array_equal(s.estimator.com_position,np.zeros(3));assert not s.monitor.lifted.any()
    np.testing.assert_array_equal(sequences[0],sequences[1])
    assert before==source_fingerprint(model_dir)
    d=dict(status='passed',physics_arrays_exact=fields,robot_geometry_arrays_exact=True,source_fingerprint=before,
        idle_no_inference_or_motion=True,space_only_starts_once=True,reset_clears_estimator_targets_monitor_and_packet=True,
        repeat_frames=len(sequences[0]),repeated_rollout_maxabs=0.,runtime_does_not_import_torch='torch' not in sys.modules,
        engine=mujoco.__version__,policy_id=s.metadata['policy_id'],outcomes=finals)
    assert d['runtime_does_not_import_torch'];args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(d,indent=2)+'\n')
    print({k:v for k,v in d.items() if k not in ['outcomes','physics_arrays_exact']})


if __name__=='__main__':main()
