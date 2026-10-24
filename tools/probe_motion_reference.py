"""Read-only fixed-v10 root versus rear-axle ground motion measurement audit."""
import pathlib
import sys
sys.path.insert(0,str(pathlib.Path.cwd()))
assert '--evaluate' in sys.argv


def motion_reference(env,valid):
    from upright.cycle_task import tensor
    robot=env.scene['robot'];d=robot.data
    ids=robot.find_bodies(['left_rear_wheel','right_rear_wheel'],preserve_order=True)[0]
    position=tensor(d.body_link_pose_w)[:,ids,:3].mean(1)-tensor(env.scene.env_origins)
    velocity=tensor(d.body_com_vel_w)[:,ids,:3].mean(1)
    return dict(axle_position=position.tolist(),axle_velocity=velocity.tolist(),motion_reference_valid=valid.tolist())


source=pathlib.Path('train_cycle.py').read_text()
def replace(old,new):
    global source
    assert source.count(old)==1,'Fixed evaluator changed'
    source=source.replace(old,new)
replace('trace.append(dict(legacy_yaw=', 'trace.append(dict(**motion_reference(env,valid&~done.bool()),legacy_yaw=')
replace("folder/'evaluation.json'","folder/'diagnostic_evaluation.json'")
replace("folder/'trace.json'","folder/'diagnostic_trace.json'")
exec(compile(source,'train_cycle.py','exec'),globals())
