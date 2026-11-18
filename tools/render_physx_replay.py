"""Visualize recorded PhysX body/joint motion using the unchanged source meshes."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MUJOCO_GL','egl')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from evaluate_mujoco import render_video
from recover.model import workspace_root,source_fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True)
    p.add_argument('--env-ids',type=int,nargs='+',required=True)
    args=p.parse_args();prefix=Path(args.evaluation);d=json.loads(prefix.with_suffix('.json').read_text())
    if d['backend']!='physx' or d['source_hash']!=source_fingerprint():raise ValueError('Wrong backend or source')
    a=np.load(prefix.with_suffix('.npz'));ids=args.env_ids
    if not 1<=len(ids)<=4 or min(ids)<0 or max(ids)>=a['root'].shape[1]:raise ValueError('Select one to four existing environment IDs')
    m=mujoco.MjModel.from_xml_path(str(workspace_root()/'robot_description/mjcf/scene.xml'))
    qpos=np.repeat(m.qpos0[None,None,:],len(a['root']),axis=0);qpos=np.repeat(qpos,len(ids),axis=1)
    qpos[:,:,:7]=a['root'][:,ids,:7]
    for j,name in enumerate(d['joint_names']):qpos[:,:,m.joint(name).qposadr[0]]=a['joint_all'][:,ids,j]
    arrays=dict(qpos=qpos,hold=a['hold'][:,ids],eligible=a['eligible'][:,ids])
    per_pose=d['per_pose_samples'];labels=[f'pose_{i//per_pose+1:02d} PhysX env {i}' for i in ids]
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    render_video(m,arrays,out,pose_labels=labels)
    out.with_suffix('.json').write_text(json.dumps(dict(source=str(prefix),env_ids=ids,
        note='Recorded PhysX motion shown with original meshes; no MuJoCo physics resimulation.'),indent=2)+'\n')


if __name__=='__main__':main()
