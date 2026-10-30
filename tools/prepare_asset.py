"""Build a step-owned derived USD from the read-only MJCF, inside the container."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step_demo.config import load_config
from step_demo.model import source_fingerprint


def unique_geometry_names(stage):
    from pxr import Usd, UsdPhysics
    paths = [p.GetPath() for p in stage.Traverse()
             if p.GetName() == p.GetParent().GetName() and not p.HasAPI(UsdPhysics.RigidBodyAPI)]
    for path in paths:
        editor = Usd.NamespaceEditor(stage)
        editor.MovePrimAtPath(path, path.GetParentPath().AppendChild('mesh_'+path.name))
        if not editor.CanApplyEdits(): raise RuntimeError(f'Cannot rename derived geometry {path}')
        editor.ApplyEdits()


def main():
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/robot'))
    parser.add_argument('--model-dir', type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    config = load_config(model_dir=args.model_dir)
    source_sha = source_fingerprint(config.model_dir)
    launcher = AppLauncher(args)
    try:
        from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
        from pxr import Usd, UsdPhysics, PhysxSchema
        import mujoco
        import numpy as np
        args.output.mkdir(parents=True, exist_ok=True)
        converter = MjcfConverter(MjcfConverterCfg(
            asset_path=str(config.model_dir / 'mjcf/robot_description.xml'),
            usd_dir=str(args.output.resolve()), usd_file_name='petbot.usd',
            force_usd_conversion=True, make_instanceable=False, fix_base=False,
            self_collision=True, collision_from_visuals=False, merge_mesh=False,
            import_physics_scene=False,
        ))
        imported = Usd.Stage.Open(converter.usd_path)
        stage = Usd.Stage.Open(imported.Flatten())
        unique_geometry_names(stage)
        bodies = []
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.append(dict(path=str(prim.GetPath()),
                                   mass=UsdPhysics.MassAPI(prim).GetMassAttr().Get()))
        model = mujoco.MjModel.from_xml_path(str(config.model_dir / 'mjcf/robot_description.xml'))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)  # source kinematics only; no dynamics or policy rollout
        mass = model.body_mass.sum()
        com = (data.xipos * model.body_mass[:, None]).sum(0) / mass
        # Repair importer omissions in the derived asset only, using source values.
        rigid_prims = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        for p in rigid_prims:
            PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.)
            if 'wheel' in p.GetName():
                pairs = UsdPhysics.FilteredPairsAPI.Apply(p).CreateFilteredPairsRel()
                pairs.SetTargets([other.GetPath() for other in rigid_prims if other != p])
        for p in stage.Traverse():
            if p.HasAPI(UsdPhysics.MaterialAPI):
                material = UsdPhysics.MaterialAPI(p)
                friction = material.GetDynamicFrictionAttr().Get()
                if friction is not None:
                    material.CreateStaticFrictionAttr(float(friction))
                    physx = PhysxSchema.PhysxMaterialAPI.Apply(p)
                    physx.CreateFrictionCombineModeAttr('min' if abs(friction - .1) < 1e-6 else 'max')
        flat = args.output / 'petbot.usd'
        stage.Flatten().Export(str(flat.resolve()))
        joints = {}
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            dof = model.jnt_dofadr[j]
            joints[name] = dict(damping=float(model.dof_damping[dof]),
                                frictionloss=float(model.dof_frictionloss[dof]),
                                armature=float(model.dof_armature[dof]))
        wheel_vertices = {}
        for name in ('left_front_wheel', 'right_front_wheel', 'left_rear_wheel', 'right_rear_wheel'):
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            mesh = model.geom_dataid[g]
            vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]].copy()
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
            vertices = vertices @ rotation.reshape(3, 3).T + model.geom_pos[g]
            from scipy.spatial import ConvexHull
            # Collapse sub-0.1 mm surface tessellation. Expand support by 0.1 mm
            # in the task, so clearance remains conservative in every direction.
            rounded = np.unique(np.round(vertices / .0001) * .0001, axis=0)
            wheel_vertices[name] = rounded[ConvexHull(rounded).vertices].tolist()
        source_bodies = []
        for b in range(1, model.nbody):
            source_bodies.append(dict(name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b),
                parent=int(model.body_parentid[b])-1, mass=float(model.body_mass[b]),
                pos=model.body_pos[b].tolist(), com=model.body_ipos[b].tolist(),
                joint=(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.body_jntadr[b])
                       if model.body_jntnum[b] else None),
                axis=(model.jnt_axis[model.body_jntadr[b]].tolist() if model.body_jntnum[b] else [0.,0.,0.])))
        metadata = dict(source_sha256=source_sha, source_mass_kg=float(mass),
                        initial_com=com.tolist(), initial_root_pos=model.qpos0[:3].tolist(),
                        bodies=bodies, source_bodies=source_bodies, joints=joints, wheel_vertices=wheel_vertices,
                        usd_file='petbot.usd', source_backend='MJCF kinematics only')
        (args.output / 'asset.json').write_text(json.dumps(metadata, indent=2) + '\n')
        assert source_fingerprint(config.model_dir) == source_sha
        print(json.dumps({k:v for k,v in metadata.items() if k not in ('wheel_vertices','source_bodies')}, indent=2))
    finally:
        launcher.app.close()


if __name__ == '__main__':
    main()
