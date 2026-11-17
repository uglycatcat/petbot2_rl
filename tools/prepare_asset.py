"""Build a task-owned USD from the immutable, compiled MJCF model.
Run only inside the project's Isaac Lab container. No upstream task code is used.
"""
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model-root', type=Path, default=Path(__file__).resolve().parents[3] / 'robot_description')
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'assets')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
import mujoco
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf, UsdShade
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

args.output.mkdir(parents=True, exist_ok=True)
model = mujoco.MjModel.from_xml_path(str(args.model_root / 'mjcf/robot_description.xml'))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
stage = Usd.Stage.CreateNew(str(args.output / 'robot.tmp.usda'))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
root = UsdGeom.Xform.Define(stage, '/Robot')
stage.SetDefaultPrim(root.GetPrim())
UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
a = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
a.CreateEnabledSelfCollisionsAttr(True)
a.CreateSolverPositionIterationCountAttr(8)
a.CreateSolverVelocityIterationCountAttr(2)
base_origin = data.xpos[1].copy()

def quat(q):
    return Gf.Quatf(float(q[0]), Gf.Vec3f(*[float(x) for x in q[1:]]))

def vec(v):
    return Gf.Vec3f(*[float(x) for x in v])

names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]
body_paths = {i: '/Robot/' + names[i] for i in range(1, model.nbody)}
geom_paths = []
meta = {'source_sha256': {}, 'mass': float(model.body_mass.sum()), 'bodies': {}, 'joints': {}, 'wheel_vertices': {},
        'locked_joints': ['left_ear', 'right_ear'], 'default_root_height': float(base_origin[2])}
for p in sorted(args.model_root.rglob('*')):
    if p.is_file() and p.suffix.lower() in {'.xml', '.stl'}:
        meta['source_sha256'][str(p.relative_to(args.model_root))] = hashlib.sha256(p.read_bytes()).hexdigest()
for i, path in body_paths.items():
    body = UsdGeom.Xform.Define(stage, path)
    body.AddTranslateOp().Set(Gf.Vec3d(*(data.xpos[i] - base_origin)))
    body.AddOrientOp().Set(quat(data.xquat[i]))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass.CreateMassAttr(float(model.body_mass[i]))
    mass.CreateCenterOfMassAttr(vec(model.body_ipos[i]))
    mass.CreateDiagonalInertiaAttr(vec(model.body_inertia[i]))
    mass.CreatePrincipalAxesAttr(quat(model.body_iquat[i]))
    PhysxSchema.PhysxRigidBodyAPI.Apply(body.GetPrim()).CreateDisableGravityAttr(False)
    # Required by Isaac Lab contact sensors.
    PhysxSchema.PhysxContactReportAPI.Apply(body.GetPrim()).CreateThresholdAttr(0.0)
    meta['bodies'][names[i]] = {'mass': float(model.body_mass[i]), 'inertia': model.body_inertia[i].tolist(),
                              'com': model.body_ipos[i].tolist(), 'default_pos': (data.xpos[i]-base_origin).tolist()}
    for g in range(int(model.body_geomadr[i]), int(model.body_geomadr[i]+model.body_geomnum[i])):
        mid = model.geom_dataid[g]
        if mid < 0:
            raise ValueError('Expected source mesh geometry')
        start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        vertices = model.mesh_vert[start:start+count].astype(float)
        q = model.geom_quat[g]
        rot = Rotation.from_quat([*q[1:], q[0]])
        vertices = rot.apply(vertices) + model.geom_pos[g]
        hull = ConvexHull(vertices)
        # Convex source collision geometry. Reuse one set of points for visual and collision mesh.
        gp = path + '/collision'
        mesh = UsdGeom.Mesh.Define(stage, gp)
        mesh.CreatePointsAttr([vec(v) for v in vertices])
        mesh.CreateFaceVertexCountsAttr([3] * len(hull.simplices))
        mesh.CreateFaceVertexIndicesAttr(hull.simplices.flatten().tolist())
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDisplayColorAttr([vec(model.geom_rgba[g, :3])])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
        phys = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        phys.CreateContactOffsetAttr(0.001)
        phys.CreateRestOffsetAttr(0.0)
        material = UsdShade.Material.Define(stage, '/Robot/materials/' + names[i])
        mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        friction = float(model.geom_friction[g, 0])
        # Source scene has explicit ground/body pair friction 0.1; wheels use source priorities.
        terrain_friction = friction if 'wheel' in names[i] else 0.1
        mat.CreateStaticFrictionAttr(terrain_friction)
        mat.CreateDynamicFrictionAttr(terrain_friction)
        mat.CreateRestitutionAttr(0.0)
        pm = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        pm.CreateFrictionCombineModeAttr('min' if model.geom_priority[g] or 'wheel' not in names[i] else 'max')
        pm.CreateRestitutionCombineModeAttr('min')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material, UsdShade.Tokens.weakerThanDescendants, 'physics')
        geom_paths.append((i, gp, 'wheel' if 'wheel' in names[i] else 'terrain'))
        if 'wheel' not in names[i]:
            # Two filtered shapes preserve source body/body (0.6) and body/terrain (0.1) pairs.
            # They never collide with the same counterpart, so no contact is counted twice.
            sp = path + '/self_collision'
            Sdf.CopySpec(stage.GetRootLayer(), gp, stage.GetRootLayer(), sp)
            sm = UsdShade.Material.Define(stage, '/Robot/materials/' + names[i] + '_self')
            sa = UsdPhysics.MaterialAPI.Apply(sm.GetPrim())
            sa.CreateStaticFrictionAttr(friction); sa.CreateDynamicFrictionAttr(friction)
            sa.CreateRestitutionAttr(0.0)
            PhysxSchema.PhysxMaterialAPI.Apply(sm.GetPrim()).CreateFrictionCombineModeAttr('max')
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(sp)).Bind(sm, UsdShade.Tokens.weakerThanDescendants, 'physics')
            UsdGeom.Imageable(stage.GetPrimAtPath(sp)).CreateVisibilityAttr('invisible')
            geom_paths.append((i, sp, 'self'))
        if 'wheel' in names[i]:
            full = vertices[hull.vertices]
            n = 8192
            t = np.arange(n)+.5
            z = 1.-2.*t/n
            phi = t*np.pi*(3.-np.sqrt(5.))
            directions = np.stack((np.sqrt(1.-z*z)*np.cos(phi),np.sqrt(1.-z*z)*np.sin(phi),z),axis=-1)
            selected = np.unique(np.concatenate([np.argmin(full@d.T,axis=0) for d in np.array_split(directions,32)]))
            support = full[selected]
            rng = np.random.default_rng(901)
            test_dirs = rng.normal(size=(4096,3));test_dirs /= np.linalg.norm(test_dirs,axis=-1,keepdims=True)
            error = max(float(np.max((support@d.T).min(0)-(full@d.T).min(0))) for d in np.array_split(test_dirs,32))
            if error > .0001: raise ValueError(f'Wheel support approximation exceeds0.1mm: {error}')
            meta.setdefault('wheel_support_error_max',{})[names[i]] = error
            meta['wheel_vertices'][names[i]] = support.tolist()
        meta['bodies'][names[i]]['source_geom_friction'] = model.geom_friction[g].tolist()
# Source collision masks: wheel shapes collide only with terrain; non-wheels retain self collisions.
for i, gp, kind in geom_paths:
    targets = [p for j, p, other in geom_paths if i != j and not (kind == other == 'self')]
    if targets:
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(gp)).CreateFilteredPairsRel().SetTargets(targets)
for i, path in body_paths.items():
    parent = int(model.body_parentid[i])
    if parent == 0:
        continue
    j = int(model.body_jntadr[i])
    has_joint = model.body_jntnum[i] > 0
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) if has_joint else names[i] + '_fixed'
    locked = name in meta['locked_joints']
    jp = '/Robot/joints/' + name
    joint = UsdPhysics.RevoluteJoint.Define(stage, jp) if has_joint and not locked else UsdPhysics.FixedJoint.Define(stage, jp)
    joint.CreateBody0Rel().SetTargets([body_paths[parent]])
    joint.CreateBody1Rel().SetTargets([path])
    axis_rot = Rotation.identity()
    anchor1 = model.jnt_pos[j].copy() if has_joint else np.zeros(3)
    if has_joint:
        axis = model.jnt_axis[j]
        axis_rot, _ = Rotation.align_vectors([axis], [[1.,0.,0.]])
    rp = Rotation.from_quat([*data.xquat[parent,1:], data.xquat[parent,0]])
    rc = Rotation.from_quat([*data.xquat[i,1:], data.xquat[i,0]])
    anchor0 = rp.inv().apply(data.xpos[i] + rc.apply(anchor1) - data.xpos[parent])
    r0 = (rp.inv()*rc*axis_rot).as_quat()
    r1 = axis_rot.as_quat()
    joint.CreateLocalPos0Attr(vec(anchor0))
    joint.CreateLocalPos1Attr(vec(anchor1))
    joint.CreateLocalRot0Attr(quat([r0[3],*r0[:3]]))
    joint.CreateLocalRot1Attr(quat([r1[3],*r1[:3]]))
    if has_joint and not locked:
        joint.CreateAxisAttr('X')
        if model.jnt_limited[j]:
            joint.CreateLowerLimitAttr(float(np.rad2deg(model.jnt_range[j,0])))
            joint.CreateUpperLimitAttr(float(np.rad2deg(model.jnt_range[j,1])))
        dof = int(model.jnt_dofadr[j])
        meta['joints'][name] = {'damping':float(model.dof_damping[dof]), 'frictionloss':float(model.dof_frictionloss[dof]),
                              'armature':float(model.dof_armature[dof]), 'range':model.jnt_range[j].tolist()}
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'angular')
        drive.CreateTypeAttr('force')
        drive.CreateStiffnessAttr(0.0);drive.CreateDampingAttr(0.0)
        drive.CreateMaxForceAttr(4.0)
        drive.CreateTargetPositionAttr(0.0);drive.CreateTargetVelocityAttr(0.0)
        p = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
        p.CreateArmatureAttr(float(model.dof_armature[dof]))
        p.CreateJointFrictionAttr(0.0)  # Coulomb torque is applied in the servo, not a dimensionless coefficient.
stage.GetRootLayer().Save()
meta['asset_sha256'] = hashlib.sha256((args.output/'robot.tmp.usda').read_bytes()).hexdigest()
(args.output / 'model.tmp.json').write_text(json.dumps(meta, indent=2))
(args.output / 'robot.tmp.usda').replace(args.output / 'robot.usda')
(args.output / 'model.tmp.json').replace(args.output / 'model.json')
print('ASSET_READY', args.output, 'mass', meta['mass'], 'joints', list(meta['joints']), flush=True)
app.close()
