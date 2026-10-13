"""Build a derived servo MJCF and, in the container, a PhysX USD asset."""

from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from .model import (
    POSITION_JOINTS,
    POSITION_KD,
    POSITION_KP,
    WHEELS,
    asset_dir,
    load_model,
    source_dir,
    source_fingerprint,
)


def prepare_mjcf(destination: Path | None = None) -> Path:
    source = source_dir()
    destination = destination or asset_dir()
    destination = destination.resolve()
    if destination.is_relative_to(source):
        raise ValueError("Derived assets must be outside the read-only model directory.")
    destination.mkdir(parents=True, exist_ok=True)
    before = source_fingerprint(source)
    shutil.copytree(source / "meshes", destination / "meshes", dirs_exist_ok=True)
    shutil.copytree(source / "mjcf", destination / "mjcf", dirs_exist_ok=True)
    spec = load_model(source)
    target = destination / "mjcf/robot_description.xml"
    tree = ET.parse(target)
    for drive in tree.getroot().find("actuator"):
        name = drive.get("joint")
        if name in POSITION_JOINTS:
            drive.tag = "position"
            drive.set("kp", str(POSITION_KP[name]))
            drive.set("kv", str(POSITION_KD[name]))
            drive.set("ctrlrange", " ".join(map(str, spec["joints"][name]["range"])))
    tree.write(target, encoding="unicode")
    if source_fingerprint(source) != before:
        raise RuntimeError("Read-only source changed during asset preparation.")
    (destination / "source.json").write_text(json.dumps(spec, indent=2) + "\n")
    return target


def adapt_usd(path: Path, spec: dict) -> dict:
    from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(str(path))
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    by_name = {p.GetName(): p for p in bodies}
    if len(by_name) != len(bodies) or not set(WHEELS).issubset(by_name):
        raise RuntimeError("USD rigid-body names are ambiguous or wheels are missing.")
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one floating articulation root; found {len(roots)}.")
    mass = sum(float(UsdPhysics.MassAPI(p).GetMassAttr().Get() or 0) for p in bodies)
    if abs(mass - spec["mass"]) > 0.001 * spec["mass"]:
        raise RuntimeError(f"MJCF/USD mass mismatch: {spec['mass']} vs {mass}")
    default = stage.GetDefaultPrim()
    if not default.IsValid():
        raise RuntimeError("Imported USD has no default prim.")
    for body in bodies:
        PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr(0.0)
    # Source wheels collide only with ground. Filter wheel-vs-robot pairs explicitly.
    for wheel in WHEELS:
        UsdPhysics.FilteredPairsAPI.Apply(by_name[wheel]).CreateFilteredPairsRel().SetTargets(
            [p.GetPath() for p in bodies if p != by_name[wheel]]
        )
    # PhysX has no MJCF pair-friction table. Match each link/ground sliding coefficient
    # via minimum combine; self-contact friction and torsional friction remain approximations.
    for body in bodies:
        coefficient = 1.4 if body.GetName() in ("left_rear_wheel", "right_rear_wheel") else 0.1
        material = UsdShade.Material.Define(stage, default.GetPath().AppendPath(f"PetBotMaterials/{body.GetName()}"))
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(coefficient)
        physics.CreateDynamicFrictionAttr(coefficient)
        physics.CreateRestitutionAttr(0.0)
        PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr("min")
        for prim in Usd.PrimRange(body):
            # Do not override a nested body's own material.
            owner = prim
            while owner.IsValid() and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
                owner = owner.GetParent()
            if owner == body and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
    stage.GetRootLayer().Save()
    return {"mass_kg": mass, "body_names": list(by_name), "articulation_root": str(roots[0].GetPath())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf-only", action="store_true", help="Prepare source adapter without Isaac Sim.")
    args, remaining = parser.parse_known_args()
    target = prepare_mjcf()
    print(f"Derived MJCF: {target}")
    if args.mjcf_only:
        if remaining:
            parser.error(f"Unrecognized options: {remaining}")
        return
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    try:
        from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

        (target.parents[1] / "asset.json").unlink(missing_ok=True)
        converter = MjcfConverter(
            MjcfConverterCfg(
                asset_path=str(target),
                usd_dir=str(target.parents[1] / "usd"),
                usd_file_name="petbot.usd",
                force_usd_conversion=True,
                fix_base=False,
                self_collision=True,
                merge_mesh=False,
                collision_type="Convex Decomposition",
                run_asset_transformer=False,
            )
        )
        spec = load_model()
        report = adapt_usd(Path(converter.usd_path), spec)
        report.update(
            source_sha256=spec["source_sha256"], usd=str(Path(converter.usd_path).relative_to(target.parents[1]))
        )
        (target.parents[1] / "asset.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        app.close()


if __name__ == "__main__":
    main()
