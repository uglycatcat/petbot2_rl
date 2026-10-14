"""实验：把派生 MJCF 里与关节同名的 geom 改名后重新转换，看导入的关节 prim 名是否保持契约名。

结论写入 /tmp/geomtest/result.txt（避免 app.close() 吞输出）。
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher

SIM_ASSET = "/workspace/demo_project/recovery_agile_01/artifacts/petbot_asset/6b7c41264d6fade533f0"

parser = __import__("argparse").ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

results = []
try:
    from pxr import Usd, UsdPhysics

    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

    root = Path("/tmp/geomtest")
    shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(SIM_ASSET, root / "asset")
    mjcf = root / "asset/mjcf/robot_description.xml"
    tree = ET.parse(mjcf)
    joints = {j.get("name") for j in tree.getroot().findall("worldbody//joint") if j.get("name")}
    renamed = 0
    for geom in tree.getroot().findall("worldbody//geom"):
        name = geom.get("name")
        if name in joints:
            geom.set("name", f"{name}_geom")
            renamed += 1
    tree.write(mjcf, encoding="unicode")
    results.append(f"renamed geoms: {renamed}")

    converter = MjcfConverter(
        MjcfConverterCfg(
            asset_path=str(mjcf),
            usd_dir=str(root / "out"),
            usd_file_name="t.usd",
            force_usd_conversion=True,
            fix_base=False,
            self_collision=True,
            merge_mesh=False,
            collision_type="Convex Decomposition",
            run_asset_transformer=False,
        )
    )
    stage = Usd.Stage.Open(converter.usd_path)
    prims = sorted(p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.Joint))
    results.append("JOINTS: " + repr(prims))
except Exception as exc:  # noqa: BLE001
    results.append(f"EXCEPTION: {type(exc).__name__}: {exc}")
finally:
    Path("/tmp/geomtest_result.txt").write_text("\n".join(results) + "\n")
    app.close()
