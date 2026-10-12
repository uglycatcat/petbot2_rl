#!/usr/bin/env python3
"""把作者的 Go2-W MJCF（论文机器人）转成 Isaac Lab 可用的 USD。"""
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# MJCF 导入器扩展默认未启用（报 "MJCFCreateImportConfig wasn't registered"），显式打开
import omni.kit.app  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "isaacsim.asset.importer.mjcf", True
)

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402

SRC = "/workspace/variants/g1_go2w_paper/assets/go2w/go2w.xml"
OUT = "/workspace/variants/g1_go2w_paper/assets/usd"

cfg = MjcfConverterCfg(
    asset_path=SRC,
    usd_dir=OUT,
    usd_file_name="go2w.usd",
    fix_base=False,
    import_sites=True,
    self_collision=False,
    make_instanceable=False,
    force_usd_conversion=True,
)
conv = MjcfConverter(cfg)
print("USD_PATH:", conv.usd_path)

# 打印转换后的关节/刚体，确认可被 Isaac Lab 使用
from isaaclab.sim import SimulationContext  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

stage = Usd.Stage.Open(conv.usd_path)
joints, bodies = [], []
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.RevoluteJoint):
        joints.append(prim.GetName())
    if prim.IsA(UsdPhysics.RigidBodyAPI.GetSchemaClass() if hasattr(UsdPhysics.RigidBodyAPI, "GetSchemaClass") else UsdPhysics.RigidBodyAPI):
        bodies.append(prim.GetName())
print("JOINTS(%d):" % len(joints), joints)
print("BODIES(%d):" % len(bodies), bodies[:25])

simulation_app.close()
