#!/usr/bin/env python3
"""规范化 go2w.usd：
1) 只保留一个 ArticulationRootAPI（放在默认 prim /go2w 上）
2) 给所有刚体加 PhysxContactReportAPI（Isaac Lab 的 ContactSensor 需要）

MJCF 导入器不写这两个东西，而 UsdFileCfg 的 activate_contact_sensors 对
嵌套层级不生效，所以直接在 USD 里落实。
"""
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
from pxr import Usd, UsdPhysics, PhysxSchema  # noqa: E402

PATH = "/workspace/variants/g1_go2w_paper/assets/usd/go2w.usd"
stage = Usd.Stage.Open(PATH)
root = stage.GetDefaultPrim()

for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and prim.GetPath() != root.GetPath():
        prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        print("移除", prim.GetPath(), "ArticulationRootAPI")
    if prim.HasAPI(UsdPhysics.RigidBodyAPI) and not prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
        PhysxSchema.PhysxContactReportAPI.Apply(prim)
        print("加接触报告 API:", prim.GetPath())
if not root.HasAPI(UsdPhysics.ArticulationRootAPI):
    UsdPhysics.ArticulationRootAPI.Apply(root)
stage.Save()

stage = Usd.Stage.Open(PATH)
print("articulation roots:", [p.GetPath().pathString for p in stage.Traverse()
                              if p.HasAPI(UsdPhysics.ArticulationRootAPI)])
print("接触报告刚体数:", sum(1 for p in stage.Traverse()
                             if p.HasAPI(PhysxSchema.PhysxContactReportAPI)))
app.close()
