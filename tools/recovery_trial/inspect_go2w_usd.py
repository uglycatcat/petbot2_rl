#!/usr/bin/env python3
"""列出 go2w.usd 中所有 prim 的完整路径与 ArticulationRootAPI 位置。"""
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
from pxr import Usd, UsdPhysics  # noqa: E402

stage = Usd.Stage.Open("/workspace/variants/g1_go2w_paper/assets/usd/go2w.usd")
print("defaultPrim:", stage.GetDefaultPrim().GetPath())
print("--- 所有 prim ---")
for prim in stage.Traverse():
    tags = []
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        tags.append("ARTICULATION_ROOT")
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        tags.append("RIGID")
    p = prim.GetPath().pathString
    if len(p.strip("/").split("/")) <= 4:
        print(f"  {p}  ({prim.GetTypeName()}) {' '.join(tags)}")
app.close()
