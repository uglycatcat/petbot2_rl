#!/usr/bin/env python3
"""检查 go2w.usd 组合后是否含 worldBody prim，以及各层里的 articulation root。"""
from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
from pxr import Usd, UsdPhysics  # noqa: E402
import glob

files = sorted(glob.glob("/workspace/variants/g1_go2w_paper/assets/usd/**/*.usd", recursive=True))
for f in files:
    st = Usd.Stage.Open(f)
    roots = [p.GetPath().pathString for p in st.GetPseudoRoot().GetChildren()]
    arts = [p.GetPath().pathString for p in st.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    wb = [p.GetPath().pathString for p in st.Traverse() if p.GetName() == "worldBody"]
    print(f"{f}\n   roots={roots}\n   articulations={arts}\n   worldBody={wb[:3]}")
app.close()
