"""在已有 USD 上直接验证 demo 的 normalize_joint_names（不启动 Kit）。"""

import shutil
import sys
from pathlib import Path

from pxr import Usd, UsdPhysics

from agile.petbot.prepare import normalize_joint_names

src = Path(sys.argv[1])
tmp = Path("/tmp/normalize_test.usda")
shutil.copy(src, tmp)
stage = Usd.Stage.Open(str(tmp))
print("改造前:", sorted(p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)))
names = normalize_joint_names(stage)
print("改造后:", names)
stage.GetRootLayer().Save()
stage2 = Usd.Stage.Open(str(tmp))
print("落盘后:", sorted(p.GetName() for p in stage2.Traverse() if p.IsA(UsdPhysics.Joint)))
print(
    "几何 prim 是否保留:",
    [p.GetName() for p in stage2.Traverse() if p.GetName().startswith("head")][:6],
)
