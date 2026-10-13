"""打印作者权重（model_7999.pt）的网络结构：MLP 每层形状，用于文档核对。

用法（容器内）：
    /isaac-sim/python.sh /workspace/recovery_env/go2w_checkpoint_shapes.py [检查点路径]
"""
import sys
import torch

path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/variants/g1_go2w_paper/assets/model_7999.pt"
d = torch.load(path, map_location="cpu")
sd = d.get("model_state_dict", d) if isinstance(d, dict) else d
print("检查点:", path)
if isinstance(d, dict):
    print("顶层键:", sorted(d.keys()))
    print("iter:", d.get("iter"))
print("参数张量形状（2 维以上）：")
for k, v in sorted(sd.items()):
    if hasattr(v, "shape") and v.dim() >= 2:
        print(f"  {k:32s} {tuple(v.shape)}")
