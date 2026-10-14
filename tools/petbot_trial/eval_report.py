"""汇总 demo_project/runs/<变体>/artifacts/petbot_eval.json：总成功率、分类成功率、成功用时分布。

用法：python3 demo_env/eval_report.py [变体...]（默认全部 11 组，缺失的跳过）
"""

import collections
import json
import statistics as st
import sys
from pathlib import Path

def _runs_dir() -> Path:
    """定位 demo_project/runs：从脚本位置向上找到含 default.xml 的仓库根，再拼 demo_project/runs。
    这样脚本放在 demo_project/demo_env/ 或版本化的 tools/petbot_trial/ 下都能用。"""
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "default.xml").is_file():
            return candidate / "demo_project" / "runs"
    raise SystemExit("找不到仓库根（default.xml）")


RUNS = _runs_dir()
ALL = [
    "base_s42", "base_s43", "gains2x_s42", "gains_down_s42", "decimation5_s42",
    "reward_still_s42", "reward_config_heavy_s42", "episode15_s42",
    "spawn1m_s42", "ppo64_s42", "ppo_explore_s42",
]
targets = sys.argv[1:] or ALL

header = f'{"变体":<24}{"总成功率":>9}{"直立":>7}{"倒置":>7}{"侧躺":>7}{"n":>6}{"被动(≤1.2s)":>12}{"1.2-2s":>8}{"2-4s":>7}{">4s":>6}{"中位用时":>9}{"末态稳定":>9}'
print(header)
print("-" * len(header))
rows = []
for v in targets:
    path = RUNS / v / "artifacts/petbot_eval.json"
    if not path.is_file():
        print(f"{v:<24} 缺结果（未评估或失败）")
        continue
    d = json.loads(path.read_text())
    ep = d["episodes"]
    s = d["summary"]
    t = [e["time_to_success_s"] for e in ep if e["success"] and e["time_to_success_s"] is not None]
    b = collections.Counter()
    for x in t:
        b["passive" if x <= 1.2 else "1.2-2" if x <= 2 else "2-4" if x <= 4 else ">4"] += 1
    n = len(t) or 1
    rows.append((s["all"]["success_rate"] or 0, v, s, len(ep), b, n, st.median(t) if t else float("nan"),
                 sum(1 for e in ep if e["stable_at_end"]) / len(ep)))
for _, v, s, count, b, n, med, stable in sorted(rows, reverse=True):
    f = lambda k: ("—" if s[k]["success_rate"] is None else f'{s[k]["success_rate"]:.3f}')
    print(f'{v:<24}{f("all"):>9}{f("upright"):>7}{f("inverted"):>7}{f("side_or_tilted"):>7}{count:>6}'
          f'{(b["passive"]/n):>12.2f}{(b["1.2-2"]/n):>8.2f}{(b["2-4"]/n):>7.2f}{(b[">4"]/n):>6.2f}{med:>9.2f}{stable:>9.3f}')
