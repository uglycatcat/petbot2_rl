# 新系列第20轮seed102同步MuJoCo（2026-11-02）

用户要求将上一轮较好策略同步到upright sim2sim。默认从旧C58改为`restart20_s102/update_001000`，来源`upright_restart_round20_s102`、训练提交40e926d、随机初始化1000更新。Isaac低速完整32/32、高范围29/32，但前进/后退满速片段约0.487/0.535m/s；不宣称优于所有历史策略。旧C58原包与选择记录完整保留，新第21轮GPU2/3训练保持。

正式sim2sim分支`rlproj/std_s2s_upright_project`固定`3695300eb64b0e5eb8510313f71f385283f57716`。main的`rl_project/manifest.json`同步该版本、默认策略及验证路径；`default.xml`仍跟踪相同独立分支，无需改pin。正式训练占位目录不变。

## 实现与验证

- 新包只加载一张ONNX，39×5观测、3动作，原30ms策略/2ms物理；旧C58仍用独立历史适配器，不混用观测或父网络。固定控制和几何源码逐字保留，观测方法提取并对照，运行时不依赖训练checkout。
- ONNX与原网络输出最大差8.35e-7；180组原观测/动作目标完全一致。27个MJCF姿态的名义高度修正后轮间隙误差<2.6e-8m，历史缓冲首帧填充/顺序与Isaac一致。
- 普通控制独立源范围，实测前0.736/后−0.739m/s；Q/E共同髋0.15rad且保持四轮接触，策略期间禁用。Space不重置物理状态、仅清理策略历史，意外退出不自动重起；普通控制继承实测速度，正常落地默认髋0。
- MuJoCo3.16静止起立保持6秒和0.15m/s、0.5rad/s混合遥控保持18秒均完成起落交接，无中断，最终实际髋误差<0.01rad。目标/力矩约束通过，实际轮速峰32.82rad/s仍有超调。4秒软件GL窗口加载正常；C58显式选择仍可运行。
- 这是两个固定短场景与工程核对，不是宽初态、长期稳定、满速能力或用户桌面验收。首次导出到只读挂载失败、适配器缺Phase导入以及检查脚本漏扣名义轮高差均保留在review，修正后重跑；无训练中断或权重更改。

详细版本化证据：`rl_project/std_s2s_upright_project/validation/20261102_restart20/`。完整本地轨迹在该项目`outputs/restart20_stationary02`、`outputs/restart20_mixed02`，摘要保存轨迹SHA。MuJoCo使用既有CPU环境，不运行Isaac/PPO、不占训练GPU，源模型未改。

## 桌面更新与启动

在主仓库更新main，再执行：

```bash
repo sync rl_project/std_s2s_upright_project
cd rl_project/std_s2s_upright_project
uv sync --locked --python 3.10
uv run --locked run.py --forward-speed 0.15 --turn-rate 0.5
```

窗口默认推荐两轮0.15/0.5，方向键与Shift继续调速，普通四轮保留完整源范围。若要低速四轮接管作对照，可追加`--ordinary-forward-speed 0.15 --ordinary-turn-rate 0.5`。旧C58对照追加`--policy policies/cycle23_course_r58`。完成用户体验后再讨论新版sim2sim结论，网络导出本身不算通过。
