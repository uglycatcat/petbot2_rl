# 第20轮正常交接修正发布到MuJoCo

2026-11-03用户明确要求同步刚验证的交接修正供桌面体验，替代此前sim2sim保持不动的边界。训练21r1固定运行继续，本次没有修改训练副本或替换训练权重。

发布分支rlproj/std_s2s_upright_project，提交773836e168ef3c20d17b084e9af34920e13aa69b。默认恢复restart20_s102原1000更新ONNX，网络SHA仍20e928cd45b4333f8213b12acf5193fbfaa597f04986606c78f960952fa5d60f；训练来源40e926d，控制器单独固定c8443fa的正常交接v1，manifest选择restart_v20_handoff，30ms/2ms时序。selection、策略manifest、控制源码provenance、根项目manifest与使用索引一起更新；default.xml仍指同名分支。

仅正常LAND确认后不再继承旧RL轮目标渐变，直接由当前普通输入产生轮速目标。原.18秒确认、髋回0/限速、异常EXIT渐变、起立/两轮、源模型/力矩限制不变。第26轮历史包和源码保留但不作默认；未迁入21r1新奖励或未完成的网络。

## 发布验证

- 三组42秒零输入/前后运动后松键均完成，后退13.32/13.95/11.73→3.15/3.59/2.60cm；首次接触记录至停稳.21/.24/.27秒。每组交接入口前800帧与原发布接口相同，前三组各1334帧（40秒）与开发候选逐帧一致。
- 持续+.15指令落地后普通控制实测约+.144m/s；交接中松键立即给零轮目标。合成异常EXIT保留首轮目标继承、后续按零输入停车，无自动重起。此项不等价任意实际跌倒验证。
- 180组源码观测/目标逐项相同，27姿态几何、普通前后满速、Q/E、Space历史重置和ONNX独占加载有效；4秒Xvfb窗口启动通过，默认包正确加载。
- 目标/髋/采样力矩有效，实测轮速峰32.82rad/s超调保留；两轮零令约.01m/s漂移与厘米级落地后退仍在。用户桌面与全能力验收未通过，不用最终停车替代落地质量。

证据见[发布review](../rl_project/std_s2s_upright_project/validation/20261103_round20_handoff/review.json)。全部本次运行退出0且产物/物理检查有效，无失败重试或训练中断；不是新训练成绩。

## 桌面更新

在主仓库根目录更新main后：

```bash
repo sync rl_project/std_s2s_upright_project
cd rl_project/std_s2s_upright_project
uv run --locked run.py --forward-speed 0.15 --turn-rate 0.5
```

默认即第20轮交接修正版；方向键仍可调整两轮上限，普通四轮范围独立。实际桌面checkout需上述同步，服务器推送不等于桌面已拉取。
