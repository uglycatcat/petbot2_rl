# PetBot upright 1104：冻结的完整训练代码

本临时工程保存用户选定的21r5 seed102版本，取自训练提交`8cac727a270fa670b182bc385b0066330b13ad42`。训练、评估、课程、奖励、资产转换、同次中断恢复和ONNX导出均在本目录；不依赖demo、历史运行目录、旧权重或教师数据。项目由`rlproj/train_upright_project_1104`分支管理。

当前训练与自动迭代全部关闭。对应桌面包已在`../std_s2s_upright_project`，默认`restart21r5_s102`；该目录不因本次收尾改变。`../std_train_upright_project`继续留空，供以后正式项目使用。

用户认可基本流程、起立与运动可用；落地接地后仍会再次抬头并有残余移动，双向连续0—0.75m/s目标未达到。1104是阶段性保留版本，不是全部需求通过。需求与问题见[冻结交接](../../docs/upright_1104.md)和[完整需求](../../docs/current_scope.md)。详细实验历史仅保存在服务器history，不是运行依赖。

## 获取与只读核验

在PetBot主仓库根目录执行`repo sync rl_project/train_upright_project_1104`。布局必须保留同级只读`robot_description/`；模型路径相对解析，不需要历史资产缓存。

```bash
cd rl_project/train_upright_project_1104
python3 project.py check
python3 tools/check_release.py
python3 train.py --dry-run
```

默认`configs/train_1104.json`与原21r5配置的有效值完全相同：seed102、2048环境、1000更新、30ms控制、2ms物理、64步rollout、同次课程与低范围回归。`release.json`记录源版本、逐文件哈希、配置/模型身份及对应ONNX身份。完整训练逻辑保留；包装变化只有默认配置文件名、宿主监督工具的位置检查、冻结权限及文档。未把21r6的髋投影混入1104。

## 重新授权后复现

本节是未来运行说明，不表示当前获得训练授权。恢复必须先明确授权，再将`project.json`中需要的执行项设为true并提交固定版本；自动迭代默认保持false。训练只能在独占可见物理GPU2或3的专用Isaac容器中，不能在宿主或桌面启动。运行期间不修改源码或同步分支。

运行环境见[环境说明](environment/README.md)。以下示例假定容器已按现有工作区布局挂载到`/workspace/petbot`，且单GPU隔离环境符合`require_gpu_isolation()`；宿主监督会核对GPU UUID、显存、日志和最终产物。未准备资产时先从只读MJCF转换，绝不复制旧检查点。

```bash
python3 tools/run_guarded.py --gpu 2 \
  --workdir /workspace/petbot/rl_project/train_upright_project_1104 \
  --log outputs/prepare.log -- -m upright.prepare --headless

python3 tools/run_guarded.py --gpu 2 \
  --workdir /workspace/petbot/rl_project/train_upright_project_1104 \
  --log outputs/seed102_train_attempt0.log -- train.py \
  --config configs/train_1104.json --run-dir outputs/seed102 \
  --seed 102 --num-envs 2048 --iterations 1000 --device cuda:0
```

新运行要求空目录，并保存随机初始网络作为证据。只有同次受迫中断才能以`--resume-interrupted outputs/seed102/checkpoints/<最新检查点>.pt`恢复；配置、源码、模型、软件和run_id必须一致，恢复优化器/课程/随机状态，未完成物理回合复位。完成的实验不能续成新实验，不能加载历史父策略。

在同一已授权容器工作目录内，分别使用`evaluate.py --run-dir outputs/seed102 --checkpoint outputs/seed102/checkpoints/update_001000.pt --scenario static_cycle --output outputs/eval_static --count 32`进行评估。另有`rise`、`low_cycle`、`capability_cycle`场景，后两者可加`--duration 60`。每个输出目录必须新建且不可覆盖；核查result与trace，不能把退出0视为策略通过。

导出入口：`export_policy.py --run-dir outputs/seed102 --checkpoint outputs/seed102/checkpoints/update_001000.pt --output outputs/export_1104`。导出包含数值对照和观测/动作配置；不会自动发布到sim2sim。现成的1104对应ONNX由已有sim2sim工程保存，本训练工程不携带权重。

## 复现边界与结构

完整代码/配置能够独立定义从零训练；封存时只做源一致性、CPU导入/配置和禁启动检查，没有重新进行1000更新或新GPU训练。原21r5的1000更新、短训/恢复/重载与MuJoCo证据属于原固定版本。不同驱动或物理求解器下不承诺逐位相同结果；完整运行环境还需按环境说明准备。

- `upright/controller.py`：单次调用、髋轮目标、普通控制权及限幅。
- `upright/runtime.py`、`curriculum.py`、`task.py`：场景、奖励、课程、Isaac与真实轨迹。
- `upright/learning.py`、`distribution.py`、`experiment.py`：单actor/critic PPO、Beta动作分布与身份恢复。
- `upright/model.py`、`prepare.py`、`asset_cfg.py`：只读模型及可重建派生资产。
- `evaluate.py`、`export_policy.py`：显式固定策略评估与ONNX导出。

普通四轮全性能、Q/E、两轮髋自主、Space独立调用及异常退出契约保留。源模型/物理参数只读，轮目标min(50,25)=±25rad/s；实际轮速超调是已知问题。无周期定时器、goal或自动事件启动入口。
