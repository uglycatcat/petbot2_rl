# 1104运行环境

原21r5训练使用Python3.12.13、PyTorch2.10.0+cu128、RSL-RL5.4.1、TensorDict0.13.0、NumPy2.4.6、ONNX1.21.0、ONNX Runtime1.31.1；Isaac Lab源码VERSION为3.0.0，而安装包元数据名isaaclab显示6.1.11，两者不能混淆。具体版本保存在项目release.json和每次新运行run.json中。依赖清单见requirements.txt；不要用pip中的同名isaaclab包替换现有仿真源码栈。

现有专用容器使用本地镜像`petbot-upright-runtime:20261016`，镜像ID为`sha256:4e593e83083ef6ce25b9a071dc397cc109d7bf88aaf1f68149e957f488a6412f`；没有上游RepoDigest。该镜像本身不在Git，不能宣称仅凭pip依赖表就能在任意机器逐位重建完整Isaac/PhysX环境。源码未带Git元数据的仿真环境需要迁移时，应从服务器现存镜像导出/保存并核对版本；相关本地身份记录在history/upright/archival_20261104/container_image.json。训练代码和模型转换不读取history。

容器须只暴露物理GPU2或3：PETBOT_HOST_GPU为2/3，PETBOT_GPU_UUID与NVIDIA_VISIBLE_DEVICES为该卡UUID，CUDA_VISIBLE_DEVICES为0；nvidia-smi只能看到该UUID。挂载PetBot工作区以保留相对模型布局。宿主只运行编排/状态检查；CPU验包不等于GPU仿真验收。本次封存不安装/升级任何软件，也不启动训练。
