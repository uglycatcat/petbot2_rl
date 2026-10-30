# Build on the Isaac Lab 3.0.0-beta2 runtime available on this training server.
# Pass a locally verified NVIDIA Isaac Lab image via --build-arg ISAACLAB_IMAGE=...
ARG ISAACLAB_IMAGE=nvcr.io/nvidia/isaac-lab:3.0.0-beta2
FROM ${ISAACLAB_IMAGE}
RUN /isaac-sim/python.sh -m pip install --no-cache-dir rsl-rl-lib==5.4.1 mujoco==3.8.1 trimesh==4.11.1 scipy tensorboard
ENV ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ALLOW_ROOT=1
WORKDIR /workspace/demo_project/step_rl_demo
