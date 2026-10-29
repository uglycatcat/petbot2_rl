ARG ISAAC_BASE=nvcr.io/nvidia/isaac-lab:3.0.0-beta2
FROM ${ISAAC_BASE}
COPY environment/requirements-scratch.txt /tmp/petbot-requirements.txt
RUN /isaac-sim/python.sh -m pip install -r /tmp/petbot-requirements.txt
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace/demo_project/upright_rl_demo
