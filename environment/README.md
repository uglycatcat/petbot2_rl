# Runtime used for code checks

The original code review used CPU checks only. Subsequent round01 engineering and full PPO runs used the dedicated single-GPU2/GPU3 containers; missing ONNX Runtime was installed without replacing the simulator stack. Reuse the existing dedicated container; never replace its simulator/PyTorch stack by blindly installing a package with a similar name.

Observed on 2026-10-30: Python 3.12.13, PyTorch 2.10.0+cu128, RSL-RL 5.4.1, TensorDict 0.13.0, NumPy 2.4.6, ONNX 1.21.0, ONNX Runtime 1.31.1. The imported Isaac Lab source reports `VERSION=3.0.0`, whereas Python distribution metadata named `isaaclab` reports `6.1.11`; the source is an editable checkout, not identified by that metadata number alone. Its container copy has no Git metadata. Both source and package versions are recorded in each future run; do not claim a fully pinned portable simulator environment from the pip list alone.

Cold asset conversion and actual PhysX/RSL integration passed the round01 gate. Changes continue to require the relevant engineering checks. Before formal delivery, capture the complete runtime image/source provenance and reproduce the pipeline from a clean checkout without historical assets or checkpoints. Source geometry and inertial properties remain in the read-only model checkout.
