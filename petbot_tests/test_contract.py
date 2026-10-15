"""CPU checks. These do not stand in for an Isaac/PhysX runtime smoke test."""

import ast
import importlib.util
import math
import sys
import xml.etree.ElementTree as ET
from types import ModuleType, SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from agile.petbot.collection import disabled_drives, validate_dataset
from agile.petbot.core import HoldTracker, SuccessThresholds, sample_joints, success_components, uniform_quaternions
from agile.petbot.model import (
    JOINT_ORDER,
    POSITION_JOINTS,
    POSITION_KP,
    PROJECT_ROOT,
    REAR_WHEELS,
    load_model,
    source_dir,
    source_fingerprint,
)
from agile.petbot.prepare import prepare_mjcf


def test_source_and_servo_adapter_preserve_physics(tmp_path):
    before = source_fingerprint()
    target = prepare_mjcf(tmp_path)
    source = mujoco.MjModel.from_xml_path(str(source_dir() / "mjcf/scene.xml"))
    adapted = mujoco.MjModel.from_xml_path(str(target.parent / "scene.xml"))
    assert (adapted.nq, adapted.nv, adapted.nu) == (16, 15, 7)
    for name in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_pos",
        "body_quat",
        "jnt_axis",
        "jnt_range",
        "jnt_limited",
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
        "geom_contype",
        "geom_conaffinity",
        "geom_friction",
        "geom_priority",
        "pair_friction",
        "actuator_forcerange",
        "mesh_vert",
    ):
        np.testing.assert_array_equal(getattr(source, name), getattr(adapted, name), err_msg=name)
    assert adapted.body_mass.sum() == pytest.approx(1.65254)
    for name in POSITION_JOINTS:
        actuator = mujoco.mj_name2id(adapted, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert adapted.actuator_gainprm[actuator, 0] == POSITION_KP[name]
        assert adapted.actuator_biasprm[actuator, 1] == -POSITION_KP[name]
    assert source_fingerprint() == before
    with pytest.raises(ValueError, match="read-only"):
        prepare_mjcf(source_dir() / "test_should_not_exist")


def test_only_position_actuators_changed(tmp_path):
    target = prepare_mjcf(tmp_path)
    original = ET.parse(source_dir() / "mjcf/robot_description.xml").getroot()
    adapted = ET.parse(target).getroot()
    for section in ("compiler", "option", "default", "asset", "worldbody", "sensor"):
        assert ET.tostring(original.find(section)) == ET.tostring(adapted.find(section))
    for name in REAR_WHEELS:
        assert (
            original.find(f"actuator/velocity[@joint='{name}']").attrib
            == adapted.find(f"actuator/velocity[@joint='{name}']").attrib
        )


def test_actual_action_configuration_matches_source_limits():
    # Execute the actual ActionsCfg body with inert config containers, not a second mapping implementation.
    tree = ast.parse((PROJECT_ROOT / "agile/petbot/env_cfg.py").read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "ActionsCfg")
    cls.decorator_list = []
    ns = {
        "SPEC": load_model(),
        "POSITION_JOINTS": POSITION_JOINTS,
        "REAR_WHEELS": REAR_WHEELS,
        "lab_mdp": SimpleNamespace(JointPositionActionCfg=SimpleNamespace, JointVelocityActionCfg=SimpleNamespace),
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "ActionsCfg", "exec"), ns)
    actions = ns["ActionsCfg"]()
    assert actions.joint_position.joint_names + actions.rear_velocity.joint_names == list(POSITION_JOINTS + REAR_WHEELS)
    for name in POSITION_JOINTS:
        cfg = actions.joint_position
        assert cfg.preserve_order and not cfg.use_default_offset
        low, high = ns["SPEC"]["joints"][name]["range"]
        assert cfg.offset[name] - cfg.scale[name] == pytest.approx(low)
        assert cfg.offset[name] + cfg.scale[name] == pytest.approx(high)
        assert cfg.clip[name] == (low, high)
    assert actions.rear_velocity.scale == dict.fromkeys(REAR_WHEELS, 25)


def test_full_orientation_sampling_and_wheel_sampling():
    rng = torch.Generator().manual_seed(1234)
    q = uniform_quaternions(50000, generator=rng)
    torch.testing.assert_close(q.norm(dim=-1), torch.ones(len(q)), atol=1e-6, rtol=0)
    # Isotropic quaternion second moments; gravity z spans both upright and inverted.
    torch.testing.assert_close(q.square().mean(0), torch.full((4,), 0.25), atol=0.005, rtol=0)
    vertical = 1 - 2 * (q[:, 0].square() + q[:, 1].square())
    assert (vertical > 0.8).float().mean() == pytest.approx(0.1, abs=0.01)
    assert (vertical < -0.8).float().mean() == pytest.approx(0.1, abs=0.01)
    spec = load_model()
    # Deliberately scramble engine joint order.
    names = list(reversed(JOINT_ORDER))
    positions = sample_joints(1000, names, spec, "cpu", rng)
    assert torch.isfinite(positions).all()
    for idx, name in enumerate(names):
        low, high = spec["joints"][name]["range"] or [-math.pi, math.pi]
        assert positions[:, idx].min() >= low and positions[:, idx].max() <= high


def nominal(count=1):
    return {
        "gravity": torch.tensor([[0.0, 0.0, -1.0]]).repeat(count, 1),
        "joint_error": torch.zeros(count, 5),
        "lin_vel": torch.zeros(count, 3),
        "ang_vel": torch.zeros(count, 3),
        "pos_vel": torch.zeros(count, 5),
        "wheel_vel": torch.zeros(count, 4),
        "height": torch.full((count,), 0.085),
        "target_height": 0.085,
        "wheel_forces": torch.ones(count, 4),
        "cfg": SuccessThresholds(),
    }


def test_success_rejects_inversion_motion_loss_of_support_and_nan():
    fields = nominal(7)
    fields["gravity"][1, 2] = 1
    fields["joint_error"][2, 0] = 0.2
    fields["wheel_vel"][3, 0] = 1
    fields["wheel_forces"][4, 0] = 0
    fields["height"][5] = 0.15
    fields["gravity"][6, 0] = float("nan")
    accepted = torch.stack(list(success_components(**fields).values())).all(0)
    assert accepted.tolist() == [True, False, False, False, False, False, False]


def test_hold_requires_consecutive_steps_and_resets_each_env():
    tracker = HoldTracker(2, dt=0.02, hold_s=1.0, device="cpu")
    for t in range(49):
        assert not tracker.update(torch.ones(2, dtype=torch.bool), torch.full((2,), (t + 1) * 0.02)).any()
    tracker.update(torch.tensor([True, False]), torch.ones(2))
    assert tracker.ever.tolist() == [True, False]
    assert tracker.first_time[0] == 1
    assert tracker.steps.tolist() == [50, 0]
    tracker.reset([0])
    assert not tracker.ever.any()
    assert torch.isnan(tracker.first_time).all()


def test_drives_restore_on_collection_failure():
    class Robot:
        data = SimpleNamespace(joint_stiffness=torch.tensor([[3.0, 0.0]]), joint_damping=torch.tensor([[0.1, 0.14]]))

        def write_joint_stiffness_to_sim(self, value):
            self.data.joint_stiffness = value.clone()

        def write_joint_damping_to_sim(self, value):
            self.data.joint_damping = value.clone()

        def set_joint_effort_target(self, value):
            assert not value.any()

    robot = Robot()
    with pytest.raises(RuntimeError):
        with disabled_drives(robot):
            assert not robot.data.joint_stiffness.any() and not robot.data.joint_damping.any()
            raise RuntimeError("failed collection")
    torch.testing.assert_close(robot.data.joint_stiffness, torch.tensor([[3.0, 0.0]]))
    torch.testing.assert_close(robot.data.joint_damping, torch.tensor([[0.1, 0.14]]))


def test_cache_rejects_invalid_or_incomplete_states():
    data = {"root": torch.zeros(3, 13), "joint_pos": torch.zeros(3, 9), "joint_vel": torch.zeros(3, 9)}
    data["root"][:, 6] = 1
    validate_dataset(data, JOINT_ORDER, 3)
    with pytest.raises(ValueError, match="Incomplete"):
        validate_dataset(data, JOINT_ORDER, 4)
    data["joint_pos"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="joint_pos"):
        validate_dataset(data, JOINT_ORDER, 3)


def load_mdp_with_fake_base(monkeypatch):
    # Simulate only the Lab lifecycle boundary. The tested methods are the real task code.
    module = ModuleType("isaaclab.envs")

    class Base:
        def _reset_idx(self, ids):
            self.extras = {"log": {"upstream": 1}}
            self.episode_length_buf[ids] = 0

    module.ManagerBasedRLEnv = Base
    monkeypatch.setitem(sys.modules, "isaaclab.envs", module)
    spec = importlib.util.spec_from_file_location("agile.petbot._mdp_test", PROJECT_ROOT / "agile/petbot/mdp.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_episode_success_log_survives_parent_reset(monkeypatch):
    mdp = load_mdp_with_fake_base(monkeypatch)
    env = mdp.PetBotRecoveryEnv()
    env.device = "cpu"
    env.episode_length_buf = torch.tensor([50, 20])
    env.petbot_hold = HoldTracker(2, 0.02, 1.0, "cpu")
    env.petbot_hold.ever[0] = True
    env.petbot_hold.first_time[0] = 1.0
    env.petbot_metric_data = ({}, {"still": torch.tensor([True, False])}, None, torch.tensor([True, False]))
    env._reset_idx(torch.tensor([0]))
    assert env.extras["log"]["Recovery/success_rate"] == 1
    assert env.extras["log"]["Recovery/time_to_success_s"] == 1
    assert env.extras["log"]["upstream"] == 1
    assert env.episode_length_buf.tolist() == [0, 20]
    assert not env.petbot_hold.ever.any()
    with pytest.raises(RuntimeError, match="pre_learn"):
        env.step(torch.zeros(2, 7))
