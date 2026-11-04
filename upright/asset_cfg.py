"""Isaac Lab 3 implicit drives. Source geometry and inertial properties stay read-only."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .model import FRONT_WHEELS, POSITION_JOINTS, POSITION_KD, POSITION_KP, REAR_WHEELS, load_model, prepared_asset


def robot_cfg():
    spec = load_model()
    usd, _ = prepared_asset()
    joints = spec["joints"]
    drives = spec["actuators"]

    def passive(names):
        return {
            "armature": {n: joints[n]["armature"] for n in names},
            "friction": {n: joints[n]["friction"] for n in names},
            "dynamic_friction": {n: joints[n]["friction"] for n in names},
            "viscous_friction": {n: joints[n]["damping"] for n in names},
        }

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd),
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(spec["root_position"]),
            joint_pos={n: item["default"] for n, item in joints.items()},
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "position": ImplicitActuatorCfg(
                joint_names_expr=list(POSITION_JOINTS),
                stiffness=POSITION_KP,
                damping=POSITION_KD,
                effort_limit_sim={n: drives[n]["effort"] for n in POSITION_JOINTS},
                **passive(POSITION_JOINTS),
            ),
            "rear_velocity": ImplicitActuatorCfg(
                joint_names_expr=list(REAR_WHEELS),
                stiffness=0.0,
                damping={n: drives[n]["kv"] for n in REAR_WHEELS},
                effort_limit_sim={n: drives[n]["effort"] for n in REAR_WHEELS},
                **passive(REAR_WHEELS),
            ),
            "passive_front": ImplicitActuatorCfg(
                joint_names_expr=list(FRONT_WHEELS),
                stiffness=0.0,
                damping=0.0,
                effort_limit_sim=0.0,
                **passive(FRONT_WHEELS),
            ),
        },
    )
