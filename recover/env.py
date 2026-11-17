"""Vectorized PhysX task, independent policies for four exact recorded starts."""
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg

from .asset_cfg import robot_cfg
from .model import POSITION_JOINTS, REAR_WHEELS, FRONT_WHEELS, prepared_asset
from .poses import load_poses
from .contract import OBS_DIM, ACTION_DIM, POSITION_LOW, POSITION_HIGH, GAMMA, HOLD_S, observation, posture


def tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = TerrainImporterCfg(prim_path='/World/ground', terrain_type='plane', collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.5,
                                                       restitution=0., friction_combine_mode='min'))
    robot = robot_cfg()
    contacts = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/Geometry/.*', history_length=10, update_period=.002)


@configclass
class EnvCfg(DirectRLEnvCfg):
    scene = SceneCfg(num_envs=4096, env_spacing=8., lazy_sensor_update=False)
    sim = sim_utils.SimulationCfg(dt=.002, render_interval=10, physics=PhysxCfg())
    decimation = 10
    episode_length_s = 16.
    action_space = ACTION_DIM
    observation_space = OBS_DIM
    state_space = 0
    seed = 501
    action_mode = 'absolute'
    delta_scale = 0.12
    capture_witnesses = False
    upright_reward = 0.
    # The task has an explicit finite time budget, included in the observation.
    is_finite_horizon = True


class RecoverEnv(DirectRLEnv):
    def __init__(self, cfg):
        if not 0 <= cfg.upright_reward <= .005:
            raise ValueError('upright_reward must stay within the audited small auxiliary range [0, 0.005]')
        _, metadata = prepared_asset()
        if cfg.scene.num_envs * metadata.get('physics_material_count', 14) + 4 >= 64000:
            raise ValueError('Environment count exceeds the audited PhysX material capacity')
        if len(metadata.get('effective_collision_materials', {})) != 10:
            raise ValueError('Prepare and audit all ten collider material bindings before simulation')
        super().__init__(cfg)
        if self.num_envs % 4:
            raise ValueError('num_envs must be divisible by four')
        self.robot = self.scene['robot']; self.sensor = self.scene['contacts']
        self.jp = self.robot.find_joints(list(POSITION_JOINTS), preserve_order=True)[0]
        self.jw = self.robot.find_joints(list(REAR_WHEELS), preserve_order=True)[0]
        self.sw = self.sensor.find_bodies(list(FRONT_WHEELS+REAR_WHEELS), preserve_order=True)[0]
        self.so = [i for i,n in enumerate(self.sensor.body_names) if n not in FRONT_WHEELS+REAR_WHEELS]
        self.low = torch.tensor(POSITION_LOW, device=self.device)
        self.high = torch.tensor(POSITION_HIGH, device=self.device)
        poses, addresses, self.placement = load_poses()
        self.initial_root = torch.zeros(4, 13, device=self.device)
        self.initial_root[:, :7] = torch.tensor(poses[:, :7], device=self.device)
        self.initial_joint = torch.tensor(np.stack([poses[:, addresses[n]] for n in self.robot.joint_names], -1),
                                          device=self.device, dtype=torch.float32)
        self.pose_ids = torch.arange(self.num_envs, device=self.device)//(self.num_envs//4)
        self.applied = torch.zeros(self.num_envs, 7, device=self.device)
        self.position = torch.zeros(self.num_envs, 5, device=self.device)
        self.velocity = torch.zeros(self.num_envs, 2, device=self.device)
        self.hold = torch.zeros(self.num_envs, device=self.device)
        self.previous_potential = torch.zeros_like(self.hold)
        self.episode_return = torch.zeros_like(self.hold)
        self.completed = torch.zeros(4, device=self.device)
        self.succeeded = torch.zeros(4, device=self.device)
        self.best_hold = torch.zeros(4, device=self.device)
        self.best_up = torch.full((4,),-1.,device=self.device)
        self.best_contacts = torch.zeros(4,device=self.device)
        self.last_step = {}; self.record_ids = None; self.record = None
        self.standing_probe = False
        self.auto_reset = True
        if cfg.capture_witnesses:
            # Applied target sequences, not policy averages. Keep two actual
            # successful episodes per pose/window so discoveries can be replayed.
            self.action_history=torch.zeros(self.max_episode_length,self.num_envs,7,device=self.device)
            self.witness_actions=torch.zeros(4,2,self.max_episode_length,7,device=self.device)
            self.witness_lengths=torch.zeros(4,2,dtype=torch.long,device=self.device)
            self.witness_env_ids=torch.full((4,2),-1,dtype=torch.long,device=self.device)
            self.witness_count=torch.zeros(4,dtype=torch.long,device=self.device)
            self.history_ids=torch.arange(self.num_envs,device=self.device)
            self.witness_pose_ids=torch.arange(4,device=self.device)

    def state(self):
        data = self.robot.data
        root = tensor(data.root_state_w).clone()
        # Isaac Lab 3 stores xyzw; snapshots, rewards and MuJoCo use wxyz.
        root[:, 3:7] = root[:, [6, 3, 4, 5]]
        root[:, :3] -= tensor(self.scene.env_origins)
        forces = tensor(self.sensor.data.net_forces_w_history)
        # At least 8/10 physics samples must carry positive force at each wheel.
        contact = (forces[:, :, self.sw, 2] > .05).float().mean(1) >= .8
        other = forces[:, :, self.so].norm(dim=-1).amax((1,2))
        return (root, tensor(data.joint_pos)[:, self.jp], tensor(data.joint_vel)[:, self.jp],
                tensor(data.joint_vel)[:, self.jw], contact, other)

    def _pre_physics_step(self, actions):
        self.requested=actions.clamp(-1,1)
        if self.cfg.action_mode == 'absolute':
            self.applied.copy_(self.requested)
        elif self.cfg.action_mode == 'delta':
            self.applied.add_(self.cfg.delta_scale*self.requested).clamp_(-1,1)
        else:
            raise ValueError(f'Unknown action mode: {self.cfg.action_mode}')
        self.position.copy_(self.low + (self.applied[:, :5]+1)*.5*(self.high-self.low))
        self.velocity.copy_(25*self.applied[:, 5:])
        if self.cfg.capture_witnesses:
            # This is before DirectRLEnv increments episode_length_buf, and
            # before any auto-reset can replace the final applied target.
            step=self.episode_length_buf.clamp_max(self.max_episode_length-1)
            self.action_history[step,self.history_ids]=self.applied

    def _apply_action(self):
        self.robot.set_joint_position_target(self.position, joint_ids=self.jp)
        self.robot.set_joint_velocity_target(self.velocity, joint_ids=self.jw)

    def _get_observations(self):
        root,q,qd,wheel,contacts,_ = self.state()
        remaining = 1-self.episode_length_buf.float()/self.max_episode_length
        return {'policy': observation(root,q,qd,wheel,self.applied,contacts,remaining)}

    def _get_dones(self):
        root,q,qd,wheel,contacts,other = self.state()
        up,four,stable,potential = posture(root,contacts,other,wheel)
        self.hold = torch.where(stable, self.hold+self.step_dt, 0.)
        invalid = (~torch.isfinite(root).all(-1) | ~torch.isfinite(q).all(-1)
                   | ~torch.isfinite(qd).all(-1) | (root[:, 2] < -.1)
                   | (root[:, :2].norm(dim=-1) > 3.))
        success = (self.hold >= HOLD_S) & ~invalid
        if self.cfg.capture_witnesses:
            found=success.reshape(4,-1).any(-1) & (self.witness_count<2)
            local=success.reshape(4,-1).long().argmax(-1)
            chosen=local+self.witness_pose_ids*(self.num_envs//4)
            slot=self.witness_count.clamp_max(1)
            old=self.witness_actions[self.witness_pose_ids,slot]
            actions=self.action_history[:,chosen].permute(1,0,2)
            self.witness_actions[self.witness_pose_ids,slot]=torch.where(found[:,None,None],actions,old)
            lengths=self.witness_lengths[self.witness_pose_ids,slot]
            self.witness_lengths[self.witness_pose_ids,slot]=torch.where(found,self.episode_length_buf[chosen],lengths)
            old_ids=self.witness_env_ids[self.witness_pose_ids,slot]
            self.witness_env_ids[self.witness_pose_ids,slot]=torch.where(found,chosen,old_ids)
            self.witness_count+=found.long()
        timeout = self.episode_length_buf >= self.max_episode_length
        terminated = success | invalid
        self.last_step = dict(root=root, q=q, qd=qd, wheel=wheel, contacts=contacts, other=other,
                              up=up, four=four, stable=stable, potential=potential,
                              success=success, invalid=invalid, timeout=timeout, hold=self.hold.clone())
        self.best_hold = torch.maximum(self.best_hold,torch.where(invalid,0.,self.hold).reshape(4,-1).amax(-1))
        self.best_up = torch.maximum(self.best_up,torch.where(invalid,-1.,up).reshape(4,-1).amax(-1))
        self.best_contacts = torch.maximum(self.best_contacts,torch.where(invalid,0,contacts.sum(-1)).reshape(4,-1).amax(-1))
        if self.record_ids is not None:
            ids = self.record_ids
            self.record = {k:v[ids].detach().cpu().numpy() for k,v in self.last_step.items()}
            self.record.update(action=self.applied[ids].detach().cpu().numpy(),
                               policy_action=self.requested[ids].detach().cpu().numpy(),
                               joint_all=tensor(self.robot.data.joint_pos)[ids].detach().cpu().numpy(),
                               effort=tensor(self.robot.data.applied_torque)[ids].detach().cpu().numpy(),
                               force=tensor(self.sensor.data.net_forces_w_history)[ids].detach().cpu().numpy())
        if not self.auto_reset:
            return torch.zeros_like(terminated), torch.zeros_like(timeout)
        return terminated, timeout

    def _get_rewards(self):
        s=self.last_step
        done=s['success']|s['invalid']|s['timeout']
        next_potential=torch.where(done, 0., s['potential'])
        # Potential shaping does not prescribe a support sequence or penalize a
        # temporary roll away from upright. Four-wheel/stability terms are small.
        # The optional upright term adds an explicit, bounded search preference;
        # potential-only A/B runs remain the comparison for that bias.
        reward = (2*(GAMMA*next_potential-self.previous_potential)-.01
                  + .02*s['four'] + .08*s['stable'] + 40*s['success'] - 5*s['invalid']
                  + self.cfg.upright_reward*(s['up']+1).clamp(0,2))
        reward = torch.nan_to_num(reward, nan=-5., posinf=-5., neginf=-5.)
        self.previous_potential.copy_(s['potential'])
        self.episode_return += reward
        return reward

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if not hasattr(self, 'initial_root'):
            return
        env_ids=env_ids.long()
        if len(env_ids)==0: return
        if self.last_step:
            self.completed.index_add_(0,self.pose_ids[env_ids],torch.ones(len(env_ids),device=self.device))
            self.succeeded.index_add_(0,self.pose_ids[env_ids],self.last_step['success'][env_ids].float())
        pose=self.pose_ids[env_ids]
        if self.standing_probe:
            root=tensor(self.robot.data.default_root_state)[env_ids].clone()
            root[:,3:7]=root[:,[6,3,4,5]]
            q=tensor(self.robot.data.default_joint_pos)[env_ids].clone()
        else:
            root=self.initial_root[pose].clone(); q=self.initial_joint[pose].clone()
        root[:, :3] += tensor(self.scene.env_origins)[env_ids]
        root[:,7:]=0
        simulator_root=root.clone()
        simulator_root[:,3:7]=root[:,[4,5,6,3]]
        self.robot.write_root_state_to_sim(simulator_root,env_ids=env_ids)
        self.robot.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=env_ids)
        self.position[env_ids]=q[:,self.jp]; self.velocity[env_ids]=0
        self.applied[env_ids,:5]=2*(self.position[env_ids]-self.low)/(self.high-self.low)-1
        self.applied[env_ids,5:]=0
        self.hold[env_ids]=0; self.episode_return[env_ids]=0
        up=1-2*(root[:,4].square()+root[:,5].square())
        self.previous_potential[env_ids]=up+1

    def summary(self):
        return dict(episodes=self.completed.detach().cpu().tolist(), successes=self.succeeded.detach().cpu().tolist(),
                    max_stable_seconds=self.best_hold.detach().cpu().tolist(),
                    max_uprightness=self.best_up.detach().cpu().tolist(),
                    max_wheels_in_contact=self.best_contacts.detach().cpu().tolist(),
                    success_rate=(self.succeeded/self.completed.clamp_min(1)).detach().cpu().tolist())
