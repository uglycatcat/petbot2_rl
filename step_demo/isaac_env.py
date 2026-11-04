"""Isaac adapter: read-only robot physics, sensor actor, explicit STEP task.

Task history/reward/termination live in task.py. No legacy objective flags,
auxiliary resets, phase-dependent control override, or online terrain updates.
Import only after AppLauncher starts.
"""
from pathlib import Path
import json
import torch
import isaaclab.sim as sim
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, Imu, ImuCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg
from .config import StepConfig
from .estimator import InertialEstimator
from .tensor_math import rotate, heading
from .terrain import convex_edges, wheel_terrain_clearance
from .task import StepTask, PhysicsSample, Phase, Failure
from .kinematics import body_pitch


def tensor(value):
    return value.torch if hasattr(value, 'torch') else value


@configclass
class StepEnvCfg(DirectRLEnvCfg):
    decimation = 10
    episode_length_s = 15.
    action_space = 2
    observation_space = 27
    state_space = 1  # Filled from the explicit task/robot state schema.
    is_finite_horizon = True
    sim = sim.SimulationCfg(dt=.002, render_interval=10,
                           physics=PhysxCfg(enable_external_forces_every_iteration=True))
    scene = InteractiveSceneCfg(num_envs=256, env_spacing=3., replicate_physics=True,
                                lazy_sensor_update=False)


class StepEnv(DirectRLEnv):
    def __init__(self, task: StepConfig, training: dict, asset_dir: Path, device='cuda:0'):
        self.task, self.training = task, training
        self.asset_dir = asset_dir.resolve()
        self.asset = json.loads((asset_dir/'asset.json').read_text())
        schema = StepTask(1, 'cpu', task.acceptance, task.terrain, torch.zeros(1), torch.zeros(1))
        self.memory_dim = schema.critic_memory().shape[-1]
        self.critic_dim = self.memory_dim + 79 + 2*len(self.asset['joints'])
        cfg = StepEnvCfg()
        cfg.state_space = self.critic_dim
        cfg.seed = training['seed']; cfg.scene.num_envs = training['num_envs']
        cfg.decimation = training['decimation']; cfg.sim.render_interval = cfg.decimation
        cfg.sim.device = device; cfg.episode_length_s = task.acceptance.timeout_s
        super().__init__(cfg)
        self.hips = self.robot.find_joints(['left_frame_1','right_frame_1'], preserve_order=True)[0]
        self.wheels = self.robot.find_joints(['left_rear_wheel_1','right_rear_wheel_1'], preserve_order=True)[0]
        self.front_joints = self.robot.find_joints(['left_front_wheel_1','right_front_wheel_1'], preserve_order=True)[0]
        self.wheel_names = ['left_front_wheel','right_front_wheel','left_rear_wheel','right_rear_wheel']
        self.wheel_bodies = self.robot.find_bodies(self.wheel_names, preserve_order=True)[0]
        self.lower_body = self.robot.find_bodies('lower_body')[0][0]
        self.joint_names = [n.removesuffix('_1') for n in self.robot.joint_names]
        self.estimator = InertialEstimator(self.num_envs, self.device, self.asset['source_bodies'], self.joint_names)
        self.vertices = [torch.tensor(self.asset['wheel_vertices'][n], device=self.device) for n in self.wheel_names]
        self.edges = [torch.tensor(convex_edges(self.asset['wheel_vertices'][n]), dtype=torch.long,
                                  device=self.device) for n in self.wheel_names]
        n, d = self.num_envs, self.device
        self.origin_com = self.scene.env_origins + torch.tensor(self.asset['initial_com'], device=d)
        self.edge_x = self.origin_com[:, 0]+task.terrain.initial_distance_m
        self.objective = StepTask(n, d, task.acceptance, task.terrain, self.edge_x,
                                  self.scene.env_origins[:, 1], self.physics_dt)
        self.actions = torch.zeros(n, 2, device=d); self.actions[:, 1] = -1.
        self.raw_actions = self.actions.clone(); self.executed = self.actions.clone()
        self.hip_target = torch.zeros(n, device=d); self.wheel_target = torch.zeros(n, device=d)
        self.packet = torch.tensor([task.terrain.height_m, task.terrain.initial_distance_m,
                                   task.terrain.edge_angle_rad], device=d).repeat(n, 1)
        self.packet_count = torch.zeros(n, dtype=torch.long, device=d)
        self.minimum = torch.zeros(n, 4, 3, device=d); self.maximum = self.minimum.clone()
        self.clearance = torch.zeros(n, 4, device=d)
        self.com = self.origin_com.clone(); self.com_vel = torch.zeros(n, 3, device=d)
        self.yaw = torch.zeros(n, device=d); self.world_pitch = self.yaw.clone(); self.world_pitch_rate = self.yaw.clone()
        self.wheel_contact_force = self.minimum.clone(); self.platform_normal = self.minimum.clone()
        self.contact_point = torch.full_like(self.minimum, float('nan'))
        self.pending_reward = torch.zeros(n, device=d); self.episode_return = self.pending_reward.clone()
        self.wheel_speed_peak = self.pending_reward.clone()
        self.trace_ids = None; self.physics_trace_ids = None
        self.trace_writer = None; self.physics_trace_writer = None; self.episode_writer = None
        self._capture_counter = self._sim_step_counter; self.captured_substeps = 0
        self.episode_serial = torch.zeros(n, dtype=torch.long, device=d)
        self.totals = dict(episodes=0, successes=0, front_collisions=0, rear_air=0, falls=0,
                           quality_failures=0, platform_exits=0, timeouts=0)
        self.learning_coverage = {name:0 for name in ('front_landing_entries','four_top_entries','drive_entries','stop_entries')}
        self.learning_coverage.update(stop_seconds=0., eligible_stop_seconds=0.)
        self._coverage_delta = torch.zeros(6, device=d)
        self.terminal_records = []; self.attempt_id = 'probe'
        self.sensor_scale = torch.tensor([.02,.3,1.,1.,.3,1.,1.,.02,1.,1.,1.,1.,
                                         1.,1.,1.,5.,5.,5.,1.,1.,10.,10.,25.,25.,1.,1.], device=d)

    def _setup_scene(self):
        t = self.training
        # Explicit controllers obey source motor effort limits. Imported joint friction/armature retained.
        actuator_cfg = {
            'hips': IdealPDActuatorCfg(joint_names_expr=['.*_frame_1'], stiffness=t['hip_kp'],
                damping=t['hip_kd'], effort_limit=1.2, effort_limit_sim=1.2, viscous_friction=.072,
                friction=.04, dynamic_friction=.04, armature=.000096),
            'head': IdealPDActuatorCfg(joint_names_expr=['head_1'], stiffness=t['head_kp'],
                damping=t['head_kd'], effort_limit=.72, effort_limit_sim=.72, viscous_friction=.048,
                friction=.025, dynamic_friction=.025, armature=.000096),
            'ears': IdealPDActuatorCfg(joint_names_expr=['.*_ear_1'], stiffness=t['ear_kp'],
                damping=t['ear_kd'], effort_limit=.10, effort_limit_sim=.10, viscous_friction=.010,
                friction=.008, dynamic_friction=.008, armature=.000096),
            'rear': ImplicitActuatorCfg(joint_names_expr=['.*_rear_wheel_1'], stiffness=0.,
                damping=.14, effort_limit=.38, effort_limit_sim=.38, viscous_friction=.0005,
                friction=.012, dynamic_friction=.012, armature=.000024),
            'front': ImplicitActuatorCfg(joint_names_expr=['.*_front_wheel_1'], stiffness=0.,
                damping=0., effort_limit_sim=0., viscous_friction=.0005,
                friction=.004, dynamic_friction=.004, armature=.000024),
        }
        robot_cfg = ArticulationCfg(prim_path='/World/envs/env_.*/Robot',
            spawn=sim.UsdFileCfg(usd_path=str(self.asset_dir / self.asset['usd_file']),
                activate_contact_sensors=True,
                rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=2.),
                articulation_props=sim.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                    solver_position_iteration_count=8, solver_velocity_iteration_count=2)),
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(self.asset['initial_root_pos']), joint_pos={'.*':0.}),
            actuators=actuator_cfg)
        self.robot = Articulation(robot_cfg)
        ground = sim.GroundPlaneCfg(physics_material=sim.RigidBodyMaterialCfg(
            static_friction=1.5, dynamic_friction=1.5, restitution=0., friction_combine_mode='min'))
        ground.func('/World/Ground', ground)
        terrain = self.task.terrain
        platform_cfg = RigidObjectCfg(prim_path='/World/envs/env_.*/Platform',
            spawn=sim.CuboidCfg(size=(terrain.platform_length_m, terrain.platform_width_m, terrain.height_m),
                rigid_props=sim.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=sim.CollisionPropertiesCfg(contact_offset=.0005, rest_offset=0.),
                mass_props=sim.MassPropertiesCfg(mass=1.),
                physics_material=sim.RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.5,
                    friction_combine_mode='min'), activate_contact_sensors=True),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(self.asset['initial_com'][0] + .30 + terrain.platform_length_m/2,
                                                         0., terrain.height_m/2)))
        self.platform = RigidObject(platform_cfg)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=['/World/Ground'])
        self.scene.articulations['robot'] = self.robot
        self.scene.rigid_objects['platform'] = self.platform
        base = '/World/envs/env_.*/Robot/Geometry/base_link/lower_body'
        self.imu = Imu(ImuCfg(prim_path=base, update_period=0.))
        self.scene.sensors['imu'] = self.imu
        self.contacts = []
        for side in ('left','right'):
            for kind in ('front','rear'):
                name = f'{side}_{kind}_wheel'
                sensor = ContactSensor(ContactSensorCfg(
                    prim_path=f'{base}/{side}_frame/{name}', update_period=0.,
                    filter_prim_paths_expr=['/World/envs/env_.*/Platform'], track_contact_points=True,
                    max_contact_data_count_per_prim=32))
                self.scene.sensors[name] = sensor
                self.contacts.append((name, sensor))
        self.body_contact = ContactSensor(ContactSensorCfg(prim_path=base+'/(head|left_frame|right_frame)', update_period=0.))
        self.scene.sensors['body_contact'] = self.body_contact
        self.lower_contact=ContactSensor(ContactSensorCfg(prim_path=base,update_period=0.))
        self.scene.sensors['lower_contact']=self.lower_contact
        light = sim.DomeLightCfg(intensity=1500.)
        light.func('/World/Light', light)

    def _pre_physics_step(self, actions):
        if actions.shape != self.actions.shape or not torch.isfinite(actions).all():
            raise ValueError('Expected finite [num_envs,2] raw policy actions')
        self.raw_actions = actions.clone()
        self.actions = actions.clamp(-1., 1.)
        self.pending_reward.zero_()
        self._coverage_delta.zero_()

    def _apply_action(self):
        self._capture(self._sim_step_counter - 1)
        desired_hip = torch.where(self.actions[:, 0] >= 0, 1.2*self.actions[:, 0], self.actions[:, 0])
        desired_wheel = (self.actions[:, 1]+1.) * 12.5
        self.hip_target += (desired_hip-self.hip_target).clamp(
            -self.training['hip_target_rate_rad_s']*self.physics_dt, self.training['hip_target_rate_rad_s']*self.physics_dt)
        self.wheel_target += (desired_wheel-self.wheel_target).clamp(
            -self.training['wheel_target_accel_rad_s2']*self.physics_dt,
            self.training['wheel_target_accel_rad_s2']*self.physics_dt)
        positions = torch.zeros_like(tensor(self.robot.data.joint_pos))
        velocities = torch.zeros_like(positions)
        positions[:, self.hips] = self.hip_target[:, None]
        velocities[:, self.wheels] = self.wheel_target[:, None]
        self.robot.set_joint_position_target_index(target=positions)
        self.robot.set_joint_velocity_target_index(target=velocities)
        self.executed[:, 0] = torch.where(self.hip_target >= 0, self.hip_target/1.2, self.hip_target)
        self.executed[:, 1] = self.wheel_target/12.5 - 1

    def _capture(self, counter):
        if counter <= self._capture_counter:
            return
        self._capture_counter = counter; self.captured_substeps += 1
        pos, vel = tensor(self.robot.data.joint_pos), tensor(self.robot.data.joint_vel)
        self.estimator.update(tensor(self.imu.data.lin_acc_b), tensor(self.imu.data.ang_vel_b),
                              pos, self.physics_dt, vel)
        mass = tensor(self.robot.data.body_mass)
        self.com = (tensor(self.robot.data.body_com_pos_w)*mass[..., None]).sum(1)/mass.sum(1)[:, None]
        self.com_vel = (tensor(self.robot.data.body_com_lin_vel_w)*mass[..., None]).sum(1)/mass.sum(1)[:, None]
        body_q, body_p = tensor(self.robot.data.body_link_quat_w), tensor(self.robot.data.body_link_pos_w)
        self.yaw = heading(body_q[:, self.lower_body])
        for i, body in enumerate(self.wheel_bodies):
            basis = torch.eye(3, device=self.device)[None].expand(self.num_envs,-1,-1)
            axes = rotate(body_q[:,body,None,:].expand(-1,3,-1), basis)
            points = torch.matmul(self.vertices[i], axes)+body_p[:,body,None,:]
            self.minimum[:,i] = points.amin(1)-.0001; self.maximum[:,i] = points.amax(1)+.0001
            self.clearance[:,i] = wheel_terrain_clearance(points, self.edges[i], self.edge_x,
                self.scene.env_origins[:,1], self.task.terrain.height_m,
                self.task.terrain.platform_length_m, self.task.terrain.platform_width_m/2)
        for name, sensor in self.contacts:
            i = self.wheel_names.index(name)
            self.wheel_contact_force[:,i] = tensor(sensor.data.net_forces_w)[:,0]
            self.platform_normal[:,i] = tensor(sensor.data.force_matrix_w)[:,0].sum(1)
            self.contact_point[:,i] = tensor(sensor.data.contact_pos_w)[:,0].nanmean(1)
        angular = tensor(self.robot.data.body_link_ang_vel_w)[:,self.lower_body]
        self.world_pitch, self.world_pitch_rate = body_pitch(body_q[:,self.lower_body], angular)
        up = rotate(body_q[:,self.lower_body], torch.tensor([0.,0.,1.],device=self.device).expand(self.num_envs,-1))[:,2]
        body_hit = tensor(self.body_contact.data.net_forces_w).norm(dim=-1).amax(-1) > .5
        body_hit |= tensor(self.lower_contact.data.net_forces_w).norm(dim=-1).amax(-1) > .5
        sample = PhysicsSample(self.com[:,0]-self.origin_com[:,0], self.com[:,1]-self.origin_com[:,1],
            self.yaw, pos[:,self.hips], vel[:,self.hips], vel[:,self.wheels], self.com_vel.norm(dim=-1),
            angular[:,2], self.world_pitch, self.world_pitch_rate, self.minimum, self.maximum,
            self.clearance, self.wheel_contact_force, self.platform_normal, self.contact_point, (up < .4)|body_hit)
        before = self.objective.phase.clone()
        reward = self.objective.advance(sample)
        self.pending_reward += reward; self.episode_return += reward
        self.wheel_speed_peak = torch.maximum(self.wheel_speed_peak, vel[:,self.wheels].abs().amax(-1))
        entered = [((before < p) & (self.objective.phase >= p) & (self.objective.failure == 0)).sum()
                   for p in (Phase.REAR_ASCENT, Phase.ALIGN, Phase.DRIVE, Phase.STOP)]
        stopping = self.objective.phase == Phase.STOP
        self._coverage_delta += torch.stack([*entered, stopping.sum()*self.physics_dt,
                                             (stopping & self.objective.ready).sum()*self.physics_dt])
        if self.physics_trace_writer is not None:
            ids = self.physics_trace_ids if self.physics_trace_ids is not None else self.trace_ids
            self.physics_trace_writer.write(json.dumps(self.trace(ids))+'\n')

    def _get_dones(self):
        self._capture(self._sim_step_counter)
        timeout = self.objective.failure == Failure.TIMEOUT
        return self.objective.terminal & ~timeout, timeout

    def _get_rewards(self):
        for key, value in zip(self.learning_coverage, self._coverage_delta.tolist()):
            self.learning_coverage[key] += value
        if self.trace_ids is not None:
            self.last_trace = self.trace(self.trace_ids)
            if self.trace_writer:
                self.trace_writer.write(json.dumps(self.last_trace)+'\n')
        return self.pending_reward.clone()

    def _get_observations(self):
        sensor = self.estimator.observation(self.packet, tensor(self.robot.data.joint_pos),
            tensor(self.robot.data.joint_vel), self.hips, self.wheels, tensor(self.imu.data.ang_vel_b), self.executed)
        # Time is available from the local controller clock, not simulator terrain truth.
        clock = (self.episode_length_buf*self.step_dt/self.task.acceptance.timeout_s)[:,None]
        policy = torch.cat((sensor/self.sensor_scale, clock), -1)
        critic = self._critic_state()
        if policy.shape[-1] != 27 or critic.shape[-1] != self.critic_dim:
            raise RuntimeError('Observation schema mismatch')
        if not torch.isfinite(policy).all() or not torch.isfinite(critic).all():
            raise RuntimeError('Nonfinite observation')
        return {'policy':policy, 'critic':critic}

    def _critic_state(self):
        # Full robot state plus every persistent task variable; never routed to actor.
        origin = self.origin_com[:,None,:]
        pieces = [self.objective.critic_memory(), self.com-self.origin_com, self.com_vel,
            tensor(self.robot.data.body_link_quat_w)[:,self.lower_body],
            tensor(self.robot.data.body_link_ang_vel_w)[:,self.lower_body]/5.,
            tensor(self.robot.data.joint_pos), tensor(self.robot.data.joint_vel)/25.,
            (self.minimum-origin).flatten(1), (self.maximum-origin).flatten(1), self.clearance,
            self.wheel_contact_force.flatten(1)/20., self.platform_normal.flatten(1)/20.,
            (self.contact_point-origin).nan_to_num().flatten(1), self.executed]
        return torch.cat(pieces, -1)

    def _reset_idx(self, env_ids):
        if env_ids is None: env_ids = torch.arange(self.num_envs,device=self.device)
        env_ids = torch.as_tensor(env_ids,device=self.device,dtype=torch.long)
        if hasattr(self, 'objective'):
            for i in env_ids[self.episode_length_buf[env_ids] > 0].tolist():
                record = dict(attempt=self.attempt_id, env=i, episode=int(self.episode_serial[i]),
                    success=bool(self.objective.completed[i]), failure=int(self.objective.failure[i]),
                    timeout=int(self.objective.failure[i]) == Failure.TIMEOUT, phase=int(self.objective.phase[i]),
                    front_flight_count=self.objective.flight_count[i].tolist(),
                    front_landed=self.objective.landed_top[i].tolist(), drive_distance=float(self.objective.drive_distance[i]),
                    hold_s=float(self.objective.hold_s[i]), elapsed_s=float(self.objective.elapsed_s[i]),
                    return_sum=float(self.episode_return[i]), quality_cost=float(self.objective.quality_cost[i]),
                    actual_wheel_peak=float(self.wheel_speed_peak[i]), estimator_error_m=float(
                        (self.estimator.com_position[i]-(self.com[i]-self.origin_com[i])).norm()))
                self.terminal_records.append(record)
                if self.episode_writer: self.episode_writer.write(json.dumps(record)+'\n')
                self.totals['episodes'] += 1; self.totals['successes'] += int(record['success'])
                for key, code in [('front_collisions',1),('rear_air',2),('falls',3),('quality_failures',5),('platform_exits',6),('timeouts',7)]:
                    self.totals[key] += int(record['failure'] == code)
            self.terminal_records = self.terminal_records[-4096:]
            self.extras['log'] = {f'Step/{k}':float(v) for k,v in self.totals.items()}
        super()._reset_idx(env_ids)
        pose = tensor(self.robot.data.default_root_pose)[env_ids].clone()
        pose[:,:3] += self.scene.env_origins[env_ids]
        joints = tensor(self.robot.data.default_joint_pos)[env_ids].clone()
        self.robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(env_ids),6,device=self.device),env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joints,env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joints),env_ids=env_ids)
        if hasattr(self, 'objective'):
            self.objective.reset(env_ids); self.estimator.reset(env_ids)
            self.actions[env_ids] = torch.tensor([0.,-1.],device=self.device)
            self.raw_actions[env_ids] = self.actions[env_ids]; self.executed[env_ids] = self.actions[env_ids]
            for name in ('hip_target','wheel_target','pending_reward','episode_return','wheel_speed_peak','minimum',
                         'maximum','clearance','com_vel','wheel_contact_force','platform_normal','yaw','world_pitch','world_pitch_rate'):
                getattr(self,name)[env_ids] = 0.
            self.com[env_ids] = self.origin_com[env_ids]; self.contact_point[env_ids] = float('nan')
            self.packet_count[env_ids] = 1; self.episode_serial[env_ids] += 1

    def trace(self, ids):
        # One GPU-to-CPU transfer for a whole sample; per-field .tolist() at
        # 500 Hz would serialize dozens of transfers and stall the simulator.
        fields = {f'task.{name}':getattr(self.objective,name) for name in self.objective._memory_names}
        fields.update(episode=self.episode_serial, completed=self.objective.completed,
            elapsed=self.objective.elapsed_s, stop_time=self.objective.hold_s,
            action=self.raw_actions, requested_action=self.actions, executed=self.executed,
            hip_target=self.hip_target, wheel_target=self.wheel_target,
            joint_position=tensor(self.robot.data.joint_pos), joint_velocity=tensor(self.robot.data.joint_vel),
            controller_effort_estimate=tensor(self.robot.data.applied_torque),
            com=self.com-self.origin_com, velocity=self.com_vel, yaw=self.yaw,
            yaw_rate=tensor(self.robot.data.body_link_ang_vel_w)[:,self.lower_body,2],
            world_pitch=self.world_pitch, world_pitch_rate=self.world_pitch_rate,
            estimated_com=self.estimator.com_position, estimated_velocity=self.estimator.com_velocity,
            imu_acceleration=tensor(self.imu.data.lin_acc_b), imu_gyro=tensor(self.imu.data.ang_vel_b),
            estimated_quaternion=self.estimator.q, true_quaternion=tensor(self.robot.data.body_link_quat_w)[:,self.lower_body],
            wheel_min=self.minimum, wheel_max=self.maximum, wheel_clearance=self.clearance,
            platform_force=self.platform_normal, contact_point=self.contact_point,
            wheel_contact_force=self.wheel_contact_force,
            body_contact_force=tensor(self.body_contact.data.net_forces_w),
            lower_contact_force=tensor(self.lower_contact.data.net_forces_w),
            reward=self.pending_reward, return_sum=self.episode_return, edge_x=self.edge_x, origin_com=self.origin_com)
        packed = torch.cat([v.reshape(self.num_envs,-1).float() for v in fields.values()],-1)[ids].cpu()
        result = dict(schema='step_trace_v2',physics_step=self._capture_counter,attempt=self.attempt_id,task={})
        offset = 0
        for name, value in fields.items():
            width = value[0].numel()
            values = packed[:,offset:offset+width].to(value.dtype).reshape(len(ids),*value.shape[1:]).tolist()
            if name.startswith('task.'): result['task'][name[5:]] = values
            else: result[name] = values
            offset += width
        return result
