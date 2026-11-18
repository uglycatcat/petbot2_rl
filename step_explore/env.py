"""Physics and task for unrestricted platform ascent. All lengths are metres."""
from pathlib import Path
import json
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse

PROJECT = Path(__file__).resolve().parents[1]
WHEELS = ['left_front_wheel', 'right_front_wheel', 'left_rear_wheel', 'right_rear_wheel']
ANGLE = ['head', 'left_frame', 'right_frame']
REAR = ['left_rear_wheel', 'right_rear_wheel']

@configclass
class ExploreCfg(DirectRLEnvCfg):
    decimation = 10
    episode_length_s = 10.0
    is_finite_horizon = True
    action_space = 5
    observation_space = 40
    state_space = 0
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=0.002, render_interval=10,
        physx=sim_utils.PhysxCfg(gpu_max_rigid_contact_count=2**20, gpu_max_rigid_patch_count=2**18,
                          gpu_found_lost_pairs_capacity=2**21))
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=4.0, replicate_physics=True)
    asset_dir: str = str(PROJECT / 'assets')
    edge: float = 0.65
    height: float = 0.07
    platform_length: float = 2.0
    platform_width: float = 2.0
    exploration_bonus: float = 10.0

class ExploreEnv(DirectRLEnv):
    def __init__(self, cfg, **kwargs):
        self.meta = json.loads((Path(cfg.asset_dir) / 'model.json').read_text())
        super().__init__(cfg, **kwargs)
        self.angle_ids, _ = self.robot.find_joints(ANGLE, preserve_order=True)
        self.rear_ids, _ = self.robot.find_joints(REAR, preserve_order=True)
        self.wheel_ids, _ = self.robot.find_bodies(WHEELS, preserve_order=True)
        self.contact_ids, _ = self.contact.find_bodies(WHEELS, preserve_order=True)
        self.head_id = self.robot.find_bodies('head')[0][0]
        self.dtype = torch.float32
        self.actions = torch.zeros(self.num_envs, 5, device=self.device)
        self.targets = self.actions.clone()
        self.servo_torque = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        self.damping = torch.tensor([self.meta['joints'][n]['damping'] for n in self.robot.joint_names], device=self.device)
        self.friction = torch.tensor([self.meta['joints'][n]['frictionloss'] for n in self.robot.joint_names], device=self.device)
        self.kp = torch.tensor([4., 6., 6.], device=self.device)
        self.kd = torch.tensor([.08, .12, .12], device=self.device)
        self.torque_limit = torch.tensor([.72, 1.2, 1.2], device=self.device)
        # Pad hull vertices only by repeating an existing vertex (does not alter support geometry).
        verts = [torch.tensor(self.meta['wheel_vertices'][n], device=self.device) for n in WHEELS]
        k = max(len(v) for v in verts)
        self.wheel_vertices = torch.stack([torch.cat((v, v[:1].repeat(k-len(v),1))) for v in verts])
        self.best_progress = torch.zeros(self.num_envs, device=self.device)
        self.seen_top = torch.zeros(self.num_envs,4,dtype=torch.bool,device=self.device)
        self.peak_top = torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.success = torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.outside = self.success.clone()
        self.total_episodes = 0
        self.total_success = 0
        self.max_top = 0
        self.max_distinct_top = 0
        self.last_transition = {}
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.init_phi = None
        self._geometry()
        self.init_phi = self._progress().clone()
        self.best_progress.copy_(self.init_phi)

    def _setup_scene(self):
        arm = {n:v['armature'] for n,v in self.meta['joints'].items()}
        robot_cfg = ArticulationCfg(prim_path='/World/envs/env_.*/Robot',
            spawn=sim_utils.UsdFileCfg(usd_path=str(Path(self.cfg.asset_dir)/'robot.usda'),
                activate_contact_sensors=True,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                    solver_position_iteration_count=8, solver_velocity_iteration_count=2)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.,0.,self.meta['default_root_height']),
                                                       joint_pos={'.*':0.},joint_vel={'.*':0.}),
            actuators={
                'angles_and_front':ImplicitActuatorCfg(joint_names_expr=ANGLE+['left_front_wheel','right_front_wheel'],
                    stiffness=0.,damping=0.,effort_limit_sim=4.,
                    armature={n:a for n,a in arm.items() if n not in REAR},friction=0.),
                'rear_velocity':ImplicitActuatorCfg(joint_names_expr=REAR,stiffness=0.,damping=.14,
                    effort_limit_sim=.38,armature={n:arm[n] for n in REAR},friction=0.)})
        self.robot = Articulation(robot_cfg)
        ground = sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.5,dynamic_friction=1.5,friction_combine_mode='min',restitution=0.))
        sim_utils.spawn_ground_plane('/World/ground', ground)
        box = sim_utils.CuboidCfg(size=(self.cfg.platform_length,self.cfg.platform_width,self.cfg.height),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0.),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.5,dynamic_friction=1.5,
                friction_combine_mode='min',restitution=0.),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.25,.4,.55)))
        box.func('/World/envs/env_0/Platform',box,
            translation=(self.cfg.edge+self.cfg.platform_length/2,0.,self.cfg.height/2))
        self.contact = ContactSensor(ContactSensorCfg(prim_path='/World/envs/env_.*/Robot/.*wheel',
            update_period=0.,history_length=0,track_air_time=False))
        from pxr import UsdPhysics
        for name in self.meta['bodies']:
            if 'wheel' not in name:
                prim = self.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/'+name+'/self_collision')
                if not prim.IsValid():
                    continue
                relation = UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
                relation.AddTarget('/World/ground')
                relation.AddTarget('/World/envs/env_0/Platform')
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=['/World/ground'])
        self.scene.articulations['robot'] = self.robot
        self.scene.sensors['contact'] = self.contact
        light = sim_utils.DomeLightCfg(intensity=1500.,color=(.8,.8,.8))
        light.func('/World/Light',light)

    def _pre_physics_step(self, actions):
        self.extras.pop('log',None)
        self.actions = actions.clone()
        x = torch.tanh(self.actions)
        self.targets[:,0] = x[:,0]*1.3
        self.targets[:,1:3] = torch.where(x[:,1:3]>=0,x[:,1:3]*1.2,x[:,1:3])
        self.targets[:,3:] = x[:,3:]*25.

    def _apply_action(self):
        q, qd = self.robot.data.joint_pos, self.robot.data.joint_vel
        # Position and velocity setpoint servo, evaluated at each 2ms physics step.
        tau = -qd*self.damping - self.friction*torch.tanh(qd/.01)
        motor = (self.kp*(self.targets[:,:3]-q[:,self.angle_ids])-self.kd*qd[:,self.angle_ids])
        motor = torch.clamp(motor, -self.torque_limit, self.torque_limit)
        tau[:,self.angle_ids] += motor
        # Rear velocity drive is implicit: explicit damping is unstable at the source wheel inertia.
        self.robot.set_joint_velocity_target(self.targets[:,3:],joint_ids=self.rear_ids)
        self.servo_torque = tau.clone()
        self.servo_torque[:,self.rear_ids] += torch.clamp(.14*(self.targets[:,3:]-qd[:,self.rear_ids]),-.38,.38)
        self.robot.set_joint_effort_target(tau)

    def _geometry(self):
        self.wheel_pos = self.robot.data.body_pos_w[:,self.wheel_ids] - self.scene.env_origins[:,None,:]
        q = self.robot.data.body_quat_w[:,self.wheel_ids]
        # min(R*v).z can be computed from R^-1*world_z without materializing all transformed vertices.
        z = torch.zeros_like(self.wheel_pos); z[:,:,2] = 1.
        local_z = quat_apply_inverse(q,z)
        height = torch.einsum('nwj,wkj->nwk',local_z,self.wheel_vertices).min(-1).values
        self.bottom = self.wheel_pos[:,:,2]+height
        self.forces = self.contact.data.net_forces_w[:,self.contact_ids]
        inside = (self.wheel_pos[:,:,0] >= self.cfg.edge+.003) & (self.wheel_pos[:,:,0] <= self.cfg.edge+self.cfg.platform_length-.003)
        inside &= self.wheel_pos[:,:,1].abs() <= self.cfg.platform_width/2-.003
        self.top = inside & ((self.bottom-self.cfg.height).abs()<=.004) & (self.forces[:,:,2]>.03)
        self.success = self.top.all(-1)

    def _progress(self):
        # Signed wheel-distance progress plus clearance near the step; no upright or symmetric-motion objective.
        horizontal = (self.wheel_pos[:,:,0]/(self.cfg.edge+.12)).clamp(-1.,1.)
        near = torch.exp(-((self.wheel_pos[:,:,0]-self.cfg.edge)/.18).square())
        clearance = (self.bottom/self.cfg.height).clamp(0.,1.)*near
        return horizontal.mean(-1)+.3*clearance.mean(-1)

    def _get_dones(self):
        self._geometry()
        self.peak_top = torch.maximum(self.peak_top,self.top.sum(-1))
        root = self.robot.data.root_pos_w - self.scene.env_origins
        self.outside = (root[:,0]<-1.) | (root[:,0]>self.cfg.edge+self.cfg.platform_length+.2) | (root[:,1].abs()>1.3) | (root[:,2]<-.2)
        # No termination for pitching, body contact, wheel lift, reversing or touching the step face.
        timeout = self.episode_length_buf >= self.max_episode_length-1
        return self.success | self.outside, timeout

    def _get_rewards(self):
        progress = self._progress()
        delta = (progress-self.best_progress).clamp(min=0.)
        self.best_progress = torch.maximum(self.best_progress,progress)
        new_top = self.top & ~self.seen_top
        self.seen_top |= self.top
        reward = self.cfg.exploration_bonus*delta + 3.*new_top.float().sum(-1) + 50.*self.success.float() - .005 - 3.*self.outside.float()
        self.episode_return += reward
        self.last_transition = {k:v.clone() for k,v in {
            'success':self.success,'top':self.top,'bottom':self.bottom,'wheel_pos':self.wheel_pos,
            'root_pose':self.robot.data.root_state_w[:,:7]-torch.cat((self.scene.env_origins,torch.zeros(self.num_envs,4,device=self.device)),dim=-1),
            'joint_pos':self.robot.data.joint_pos,'joint_vel':self.robot.data.joint_vel,'targets':self.targets,
            'forces':self.forces,'servo_torque':self.servo_torque,'outside':self.outside}.items()}
        return reward

    def _get_observations(self):
        # Geometry must be refreshed after a reset, too.
        self._geometry()
        d = self.robot.data
        root = d.root_pos_w-self.scene.env_origins
        target = torch.tensor([self.cfg.edge,0.,self.cfg.height],device=self.device).expand(self.num_envs,-1)
        relative = quat_apply_inverse(d.root_quat_w,target-root)
        forward = torch.zeros_like(root);forward[:,0] = 1.
        forward = quat_apply_inverse(d.root_quat_w,forward)
        wheel_delta = torch.stack((self.cfg.edge-self.wheel_pos[:,:,0], self.cfg.height-self.bottom),dim=-1).flatten(1)
        obs = torch.cat((d.root_lin_vel_b,d.root_ang_vel_b*.2,d.projected_gravity_b,
            d.joint_pos[:,self.angle_ids],d.joint_vel*.04,relative,forward,
            torch.full((self.num_envs,1),self.cfg.height,device=self.device),wheel_delta,
            torch.tanh(self.actions),self.episode_length_buf[:,None]/self.max_episode_length),-1)
        if obs.shape[-1] != 40:
            raise RuntimeError(f'Observation dimension mismatch: {obs.shape}')
        return {'policy':obs.clamp(-20.,20.)}

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        ended = env_ids[self.episode_length_buf[env_ids]>0]
        if hasattr(self,'seen_top') and len(ended):
            succ = int(self.success[ended].sum().item())
            self.total_episodes += len(ended);self.total_success += succ
            self.max_top = max(self.max_top,int(self.peak_top[ended].max().item()))
            self.max_distinct_top = max(self.max_distinct_top,int(self.seen_top[ended].sum(-1).max().item()))
            self.extras['log'] = {'Task/success':self.success[ended].float().mean(),
                'Task/max_simultaneous_wheels_on_top':self.peak_top[ended].float().mean(),
                'Task/distinct_wheels_reached':self.seen_top[ended].float().sum(-1).mean(),
                'Task/progress':self.best_progress[ended].mean(),
                'Task/return':self.episode_return[ended].mean()}
        super()._reset_idx(env_ids)
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:,:3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root[:,:7],env_ids)
        self.robot.write_root_velocity_to_sim(root[:,7:],env_ids)
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos[env_ids],
                                            self.robot.data.default_joint_vel[env_ids],env_ids=env_ids)
        if hasattr(self,'actions'):
            self.actions[env_ids]=0.;self.targets[env_ids]=0.;self.seen_top[env_ids]=False;self.peak_top[env_ids]=0
            self.episode_return[env_ids]=0.;self.success[env_ids]=False
            self.best_progress[env_ids]=self.init_phi[env_ids] if self.init_phi is not None else 0.
