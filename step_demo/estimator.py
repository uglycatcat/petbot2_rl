"""Dead reckoning driven only by IMU samples and joint encoders.

No true position, orientation, velocity, contact state or updated terrain input.
Ideal simulator IMU accuracy does not establish real sensor drift robustness.
"""
import torch
from .tensor_math import quat_mul, rotate, inverse_rotate, rotation_step, heading


class InertialEstimator:
    def __init__(self, num_envs, device, source_bodies, joint_names):
        self.q = torch.zeros(num_envs, 4, device=device); self.q[:, 3] = 1
        self.position = torch.zeros(num_envs, 3, device=device)
        self.velocity = torch.zeros_like(self.position)
        self.com_position = torch.zeros_like(self.position)
        self.com_velocity = torch.zeros_like(self.position)
        self.gravity = torch.tensor([0., 0., -9.81], device=device).expand_as(self.position)
        self.source_bodies = source_bodies
        self.joint_names = joint_names
        self.fixed = [(torch.tensor(b['pos'], device=device), torch.tensor(b['com'], device=device),
                       torch.tensor(b['axis'], device=device)) for b in source_bodies]
        self.initial_offset = self.com_offset(torch.zeros(num_envs, len(joint_names), device=device))
        self.previous_offset = self.initial_offset.clone()
        self.rear_ids = [i for i,b in enumerate(source_bodies) if b.get('name') in ('left_rear_wheel','right_rear_wheel')]
        self.encoder_ids = [joint_names.index(n) for n in ('left_frame','right_frame','left_rear_wheel','right_rear_wheel')
                            if n in joint_names]
        self.previous_rear_offset = self.rear_offset.clone()
        self.samples = torch.zeros(num_envs, dtype=torch.long, device=device)

    def com_offset(self, joint_pos):
        positions, rotations = [], []
        identity = self.q.new_zeros(self.q.shape); identity[:, 3] = 1
        result = torch.zeros_like(self.position)
        mass = 0.
        for i, b in enumerate(self.source_bodies):
            local_pos, local_com, axis = self.fixed[i]
            if b['parent'] < 0:
                pos, q = torch.zeros_like(result), identity  # root translation excluded
            else:
                parent = b['parent']
                pos = positions[parent] + rotate(rotations[parent], local_pos.expand_as(result))
                q = rotations[parent]
                if b['joint'] in self.joint_names:
                    angle = joint_pos[:, self.joint_names.index(b['joint'])]
                    q = quat_mul(q, rotation_step(axis[None, :] * angle[:, None], 1.))
            result += b['mass'] * (pos + rotate(q, local_com.expand_as(result)))
            mass += b['mass']; positions.append(pos); rotations.append(q)
        rear = [positions[i] for i,b in enumerate(self.source_bodies) if b.get('name') in ('left_rear_wheel','right_rear_wheel')]
        self.rear_offset = torch.stack(rear).mean(0) if rear else torch.zeros_like(result)
        return result / mass

    def reset(self, ids):
        self.q[ids] = 0.; self.q[ids, 3] = 1.
        self.position[ids] = 0.; self.velocity[ids] = 0.
        self.com_position[ids] = 0.; self.com_velocity[ids] = 0.
        self.previous_offset[ids] = self.initial_offset[ids]; self.samples[ids] = 0
        zero = self.com_offset(torch.zeros(len(self.q),len(self.joint_names),device=self.q.device))
        self.previous_rear_offset[ids] = self.rear_offset[ids]

    def update(self, acceleration, gyro, joint_pos, dt, joint_vel=None):
        stationary=None
        if len(self.encoder_ids)==4 and joint_vel is not None:
            stationary=(joint_vel[:,self.encoder_ids].abs().amax(-1)<.15) & (gyro.norm(dim=-1)<.15)
            stationary &= (acceleration.norm(dim=-1)-9.81).abs()<.4
            up = inverse_rotate(self.q,-self.gravity/9.81)
            measured=acceleration/acceleration.norm(dim=-1,keepdim=True).clamp_min(1e-6)
            correction=torch.cross(measured,up,dim=-1)*2.
            gyro=gyro+stationary[:,None]*correction
        self.q = quat_mul(self.q, rotation_step(gyro, dt))
        self.q /= self.q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        acceleration_world = rotate(self.q, acceleration) + self.gravity
        new_velocity = self.velocity + acceleration_world * dt
        offset = rotate(self.q, self.com_offset(joint_pos))
        rear_offset = rotate(self.q,self.rear_offset)
        if len(self.encoder_ids)==4 and joint_vel is not None:
            # Rear wheels are required to stay supported. Fuse their rolling odometry
            # with IMU integration; no simulator pose, velocity or contacts enter here.
            omega=gyro.clone()
            omega[:,1] += joint_vel[:,self.encoder_ids[:2]].mean(-1)+joint_vel[:,self.encoder_ids[2:]].mean(-1)
            omega_world=rotate(self.q,omega)
            rear_velocity=torch.stack((.03*omega_world[:,1],-.03*omega_world[:,0]),dim=-1)
            odometry=rear_velocity-(rear_offset-self.previous_rear_offset)[:,:2]/dt
            blend=1.-__import__('math').exp(-dt/.10)
            new_velocity[:,:2]=(1-blend)*new_velocity[:,:2]+blend*odometry
            new_velocity=torch.where(stationary[:,None],torch.zeros_like(new_velocity),new_velocity)
        self.position += .5 * (self.velocity + new_velocity) * dt
        self.velocity = new_velocity
        self.com_position = self.position + offset - self.initial_offset
        self.com_velocity = self.velocity + (offset - self.previous_offset) / dt
        self.previous_offset = offset
        self.previous_rear_offset = rear_offset
        self.samples += 1

    def observation(self, packet, joint_pos, joint_vel, hips, wheels, gyro, previous_action):
        angle = packet[:, 2] - heading(self.q)
        distance = ((packet[:, 1] - self.com_position[:, 0]) * packet[:, 2].sin()
                    + self.com_position[:, 1] * packet[:, 2].cos())
        yaw = heading(self.q)
        return torch.cat((packet[:, :2], packet[:, 2:3].sin(), packet[:, 2:3].cos(),
            distance[:, None], angle[:, None].sin(), angle[:, None].cos(),
            self.com_position[:, 1:2], yaw[:, None].sin(), yaw[:, None].cos(), self.com_velocity[:, :2],
            inverse_rotate(self.q, self.gravity / 9.81), gyro,
            joint_pos[:, hips], joint_vel[:, hips], joint_vel[:, wheels], previous_action), dim=-1)
