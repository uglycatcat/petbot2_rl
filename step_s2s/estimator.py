"""NumPy port of the frozen Step inertial/encoder estimator (xyzw quaternions)."""
import numpy as np


def cross3(a,b):
    """The estimator uses only 3-vectors; avoid NumPy's general axis dispatch."""
    return np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],
                     a[0]*b[1]-a[1]*b[0]],dtype=np.result_type(a,b))


def quat_mul(a, b):
    return np.concatenate((a[3]*b[:3]+b[3]*a[:3]+cross3(a[:3],b[:3]),
                           np.array([a[3]*b[3]-np.dot(a[:3],b[:3])],dtype=np.float32)))


def rotate(q,v):
    uv=cross3(q[:3],v)
    return v+2*(q[3]*uv+cross3(q[:3],uv))


def inverse_rotate(q,v):
    conjugate=q.copy();conjugate[:3]*=-1
    return rotate(conjugate,v)


def rotation_step(omega,dt):
    angle=np.linalg.norm(omega)*dt
    return np.r_[omega*(.5*dt*np.sinc(angle/(2*np.pi))),np.cos(angle/2)].astype(np.float32)


def heading(q):
    return np.arctan2(2*(q[3]*q[2]+q[0]*q[1]),1-2*(q[1]**2+q[2]**2))


class Estimator:
    def __init__(self,bodies,joint_names):
        self.bodies=bodies;self.names=joint_names
        self.ids={name:i for i,name in enumerate(joint_names)}
        self.fixed=[tuple(np.array(b[k],np.float32) for k in ('pos','com','axis')) for b in bodies]
        self.enc=[self.ids[x] for x in ('left_frame','right_frame','left_rear_wheel','right_rear_wheel')]
        self.gravity=np.array([0,0,-9.81],np.float32)
        self.initial_offset=self.com_offset(np.zeros(len(joint_names),np.float32))
        self.reset()

    def com_offset(self,joints):
        positions=[];rotations=[];result=np.zeros(3,np.float32);mass=0.;rear=[]
        for i,b in enumerate(self.bodies):
            lp,lc,axis=self.fixed[i]
            if b['parent']<0:
                pos=np.zeros(3,np.float32);q=np.array([0,0,0,1],np.float32)
            else:
                parent=b['parent'];pos=positions[parent]+rotate(rotations[parent],lp);q=rotations[parent]
                if b['joint'] in self.ids:q=quat_mul(q,rotation_step(axis*joints[self.ids[b['joint']]],1.))
            result+=b['mass']*(pos+rotate(q,lc));mass+=b['mass'];positions.append(pos);rotations.append(q)
            if b['name'] in ('left_rear_wheel','right_rear_wheel'):rear.append(pos)
        self.rear_offset=np.mean(rear,axis=0).astype(np.float32)
        return (result/mass).astype(np.float32)

    def reset(self):
        self.q=np.array([0,0,0,1],np.float32)
        self.position=np.zeros(3,np.float32);self.velocity=np.zeros(3,np.float32)
        self.com_position=np.zeros(3,np.float32);self.com_velocity=np.zeros(3,np.float32)
        self.previous_offset=self.initial_offset.copy()
        self.com_offset(np.zeros(len(self.names),np.float32));self.previous_rear_offset=self.rear_offset.copy()
        self.samples=0

    def update(self,acceleration,gyro,joint_pos,dt,joint_vel):
        acceleration=np.asarray(acceleration,np.float32);gyro=np.asarray(gyro,np.float32).copy()
        joint_pos=np.asarray(joint_pos,np.float32);joint_vel=np.asarray(joint_vel,np.float32)
        stationary=(np.max(np.abs(joint_vel[self.enc]))<.15 and np.linalg.norm(gyro)<.15
                    and abs(np.linalg.norm(acceleration)-9.81)<.4)
        if stationary:
            up=inverse_rotate(self.q,-self.gravity/9.81)
            gyro+=cross3(acceleration/max(np.linalg.norm(acceleration),1e-6),up)*2
        self.q=quat_mul(self.q,rotation_step(gyro,dt));self.q/=max(np.linalg.norm(self.q),1e-9)
        new_velocity=self.velocity+(rotate(self.q,acceleration)+self.gravity)*dt
        offset=rotate(self.q,self.com_offset(joint_pos));rear_offset=rotate(self.q,self.rear_offset)
        omega=gyro.copy();omega[1]+=joint_vel[self.enc[:2]].mean()+joint_vel[self.enc[2:]].mean()
        world=rotate(self.q,omega)
        odometry=np.array([.03*world[1],-.03*world[0]],np.float32)-(rear_offset-self.previous_rear_offset)[:2]/dt
        blend=1-np.exp(-dt/.10);new_velocity[:2]=(1-blend)*new_velocity[:2]+blend*odometry
        if stationary:new_velocity[:]=0
        self.position+=.5*(self.velocity+new_velocity)*dt;self.velocity=new_velocity.astype(np.float32)
        self.com_position=(self.position+offset-self.initial_offset).astype(np.float32)
        self.com_velocity=(self.velocity+(offset-self.previous_offset)/dt).astype(np.float32)
        self.previous_offset=offset;self.previous_rear_offset=rear_offset;self.samples+=1

    def observation(self,packet,joint_pos,joint_vel,gyro,executed):
        yaw=heading(self.q);angle=packet[2]-yaw
        distance=(packet[1]-self.com_position[0])*np.sin(packet[2])+self.com_position[1]*np.cos(packet[2])
        hips=self.enc[:2];wheels=self.enc[2:]
        return np.clip(np.r_[packet[:2],np.sin(packet[2]),np.cos(packet[2]),distance,np.sin(angle),np.cos(angle),
            self.com_position[1],np.sin(yaw),np.cos(yaw),self.com_velocity[:2],inverse_rotate(self.q,self.gravity/9.81),
            gyro,joint_pos[hips],joint_vel[hips],joint_vel[wheels],executed],-100.,100.).astype(np.float32)
