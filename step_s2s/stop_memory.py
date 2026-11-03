"""Sensor-only task memory, matching the selected training policy contract."""
import numpy as np


class SensorStopMemory:
    def __init__(self, drive_distance=.15, hip_tolerance=.05):
        self.drive_distance=np.float32(drive_distance);self.hip_tolerance=hip_tolerance
        self.reset()

    def reset(self):
        self.home_time=np.float32(0.);self.anchor=np.float32(0.)
        self.started=False;self.fraction=np.float32(0.);self.stop_intent=False

    def update(self,packet,com_position,rear_position,hip_position,dt):
        projection=rear_position[0]*np.sin(packet[2])-rear_position[1]*np.cos(packet[2])
        passed=projection>packet[1]+np.float32(.04)
        home=np.max(np.abs(hip_position))<=self.hip_tolerance
        self.home_time=self.home_time+np.float32(dt) if passed and home else np.float32(0.)
        if not self.started and self.home_time>=.06-1e-7:
            self.anchor=np.float32(com_position[0]);self.started=True
        progress=np.clip((com_position[0]-self.anchor)/self.drive_distance,0.,1.)
        self.fraction=np.float32(max(self.fraction,progress*self.started))
        self.stop_intent=bool(self.stop_intent or (self.started and self.fraction>=1.))

    def observation(self):
        return np.array([self.started,self.fraction,self.stop_intent],np.float32)
