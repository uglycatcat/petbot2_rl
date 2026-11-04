"""Independent scalar verification of 2 ms physical traces.

Does not import the tensor task or use its phase/completed/support labels to
make decisions. Those labels are compared only after physical reconstruction.
"""
import math


def norm(v):
    return math.sqrt(sum(x*x for x in v))


class PhysicsVerifier:
    def __init__(self, acceptance, terrain, hip_indices, dt=.002):
        self.a, self.t, self.dt = acceptance, terrain, dt
        self.hip_indices = hip_indices
        self.ticks = 0
        self.flight = [0, 0]; self.air = [False, False]; self.off = [0, 0]
        self.landed = [False, False]; self.rear_off = [0, 0]
        self.age = [1., 1., 1., 1.]
        self.phase = 0; self.reason = 0; self.distance = 0.; self.drive_origin = 0.
        self.previous_x = 0.; self.previous_ready = False; self.hold = 0
        self.result = None

    def update(self, row, i):
        if self.result is not None: return self.result
        self.ticks += 1
        lo, hi = row['wheel_min'][i], row['wheel_max'][i]
        force, normal = row['wheel_contact_force'][i], row['platform_force'][i]
        points, gap = row['contact_point'][i], row['wheel_clearance'][i]
        edge, center_y = row['edge_x'][i], row['origin_com'][i][1]
        loaded, support = [], []
        failures = []
        for k in range(4):
            p = points[k]
            top = (all(math.isfinite(x) for x in p) and p[0] > edge+.0005
                   and p[0] < edge+self.t.platform_length_m
                   and abs(p[1]-center_y) < self.t.platform_width_m/2
                   and abs(p[2]-self.t.height_m) <= .004 and normal[k][2] > .02)
            loaded.append(top)
            if k < 2 and ((norm(normal[k]) > .02 and not top)
                          or (lo[k][0] <= edge+.0005 and norm(normal[k][:2]) > .01)):
                failures.append(1)
            self.age[k] = 0. if top else min(1., self.age[k]+self.dt)
            inside = (lo[k][0] > edge+.0005 and hi[k][0] < edge+self.t.platform_length_m
                      and abs(lo[k][1]-center_y) < self.t.platform_width_m/2
                      and abs(hi[k][1]-center_y) < self.t.platform_width_m/2)
            support.append(inside and (top or (self.age[k] <= .02+1e-7 and gap[k] <= .004)))
            if k < 2:
                plane = self.t.height_m if self.landed[k] else 0.
                airborne = lo[k][2]-plane > .004 and norm(force[k]) < .02
                self.off[k] = self.off[k]+1 if airborne else 0
                if not self.air[k] and airborne:
                    self.flight[k] += 1; self.air[k] = True
                if self.air[k] and top:
                    self.air[k] = False; self.landed[k] = True
                elif self.air[k] and lo[k][2] <= .004 and norm(force[k]) > .02:
                    self.air[k] = False; failures.append(5)
                if self.flight[k] > 1 or (top and self.flight[k] == 0): failures.append(5)
            else:
                airborne = gap[k] > .004 and norm(force[k]) < .02
                self.rear_off[k-2] = self.rear_off[k-2]+1 if airborne else 0
                if airborne: failures.append(2)
        q = row['true_quaternion'][i]
        up_z = 1.-2.*(q[0]*q[0]+q[1]*q[1])
        body_hit = any(norm(v) > .5 for v in row['body_contact_force'][i]+row['lower_contact_force'][i])
        if up_z < .4 or body_hit: failures.append(3)
        x, y = row['com'][i][:2]
        if x < -.05 or abs(y) > .15: failures.append(4)
        if any(v[0] >= edge+self.t.platform_length_m for v in hi): failures.append(6)
        # Explicit failure priority agrees with the published contract.
        self.reason = next((reason for reason in (1,2,3,4,5,6) if reason in failures), 0)
        old = self.phase
        if not self.reason:
            if any(self.flight): self.phase = max(self.phase,1)
            if all(self.landed) and not any(self.air): self.phase = max(self.phase,2)
            if self.phase >= 2 and all(support): self.phase = max(self.phase,3)
        hip = [row['joint_position'][i][j] for j in self.hip_indices]
        aligned = abs(y) <= self.a.line_tolerance_m and abs(row['yaw'][i]) <= self.a.heading_tolerance_rad
        ready = (not self.reason and all(self.landed) and not any(self.air) and all(support)
                 and aligned and max(abs(v) for v in hip) <= self.a.hip_tolerance_rad)
        if ready: self.phase = max(self.phase,4)
        if not self.reason and old < 4 and self.phase == 4: self.drive_origin = x
        if not self.reason and old == 4 and not (ready and self.previous_ready):
            self.drive_origin += max(0.,x-self.previous_x)
        if not self.reason and self.phase >= 4:
            self.distance = max(0.,min(self.a.post_climb_distance_m,x-self.drive_origin))
        self.previous_x, self.previous_ready = x, ready
        drove = self.distance+1e-7 >= self.a.post_climb_distance_m
        if not self.reason and self.phase == 4 and ready and drove:
            self.phase = 5
        stopped = (self.phase == 5 and ready and drove and norm(row['velocity'][i]) <= self.a.stop_speed_m_s
                   and abs(row['yaw_rate'][i]) <= self.a.stop_yaw_rate_rad_s)
        self.hold = self.hold+1 if stopped else 0
        success = stopped and self.hold*self.dt+1e-7 >= self.a.stop_hold_s+2.
        if success: self.phase = 6
        if not success and not self.reason and self.ticks >= round(self.a.timeout_s/self.dt): self.reason = 7
        if self.reason: self.phase = 7
        if self.reason or success:
            self.result = dict(success=bool(success), failure=self.reason, phase=self.phase,
                               front_flight_count=self.flight.copy(), drive_distance=self.distance,
                               hold_s=self.hold*self.dt, elapsed_s=self.ticks*self.dt)
        return self.result
