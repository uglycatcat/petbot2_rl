"""Independent scalar checks use physics fields, never reported task success."""
import json
from pathlib import Path
import tempfile
import unittest

from step_demo.review import verify_physics
from step_demo.task import Phase
from test_task import task, sample, fly_front, put_on_top


def row(t, s, k):
    joint = [[0.]*9]
    joint[0][1:3] = s.hip[0].tolist()
    return dict(schema='step_trace_v2',physics_step=k,episode=[1],task=t.snapshot([0]),
        wheel_min=s.minimum.tolist(),wheel_max=s.maximum.tolist(),wheel_clearance=s.clearance.tolist(),
        wheel_contact_force=s.force.tolist(),platform_force=s.platform_force.tolist(),contact_point=s.contact_point.tolist(),
        edge_x=[.3],origin_com=[[0.,0.,0.]],true_quaternion=[[0.,0.,0.,1.]],
        body_contact_force=[[[0.,0.,1. if s.body_invalid.item() else 0.]]],lower_contact_force=[[[0.,0.,0.]]],
        com=[[s.forward.item(),s.lateral.item(),0.]],velocity=[[s.speed.item(),0.,0.]],
        yaw=s.yaw.tolist(),yaw_rate=s.yaw_rate.tolist(),joint_position=joint)


class PhysicalVerifierTests(unittest.TestCase):
    def test_full_route_support_gap_substep_spike_and_actual_hold(self):
        t,s=task(),sample(); rows=[]
        def emit():
            t.advance(s);rows.append(row(t,s,len(rows)+1))
        fly_front(s)
        for _ in range(15):emit()
        put_on_top(s,.5);emit()
        for k in range(1,17):
            s.forward[:]=.5+k*.01;s.speed[:]=.5;emit()
        self.assertEqual(t.phase.item(),Phase.STOP)
        s.speed[:]=0.
        for _ in range(200):emit()
        s.hip[:]=.06;emit();s.hip.zero_()
        for _ in range(1249):emit()
        s.speed[:]=.021;emit();s.speed.zero_()
        # Loaded penetrated geometry from the independent code audit.
        s.minimum[:,:2,2]=.017984977
        for _ in range(1250):emit()
        self.assertTrue(t.completed.item())
        with tempfile.TemporaryDirectory(prefix='step-physical-replay-') as directory:
            path=Path(directory)/'physics.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result=verify_physics(path,[0],[1,2])
            self.assertTrue(result['episodes']['0']['success'])
            self.assertEqual(result['episodes']['0']['hold_s'],2.5)
            # Corrupting only a reported label must be detected.
            rows[-1]['task']['phase'][0]=Phase.DRIVE
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'differs'):verify_physics(path,[0],[1,2])

    def test_missing_physics_frame_is_rejected(self):
        t,s=task(),sample();t.advance(s);a=row(t,s,1);t.advance(s);b=row(t,s,3)
        with tempfile.TemporaryDirectory(prefix='step-physical-gap-') as directory:
            path=Path(directory)/'physics.jsonl';path.write_text(json.dumps(a)+'\n'+json.dumps(b)+'\n')
            with self.assertRaisesRegex(ValueError,'Missing physics'):verify_physics(path,[0],[1,2])

    def test_qualified_forward_unqualified_reverse_never_certifies_short_drive(self):
        t,s=task(),sample();rows=[]
        def emit():
            t.advance(s);rows.append(row(t,s,len(rows)+1))
        fly_front(s);emit();put_on_top(s,.5);emit();s.speed[:]=.25
        for _ in range(16):
            s.hip.zero_()
            for k in range(1,21):s.forward[:]=.5+k*.0005;emit()
            s.hip[:]=.06
            for k in range(19,-1,-1):s.forward[:]=.5+k*.0005;emit()
            s.hip.zero_();emit()
        s.speed.zero_()
        for _ in range(1250):emit()
        s.body_invalid[:]=True;emit()
        with tempfile.TemporaryDirectory(prefix='step-net-drive-') as directory:
            path=Path(directory)/'physics.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result=verify_physics(path,[0],[1,2])
            self.assertFalse(result['episodes']['0']['success'])
            self.assertEqual(result['episodes']['0']['hold_s'],0.)


if __name__=='__main__':unittest.main()
