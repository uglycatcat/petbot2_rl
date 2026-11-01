import itertools
import math
import unittest

import torch
from step_demo.terrain import convex_edges, wheel_terrain_clearance, rear_airborne


class TerrainClearanceTests(unittest.TestCase):
    def box(self, xmin, xmax, zmin, zmax, ymin=-.01, ymax=.01):
        return torch.tensor(list(itertools.product((xmin,xmax),(ymin,ymax),(zmin,zmax))), dtype=torch.float64)

    def distance(self, points):
        edge=torch.tensor(convex_edges(points.numpy()),dtype=torch.long)
        return wheel_terrain_clearance(points[None],edge,torch.zeros(1),torch.zeros(1),.02,2.,1.).item()

    def test_floor_and_platform_flat_clearances(self):
        self.assertAlmostEqual(self.distance(self.box(-.2,-.1,.01,.07)),.0099)
        self.assertAlmostEqual(self.distance(self.box(.2,.3,.03,.09)),.0099)

    def test_front_riser_gap_not_lower_floor_height(self):
        self.assertAlmostEqual(self.distance(self.box(-.062,-.002,.008,.068)),.0019)

    def test_corner_distance_not_vertex_or_bounding_box_distance(self):
        # Diamond's nearest edge to the top corner is x+z=.025.
        poly=[(-.015,.04),(.01,.015),(-.015,-.01),(-.04,.015)]
        # Shift up so the lower vertex cannot beat the corner distance.
        poly=[(x,z+.03) for x,z in poly]
        pts=torch.tensor([(x,y,z) for x,z in poly for y in [-.01,.01]],dtype=torch.float64)
        # Closest lower-right edge: z-x=.035; corner (0,.02).
        self.assertAlmostEqual(self.distance(pts),.015/math.sqrt(2)-.0001)

    def test_rolling_across_edge_and_penetrating_top_are_near(self):
        self.assertEqual(self.distance(self.box(-.02,.02,.008,.068)),0.)
        self.assertEqual(self.distance(self.box(.2,.3,.019,.079)),0.)

    def test_edge_crossing_with_no_vertex_inside_rectangle(self):
        self.assertEqual(self.distance(self.box(-.1,2.1,.01,.03)),0.)

    def test_finite_platform_back_face_and_ground_beyond(self):
        self.assertAlmostEqual(self.distance(self.box(2.001,2.061,.01,.07)),.0009)
        self.assertAlmostEqual(self.distance(self.box(2.2,2.26,.01,.07)),.0099)

    def test_lateral_empty_space_does_not_extend_platform_infinitely(self):
        self.assertAlmostEqual(self.distance(self.box(.2,.3,.03,.09,1.1,1.12)),.0299)

    def test_world_translation_invariance(self):
        p=self.box(-.062,-.002,.008,.068);edges=torch.tensor(convex_edges(p.numpy()))
        q=p+torch.tensor([12.,-9.,0.])
        d=wheel_terrain_clearance(q[None],edges,torch.tensor([12.]),torch.tensor([-9.]),.02,2.,1.)
        self.assertAlmostEqual(d.item(),self.distance(p))

    def test_true_flight_and_force_threshold_unchanged(self):
        clearance=torch.tensor([[.004,.00401,.01,.01]])
        forces=torch.tensor([[[0.,0.,0.],[0.,0.,0.],[0.,0.,.02],[0.,0.,.019]]])
        self.assertEqual(rear_airborne(clearance,forces).tolist(),[[False,True,False,True]])


if __name__=='__main__':unittest.main()
