"""Wheel clearance from the finite step, used only for physical termination.

The XZ distance is exact for convex wheels contained within the platform width.
For wheels outside that band, combining it with the lateral interval gap is a
conservative distance certificate: it cannot invent flight. Such side departures
are independently rejected by the unchanged COM lateral boundary.
"""
import torch


def convex_edges(vertices):
    """Include the projected silhouette at every orientation; called once on CPU."""
    from scipy.spatial import ConvexHull
    faces = ConvexHull(vertices).simplices
    return sorted({tuple(sorted((int(a), int(b)))) for f in faces
                   for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))})


def _wheel_terrain_clearance(points: torch.Tensor, edges: torch.Tensor,
                            edge_x: torch.Tensor, center_y: torch.Tensor,
                            height: float, length: float, half_width: float):
    """Distance certificate in metres; retain the existing 0.1 mm geometry pad.

points: [N,V,3] world vertices; edges: [E,2] convex-hull edges. The ground
plane continues under and beyond the finite platform. No contact force, target,
reward, actor observation or physical parameter is changed here.
    """
    x = points[:, :, 0] - edge_x[:, None]
    z = points[:, :, 2]
    floor = z.amin(1).clamp_min(0.)
    # Vertex-to-rectangle distances cover closest points on all four faces.
    dx = torch.maximum(-x, x-length).clamp_min(0.)
    dz = torch.maximum(-z, z-height).clamp_min(0.)
    distance2 = (dx.square()+dz.square()).amin(1)
    ax = x[:, edges[:, 0]]; az = z[:, edges[:, 0]]
    vx = x[:, edges[:, 1]]-ax; vz = z[:, edges[:, 1]]-az
    norm2 = (vx.square()+vz.square()).clamp_min(1e-20)
    # Rectangle-to-edge distances. Bottom corners cannot beat the ground plane.
    for corner_x in (0., length):
        t = (((corner_x-ax)*vx+(height-az)*vz)/norm2).clamp(0., 1.)
        d2 = (ax+t*vx-corner_x).square()+(az+t*vz-height).square()
        distance2 = torch.minimum(distance2, d2.amin(1))
    # Slab clipping catches edge intersections even when neither vertex is inside.
    inv_x = torch.where(vx.abs()>1e-12, vx, torch.ones_like(vx))
    inv_z = torch.where(vz.abs()>1e-12, vz, torch.ones_like(vz))
    tx0 = -ax/inv_x; tx1 = (length-ax)/inv_x
    tz0 = -az/inv_z; tz1 = (height-az)/inv_z
    lo_x = torch.where(vx.abs()>1e-12, torch.minimum(tx0,tx1), torch.zeros_like(vx))
    hi_x = torch.where(vx.abs()>1e-12, torch.maximum(tx0,tx1), torch.ones_like(vx))
    lo_z = torch.where(vz.abs()>1e-12, torch.minimum(tz0,tz1), torch.zeros_like(vz))
    hi_z = torch.where(vz.abs()>1e-12, torch.maximum(tz0,tz1), torch.ones_like(vz))
    intersects = (torch.maximum(lo_x,lo_z).clamp_min(0.) <= torch.minimum(hi_x,hi_z).clamp_max(1.))
    intersects &= ((vx.abs()>1e-12) | ((ax>=0.) & (ax<=length)))
    intersects &= ((vz.abs()>1e-12) | ((az>=0.) & (az<=height)))
    distance2 = torch.where(intersects.any(1), torch.zeros_like(distance2), distance2)
    y = points[:, :, 1]-center_y[:, None]
    lateral = torch.maximum(y.amin(1)-half_width, -half_width-y.amax(1)).clamp_min(0.)
    platform = (distance2+lateral.square()).sqrt()
    return (torch.minimum(floor, platform)-.0001).clamp_min(0.)


wheel_terrain_clearance = torch.jit.script(_wheel_terrain_clearance)


def compiled_clearance():
    """Fuse the same geometry algebra without changing meshes or thresholds."""
    return torch.compile(_wheel_terrain_clearance, fullgraph=True)


def rear_airborne(clearance, force):
    """Original distance and force thresholds, now relative to actual terrain."""
    return (clearance > .004) & (force.norm(dim=-1) < .02)
