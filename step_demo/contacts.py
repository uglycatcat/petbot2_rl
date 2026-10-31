"""Geometry-aware front face/edge contact classification."""


def forbidden_front_contact(normal, contact_x, minimum_x, edge_x):
    # A wheel entirely beyond the edge cannot touch the front face. PhysX top
    # contacts can have small horizontal components, especially under preload.
    near_face=minimum_x <= edge_x[:,None]+.0005
    horizontal=(normal[:,:,:2].norm(dim=-1)>.01) & near_face
    edge_point=(normal.norm(dim=-1)>.02) & (contact_x<=edge_x[:,None]+.0005)
    return horizontal | edge_point


def early_top_contact(normal, contact_x, edge_x, forbidden):
    """Reward-only top contact, without waiting for the entire wheel to clear."""
    return (normal[:, :, 2] > .02) & (contact_x > edge_x[:, None]+.002) & ~forbidden
