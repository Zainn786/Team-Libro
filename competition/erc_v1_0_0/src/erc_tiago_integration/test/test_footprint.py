import numpy as np
from geometry_msgs.msg import Quaternion
from erc_tiago_integration.footprint_guard import convex_hull, rotate


def test_hull_keeps_extended_gripper_not_only_base():
    points=[[-.4,-.3],[-.4,.3],[.4,.3],[.4,-.3],[1.02,.5],[1.02,-.5],[0.,0.]]
    hull=convex_hull(points)
    assert (1.02,.5) in hull and (1.02,-.5) in hull
    assert (0.,0.) not in hull


def test_quaternion_projection_rotates_collision_geometry():
    p=rotate(np.array([[1.,0.,0.]]),Quaternion(z=2**-.5,w=2**-.5))
    assert np.allclose(p,[[0.,1.,0.]])
