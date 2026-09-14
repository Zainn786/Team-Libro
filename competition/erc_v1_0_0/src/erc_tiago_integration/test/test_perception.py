from types import SimpleNamespace

import numpy as np
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from erc_tiago_integration.perception import LivePerception


def test_backprojection_uses_depth_optical_origin_and_timestamp():
    frames=[]
    transform=TransformStamped()
    transform.transform.rotation.w=1.
    def lookup(target,source,stamp):
        frames.append((source,stamp.nanoseconds))
        transform.transform.translation.x=.015 if source=='color_optical' else 0.
        return transform
    colour=Image(header=Header(frame_id='color_optical'))
    depth=Image(header=Header(frame_id='depth_optical'))
    depth.header.stamp.sec=7
    node=SimpleNamespace(ready=lambda:True,depth=np.full((20,20),2.,dtype=np.float32),
        rgb_msg=colour,depth_msg=depth,depth_info=CameraInfo(k=[100.,0.,10.,0.,100.,10.,0.,0.,1.]),
        tf=SimpleNamespace(lookup_transform=lookup))
    point=LivePerception.point(node,(5,5,10,10))
    assert np.allclose(point,[0.,0.,2.])
    assert frames==[('depth_optical',7_000_000_000)]
