from types import SimpleNamespace
import math

import pytest
from geometry_msgs.msg import TransformStamped

from erc_tiago_integration.solution import Trial
from erc_tiago_integration.manipulation import Manipulator


@pytest.mark.parametrize('tilt',[.65,-1.2,float('nan')])
def test_head_rejects_invalid_targets_before_sending_motion(tilt):
    with pytest.raises(ValueError,match='tilt limits'):
        Trial.point_head(SimpleNamespace(),0.,tilt)


def test_head_action_success_requires_measured_alignment():
    node=SimpleNamespace(head=object(),action=lambda *args:None,
        manipulator=SimpleNamespace(head_positions={'head_1_joint':0.,'head_2_joint':.34907}),
        wait=lambda predicate,timeout:predicate())
    with pytest.raises(RuntimeError,match='did not reach'):
        Trial.point_head(node,0.,-.85)


def test_camera_aims_down_toward_a_lower_collection_bin():
    transform=TransformStamped()
    transform.transform.rotation.w=1.
    commands=[]
    node=SimpleNamespace(wait=lambda predicate,timeout:predicate(),
        vision=SimpleNamespace(ready=lambda:True,
            depth_msg=SimpleNamespace(header=SimpleNamespace(frame_id='depth_optical')),
            tf=SimpleNamespace(lookup_transform=lambda *args:transform)),
        manipulator=SimpleNamespace(head_positions={'head_2_joint':0.},
                                    _point_pose=Manipulator._point_pose),
        point_head=lambda pan,tilt:commands.append((pan,tilt)))
    Trial.aim_at_point(node,[0.,.5,1.])
    assert commands==[(0.,-math.atan2(.5,1.))]
