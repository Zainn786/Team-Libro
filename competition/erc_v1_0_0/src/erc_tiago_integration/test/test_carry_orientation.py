import math
from types import SimpleNamespace

import numpy as np

from erc_tiago_integration.manipulation import Manipulator


def _manipulator():
    manipulator = Manipulator.__new__(Manipulator)
    manipulator.tip = 'gripper_left_grasping_link'
    goals = []
    manipulator.node = SimpleNamespace(
        move=object(),
        action=lambda client, goal, timeout: (goals.append(goal) or
                                              SimpleNamespace(error_code=SimpleNamespace(val=1))),
        event=lambda state, **data: None)
    return manipulator, goals


def test_plain_motion_has_no_path_constraint():
    manipulator, goals = _manipulator()
    manipulator.go(Manipulator._base_pose(np.array([.4, .2, 1.]), 0.))
    assert not goals[0].request.path_constraints.orientation_constraints


def test_carry_motion_keeps_the_book_level_but_may_turn_about_vertical():
    manipulator, goals = _manipulator()
    manipulator.go(Manipulator._base_pose(np.array([.4, .2, 1.]), .7),
                   path_tilt_tolerance=Manipulator.CARRY_TILT_TOLERANCE)
    keep = goals[0].request.path_constraints.orientation_constraints[0]
    assert keep.link_name == 'gripper_left_grasping_link'
    assert keep.absolute_x_axis_tolerance == Manipulator.CARRY_TILT_TOLERANCE
    assert keep.absolute_y_axis_tolerance == Manipulator.CARRY_TILT_TOLERANCE
    assert keep.absolute_z_axis_tolerance == math.pi


def test_carry_and_place_poses_point_the_gripper_straight_down():
    """z is only a vertical axis, and so safe to leave free, if roll is pi."""
    for yaw in (0., .7, -2.):
        q = Manipulator._base_pose(np.array([0., 0., 0.]), yaw).orientation
        # third column of the rotation matrix = link z axis in base frame
        z_axis = (2 * (q.x * q.z + q.w * q.y), 2 * (q.y * q.z - q.w * q.x),
                  1 - 2 * (q.x * q.x + q.y * q.y))
        assert np.allclose(z_axis, (0., 0., -1.), atol=1e-9)


def _carry_manipulator(straight_fails):
    manipulator = Manipulator.__new__(Manipulator)
    calls = []
    manipulator.update_fixed_scene = lambda: None
    manipulator.command_torso = lambda height: calls.append(('torso', height))
    manipulator.tip_in_base = lambda: (.45, .01, 1.40)

    def straight(poses, **kwargs):
        calls.append(('straight', round(poses[0].position.x, 3), kwargs))
        if straight_fails:
            raise RuntimeError('Incomplete collision-free Cartesian path: 0.400')

    manipulator.straight = straight
    manipulator.go = lambda pose, **kwargs: calls.append(('go', kwargs))
    events = []
    manipulator.node = SimpleNamespace(wait=lambda predicate, timeout: True,
                                       event=lambda state, **data: events.append(state))
    return manipulator, calls, events


def test_carry_only_lifts_and_backs_straight_out():
    """The book slipped at the first sideways swing; only depth-axis and vertical moves held."""
    manipulator, calls, _ = _carry_manipulator(straight_fails=False)
    manipulator.carry()
    assert calls[0] == ('torso', Manipulator.PLACE_TORSO)
    assert calls[1] == ('straight', round(.45 - Manipulator.CARRY_BACKOFF, 3),
                        {'time_scale': Manipulator.CARRY_TIME_SCALE})
    assert not any(call[0] == 'go' for call in calls)


def test_blocked_backoff_is_skipped_not_fatal():
    manipulator, calls, events = _carry_manipulator(straight_fails=True)
    manipulator.carry()
    assert 'CARRY_BACKOFF_SKIPPED' in events
    assert 'BOOK_STOWED_FOR_TRANSPORT' in events
