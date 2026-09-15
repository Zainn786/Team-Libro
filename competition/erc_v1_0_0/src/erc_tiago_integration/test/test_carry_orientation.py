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

    def straight(poses, **kwargs):
        calls.append(('straight', kwargs))
        if straight_fails:
            raise RuntimeError('Incomplete collision-free Cartesian path: 0.400')

    manipulator.straight = straight
    manipulator.go = lambda pose, **kwargs: calls.append(('go', kwargs))
    events = []
    manipulator.node = SimpleNamespace(wait=lambda predicate, timeout: True,
                                       event=lambda state, **data: events.append(state))
    return manipulator, calls, events


def test_book_carrying_moves_are_never_faster_than_the_extraction_that_held():
    """A faster straight-line carry shook the book out of the pads within 4 s."""
    import inspect
    assert Manipulator.CARRY_TIME_SCALE > Manipulator.EXTRACTION_TIME_SCALE
    straight_default = inspect.signature(Manipulator.straight).parameters['time_scale'].default
    assert straight_default == Manipulator.EXTRACTION_TIME_SCALE


def test_carry_height_is_the_one_a_straight_line_reaches_from_the_top_row():
    assert Manipulator.CARRY_POSITION[2] >= 1.20


def test_carry_prefers_a_straight_line_that_holds_orientation():
    manipulator, calls, _ = _carry_manipulator(straight_fails=False)
    manipulator.carry()
    assert calls == [('torso', Manipulator.PLACE_TORSO),
                     ('straight', {'time_scale': Manipulator.CARRY_TIME_SCALE})]


def test_carry_falls_back_to_the_tilt_constrained_plan():
    manipulator, calls, events = _carry_manipulator(straight_fails=True)
    manipulator.carry()
    assert calls[-1] == ('go', {'path_tilt_tolerance': Manipulator.CARRY_TILT_TOLERANCE})
    assert 'CARRY_STRAIGHT_LINE_UNAVAILABLE' in events
