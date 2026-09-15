import math
from types import SimpleNamespace

import numpy as np
import pytest

from erc_tiago_integration.manipulation import Manipulator


def test_place_target_is_clamped_to_the_measured_reach():
    target = Manipulator.place_position(.85, 0.)
    assert target[0] == pytest.approx(Manipulator.PLACE_MAX_REACH)
    assert target[2] == pytest.approx(Manipulator.PLACE_HEIGHT)


def test_place_target_keeps_the_bearing_to_the_bin():
    target = Manipulator.place_position(.9, .3)
    assert math.atan2(target[1], target[0]) == pytest.approx(math.atan2(.3, .9))
    assert math.hypot(target[0], target[1]) == pytest.approx(Manipulator.PLACE_MAX_REACH)


def test_a_bin_already_within_reach_is_not_pushed_further():
    target = Manipulator.place_position(.6, .1)
    assert target[:2] == pytest.approx([.6, .1])


def test_carry_height_matches_placement_height():
    """Placement is then a level horizontal reach from the carry pose."""
    assert Manipulator.CARRY_POSITION[2] == pytest.approx(Manipulator.PLACE_HEIGHT)


def test_carry_raises_the_torso_before_moving_the_arm():
    manipulator = Manipulator.__new__(Manipulator)
    order = []
    manipulator.command_torso = lambda height: order.append(('torso', height))
    manipulator.tip_in_base = lambda: (.45, 0., 1.40)
    manipulator.update_fixed_scene = lambda: order.append(('scene',))
    manipulator.straight = lambda poses, **kwargs: order.append(('straight',))
    manipulator.node = SimpleNamespace(wait=lambda predicate, timeout: True,
                                       event=lambda state, **data: None)
    manipulator.carry()
    assert order[0] == ('torso', Manipulator.PLACE_TORSO)
    assert order.index(('straight',)) > 0


def test_place_height_stays_near_the_carried_height():
    assert Manipulator.place_position(.8, 0., 1.45)[2] == pytest.approx(Manipulator.PLACE_MAX_HEIGHT)
    assert Manipulator.place_position(.8, 0., .95)[2] == pytest.approx(Manipulator.PLACE_HEIGHT)


def test_released_book_lies_inside_the_bin_opening():
    from erc_tiago_integration.solution import Trial
    half_opening = .155
    near_edge = Trial.BIN_STANDOFF - half_opening
    far_edge = Trial.BIN_STANDOFF + half_opening
    book_near = Manipulator.PLACE_MAX_REACH - .030
    book_far = Manipulator.PLACE_MAX_REACH + .130
    assert near_edge < book_near and book_far < far_edge
