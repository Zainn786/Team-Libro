import pytest

from erc_tiago_integration.solution import Trial


def test_no_approach_when_already_at_the_standoff():
    assert Trial.bin_gap((0., 0.), (Trial.BIN_STANDOFF + .01, 0.)) == 0.


def test_nav_tolerance_gap_is_closed():
    assert Trial.bin_gap((0., 0.), (.88, 0.)) == pytest.approx(.08)


def test_approach_is_capped_so_the_base_never_reaches_the_table():
    assert Trial.bin_gap((0., 0.), (1.5, 0.)) == pytest.approx(Trial.BIN_APPROACH_MAX)


def test_standoff_keeps_the_bin_within_the_arms_placement_reach():
    from erc_tiago_integration.manipulation import Manipulator
    assert Trial.BIN_STANDOFF <= Manipulator.PLACE_MAX_REACH
