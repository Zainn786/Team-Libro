import math

import pytest

from erc_tiago_integration.solution import Trial


def test_heading_error_takes_the_short_way_round():
    assert Trial.heading_error(-math.pi/2,-1.4159) == pytest.approx(-math.pi/2+1.4159)
    # Across the +/-pi seam the error must stay small, not approach 2*pi.
    assert Trial.heading_error(math.pi-.01,-math.pi+.01) == pytest.approx(-.02)
    assert Trial.heading_error(1.,1.) == 0.


def test_square_up_turns_towards_the_target_heading():
    # A trial arrived at -1.4159 rad against a -pi/2 goal: it must turn clockwise.
    error=Trial.heading_error(-math.pi/2,-1.4159)
    assert Trial.square_up_rate(error) < 0.
    assert Trial.square_up_rate(-error) > 0.


def test_square_up_rate_is_bounded_both_ways():
    assert abs(Trial.square_up_rate(3.)) == pytest.approx(Trial.SQUARE_UP_MAX_RATE)
    # Tiny residual errors still command enough to overcome the smoother deadband.
    assert abs(Trial.square_up_rate(.001)) == pytest.approx(Trial.SQUARE_UP_MIN_RATE)
    # The floor rate cannot overshoot the tolerance in one 50 ms control step.
    assert Trial.SQUARE_UP_MIN_RATE*.05 < Trial.SQUARE_UP_TOLERANCE
