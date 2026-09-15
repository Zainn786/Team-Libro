from types import SimpleNamespace

import pytest

from erc_tiago_integration.manipulation import GripperJammed, Manipulator, NoBookContact


def _closing_manipulator(position_step, contacts=False):
    """A gripper whose finger joint moves by position_step per 100 ms of sim time."""
    manipulator = Manipulator.__new__(Manipulator)
    now = SimpleNamespace(nanoseconds=10_000_000_000)
    cancelled, events = [], []
    handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: SimpleNamespace(done=lambda: False),
        cancel_goal_async=lambda: (cancelled.append(True) or SimpleNamespace(done=lambda: True)))
    manipulator.gripper = SimpleNamespace(
        send_goal_async=lambda goal: SimpleNamespace(done=lambda: True, result=lambda: handle))
    manipulator.finger_contacts = {}
    manipulator.gripper_position = .07

    def wait(predicate, timeout):
        for _ in range(120):
            now.nanoseconds += 100_000_000
            manipulator.gripper_position = max(0., manipulator.gripper_position - position_step)
            if contacts:
                manipulator.finger_contacts = {'left': now.nanoseconds, 'right': now.nanoseconds}
            if predicate():
                return True
        return False

    manipulator.node = SimpleNamespace(
        wait=wait, abort_reason='',
        get_clock=lambda: SimpleNamespace(now=lambda: now),
        event=lambda state, **data: events.append((state, data)))
    return manipulator, cancelled, events


def test_frozen_finger_joint_is_reported_as_a_jam_and_the_goal_cancelled():
    manipulator, cancelled, events = _closing_manipulator(position_step=0.)
    with pytest.raises(GripperJammed):
        manipulator.close_on_book(object())
    assert cancelled
    assert events[-1][0] == 'GRIPPER_JAMMED'
    assert events[-1][1]['waited_s'] >= Manipulator.GRIPPER_JAM_WINDOW_NS / 1e9


def test_a_jam_is_not_mistaken_for_a_missed_book():
    """NoBookContact triggers lateral re-alignment; a jam must end the attempt."""
    assert not issubclass(GripperJammed, NoBookContact)


def test_moving_jaws_without_contact_are_not_a_jam():
    manipulator, _, events = _closing_manipulator(position_step=.002)
    with pytest.raises(NoBookContact):
        manipulator.close_on_book(object())
    assert all(state != 'GRIPPER_JAMMED' for state, _ in events)


def test_stationary_jaws_already_touching_the_book_are_not_a_jam():
    manipulator, _, events = _closing_manipulator(position_step=0., contacts=True)
    try:
        manipulator.close_on_book(object())
    except GripperJammed:
        pytest.fail('bilateral contact with no travel is a grip, not a jam')
    except Exception:
        pass
    assert all(state != 'GRIPPER_JAMMED' for state, _ in events)


def test_open_command_stays_clear_of_the_joint_limit():
    """Opening to exactly the 0.070 limit jammed the left finger joint permanently."""
    assert Manipulator.GRIPPER_OPEN_COMMAND < Manipulator.OPEN_FINGERTIP_GAP_JOINT - .001


def test_reduced_opening_still_centres_a_thirty_millimetre_book():
    span = Manipulator.width_for_finger_joint(Manipulator.GRIPPER_OPEN_COMMAND)
    margin = (span - .030) / 2
    assert margin > Manipulator.ALIGNMENT_STEP
