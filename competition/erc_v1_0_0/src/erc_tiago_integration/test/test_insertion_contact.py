import math
from types import SimpleNamespace

import pytest

from erc_tiago_integration.manipulation import (ActionInterrupted, InsertionContact,
                                                Manipulator)
from erc_tiago_integration.solution import Trial


def test_finger_side_moves_the_jaws_towards_the_book():
    """Gazebo link poses with the base squared to the shelf put the left
    fingertip at base +y, which is odom +x when facing -pi/2. A left-finger
    touch means the book is on the +x side, so the offset must increase."""
    assert Manipulator.FINGER_SIDE_OFFSET_SIGN['left'] > 0
    assert Manipulator.FINGER_SIDE_OFFSET_SIGN['right'] < 0


def _manipulator(now_ns=1_000):
    manipulator = Manipulator.__new__(Manipulator)
    manipulator.finger_contacts = {}
    events = []
    manipulator.node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=now_ns)),
        last_grasp_contact=-math.inf,
        event=lambda state, **data: events.append((state, data)))
    return manipulator, events


def test_insertion_stops_and_reports_the_finger_that_struck_the_cover():
    manipulator, events = _manipulator()

    def straight(poses, min_fraction=.999, interrupt=None, **kwargs):
        assert interrupt is not None and not interrupt()
        manipulator.finger_contacts['right'] = 2_000
        manipulator.node.last_grasp_contact = math.inf
        assert interrupt()
        raise ActionInterrupted('stopped')

    manipulator.straight = straight
    with pytest.raises(InsertionContact) as info:
        manipulator.insert_to(object())
    assert info.value.side == 'right'
    assert events[-1] == ('INSERTION_CONTACT', {'side': 'right'})


def test_insertion_contact_on_both_fingers_is_unlocated():
    manipulator, _ = _manipulator()

    def straight(poses, min_fraction=.999, interrupt=None, **kwargs):
        manipulator.finger_contacts.update(left=2_000, right=2_000)
        raise ActionInterrupted('stopped')

    manipulator.straight = straight
    with pytest.raises(InsertionContact) as info:
        manipulator.insert_to(object())
    assert info.value.side is None


def test_contacts_from_before_the_insertion_do_not_count():
    manipulator, _ = _manipulator(now_ns=5_000)
    manipulator.finger_contacts = {'left': 4_000}

    def straight(poses, min_fraction=.999, interrupt=None, **kwargs):
        manipulator.finger_contacts['right'] = 6_000
        raise ActionInterrupted('stopped')

    manipulator.straight = straight
    with pytest.raises(InsertionContact) as info:
        manipulator.insert_to(object())
    assert info.value.side == 'right'


def test_clean_insertion_returns_without_raising():
    manipulator, events = _manipulator()
    manipulator.straight = lambda poses, **kwargs: None
    manipulator.insert_to(object())
    assert events == []


def _trial_with_goal(result_done):
    trial = Trial.__new__(Trial)
    trial.abort_reason = ''
    trial.active_goal = None
    cancelled = []
    result = SimpleNamespace(
        done=result_done,
        result=lambda: SimpleNamespace(status=4, result=SimpleNamespace(error_code=0)))
    handle = SimpleNamespace(
        accepted=True, get_result_async=lambda: result,
        cancel_goal_async=lambda: (cancelled.append(True) or SimpleNamespace(done=lambda: True)))
    client = SimpleNamespace(
        wait_for_server=lambda timeout_sec: True, _action_name='execute',
        send_goal_async=lambda goal: SimpleNamespace(done=lambda: True, result=lambda: handle))
    trial.wait = lambda predicate, timeout: predicate()
    trial.event = lambda state, **data: None
    return trial, client, cancelled


def test_action_cancels_the_goal_when_the_interrupt_fires():
    trial, client, cancelled = _trial_with_goal(result_done=lambda: False)
    with pytest.raises(ActionInterrupted):
        trial.action(client, object(), 10., interrupt=lambda: True)
    assert cancelled
    assert trial.active_goal is None


def test_action_prefers_a_completed_result_over_a_late_interrupt():
    trial, client, cancelled = _trial_with_goal(result_done=lambda: True)
    trial.action(client, object(), 10., interrupt=lambda: True)
    assert not cancelled


def test_unacknowledged_goal_is_resent_before_failing():
    trial = Trial.__new__(Trial)
    trial.abort_reason = ''
    trial.active_goal = None
    sends, events = [], []
    handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: SimpleNamespace(
            done=lambda: True,
            result=lambda: SimpleNamespace(status=4, result=SimpleNamespace(error_code=0))))

    def send_goal_async(goal):
        sends.append(goal)
        acknowledged = len(sends) >= 2
        return SimpleNamespace(done=lambda: acknowledged, result=lambda: handle,
                               add_done_callback=lambda cb: None)

    client = SimpleNamespace(wait_for_server=lambda timeout_sec: True,
                             _action_name='execute_trajectory', send_goal_async=send_goal_async)
    trial.wait = lambda predicate, timeout: predicate()
    trial.event = lambda state, **data: events.append(state)
    trial.action(client, object(), 10.)
    assert len(sends) == 2
    assert events == ['ACTION_ACK_RETRY']


def test_goal_that_is_never_acknowledged_still_fails():
    trial = Trial.__new__(Trial)
    trial.abort_reason = ''
    trial.active_goal = None
    client = SimpleNamespace(
        wait_for_server=lambda timeout_sec: True, _action_name='execute_trajectory',
        send_goal_async=lambda goal: SimpleNamespace(done=lambda: False, add_done_callback=lambda cb: None))
    trial.wait = lambda predicate, timeout: predicate()
    trial.event = lambda state, **data: None
    with pytest.raises(RuntimeError, match='acknowledgement timeout'):
        trial.action(client, object(), 10.)


def test_interrupt_waits_for_the_cancelled_goal_to_finish():
    trial, client, cancelled = _trial_with_goal(result_done=lambda: False)
    waits = []
    trial.wait = lambda predicate, timeout: (waits.append(timeout) or predicate())
    trial.event = lambda state, **data: None
    with pytest.raises(ActionInterrupted):
        trial.action(client, object(), 10., interrupt=lambda: True)
    assert Trial.CANCEL_SETTLE_TIMEOUT in waits


def test_interrupt_stops_moveit_execution_not_just_the_action_goal():
    """A cancel alone left the insertion running for 55 s, still pushing the book."""
    trial, client, cancelled = _trial_with_goal(result_done=lambda: False)
    published = []
    trial.execution_event = SimpleNamespace(publish=lambda message: published.append(message.data))
    trial.event = lambda state, **data: None
    with pytest.raises(ActionInterrupted):
        trial.action(client, object(), 10., interrupt=lambda: True)
    assert published == ['stop']
    assert cancelled
