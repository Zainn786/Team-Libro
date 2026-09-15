from erc_tiago_integration.manipulation import Manipulator


def test_face_grips_seen_in_trials_are_accepted():
    for contact_opening in (.0273, .0283, .0291):
        assert Manipulator.width_for_finger_joint(contact_opening) < Manipulator.GRIP_MAX_WIDTH


def test_the_edge_grip_that_slipped_is_rejected():
    assert Manipulator.width_for_finger_joint(.0407) > Manipulator.GRIP_MAX_WIDTH


def test_squeeze_stays_inside_the_joint_range():
    assert 0. < Manipulator.GRIP_SQUEEZE < .0273
