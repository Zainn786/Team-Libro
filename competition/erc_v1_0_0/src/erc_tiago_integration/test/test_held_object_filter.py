from erc_tiago_integration.perception import LivePerception


def test_red_blob_held_in_the_gripper_is_not_taken_for_the_bin():
    grasp_frame = [1.05, 1.02, 1.0]
    carried_book_centroid = [1.00, 1.00, 1.05]
    assert LivePerception.within(carried_book_centroid, grasp_frame,
                                 LivePerception.HELD_OBJECT_RADIUS)


def test_bin_at_the_placement_distance_is_kept():
    grasp_frame = [1.05, 1.02, 1.0]
    bin_centroid = [1.45, 1.02, .90]
    assert not LivePerception.within(bin_centroid, grasp_frame,
                                     LivePerception.HELD_OBJECT_RADIUS)


def test_missing_gripper_transform_filters_nothing():
    assert not LivePerception.within([1., 1., 1.], None, LivePerception.HELD_OBJECT_RADIUS)
