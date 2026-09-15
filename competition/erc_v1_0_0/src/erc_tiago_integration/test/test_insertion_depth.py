import numpy as np
import pytest

from erc_tiago_integration.manipulation import Manipulator


def test_depths_start_over_the_centre_of_mass_and_keep_both_pads_on_the_cover():
    depths = Manipulator.INSERTION_DEPTHS
    assert depths[0] == pytest.approx(Manipulator.FINGERTIP_BOOK_OVERLAP)
    assert list(depths) == sorted(depths, reverse=True)
    assert min(depths) > Manipulator.FAR_PAD_DEPTH


def test_a_reduced_depth_is_not_retried_deeper():
    assert Manipulator.insertion_depths(.045) == [.045]
    assert Manipulator.insertion_depths(Manipulator.INSERTION_DEPTHS[0]) == list(Manipulator.INSERTION_DEPTHS)


def test_overlap_moves_only_the_grasp_frame():
    book = [.9, -2.8, .95]
    pre_a, grasp_a, _, _ = Manipulator.shelf_grasp_points(book)
    pre_b, grasp_b, _, _ = Manipulator.shelf_grasp_points(book, overlap=.045)
    assert np.allclose(pre_a, pre_b)
    assert grasp_b[1] - grasp_a[1] == pytest.approx(Manipulator.FINGERTIP_BOOK_OVERLAP - .045)
