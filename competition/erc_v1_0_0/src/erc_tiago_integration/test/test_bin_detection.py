import numpy as np

from erc_tiago_integration.vision import ShelfVision, robust_depth


def _bin_frame():
    """A red square blob the size and shape of the bin seen at 1.1 m (143x149 px)."""
    image = np.zeros((360, 640, 3), np.uint8)
    image[153:302, 284:427] = (0, 0, 200)  # BGR red
    return image


def test_bin_seen_from_close_range_passes_the_shape_filter():
    vision = ShelfVision.__new__(ShelfVision)
    found = vision.bins(_bin_frame())
    assert len(found) == 1
    x, y, w, h = found[0]['bbox']
    assert (w, h) == (143, 149)


def test_narrow_book_shaped_blob_is_still_not_a_bin():
    vision = ShelfVision.__new__(ShelfVision)
    image = np.zeros((360, 640, 3), np.uint8)
    image[100:190, 300:330] = (0, 0, 200)   # 30x90: a standing book
    assert vision.bins(image) == []


def test_bin_depth_tolerates_rim_to_floor_spread():
    # Rim to floor across the whole blob, as measured live (0.32 m spread in the sample).
    depth = np.tile(np.linspace(.70, 1.50, 149, dtype=np.float32)[:, None], (1, 143))
    frame = np.zeros((360, 640), np.float32); frame[153:302, 284:427] = depth
    bbox = [284, 153, 143, 149]
    assert robust_depth(frame, bbox) is None
    median = robust_depth(frame, bbox, None)
    assert .9 < median < 1.3
