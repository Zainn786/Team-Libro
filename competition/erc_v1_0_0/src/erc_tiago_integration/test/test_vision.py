from pathlib import Path
import cv2
import numpy as np
from erc_tiago_integration.vision import ShelfVision, robust_depth

ROOT=Path(__file__).resolve().parents[1]


def test_colours_and_distractor_geometry():
    image=np.full((180,320,3),210,np.uint8)
    colours=[(0,0,255),(255,0,0),(0,255,0),(0,255,255)]
    for i,colour in enumerate(colours):
        cv2.rectangle(image,(20+i*65,40),(32+i*65,120),colour,-1)
    # Wide red bin must not be confused with an upright book.
    cv2.rectangle(image,(10,145),(160,170),(0,0,255),-1)
    vision=ShelfVision(ROOT/'templates')
    found=vision.books(image)
    assert sorted(x['colour'] for x in found)==['blue','green','red','yellow']
    bins=vision.bins(image)
    assert len(bins)==1 and bins[0]['bbox']==[10,145,151,26]


def test_bin_rejects_red_book_and_small_noise():
    image=np.full((180,320,3),210,np.uint8)
    cv2.rectangle(image,(30,30),(42,125),(0,0,255),-1)
    cv2.rectangle(image,(200,20),(205,24),(0,0,255),-1)
    vision=ShelfVision(ROOT/'templates')
    assert vision.bins(image)==[]


def test_unreadable_marker_is_not_a_number():
    vision=ShelfVision(ROOT/'templates')
    image=np.full((200,300,3),255,np.uint8)
    cv2.rectangle(image,(40,40),(60,65),(0,0,0),-1)
    assert vision.numbers(image)==[]


def test_depth_rejects_missing_mixed_surfaces_and_range():
    box=[0,0,20,20]
    assert robust_depth(np.full((20,20),np.nan),box) is None
    assert robust_depth(np.zeros((20,20)),box) is None
    assert robust_depth(np.full((20,20),9.),box) is None
    depth=np.ones((20,20));depth[:,10:]=2.
    assert robust_depth(depth,box) is None
    assert robust_depth(np.full((20,20),1.25),box)==1.25


def test_digits_under_scale_and_brightness_changes():
    vision=ShelfVision(ROOT/'templates')
    for number in range(1,6):
        src=cv2.imread(str(ROOT/'templates'/f'{number}.png'))
        src=cv2.resize(src,(25,32))
        src=np.clip(src.astype(float)*.85+12,0,255).astype(np.uint8)
        image=np.full((100,100,3),230,np.uint8)
        image[30:62,40:65]=src
        found=vision.numbers(image)
        assert len(found)==1 and found[0]['number']==number
