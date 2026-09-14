import numpy as np
import pytest
from erc_tiago_integration.catalogue import ObservationCatalogue, assign_rows, approach_pose, shelf_axes


def detection(point, stamp, **extra):
    return {'point':point,'stamp_ns':stamp,**extra}


def test_confirmation_requires_independent_frames_and_rejects_jumps():
    store=ObservationCatalogue()
    for _ in range(5):store.add('red',detection([0.,1.,1.2],10))
    assert not store.confirmed()
    store.add('red',detection([.01,1.,1.2],11))
    store.add('red',detection([.02,1.,1.2],12))
    assert 'red' in store.confirmed()
    store.add('red',detection([1.,1.,1.2],13))
    assert not store.confirmed()


def test_shelf_labels_do_not_determine_physical_order():
    markers={5:detection([-2.,-2.7,2.25],1),2:detection([0.,-2.7,2.25],1),
             1:detection([2.,-2.7,2.25],1)}
    tangent,normal=shelf_axes(markers,[0.,0.,0.])
    assert np.allclose(normal,[0.,-1.])
    goal=approach_pose(markers[5],normal)
    assert np.allclose(goal[:2],[-2.,-1.6])
    assert abs(np.dot(tangent,normal))<1e-8


def test_rows_come_from_observed_vertical_order():
    books={c:detection([0.,-2.8,z],1,colour=c) for c,z in
           [('red',.7),('blue',1.69),('green',1.36),('yellow',1.03)]}
    result=assign_rows(books)
    assert result['blue']['row']==1
    assert result['red']['row']==4
    del books['green']
    with pytest.raises(ValueError):assign_rows(books)


def test_same_row_and_invalid_pose_are_rejected():
    books={c:detection([0.,-2.8,1.],1,colour=c) for c in ('red','blue','green','yellow')}
    with pytest.raises(ValueError):assign_rows(books)
    with pytest.raises(ValueError):approach_pose(books['red'],[0.,0.])
    with pytest.raises(ValueError):approach_pose(books['red'],[0.,-1.],.2)
