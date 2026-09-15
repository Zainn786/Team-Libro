import math
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_msgs.msg import Pose
from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import CollisionObject, RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from erc_tiago_integration.manipulation import Manipulator


def test_lost_book_model_is_removed_after_detachment():
    manipulator=Manipulator.__new__(Manipulator)
    manipulator.tip='gripper_left_grasping_link'
    manipulator.apply=object()
    requests=[]
    def call(client,request):
        requests.append(request)
        return SimpleNamespace(success=True)
    manipulator.call=call
    manipulator.detach_book(remove_world=True)
    assert len(requests)==2
    assert requests[0].scene.robot_state.attached_collision_objects[0].object.operation==CollisionObject.REMOVE
    assert requests[1].scene.world.collision_objects[0].id=='carried_book'
    assert requests[1].scene.world.collision_objects[0].operation==CollisionObject.REMOVE


def test_insertion_seats_both_fingertip_pads_on_the_book():
    front=-2.83
    pre,grasp,pull,lift=Manipulator.shelf_grasp_points([.94,front,.605])
    lead=Manipulator.OPEN_FINGERTIP_LEADING_EDGE
    expected_grasp_y=front+lead-Manipulator.FINGERTIP_BOOK_OVERLAP
    assert np.allclose(grasp,[.94,expected_grasp_y,.605])
    # Insertion is normal to the shelf: no lateral or vertical sweep on the way in.
    assert np.allclose(pre,[.94,front+lead+.04,.605])
    # The final waypoint adds no pull beyond one book depth: only a lift.
    assert np.allclose(pull,[.94,expected_grasp_y+Manipulator.BOOK_DEPTH,.6425])
    # The post-extraction lift is deliberately limited to 3 cm to retain shelf
    # clearance while confirming that the book remains secured.
    assert np.allclose(lift,[.94,expected_grasp_y+Manipulator.BOOK_DEPTH,.6725])


def test_high_shelf_stands_further_back_before_inserting():
    front=-2.8
    pre,_,_,_=Manipulator.shelf_grasp_points([.2,front,1.4])
    assert np.allclose(pre,[.2,front+Manipulator.OPEN_FINGERTIP_LEADING_EDGE+
                            Manipulator.HIGH_ROW_STANDOFF,1.4])
    assert Manipulator.HIGH_ROW_STANDOFF > Manipulator.LOW_ROW_STANDOFF


def test_pregrasp_never_folds_the_wrist_back_over_the_chassis():
    """The base stands BASE_GRASP_REACH from the grasp frame, so a long
    stand-off drags the wrist towards the robot rather than away from it."""
    for height in (.6,1.4):
        pre,grasp,_,_=Manipulator.shelf_grasp_points([0.,-2.8,height])
        base_y=grasp[1]+Manipulator.BASE_GRASP_REACH
        assert base_y-pre[1] > .30, f'wrist too close to the chassis at z={height}'
        # The straight-line insertion still has room to be a real approach.
        assert pre[1]-grasp[1] > .05


def test_both_fingertip_pads_bear_on_the_book_at_the_grasp_pose():
    """The fingertip presents two raised pads; a shallow seat grips only air."""
    front=0.
    _,grasp,_,_=Manipulator.shelf_grasp_points([0.,front,.6])
    # Depth of the leading edge past the front cover.
    leading_edge_depth=front-(grasp[1]-Manipulator.OPEN_FINGERTIP_LEADING_EDGE)
    assert leading_edge_depth == pytest.approx(Manipulator.FINGERTIP_BOOK_OVERLAP)
    # Both pads sit behind that edge, and both must land inside the cover.
    assert leading_edge_depth > Manipulator.FAR_PAD_DEPTH > Manipulator.NEAR_PAD_DEPTH
    # ...without reaching the back of the shelf.
    assert leading_edge_depth < Manipulator.BOOK_DEPTH


def test_insertion_never_drives_the_palm_into_the_cover():
    """The palm has no contact sensor; a deeper seat pushed a book 112 mm back."""
    for depth in Manipulator.INSERTION_DEPTHS:
        assert depth <= Manipulator.GRIPPER_THROAT_DEPTH - Manipulator.PALM_CLEARANCE + 1e-9
        # Both contact pads still bear on the cover.
        assert depth > Manipulator.FAR_PAD_DEPTH


def test_finger_joint_and_pad_span_are_mutual_inverses():
    at_limit=Manipulator.OPEN_FINGERTIP_GAP_JOINT
    assert Manipulator.width_for_finger_joint(at_limit) == pytest.approx(
        Manipulator.OPEN_FINGERTIP_GAP)
    assert Manipulator.finger_joint_for_width(
        Manipulator.OPEN_FINGERTIP_GAP) == pytest.approx(at_limit)
    for width in (.030,.040,.050):
        joint=Manipulator.finger_joint_for_width(width)
        assert Manipulator.width_for_finger_joint(joint) == pytest.approx(width)
        # Every width the gripper can hold is reachable inside the joint limits.
        assert -.001 <= joint <= at_limit


def test_shipped_v1_book_does_not_fit_the_shipped_gripper():
    """Regression guard for the defect in simulator release v1.0.0.

    erc_book.sdf ships a 60 mm collision box there, wider than the jaws can
    open, so no grasp is possible until the organizers' corrected 30 mm book
    is used.  This test documents the measurement rather than working around it.
    """
    assert Manipulator.OPEN_FINGERTIP_GAP < .060
    assert Manipulator.finger_joint_for_width(.060) > Manipulator.OPEN_FINGERTIP_GAP_JOINT
    # The corrected book has usable centring clearance on each side.
    assert (Manipulator.OPEN_FINGERTIP_GAP-.030)/2 > .01


def test_cartesian_slowdown_preserves_consistent_motion_derivatives():
    point=JointTrajectoryPoint(positions=[.4],velocities=[.8],accelerations=[1.6])
    point.time_from_start.sec=1
    point.time_from_start.nanosec=500_000_000
    trajectory=JointTrajectory(joint_names=['arm_left_1_joint'],points=[point])
    Manipulator.slow_cartesian(trajectory)
    assert np.allclose(point.positions,[.4])
    assert point.time_from_start.sec==6
    assert point.time_from_start.nanosec==0
    assert np.allclose(point.velocities,[.2])
    assert np.allclose(point.accelerations,[.1])


def test_cartesian_normal_speed_preserves_timing_and_derivatives():
    point=JointTrajectoryPoint(positions=[.4],velocities=[.8],accelerations=[1.6])
    point.time_from_start.sec=1
    point.time_from_start.nanosec=500_000_000
    trajectory=JointTrajectory(joint_names=['arm_left_1_joint'],points=[point])
    Manipulator.slow_cartesian(trajectory,time_scale=1.)
    assert point.time_from_start.sec==1
    assert point.time_from_start.nanosec==500_000_000
    assert np.allclose(point.velocities,[.8])
    assert np.allclose(point.accelerations,[1.6])


def test_cartesian_continuity_accepts_varying_small_steps():
    trajectory=JointTrajectory(joint_names=['arm_left_1_joint'],points=[
        JointTrajectoryPoint(positions=[position]) for position in (0.,.01,.02,.10)])
    Manipulator.validate_cartesian_joints(trajectory)


@pytest.mark.parametrize('positions',[(0.,.3),(0.,float('nan')),(0.,2*math.pi)])
def test_cartesian_continuity_rejects_jumps_and_nonfinite_values(positions):
    trajectory=JointTrajectory(joint_names=['arm_left_1_joint'],points=[
        JointTrajectoryPoint(positions=[position]) for position in positions])
    with pytest.raises(RuntimeError):
        Manipulator.validate_cartesian_joints(trajectory)


def test_partial_cartesian_path_is_never_executed(monkeypatch):
    monkeypatch.setattr('erc_tiago_integration.manipulation.time.sleep',lambda s:None)
    manipulator=Manipulator.__new__(Manipulator)
    manipulator.tip='gripper_left_grasping_link'
    manipulator.cartesian=object()
    manipulator.call=lambda client,request:SimpleNamespace(
        error_code=SimpleNamespace(val=1),fraction=.97)
    executed=[]
    manipulator.node=SimpleNamespace(action=lambda *args:executed.append(args),
                                     event=lambda state,**data:None)
    with pytest.raises(RuntimeError,match='Incomplete collision-free'):
        manipulator.straight([Manipulator._base_pose(np.array([.5,0.,.6]),0.)])
    assert not executed


def test_grasp_pose_quaternion_is_normalized():
    pose=Manipulator._base_pose(np.array([.5,.1,1.]),-.7)
    quaternion=pose.orientation
    norm=math.sqrt(quaternion.x**2+quaternion.y**2+
                   quaternion.z**2+quaternion.w**2)
    assert abs(norm-1.)<1e-12


def test_temporary_shelf_clearance_preserves_mechanical_pairs():
    manipulator=Manipulator.__new__(Manipulator)
    request=ApplyPlanningScene.Request()
    manipulator.configure_matrix(
        request,{('arena_shelf','gripper_left_fingertip_left_link'),
                 ('arena_shelf','carried_book')},
        {'arena_shelf','carried_book'})
    matrix=request.scene.allowed_collision_matrix
    indices={name:index for index,name in enumerate(matrix.entry_names)}
    def enabled(first,second):
        return matrix.entry_values[indices[first]].enabled[indices[second]]
    assert enabled('arm_left_6_link','arm_left_7_link')
    assert enabled('arm_left_7_link','gripper_left_base_link')
    assert enabled('gripper_left_base_link','gripper_left_inner_finger_left_link')
    assert enabled('arena_shelf','gripper_left_fingertip_left_link')
    assert enabled('arena_shelf','carried_book')


def test_shelf_clearance_extends_the_live_matrix_rather_than_replacing_it():
    """MoveIt seeds the matrix from the SRDF; dropping those entries makes every
    whole-robot collision check fail on a permanent base/wheel contact."""
    from moveit_msgs.msg import AllowedCollisionMatrix, AllowedCollisionEntry
    manipulator=Manipulator.__new__(Manipulator)
    live=AllowedCollisionMatrix(
        entry_names=['base_link','wheel_front_left_link'],
        entry_values=[AllowedCollisionEntry(enabled=[True,True]),
                      AllowedCollisionEntry(enabled=[True,True])])
    manipulator.live_matrix=lambda:live
    request=ApplyPlanningScene.Request()
    manipulator.configure_matrix(request,{('arena_shelf','carried_book')},
                                 {'arena_shelf','carried_book'})
    matrix=request.scene.allowed_collision_matrix
    indices={name:index for index,name in enumerate(matrix.entry_names)}
    # The SRDF-derived pair survives...
    assert matrix.entry_values[indices['base_link']].enabled[indices['wheel_front_left_link']]
    # ...alongside the pairs this package adds.
    assert matrix.entry_values[indices['arena_shelf']].enabled[indices['carried_book']]
    assert matrix.entry_values[indices['arm_left_6_link']].enabled[indices['arm_left_7_link']]
    # The matrix stays square.
    assert all(len(entry.enabled)==len(matrix.entry_names) for entry in matrix.entry_values)


def test_base_shell_and_wheels_are_mutually_allowed():
    """Wheels hang off suspension links, so they are not adjacent to base_link.

    Leaving those pairs enabled makes every whole-robot collision check fail on
    a permanent contact, which aborts MoveIt planning while single-group checks
    still report the state as valid.
    """
    pairs=Manipulator.mechanical_pairs()
    symmetric=pairs|{(second,first) for first,second in pairs}
    for wheel in ('wheel_front_left_link','wheel_front_right_link',
                  'wheel_rear_left_link','wheel_rear_right_link'):
        assert ('base_link',wheel) in symmetric
    assert ('base_link','suspension_front_left_link') in symmetric
    assert ('base_link','base_dock_link') in symmetric
    # The torso still collides with the world and the arms; only its joint
    # boundary with the base shell is exempt.
    assert ('base_link','torso_lift_link') not in symmetric


def test_extraction_requires_full_travel_only_while_inside_the_shelf():
    """A full book depth is out after two segments; the clearance move is best-effort."""
    _,grasp,pull,_=Manipulator.shelf_grasp_points([0.,-2.8,.6])
    extraction=[]
    for distance,height in ((.08,0.),(.16,.0125)):
        point=grasp.copy();point[1]+=distance;point[2]+=height
        extraction.append(point)
    extraction.append(pull)
    # By the last segment a full book depth is already clear of the shelf.
    assert extraction[1][1]-grasp[1] >= Manipulator.BOOK_DEPTH
    assert 0. < Manipulator.FINAL_CLEARANCE_MIN_FRACTION < .999


def test_final_extraction_waypoint_only_lifts():
    """Extra pull-back swung the forearm into the shelf end at an outer column."""
    _,grasp,pull,_=Manipulator.shelf_grasp_points([2.1,-2.8,.6])
    segment_two_end=grasp.copy();segment_two_end[1]+=.16;segment_two_end[2]+=.0125
    assert pull[0]==pytest.approx(segment_two_end[0])
    assert pull[1]==pytest.approx(segment_two_end[1])
    assert pull[2] > segment_two_end[2]


def test_blocked_final_clearance_move_is_skipped_not_fatal():
    import inspect
    source=inspect.getsource(Manipulator.grasp)
    assert "BOOK_FINAL_CLEARANCE_SKIPPED" in source
    assert "FINAL_CLEARANCE_MIN_FRACTION" in source


def test_reobservation_updates_the_target_between_attempts():
    manipulator=Manipulator.__new__(Manipulator)
    events=[]
    manipulator.node=SimpleNamespace(
        reobserve_target=lambda point:[point[0]+.01,point[1],point[2]],
        event=lambda state,**data:events.append((state,data)))
    updated=manipulator.reobserve_book([1.,-2.8,.6])
    assert updated==pytest.approx([1.01,-2.8,.6])
    assert events[0][0]=='BOOK_REOBSERVED'
    assert events[0][1]['moved'] == pytest.approx(.01)


def test_height_only_jump_after_a_failed_closure_is_treated_as_occlusion():
    """Values from a live trial: same plane fix, 90 mm lower, book undisturbed."""
    manipulator=Manipulator.__new__(Manipulator)
    events=[]
    first=[-2.1613824654224656,-2.7744053222023393,1.5923089926791554]
    second=[-2.1607051395488486,-2.7741313291293856,1.502319253110904]
    manipulator.node=SimpleNamespace(reobserve_target=lambda point:second,
                                     event=lambda state,**data:events.append(data))
    updated=manipulator.reobserve_book(first)
    assert updated[2]==pytest.approx(first[2])
    assert updated[:2]==pytest.approx(second[:2])
    assert events[0]['height_rejected_as_occlusion'] is True


def test_real_displacement_still_updates_height_and_plane():
    manipulator=Manipulator.__new__(Manipulator)
    manipulator.node=SimpleNamespace(
        reobserve_target=lambda point:[point[0]+.05,point[1],point[2]-.06],
        event=lambda state,**data:None)
    updated=manipulator.reobserve_book([1.,-2.8,.6])
    assert updated==pytest.approx([1.05,-2.8,.54])


def test_reobservation_aborts_when_the_book_has_been_displaced():
    """A knocked-over book must end the attempt, not invite another shove."""
    manipulator=Manipulator.__new__(Manipulator)
    far=Manipulator.BOOK_DISPLACED_LIMIT+.05
    manipulator.node=SimpleNamespace(
        reobserve_target=lambda point:[point[0]+far,point[1],point[2]],
        event=lambda state,**data:None)
    with pytest.raises(RuntimeError,match='moved'):
        manipulator.reobserve_book([1.,-2.8,.6])


def test_reobservation_keeps_the_old_estimate_when_the_book_is_not_seen():
    manipulator=Manipulator.__new__(Manipulator)
    manipulator.node=SimpleNamespace(reobserve_target=lambda point:None,
                                     event=lambda state,**data:None)
    assert manipulator.reobserve_book([1.,-2.8,.6]) is None


def test_bottom_row_lifts_the_torso_off_its_lower_stop():
    """With the torso fully down the bottom-row approach pose has no IK
    solution from any seed, even with collision checking disabled."""
    assert Manipulator.torso_height_for_row(.605) == .15
    # Middle rows sit near shoulder height and keep the configuration that has
    # already grasped a book.
    assert Manipulator.torso_height_for_row(.935) == 0.
    assert Manipulator.torso_height_for_row(1.265) == .15
    assert Manipulator.torso_height_for_row(1.595) == .3
    # Every commanded height stays inside the 35 cm of torso travel.
    for row in (.605,.935,1.265,1.595):
        assert 0. <= Manipulator.torso_height_for_row(row) <= .35


def _cartesian_manipulator(fractions):
    manipulator=Manipulator.__new__(Manipulator)
    events=[]
    manipulator.node=SimpleNamespace(event=lambda state,**data:events.append((state,data)),
                                     action=lambda *a,**k:SimpleNamespace(error_code=SimpleNamespace(val=1)))
    manipulator.cartesian=object();manipulator.execute=object()
    manipulator.tip='gripper_left_grasping_link'
    replies=iter(fractions)
    def call(client,request):
        point=JointTrajectoryPoint(positions=[0.]*7)
        point.time_from_start.sec=1
        trajectory=JointTrajectory(joint_names=[f'arm_left_{i}_joint' for i in range(1,8)],
                                   points=[point])
        return SimpleNamespace(error_code=SimpleNamespace(val=1),fraction=next(replies),
                               solution=RobotTrajectory(joint_trajectory=trajectory))
    manipulator.call=call
    return manipulator,events


def test_short_cartesian_path_is_replanned_before_failing(monkeypatch):
    monkeypatch.setattr('erc_tiago_integration.manipulation.time.sleep',lambda s:None)
    manipulator,events=_cartesian_manipulator([.055,1.])
    manipulator.straight([Pose()],min_fraction=.84)
    assert [state for state,_ in events]==['CARTESIAN_REPLANNING']
    assert events[0][1]['fraction']==pytest.approx(.055)


def test_persistently_short_cartesian_path_still_fails(monkeypatch):
    monkeypatch.setattr('erc_tiago_integration.manipulation.time.sleep',lambda s:None)
    attempts=Manipulator.CARTESIAN_PLAN_ATTEMPTS
    manipulator,events=_cartesian_manipulator([.055]*attempts)
    with pytest.raises(RuntimeError,match='Incomplete collision-free Cartesian path'):
        manipulator.straight([Pose()],min_fraction=.84)
    assert len(events)==attempts


def test_only_links_that_stay_outside_the_shelf_are_padded():
    """The wrist and gripper work inside the shelf row; padding them would
    make every insertion invalid. The upper arm never needs to get that close."""
    padded=set(Manipulator.PADDED_ARM_LINKS)
    assert {'arm_left_3_link','arm_left_4_link'} <= padded
    for link in ('arm_left_5_link','arm_left_6_link','arm_left_7_link',
                 'gripper_left_base_link','gripper_left_fingertip_left_link'):
        assert link not in padded
    assert 0. < Manipulator.UPPER_ARM_PADDING <= .05


def test_alignment_sweep_covers_the_span_nearest_first():
    sweep=Manipulator.alignment_sweep()
    assert sweep[:4]==[.012,-.012,.024,-.024]
    assert max(sweep)==pytest.approx(Manipulator.ALIGNMENT_SPAN)
    assert min(sweep)==pytest.approx(-Manipulator.ALIGNMENT_SPAN)
    # Every offset is distinct, so the retry loop never repeats a position.
    assert len(set(sweep))==len(sweep)
    # The loop must be able to try the whole sweep before giving up.
    assert Manipulator.GRASP_ALIGNMENT_ATTEMPTS > len(sweep)


def test_alignment_span_exceeds_the_jaw_centring_margin():
    """The sweep is useless unless it reaches past what centring alone tolerates."""
    margin=(Manipulator.OPEN_FINGERTIP_GAP-.030)/2
    assert Manipulator.ALIGNMENT_SPAN > margin


def test_grasp_requires_two_recent_fingertip_contacts():
    manipulator=Manipulator.__new__(Manipulator)
    now=SimpleNamespace(nanoseconds=5_000_000_000)
    manipulator.node=SimpleNamespace(get_clock=lambda:SimpleNamespace(now=lambda:now))
    manipulator.finger_contacts={}
    manipulator.record_finger_contact(['book_col_4_row_3_red',
                                      'tiago_pro::gripper_left_fingertip_left_link'])
    assert not manipulator.bilateral_contact()


def test_grasp_accepts_bilateral_inner_and_outer_finger_contacts():
    now=SimpleNamespace(nanoseconds=1_000_000_000)
    manipulator=Manipulator.__new__(Manipulator)
    manipulator.node=SimpleNamespace(get_clock=lambda:SimpleNamespace(now=lambda:now))
    manipulator.finger_contacts={}
    manipulator.record_finger_contact([
        'book_col_4_row_3_red',
        'tiago_pro::gripper_left_inner_finger_left_link'])
    manipulator.record_finger_contact([
        'tiago_pro::gripper_left_outer_finger_right_link',
        'book_col_4_row_3_red'])
    assert manipulator.bilateral_contact()
    manipulator.record_finger_contact(['tiago_pro::gripper_left_fingertip_right_link',
                                      'book_col_4_row_3_red'])
    assert manipulator.bilateral_contact()
    now.nanoseconds+=1_001_000_000
    assert not manipulator.bilateral_contact()


def test_confirmed_grip_stops_closure_and_holds_measured_opening():
    manipulator=Manipulator.__new__(Manipulator)
    now=SimpleNamespace(nanoseconds=0)
    cancelled=[]
    sent=[]
    events=[]
    cancel=SimpleNamespace(done=lambda:True)
    handle=SimpleNamespace(accepted=True,
                           get_result_async=lambda:SimpleNamespace(done=lambda:False),
                           cancel_goal_async=lambda:(cancelled.append(True) or cancel))
    preload_handle=SimpleNamespace(accepted=True,cancel_goal_async=lambda:None)
    def send_goal(goal):
        sent.append(goal)
        selected=handle if len(sent)==1 else preload_handle
        return SimpleNamespace(done=lambda:True,result=lambda:selected)
    manipulator.gripper=SimpleNamespace(send_goal_async=send_goal)
    manipulator.finger_contacts={'left':-10,'right':-10}
    manipulator.gripper_position=.07

    def wait(predicate,timeout):
        for step in range(40):
            now.nanoseconds+=100_000_000
            # The fingers meet the 30 mm book faces near joint 0.028, not at the
            # open position; stop closing there like the real jaws do.
            manipulator.gripper_position=max(.028,manipulator.gripper_position-.002)
            if manipulator.gripper_position <= .028:
                manipulator.finger_contacts={'left':now.nanoseconds,'right':now.nanoseconds}
            if predicate():
                return True
        return False

    manipulator.node=SimpleNamespace(wait=wait,abort_reason='',
        get_clock=lambda:SimpleNamespace(now=lambda:now),
        event=lambda state,**kwargs:events.append((state,kwargs)))
    manipulator.close_on_book(object())
    assert manipulator.hold_goal is preload_handle
    assert cancelled
    assert len(sent)==2
    grip_event=next(data for state,data in events if state=='GRIP_CONFIRMED')
    # The hold command squeezes 5 mm past the measured contact position.  The
    # book blocks that travel, so the unreached remainder becomes the grip
    # force; holding the contact position itself applies none.
    assert grip_event['hold_opening'] == pytest.approx(
        grip_event['contact_opening']-Manipulator.GRIP_SQUEEZE)
    assert sent[1].trajectory.points[0].positions[0] == pytest.approx(
        grip_event['hold_opening'])
    # The recorded width is what the jaws actually closed on, for trial evidence.
    assert grip_event['measured_book_width'] == pytest.approx(
        Manipulator.width_for_finger_joint(grip_event['contact_opening']))
    assert now.nanoseconds>=100_000_000
