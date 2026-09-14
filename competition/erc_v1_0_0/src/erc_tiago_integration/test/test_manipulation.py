import math
from types import SimpleNamespace

import numpy as np
import pytest
from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import CollisionObject
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


def test_insertion_is_normal_to_shelf_without_lateral_sweep():
    pre,grasp,pull,lift=Manipulator.shelf_grasp_points([.94,-2.83,.605])
    assert np.allclose(pre,[.94,-2.53,.605])
    assert np.allclose(grasp,[.94,-2.93,.605])
    assert np.allclose(pull,[.94,-2.73,.630])
    # The post-extraction lift is deliberately limited to 3 cm to retain shelf
    # clearance while confirming that the book remains secured.
    assert np.allclose(lift,[.94,-2.73,.660])


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


def test_partial_cartesian_path_is_never_executed():
    manipulator=Manipulator.__new__(Manipulator)
    manipulator.tip='gripper_left_grasping_link'
    manipulator.cartesian=object()
    manipulator.call=lambda client,request:SimpleNamespace(
        error_code=SimpleNamespace(val=1),fraction=.97)
    executed=[]
    manipulator.node=SimpleNamespace(action=lambda *args:executed.append(args))
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
            manipulator.gripper_position=max(0.,manipulator.gripper_position-.002)
            manipulator.finger_contacts={'left':now.nanoseconds,'right':now.nanoseconds}
            if predicate():
                return True
        return False

    manipulator.node=SimpleNamespace(wait=wait,abort_reason='',
        get_clock=lambda:SimpleNamespace(now=lambda:now),event=lambda *args,**kwargs:None)
    manipulator.close_on_book(object())
    assert manipulator.hold_goal is preload_handle
    assert cancelled
    assert len(sent)==2
    assert sent[1].trajectory.points[0].positions[0] < .062
    assert now.nanoseconds>=600_000_000
