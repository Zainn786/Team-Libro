"""Can MoveIt plan the carry() transport pose from the post-extraction state, holding the book?"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, Point
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (Constraints, PositionConstraint, OrientationConstraint, BoundingVolume,
                             RobotState)
from moveit_msgs.srv import GetStateValidity
from shape_msgs.msg import SolidPrimitive
rclpy.init(); node = Node('probe_carry'); node.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
move = ActionClient(node, MoveGroup, '/move_action')
print('move_action up:', move.wait_for_server(timeout_sec=15.))
val = node.create_client(GetStateValidity, '/check_state_validity'); val.wait_for_service(timeout_sec=10.)
f = val.call_async(GetStateValidity.Request(group_name='', robot_state=RobotState(is_diff=True)))
rclpy.spin_until_future_complete(node, f, timeout_sec=20.)
r = f.result(); print('start state valid:', r.valid if r else None, [(c.contact_body_1, c.contact_body_2) for c in (r.contacts[:4] if r else [])])
pose = Pose(); pose.position = Point(x=.38, y=.28, z=1.05)
pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = 1., 0., 0., 0.
for attempt in range(2):
    g = MoveGroup.Goal(); g.request.group_name = 'left_arm'
    g.request.allowed_planning_time = 20.; g.request.num_planning_attempts = 5
    g.request.max_velocity_scaling_factor = .2; g.request.max_acceleration_scaling_factor = .2
    g.request.start_state.is_diff = True
    pc = PositionConstraint(); pc.header.frame_id = 'base_footprint'; pc.link_name = 'gripper_left_grasping_link'; pc.weight = 1.
    pc.constraint_region = BoundingVolume(primitives=[SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[.008])], primitive_poses=[pose])
    oc = OrientationConstraint(); oc.header.frame_id = 'base_footprint'; oc.link_name = 'gripper_left_grasping_link'
    oc.orientation = pose.orientation; oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = oc.absolute_z_axis_tolerance = .08; oc.weight = 1.
    g.request.goal_constraints = [Constraints(position_constraints=[pc], orientation_constraints=[oc])]
    g.planning_options.plan_only = True
    g.planning_options.planning_scene_diff.is_diff = True
    g.planning_options.planning_scene_diff.robot_state.is_diff = True
    gf = move.send_goal_async(g); rclpy.spin_until_future_complete(node, gf, timeout_sec=15.)
    h = gf.result(); rf = h.get_result_async(); rclpy.spin_until_future_complete(node, rf, timeout_sec=60.)
    res = rf.result().result
    print(f'carry plan attempt {attempt+1}: error_code={res.error_code.val} planned_points={len(res.planned_trajectory.joint_trajectory.points)}')
node.destroy_node(); rclpy.shutdown()
