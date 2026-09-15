"""Throwaway diagnostic: how deep can the gripper actually insert, right now?

Sweeps the Cartesian insertion depth from the live robot state and reports the
fraction MoveIt can achieve for each, plus whether each pose is IK-reachable and
collision-free at all.
"""
import math, sys
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Pose, Point
from moveit_msgs.srv import GetCartesianPath, GetPositionIK, GetStateValidity
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

FRONT_Y = -2.8141704172539423
X = -0.13064904112970835
Z = 0.6010726805664737
LEAD = .011859


def make_pose(point, yaw, tf):
    p = Pose()
    p.position = Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
    roll, pitch = math.pi, 0.
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    p.orientation.x = sr*cp*cy-cr*sp*sy
    p.orientation.y = cr*sp*cy+sr*cp*sy
    p.orientation.z = cr*cp*sy-sr*sp*cy
    p.orientation.w = cr*cp*cy+sr*sp*sy
    return do_transform_pose(p, tf)


def main():
    rclpy.init()
    node = Node('probe_insertion')
    buffer = Buffer(); TransformListener(buffer, node)
    for _ in range(200):
        rclpy.spin_once(node, timeout_sec=.05)
        if buffer.can_transform('base_footprint', 'odom', Time()):
            break
    tf = buffer.lookup_transform('base_footprint', 'odom', Time())

    cartesian = node.create_client(GetCartesianPath, '/compute_cartesian_path')
    ik = node.create_client(GetPositionIK, '/compute_ik')
    validity = node.create_client(GetStateValidity, '/check_state_validity')
    for client in (cartesian, ik, validity):
        client.wait_for_service(timeout_sec=20.)

    def call(client, request):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=30.)
        return future.result()

    print(f'{"overlap_mm":>10} {"grasp_y":>10} {"ik":>6} {"valid":>6} {"fraction":>9}')
    for overlap_mm in (10, 20, 25, 30, 35, 40, 45, 50):
        grasp_y = FRONT_Y + LEAD - overlap_mm/1000.
        pose = make_pose([X, grasp_y, Z], -math.pi/2, tf)

        ik_request = GetPositionIK.Request()
        ik_request.ik_request.group_name = 'left_arm'
        ik_request.ik_request.ik_link_name = 'gripper_left_grasping_link'
        ik_request.ik_request.pose_stamped.header.frame_id = 'base_footprint'
        ik_request.ik_request.pose_stamped.pose = pose
        ik_request.ik_request.avoid_collisions = True
        ik_request.ik_request.timeout.sec = 2
        ik_result = call(ik, ik_request)
        ik_ok = ik_result.error_code.val if ik_result else -999

        valid = '-'
        if ik_result and ik_result.error_code.val == 1:
            v_request = GetStateValidity.Request()
            v_request.group_name = 'left_arm'
            v_request.robot_state = ik_result.solution
            v_result = call(validity, v_request)
            if v_result:
                valid = 'yes' if v_result.valid else 'NO'
                for contact in v_result.contacts[:4]:
                    print(f'            contact: {contact.contact_body_1} <-> {contact.contact_body_2}')

        c_request = GetCartesianPath.Request()
        c_request.header.frame_id = 'base_footprint'
        c_request.group_name = 'left_arm'
        c_request.link_name = 'gripper_left_grasping_link'
        c_request.start_state.is_diff = True
        c_request.waypoints = [pose]
        c_request.max_step = .005
        c_request.jump_threshold = 0.
        c_request.avoid_collisions = True
        c_result = call(cartesian, c_request)
        fraction = c_result.fraction if c_result else float('nan')
        print(f'{overlap_mm:>10} {grasp_y:>10.4f} {ik_ok:>6} {valid:>6} {fraction:>9.3f}')

    node.destroy_node(); rclpy.shutdown()


main()
