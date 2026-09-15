"""Probe the live scene: ACM size, whole-robot validity, IK for pre/grasp poses."""
import math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Pose, Point
from moveit_msgs.srv import (GetPositionIK, GetStateValidity, GetPlanningScene,
                             GetCartesianPath)
from moveit_msgs.msg import PlanningSceneComponents, RobotState
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

FRONT = -2.7806077711542336
X = -1.86394086833557
Z = 0.5980225575804718
LEAD = .011859

rclpy.init()
node = Node('probe2')
node.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
buffer = Buffer(); TransformListener(buffer, node)
for _ in range(600):
    rclpy.spin_once(node, timeout_sec=.05)
    if buffer.can_transform('base_footprint', 'odom', Time()):
        break
else:
    raise SystemExit('no TF')
tf = buffer.lookup_transform('base_footprint', 'odom', Time())

def call(client, request, timeout=30.):
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    return future.result()

scene = node.create_client(GetPlanningScene, '/get_planning_scene')
ik = node.create_client(GetPositionIK, '/compute_ik')
validity = node.create_client(GetStateValidity, '/check_state_validity')
cartesian = node.create_client(GetCartesianPath, '/compute_cartesian_path')
for c in (scene, ik, validity, cartesian):
    c.wait_for_service(timeout_sec=20.)

req = GetPlanningScene.Request()
req.components.components = (PlanningSceneComponents.ALLOWED_COLLISION_MATRIX |
                             PlanningSceneComponents.WORLD_OBJECT_NAMES)
res = call(scene, req)
acm = res.scene.allowed_collision_matrix
print('ACM entries:', len(acm.entry_names))
print('world objects:', [o.id for o in res.scene.world.collision_objects])
for probe in ('base_link', 'wheel_front_left_link', 'arena_shelf'):
    print(f'  {probe} in ACM:', probe in acm.entry_names)

state = RobotState(); state.is_diff = True
for group in ('left_arm', ''):
    r = call(validity, GetStateValidity.Request(group_name=group, robot_state=state))
    print(f'current state valid (group={group or "whole"}):', r.valid if r else 'TIMEOUT')
    if r:
        for c in list({(c.contact_body_1, c.contact_body_2) for c in r.contacts})[:8]:
            print('    contact:', c[0], '<->', c[1])

def make(y):
    p = Pose(); p.position = Point(x=float(X), y=float(y), z=float(Z))
    roll, pitch, yaw = math.pi, 0., -math.pi/2
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    p.orientation.x = sr*cp*cy-cr*sp*sy; p.orientation.y = cr*sp*cy+sr*cp*sy
    p.orientation.z = cr*cp*sy-sr*sp*cy; p.orientation.w = cr*cp*cy+sr*sp*sy
    return do_transform_pose(p, tf)

for label, y in (('pre(+40)', FRONT+LEAD+.04), ('grasp(-45)', FRONT+LEAD-.045),
                 ('grasp(-10)', FRONT+LEAD-.010)):
    pose = make(y)
    r = GetPositionIK.Request()
    r.ik_request.group_name = 'left_arm'
    r.ik_request.ik_link_name = 'gripper_left_grasping_link'
    r.ik_request.pose_stamped.header.frame_id = 'base_footprint'
    r.ik_request.pose_stamped.pose = pose
    r.ik_request.avoid_collisions = True
    r.ik_request.timeout.sec = 3
    ik_res = call(ik, r)
    code = ik_res.error_code.val if ik_res else -999
    whole = '-'
    if ik_res and code == 1:
        v = call(validity, GetStateValidity.Request(group_name='', robot_state=ik_res.solution))
        whole = (v.valid if v else 'TIMEOUT')
        if v and not v.valid:
            for c in list({(c.contact_body_1, c.contact_body_2) for c in v.contacts})[:5]:
                print('      contact:', c[0], '<->', c[1])
    print(f'{label:>12} base_y={pose.position.y:7.4f} reach={pose.position.y:.3f} ik={code} whole_valid={whole}')

node.destroy_node(); rclpy.shutdown()
