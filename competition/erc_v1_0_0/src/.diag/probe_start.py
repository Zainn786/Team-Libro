"""Is the CURRENT robot state valid, and where is the tip relative to the waypoint?"""
import math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from moveit_msgs.srv import GetStateValidity, GetCartesianPath
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

rclpy.init()
node = Node('probe_start')
node.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
buffer = Buffer(); TransformListener(buffer, node)
joints = {}
node.create_subscription(JointState, '/joint_states',
                         lambda m: joints.update(zip(m.name, m.position)), 10)
for _ in range(1200):
    rclpy.spin_once(node, timeout_sec=.05)
    if joints and buffer.can_transform('base_footprint', 'gripper_left_grasping_link', Time()):
        break
else:
    raise SystemExit('TF never became available')

tip = buffer.lookup_transform('base_footprint', 'gripper_left_grasping_link', Time())
t = tip.transform.translation; r = tip.transform.rotation
print(f'tip in base_footprint: x={t.x:.4f} y={t.y:.4f} z={t.z:.4f}')
print(f'tip quat: x={r.x:.4f} y={r.y:.4f} z={r.z:.4f} w={r.w:.4f}')
print('gripper joint:', round(joints.get('gripper_left_finger_joint', float("nan")), 4))
print('torso:', round(joints.get('torso_lift_joint', float("nan")), 4))
print('arm_left:', [round(joints.get(f'arm_left_{i}_joint', float("nan")), 3) for i in range(1, 8)])

validity = node.create_client(GetStateValidity, '/check_state_validity')
validity.wait_for_service(timeout_sec=20.)
state = RobotState(); state.is_diff = True
request = GetStateValidity.Request(); request.group_name = 'left_arm'; request.robot_state = state
future = validity.call_async(request)
rclpy.spin_until_future_complete(node, future, timeout_sec=30.)
result = future.result()
print('current state valid:', result.valid)
for c in result.contacts[:10]:
    print('  contact:', c.contact_body_1, '<->', c.contact_body_2)

# Zero-length Cartesian request: ask for the CURRENT pose as the only waypoint.
cart = node.create_client(GetCartesianPath, '/compute_cartesian_path')
cart.wait_for_service(timeout_sec=20.)
here = Pose()
here.position = Point(x=t.x, y=t.y, z=t.z); here.orientation = r
req = GetCartesianPath.Request()
req.header.frame_id = 'base_footprint'; req.group_name = 'left_arm'
req.link_name = 'gripper_left_grasping_link'; req.start_state.is_diff = True
req.waypoints = [here]; req.max_step = .005; req.jump_threshold = 0.
req.avoid_collisions = True
future = cart.call_async(req)
rclpy.spin_until_future_complete(node, future, timeout_sec=30.)
print('fraction to CURRENT pose:', future.result().fraction,
      'error', future.result().error_code.val)

req.avoid_collisions = False
future = cart.call_async(req)
rclpy.spin_until_future_complete(node, future, timeout_sec=30.)
print('fraction to CURRENT pose, collisions ignored:', future.result().fraction)
node.destroy_node(); rclpy.shutdown()
