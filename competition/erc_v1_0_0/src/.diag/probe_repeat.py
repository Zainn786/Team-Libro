"""Walk the insertion path and report the first whole-robot collision and its links."""
import math, sys
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Pose, Point
from moveit_msgs.srv import GetCartesianPath, GetStateValidity
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

X, FRONT, Z = map(float, sys.argv[1:4])
LEAD, OVERLAP = .011859, .045
rclpy.init(); node = Node('probe_path')
node.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
buf = Buffer(); TransformListener(buf, node)
live = {}
node.create_subscription(JointState, '/joint_states', lambda m: live.update(zip(m.name, m.position)), 10)
for _ in range(600):
    rclpy.spin_once(node, timeout_sec=.05)
    if live and buf.can_transform('base_footprint', 'odom', Time()): break
else: raise SystemExit('no TF/joints')
tf = buf.lookup_transform('base_footprint', 'odom', Time())
try:
    tip = buf.lookup_transform('base_footprint', 'gripper_left_grasping_link', Time()).transform.translation
except Exception:
    tip = None
print(f'torso={live.get("torso_lift_joint", float("nan")):.3f}  gripper={live.get("gripper_left_finger_joint", float("nan")):.4f}')
print(f'tip now (base_footprint): x={tip.x:.3f} y={tip.y:.3f} z={tip.z:.3f}' if tip else 'tip TF unavailable')
def call(c, r, t=30.):
    f = c.call_async(r); rclpy.spin_until_future_complete(node, f, timeout_sec=t); return f.result()
cart = node.create_client(GetCartesianPath, '/compute_cartesian_path')
val = node.create_client(GetStateValidity, '/check_state_validity')
for c in (cart, val): c.wait_for_service(timeout_sec=20.)
p = Pose(); p.position = Point(x=X, y=FRONT+LEAD-OVERLAP, z=Z)
roll, yaw = math.pi, -math.pi/2
p.orientation.x = math.sin(roll/2)*math.cos(yaw/2); p.orientation.y = math.sin(roll/2)*math.sin(yaw/2)
p.orientation.z = math.cos(roll/2)*math.sin(yaw/2); p.orientation.w = math.cos(roll/2)*math.cos(yaw/2)
goal = do_transform_pose(p, tf)
print(f'grasp goal (base_footprint): x={goal.position.x:.3f} y={goal.position.y:.3f} z={goal.position.z:.3f}')
fractions=[]
for trial in range(10):
    r = GetCartesianPath.Request(); r.header.frame_id = 'base_footprint'; r.group_name = 'left_arm'
    r.link_name = 'gripper_left_grasping_link'; r.start_state.is_diff = True
    r.waypoints = [goal]; r.max_step = .005; r.jump_threshold = 0.; r.avoid_collisions = True
    res = call(cart, r)
    fractions.append(round(res.fraction,3) if res else None)
print('fractions over 10 identical requests:', fractions)
print('short (<.84):', sum(1 for f in fractions if f is not None and f < .84), 'of 10')
node.destroy_node(); rclpy.shutdown()
