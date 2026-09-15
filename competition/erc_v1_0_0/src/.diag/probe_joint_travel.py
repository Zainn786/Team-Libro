"""Joint-space cost of the ~80 mm extraction segment, via the reverse Cartesian path from here."""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Pose, Point
from moveit_msgs.srv import GetCartesianPath
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

GX, FRONT, GZ = -1.005337740206651, -2.79483208834697, 0.5981726409045302
GRASP_Y = FRONT + .011859 - .045
rclpy.init(); node = Node('probe_joint_travel')
node.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
buf = Buffer(); TransformListener(buf, node)
for _ in range(600):
    rclpy.spin_once(node, timeout_sec=.05)
    if buf.can_transform('base_footprint', 'odom', Time()): break
else: raise SystemExit('no TF')
tf = buf.lookup_transform('base_footprint', 'odom', Time())
cart = node.create_client(GetCartesianPath, '/compute_cartesian_path'); print('service', cart.wait_for_service(timeout_sec=15.))
def pose(y, z):
    p = Pose(); p.position = Point(x=GX, y=y, z=z)
    roll, yaw = math.pi, -math.pi/2
    p.orientation.x = math.sin(roll/2)*math.cos(yaw/2); p.orientation.y = math.sin(roll/2)*math.sin(yaw/2)
    p.orientation.z = math.cos(roll/2)*math.sin(yaw/2); p.orientation.w = math.cos(roll/2)*math.cos(yaw/2)
    return do_transform_pose(p, tf)
for label, waypoints in (('segment2 reverse (seg2 end -> seg1 end)', [pose(GRASP_Y+.08, GZ)]),
                         ('segment1 reverse (-> grasp)', [pose(GRASP_Y+.08, GZ), pose(GRASP_Y, GZ)])):
    r = GetCartesianPath.Request(); r.header.frame_id = 'base_footprint'; r.group_name = 'left_arm'
    r.link_name = 'gripper_left_grasping_link'; r.start_state.is_diff = True
    r.waypoints = waypoints; r.max_step = .005; r.jump_threshold = 0.; r.avoid_collisions = False
    f = cart.call_async(r); rclpy.spin_until_future_complete(node, f, timeout_sec=30.); res = f.result()
    jt = res.solution.joint_trajectory
    q = np.array([p.positions for p in jt.points])
    if len(q) < 2:
        print(label, 'fraction', res.fraction, 'no points'); continue
    steps = np.abs(np.diff(q, axis=0))
    print(f'{label}: fraction={res.fraction:.3f} points={len(q)}')
    for i, name in enumerate(jt.joint_names):
        print(f'   {name:18s} total travel={np.sum(steps[:, i]):.3f} rad  net={q[-1, i]-q[0, i]:+.3f}  max step={np.max(steps[:, i]):.3f}')
node.destroy_node(); rclpy.shutdown()
