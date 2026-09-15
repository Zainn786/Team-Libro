"""Whole-robot validity + IK for the current pre/grasp poses at a low row."""
import math, sys
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Pose, Point
from moveit_msgs.srv import GetPositionIK, GetStateValidity
from moveit_msgs.msg import RobotState
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

X, FRONT, Z = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
LEAD = .011859

rclpy.init(); node = Node('probe_pre')
node.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
buf = Buffer(); TransformListener(buf, node)
for _ in range(600):
    rclpy.spin_once(node, timeout_sec=.05)
    if buf.can_transform('base_footprint', 'odom', Time()):
        break
else:
    raise SystemExit('no TF')
tf = buf.lookup_transform('base_footprint', 'odom', Time())

def call(c, r, t=25.):
    f = c.call_async(r); rclpy.spin_until_future_complete(node, f, timeout_sec=t); return f.result()

ik = node.create_client(GetPositionIK, '/compute_ik')
val = node.create_client(GetStateValidity, '/check_state_validity')
for c in (ik, val): c.wait_for_service(timeout_sec=20.)

r = call(val, GetStateValidity.Request(group_name='', robot_state=RobotState(is_diff=True)))
print('current whole-robot valid:', r.valid if r else 'TIMEOUT')
if r:
    for c in list({(c.contact_body_1, c.contact_body_2) for c in r.contacts})[:6]:
        print('   contact:', c[0], '<->', c[1])

def make(y):
    p = Pose(); p.position = Point(x=X, y=y, z=Z)
    roll, pitch, yaw = math.pi, 0., -math.pi/2
    cr,sr = math.cos(roll/2), math.sin(roll/2); cp,sp = math.cos(pitch/2), math.sin(pitch/2)
    cy,sy = math.cos(yaw/2), math.sin(yaw/2)
    p.orientation.x=sr*cp*cy-cr*sp*sy; p.orientation.y=cr*sp*cy+sr*cp*sy
    p.orientation.z=cr*cp*sy-sr*sp*cy; p.orientation.w=cr*cp*cy+sr*sp*sy
    return do_transform_pose(p, tf)

for label, y in (('pre', FRONT+LEAD+.04), ('grasp', FRONT+LEAD-.045)):
    pose = make(y)
    req = GetPositionIK.Request()
    req.ik_request.group_name='left_arm'
    req.ik_request.ik_link_name='gripper_left_grasping_link'
    req.ik_request.pose_stamped.header.frame_id='base_footprint'
    req.ik_request.pose_stamped.pose=pose
    req.ik_request.avoid_collisions=True; req.ik_request.timeout.sec=3
    res = call(ik, req)
    code = res.error_code.val if res else -999
    whole='-'
    if res and code==1:
        v = call(val, GetStateValidity.Request(group_name='', robot_state=res.solution))
        whole = v.valid if v else 'TIMEOUT'
        if v and not v.valid:
            for c in list({(c.contact_body_1,c.contact_body_2) for c in v.contacts})[:6]:
                print('     contact:', c[0], '<->', c[1])
    print(f'{label:>6} reach={pose.position.y:6.3f} z={pose.position.z:6.3f} ik={code} whole_valid={whole}')
node.destroy_node(); rclpy.shutdown()
