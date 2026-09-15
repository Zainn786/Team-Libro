"""Sweep pre-grasp stand-off and torso height for IK reachability at a given row."""
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
rclpy.init(); node = Node('probe_sweep')
node.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
buf = Buffer(); TransformListener(buf, node)
for _ in range(600):
    rclpy.spin_once(node, timeout_sec=.05)
    if buf.can_transform('base_footprint', 'odom', Time()): break
else: raise SystemExit('no TF')
tf = buf.lookup_transform('base_footprint', 'odom', Time())
def call(c, r, t=20.):
    f=c.call_async(r); rclpy.spin_until_future_complete(node,f,timeout_sec=t); return f.result()
ik = node.create_client(GetPositionIK, '/compute_ik'); ik.wait_for_service(timeout_sec=20.)
def make(y):
    p=Pose(); p.position=Point(x=X,y=y,z=Z)
    roll,pitch,yaw=math.pi,0.,-math.pi/2
    cr,sr=math.cos(roll/2),math.sin(roll/2); cp,sp=math.cos(pitch/2),math.sin(pitch/2)
    cy,sy=math.cos(yaw/2),math.sin(yaw/2)
    p.orientation.x=sr*cp*cy-cr*sp*sy; p.orientation.y=cr*sp*cy+sr*cp*sy
    p.orientation.z=cr*cp*sy-sr*sp*cy; p.orientation.w=cr*cp*cy+sr*sp*sy
    return do_transform_pose(p,tf)
print(f'{"standoff":>9} {"fwd(x)":>8} {"ik_coll":>8} {"ik_free":>8}')
for mm in (-45,-20,0,20,40,60,80,100,140,180):
    pose = make(FRONT+LEAD+mm/1000.)
    codes=[]
    for avoid in (True, False):
        r=GetPositionIK.Request()
        r.ik_request.group_name='left_arm'
        r.ik_request.ik_link_name='gripper_left_grasping_link'
        r.ik_request.pose_stamped.header.frame_id='base_footprint'
        r.ik_request.pose_stamped.pose=pose
        r.ik_request.avoid_collisions=avoid; r.ik_request.timeout.sec=3
        res=call(ik,r); codes.append(res.error_code.val if res else -999)
    print(f'{mm:>9} {pose.position.x:8.3f} {codes[0]:>8} {codes[1]:>8}')
node.destroy_node(); rclpy.shutdown()
