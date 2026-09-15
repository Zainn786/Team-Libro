"""Retry IK from many random seed states to separate solver weakness from reach."""
import math, sys, random
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Pose, Point
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

X, FRONT, Z = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
LEAD=.011859
ARM=[f'arm_left_{i}_joint' for i in range(1,8)]
rclpy.init(); node=Node('probe_seed')
node.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
buf=Buffer(); TransformListener(buf,node)
live={}
node.create_subscription(JointState,'/joint_states',
                         lambda m: live.update(zip(m.name,m.position)),10)
for _ in range(600):
    rclpy.spin_once(node,timeout_sec=.05)
    if live and buf.can_transform('base_footprint','odom',Time()): break
else: raise SystemExit('no TF/joints')
tf=buf.lookup_transform('base_footprint','odom',Time())
print('torso now:', round(live.get('torso_lift_joint',float('nan')),4))
def call(c,r,t=20.):
    f=c.call_async(r); rclpy.spin_until_future_complete(node,f,timeout_sec=t); return f.result()
ik=node.create_client(GetPositionIK,'/compute_ik'); ik.wait_for_service(timeout_sec=20.)
def make(y,Z=None):
    p=Pose(); p.position=Point(x=X,y=y,z=Z)
    roll,pitch,yaw=math.pi,0.,-math.pi/2
    cr,sr=math.cos(roll/2),math.sin(roll/2); cp,sp=math.cos(pitch/2),math.sin(pitch/2)
    cy,sy=math.cos(yaw/2),math.sin(yaw/2)
    p.orientation.x=sr*cp*cy-cr*sp*sy; p.orientation.y=cr*sp*cy+sr*cp*sy
    p.orientation.z=cr*cp*sy-sr*sp*cy; p.orientation.w=cr*cp*cy+sr*sp*sy
    return do_transform_pose(p,tf)
random.seed(0)
ZS=[float(v) for v in sys.argv[4].split(',')] if len(sys.argv)>4 else [Z]
print(f'{"z":>7} {"standoff":>9} {"fwd":>7} {"solved/seeds":>13}')
for Z in ZS:
  for mm in (-45,40):
    pose=make(FRONT+LEAD+mm/1000.,Z)
    solved=0; tries=12
    for attempt in range(tries):
        r=GetPositionIK.Request()
        r.ik_request.group_name='left_arm'
        r.ik_request.ik_link_name='gripper_left_grasping_link'
        r.ik_request.pose_stamped.header.frame_id='base_footprint'
        r.ik_request.pose_stamped.pose=pose
        r.ik_request.avoid_collisions=False
        r.ik_request.timeout.sec=1
        if attempt:
            state=RobotState(); state.is_diff=False
            js=JointState()
            js.name=list(live.keys()); js.position=[live[n] for n in js.name]
            for i,name in enumerate(ARM):
                js.position[js.name.index(name)]=random.uniform(-2.0,2.0)
            state.joint_state=js
            r.ik_request.robot_state=state
        res=call(ik,r)
        if res and res.error_code.val==1: solved+=1
    print(f'{Z:7.3f} {mm:>9} {pose.position.x:7.3f} {solved:>6}/{tries}')
node.destroy_node(); rclpy.shutdown()
