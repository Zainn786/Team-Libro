"""Does padding the forearm keep insertion poses solvable while flagging the real shelf contact?"""
import math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Pose, Point
from moveit_msgs.srv import ApplyPlanningScene, GetPositionIK, GetStateValidity
from moveit_msgs.msg import LinkPadding, RobotState
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

GX, FRONT, GZ = 2.0985003059104375, -2.817142121582593, 0.5906468231314059
LEAD, OVERLAP = .011859, .102
rclpy.init(); node = Node('probe_padding')
node.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
buf = Buffer(); TransformListener(buf, node)
for _ in range(600):
    rclpy.spin_once(node, timeout_sec=.05)
    if buf.can_transform('base_footprint', 'odom', Time()): break
else: raise SystemExit('no TF')
tf = buf.lookup_transform('base_footprint', 'odom', Time())
def call(c, r, t=25.):
    f = c.call_async(r); rclpy.spin_until_future_complete(node, f, timeout_sec=t); return f.result()
apply = node.create_client(ApplyPlanningScene, '/apply_planning_scene')
ik = node.create_client(GetPositionIK, '/compute_ik')
val = node.create_client(GetStateValidity, '/check_state_validity')
for c in (apply, ik, val): print('service', c.srv_name, c.wait_for_service(timeout_sec=15.))
def pad(links, value):
    r = ApplyPlanningScene.Request(); r.scene.is_diff = True; r.scene.robot_state.is_diff = True
    r.scene.link_padding = [LinkPadding(link_name=l, padding=value) for l in links]
    return call(apply, r).success
def make(y):
    p = Pose(); p.position = Point(x=GX, y=y, z=GZ)
    roll, yaw = math.pi, -math.pi/2
    p.orientation.x = math.sin(roll/2)*math.cos(yaw/2); p.orientation.y = math.sin(roll/2)*math.sin(yaw/2)
    p.orientation.z = math.cos(roll/2)*math.sin(yaw/2); p.orientation.w = math.cos(roll/2)*math.cos(yaw/2)
    return do_transform_pose(p, tf)
poses = {'pre(+40)': make(FRONT + LEAD + .04), 'grasp(-102)': make(FRONT + LEAD - OVERLAP)}
ARM = ['arm_left_1_link', 'arm_left_2_link', 'arm_left_3_link', 'arm_left_4_link']
for value in (0.0, .03, .05):
    print(f'--- padding arm_left_1..4 = {value:.2f} m (applied: {pad(ARM, value)}) ---')
    v = call(val, GetStateValidity.Request(group_name='', robot_state=RobotState(is_diff=True)))
    print('   current (touching) state valid:', v.valid,
          sorted({(c.contact_body_1, c.contact_body_2) for c in v.contacts})[:3])
    for label, pose in poses.items():
        solved = 0
        for attempt in range(4):
            r = GetPositionIK.Request(); r.ik_request.group_name = 'left_arm'
            r.ik_request.ik_link_name = 'gripper_left_grasping_link'
            r.ik_request.pose_stamped.header.frame_id = 'base_footprint'
            r.ik_request.pose_stamped.pose = pose; r.ik_request.avoid_collisions = True
            r.ik_request.timeout.sec = 2
            res = call(ik, r); solved += int(bool(res) and res.error_code.val == 1)
        print(f'   {label:12s} collision-free IK solved {solved}/4')
print('restore production padding (arm_left_1..3 at .03, arm_left_4 at 0):',
      pad(ARM[:3], .03), pad(ARM[3:], 0.))
node.destroy_node(); rclpy.shutdown()
