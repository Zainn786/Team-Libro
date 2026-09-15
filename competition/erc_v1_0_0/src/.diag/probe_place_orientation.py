"""Reach over the bin with the book rotated 90 deg about vertical (grasp roll pi, yaw pi/2)."""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point
from moveit_msgs.srv import GetPositionIK
rclpy.init(); node = Node('probe_place_orientation'); node.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
def call(c, r, t=20.):
    f = c.call_async(r); rclpy.spin_until_future_complete(node, f, timeout_sec=t); return f.result()
ik = node.create_client(GetPositionIK, '/compute_ik'); ik.wait_for_service(timeout_sec=15.)
def base_pose(d, z, yaw):
    # Same construction as Manipulator._base_pose: roll pi, rotated about vertical by yaw.
    p = Pose(); p.position = Point(x=d, y=0., z=z)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = math.cos(yaw/2), math.sin(yaw/2), 0., 0.
    return p
def solved(pose, attempts=4):
    ok = 0
    for _ in range(attempts):
        r = GetPositionIK.Request(); r.ik_request.group_name = 'left_arm'
        r.ik_request.ik_link_name = 'gripper_left_grasping_link'
        r.ik_request.pose_stamped.header.frame_id = 'base_footprint'
        r.ik_request.pose_stamped.pose = pose; r.ik_request.avoid_collisions = True; r.ik_request.timeout.sec = 1
        res = call(ik, r); ok += int(bool(res) and res.error_code.val == 1)
    return ok
for label, yaw in (('book rotated (yaw +pi/2)', math.pi/2), ('book rotated (yaw -pi/2)', -math.pi/2), ('book straight (yaw 0)', 0.)):
    print(f'--- {label} ---')
    print(f'{"forward":>8} {"z=1.15":>7} {"z=1.20":>7} {"z=1.25":>7}')
    for d in (.65, .70, .75, .80, .85):
        print(f'{d:8.2f} ' + ' '.join(f'{solved(base_pose(d, z, yaw)):>5}/4' for z in (1.15, 1.20, 1.25)))
node.destroy_node(); rclpy.shutdown()
