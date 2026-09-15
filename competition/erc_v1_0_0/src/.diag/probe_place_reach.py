"""How far forward can the grasp frame reach over the bin, holding the book level?"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK, GetPlanningScene
from moveit_msgs.msg import PlanningSceneComponents
rclpy.init(); node = Node('probe_place_reach'); node.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
def call(c, r, t=20.):
    f = c.call_async(r); rclpy.spin_until_future_complete(node, f, timeout_sec=t); return f.result()
ik = node.create_client(GetPositionIK, '/compute_ik'); ik.wait_for_service(timeout_sec=15.)
scene = node.create_client(GetPlanningScene, '/get_planning_scene'); scene.wait_for_service(timeout_sec=15.)
req = GetPlanningScene.Request(); req.components.components = PlanningSceneComponents.ROBOT_STATE
js = dict(zip(call(scene, req).scene.robot_state.joint_state.name, call(scene, req).scene.robot_state.joint_state.position))
print('torso now:', round(js.get('torso_lift_joint', float('nan')), 3))
print(f'{"forward":>8} {"z=1.10":>8} {"z=1.20":>8}   (collision-free IK solutions out of 4 attempts)')
for d in (.55, .60, .65, .70, .75, .80, .85):
    row = []
    for z in (1.10, 1.20):
        p = Pose(); p.position = Point(x=d, y=0., z=z)
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = 1., 0., 0., 0.
        ok = 0
        for _ in range(4):
            r = GetPositionIK.Request(); r.ik_request.group_name = 'left_arm'
            r.ik_request.ik_link_name = 'gripper_left_grasping_link'
            r.ik_request.pose_stamped.header.frame_id = 'base_footprint'
            r.ik_request.pose_stamped.pose = p; r.ik_request.avoid_collisions = True; r.ik_request.timeout.sec = 1
            res = call(ik, r); ok += int(bool(res) and res.error_code.val == 1)
        row.append(ok)
    print(f'{d:8.2f} {row[0]:>6}/4 {row[1]:>6}/4')
node.destroy_node(); rclpy.shutdown()
