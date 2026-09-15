"""Can the arm reach the 'book hanging straight down' orientations (wrist pitched +/-90 deg)?"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point
from moveit_msgs.srv import GetPositionIK, GetPlanningScene, GetPositionFK
from moveit_msgs.msg import PlanningSceneComponents
rclpy.init(); node = Node('probe_hanging'); node.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
def call(c, r, t=20.):
    f = c.call_async(r); rclpy.spin_until_future_complete(node, f, timeout_sec=t); return f.result()
ik = node.create_client(GetPositionIK, '/compute_ik'); scene = node.create_client(GetPlanningScene, '/get_planning_scene')
for c in (ik, scene): c.wait_for_service(timeout_sec=15.)
req = GetPlanningScene.Request(); req.components.components = PlanningSceneComponents.ROBOT_STATE | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
sc = call(scene, req).scene
js = dict(zip(sc.robot_state.joint_state.name, sc.robot_state.joint_state.position))
print('torso:', round(js.get('torso_lift_joint', float('nan')), 3), ' attached:', [a.object.id for a in sc.robot_state.attached_collision_objects])
orients = {'level (current carry)': (1., 0., 0., 0.), 'pitched +90': (.7071068, 0., -.7071068, 0.), 'pitched -90': (.7071068, 0., .7071068, 0.)}
targets = [('post-extraction', .45, 0., 1.45), ('carry', .38, .28, 1.20), ('carry high', .38, .28, 1.35),
           ('bin .70', .70, 0., 1.20), ('bin .75', .75, 0., 1.30), ('bin .80', .80, 0., 1.30)]
print(f'{"target":>16} ' + ' '.join(f'{k:>22}' for k in orients))
for label, x, y, z in targets:
    row = []
    for q in orients.values():
        p = Pose(); p.position = Point(x=x, y=y, z=z)
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
        ok = 0
        for _ in range(3):
            r = GetPositionIK.Request(); r.ik_request.group_name = 'left_arm'; r.ik_request.ik_link_name = 'gripper_left_grasping_link'
            r.ik_request.pose_stamped.header.frame_id = 'base_footprint'; r.ik_request.pose_stamped.pose = p
            r.ik_request.avoid_collisions = True; r.ik_request.timeout.sec = 1
            res = call(ik, r); ok += int(bool(res) and res.error_code.val == 1)
        row.append(f'{ok}/3')
    print(f'{label:>16} ' + ' '.join(f'{v:>22}' for v in row))
node.destroy_node(); rclpy.shutdown()
