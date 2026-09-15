"""Is a straight-line carry to the transport pose plannable from the top-row post-extraction state?"""
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point
from moveit_msgs.srv import GetCartesianPath, GetPositionFK, GetPlanningScene
from moveit_msgs.msg import RobotState, PlanningSceneComponents
rclpy.init(); node = Node('probe_carry_line'); node.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
def call(c, r, t=30.):
    f = c.call_async(r); rclpy.spin_until_future_complete(node, f, timeout_sec=t); return f.result()
cart = node.create_client(GetCartesianPath, '/compute_cartesian_path')
fk = node.create_client(GetPositionFK, '/compute_fk')
scene = node.create_client(GetPlanningScene, '/get_planning_scene')
for c in (cart, fk, scene): print('service', c.srv_name, c.wait_for_service(timeout_sec=15.))
req = GetPlanningScene.Request(); req.components.components = PlanningSceneComponents.ROBOT_STATE | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
sc = call(scene, req).scene
print('attached:', [a.object.id for a in sc.robot_state.attached_collision_objects])
fr = GetPositionFK.Request(); fr.header.frame_id = 'base_footprint'; fr.fk_link_names = ['gripper_left_grasping_link']; fr.robot_state = sc.robot_state
tip = call(fk, fr).pose_stamped[0].pose.position
print(f'tip now (base_footprint): x={tip.x:.3f} y={tip.y:.3f} z={tip.z:.3f}')
def pose(x, y, z):
    p = Pose(); p.position = Point(x=x, y=y, z=z)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = 1., 0., 0., 0.
    return p
target = pose(.38, .28, 1.05)
variants = {
    'V1 direct': [target],
    'V2 down then across': [pose(tip.x, tip.y, 1.05), target],
    'V3 across then down': [pose(.38, .28, tip.z), target],
    'V4 back, down, across': [pose(.38, tip.y, tip.z), pose(.38, tip.y, 1.05), target],
}
for label, waypoints in variants.items():
    r = GetCartesianPath.Request(); r.header.frame_id = 'base_footprint'; r.group_name = 'left_arm'
    r.link_name = 'gripper_left_grasping_link'; r.start_state.is_diff = True
    r.waypoints = waypoints; r.max_step = .005; r.jump_threshold = 0.; r.avoid_collisions = True
    res = call(cart, r)
    print(f'{label:24s} fraction={res.fraction:.3f} points={len(res.solution.joint_trajectory.points)}')
node.destroy_node(); rclpy.shutdown()
