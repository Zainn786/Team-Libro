"""Which carry height gives a fully plannable straight line, and what stops the low one?"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetCartesianPath, GetPositionFK, GetPlanningScene, GetStateValidity
from moveit_msgs.msg import RobotState, PlanningSceneComponents
rclpy.init(); node = Node('probe_carry_height'); node.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
def call(c, r, t=30.):
    f = c.call_async(r); rclpy.spin_until_future_complete(node, f, timeout_sec=t); return f.result()
cart = node.create_client(GetCartesianPath, '/compute_cartesian_path')
fk = node.create_client(GetPositionFK, '/compute_fk')
scene = node.create_client(GetPlanningScene, '/get_planning_scene')
val = node.create_client(GetStateValidity, '/check_state_validity')
for c in (cart, fk, scene, val): c.wait_for_service(timeout_sec=15.)
req = GetPlanningScene.Request(); req.components.components = PlanningSceneComponents.ROBOT_STATE | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
sc = call(scene, req).scene
fr = GetPositionFK.Request(); fr.header.frame_id = 'base_footprint'; fr.fk_link_names = ['gripper_left_grasping_link']; fr.robot_state = sc.robot_state
tip = call(fk, fr).pose_stamped[0].pose.position
print(f'tip now: x={tip.x:.3f} y={tip.y:.3f} z={tip.z:.3f}')
def pose(x, y, z):
    p = Pose(); p.position = Point(x=x, y=y, z=z)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = 1., 0., 0., 0.
    return p
def cartesian(waypoints, avoid=True):
    r = GetCartesianPath.Request(); r.header.frame_id = 'base_footprint'; r.group_name = 'left_arm'
    r.link_name = 'gripper_left_grasping_link'; r.start_state.is_diff = True
    r.waypoints = waypoints; r.max_step = .005; r.jump_threshold = 0.; r.avoid_collisions = avoid
    return call(cart, r)
print(f'{"carry z":>8} {"direct":>8} {"across+down":>12}')
for z in (1.05, 1.15, 1.20, 1.30, 1.40):
    target = pose(.38, .28, z)
    d = cartesian([target]).fraction
    a = cartesian([pose(.38, .28, tip.z), target]).fraction
    print(f'{z:8.2f} {d:8.3f} {a:12.3f}')
# Why does across+down to 1.05 stop? Plan it ignoring collisions, then check each state.
res = cartesian([pose(.38, .28, tip.z), pose(.38, .28, 1.05)], avoid=False)
jt = res.solution.joint_trajectory
print(f'ignoring collisions: fraction={res.fraction:.3f} points={len(jt.points)}')
first = None; seen = {}
for i, pt in enumerate(jt.points):
    js = JointState(); js.name = list(jt.joint_names); js.position = list(pt.positions)
    st = RobotState(); st.is_diff = True; st.joint_state = js
    v = call(val, GetStateValidity.Request(group_name='', robot_state=st))
    if v and not v.valid:
        first = i if first is None else first
        for c in v.contacts: seen.setdefault((c.contact_body_1, c.contact_body_2), i)
print('first colliding step:', first, 'of', len(jt.points))
for k, i in sorted(seen.items(), key=lambda kv: kv[1])[:8]: print('   from step', i, k)
node.destroy_node(); rclpy.shutdown()
