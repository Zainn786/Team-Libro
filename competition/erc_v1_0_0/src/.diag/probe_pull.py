"""Why does the final extraction segment (pull + lift) come up short? Walk the path."""
import math
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Pose, Point
from moveit_msgs.srv import GetCartesianPath, GetStateValidity, GetPlanningScene
from moveit_msgs.msg import RobotState, PlanningSceneComponents
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose

GX, FRONT, GZ = -2.0507794830343657, -2.7756578967683954, 0.5979407817766735
PULL_Y, PULL_Z = FRONT + .160 + .02, GZ + .025
rclpy.init(); node = Node('probe_pull')
node.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
buf = Buffer(); TransformListener(buf, node)
for _ in range(600):
    rclpy.spin_once(node, timeout_sec=.05)
    if buf.can_transform('base_footprint', 'odom', Time()): break
else: raise SystemExit('no TF')
tf = buf.lookup_transform('base_footprint', 'odom', Time())
def call(c, r, t=30.):
    f = c.call_async(r); rclpy.spin_until_future_complete(node, f, timeout_sec=t); return f.result()
cart = node.create_client(GetCartesianPath, '/compute_cartesian_path')
val = node.create_client(GetStateValidity, '/check_state_validity')
scene = node.create_client(GetPlanningScene, '/get_planning_scene')
for c in (cart, val, scene): print('service', c.srv_name, c.wait_for_service(timeout_sec=15.))
req = GetPlanningScene.Request(); req.components.components = (PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS | PlanningSceneComponents.WORLD_OBJECT_NAMES)
sc = call(scene, req)
print('attached:', [a.object.id for a in sc.scene.robot_state.attached_collision_objects], ' world:', [o.id for o in sc.scene.world.collision_objects])
v = call(val, GetStateValidity.Request(group_name='', robot_state=RobotState(is_diff=True)))
print('current whole-robot valid:', v.valid, [(c.contact_body_1, c.contact_body_2) for c in v.contacts[:6]])
p = Pose(); p.position = Point(x=GX, y=PULL_Y, z=PULL_Z)
roll, yaw = math.pi, -math.pi/2
p.orientation.x = math.sin(roll/2)*math.cos(yaw/2); p.orientation.y = math.sin(roll/2)*math.sin(yaw/2)
p.orientation.z = math.cos(roll/2)*math.sin(yaw/2); p.orientation.w = math.cos(roll/2)*math.cos(yaw/2)
goal = do_transform_pose(p, tf)
for avoid in (True, False):
    r = GetCartesianPath.Request(); r.header.frame_id = 'base_footprint'; r.group_name = 'left_arm'
    r.link_name = 'gripper_left_grasping_link'; r.start_state.is_diff = True
    r.waypoints = [goal]; r.max_step = .005; r.jump_threshold = 0.; r.avoid_collisions = avoid
    res = call(cart, r)
    print(f'avoid_collisions={avoid}: fraction={res.fraction:.3f} points={len(res.solution.joint_trajectory.points)}')
traj = res.solution.joint_trajectory
seen = {}; first = None
for i, pt in enumerate(traj.points):
    js = JointState(); js.name = list(traj.joint_names); js.position = list(pt.positions)
    st = RobotState(); st.is_diff = True; st.joint_state = js
    vv = call(val, GetStateValidity.Request(group_name='', robot_state=st))
    if vv and not vv.valid:
        first = i if first is None else first
        for c in vv.contacts: seen.setdefault((c.contact_body_1, c.contact_body_2), i)
print('first invalid step (ignoring-collision path):', first, 'of', len(traj.points))
for k, i in sorted(seen.items(), key=lambda kv: kv[1])[:10]: print('   from step', i, k)
node.destroy_node(); rclpy.shutdown()
