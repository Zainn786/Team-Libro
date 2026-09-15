"""Planning-scene view of the shelf: object pose, ACM entry for arm_left_3, current validity."""
import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPlanningScene, GetStateValidity
from moveit_msgs.msg import PlanningSceneComponents, RobotState
rclpy.init(); node = Node('probe_scene')
def call(c, r, t=20.):
    f = c.call_async(r); rclpy.spin_until_future_complete(node, f, timeout_sec=t); return f.result()
scene = node.create_client(GetPlanningScene, '/get_planning_scene')
val = node.create_client(GetStateValidity, '/check_state_validity')
print('services:', scene.wait_for_service(timeout_sec=15.), val.wait_for_service(timeout_sec=15.))
req = GetPlanningScene.Request()
req.components.components = (PlanningSceneComponents.WORLD_OBJECT_GEOMETRY |
                              PlanningSceneComponents.ALLOWED_COLLISION_MATRIX |
                              PlanningSceneComponents.ROBOT_STATE)
res = call(scene, req)
if res is None: raise SystemExit('get_planning_scene timeout')
sc = res.scene
print('planning frame:', sc.robot_model_name, '/', sc.world.collision_objects[0].header.frame_id if sc.world.collision_objects else '-')
for o in sc.world.collision_objects:
    mp = o.mesh_poses[0] if o.mesh_poses else o.pose
    print(f'  {o.id:22s} pos=({mp.position.x:.3f},{mp.position.y:.3f},{mp.position.z:.3f}) '
          f'q=({mp.orientation.x:.3f},{mp.orientation.y:.3f},{mp.orientation.z:.3f},{mp.orientation.w:.3f}) '
          f'verts={len(o.meshes[0].vertices) if o.meshes else 0}')
acm = sc.allowed_collision_matrix; idx = {n: i for i, n in enumerate(acm.entry_names)}
for link in ('arm_left_3_link', 'arm_left_4_link', 'arm_left_5_link', 'gripper_left_fingertip_left_link'):
    a = idx.get('arena_shelf'); b = idx.get(link)
    allowed = acm.entry_values[a].enabled[b] if a is not None and b is not None else 'n/a'
    print(f'  ACM arena_shelf <-> {link}: allowed={allowed}')
v = call(val, GetStateValidity.Request(group_name='', robot_state=RobotState(is_diff=True)))
print('current whole-robot valid:', v.valid if v else 'TIMEOUT')
if v:
    for c in list({(c.contact_body_1, c.contact_body_2) for c in v.contacts})[:8]:
        print('   contact:', c[0], '<->', c[1])
node.destroy_node(); rclpy.shutdown()
