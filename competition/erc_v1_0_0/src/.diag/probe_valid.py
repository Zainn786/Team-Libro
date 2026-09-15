"""Is the current state in collision, and with what? No TF needed."""
import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetStateValidity
from moveit_msgs.msg import RobotState

rclpy.init()
node = Node('probe_valid')
client = node.create_client(GetStateValidity, '/check_state_validity')
client.wait_for_service(timeout_sec=20.)
for group in ('left_arm', ''):
    state = RobotState(); state.is_diff = True
    request = GetStateValidity.Request()
    request.group_name = group
    request.robot_state = state
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=30.)
    result = future.result()
    print(f'group={group or "<whole robot>"} valid={result.valid} contacts={len(result.contacts)}')
    seen = set()
    for c in result.contacts:
        key = (c.contact_body_1, c.contact_body_2)
        if key not in seen:
            seen.add(key)
            print('   ', c.contact_body_1, '<->', c.contact_body_2)
node.destroy_node(); rclpy.shutdown()
