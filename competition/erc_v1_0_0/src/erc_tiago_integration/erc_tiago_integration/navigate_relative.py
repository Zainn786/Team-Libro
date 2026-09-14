#!/usr/bin/env python3

import math
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node


def _yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class RelativeNavigator(Node):
    """Send a bounded relative NavigateToPose goal in the odom frame."""

    def __init__(self):
        super().__init__('tiago_relative_navigator')
        self.declare_parameter('forward', 0.50)
        self.declare_parameter('lateral', 0.0)
        self.declare_parameter('yaw', 0.0)
        self.declare_parameter('timeout', 45.0)
        self.forward = float(self.get_parameter('forward').value)
        self.lateral = float(self.get_parameter('lateral').value)
        self.yaw_offset = float(self.get_parameter('yaw').value)
        self.timeout = float(self.get_parameter('timeout').value)

        if math.hypot(self.forward, self.lateral) > 2.0:
            raise ValueError('relative navigation distance must not exceed 2.0 m')
        if abs(self.yaw_offset) > math.pi:
            raise ValueError('relative yaw must be within [-pi, pi]')

        self.odom = None
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.stop_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

    def _on_odom(self, message):
        self.odom = message

    def _stop(self):
        for _ in range(5):
            self.stop_publisher.publish(Twist())
            rclpy.spin_once(self, timeout_sec=0.02)

    def _wait_for_odom(self):
        deadline = time.monotonic() + 10.0
        while rclpy.ok() and self.odom is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.odom is not None

    def run(self):
        if not self._wait_for_odom():
            self.get_logger().error('No /odom received; refusing to navigate')
            return False
        if not self.client.wait_for_server(timeout_sec=15.0):
            self.get_logger().error('Nav2 /navigate_to_pose action is unavailable')
            return False

        pose = self.odom.pose.pose
        start_yaw = _yaw_from_quaternion(pose.orientation)
        goal_x = (pose.position.x + self.forward * math.cos(start_yaw)
                  - self.lateral * math.sin(start_yaw))
        goal_y = (pose.position.y + self.forward * math.sin(start_yaw)
                  + self.lateral * math.cos(start_yaw))
        goal_yaw = start_yaw + self.yaw_offset

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'odom'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = goal_x
        goal.pose.pose.position.y = goal_y
        goal.pose.pose.orientation.z = math.sin(goal_yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(goal_yaw / 2.0)

        self.get_logger().info(
            f'Sending relative goal: forward={self.forward:.2f} m, '
            f'lateral={self.lateral:.2f} m, yaw={self.yaw_offset:.2f} rad')
        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Nav2 rejected the goal')
            self._stop()
            return False

        result_future = goal_handle.get_result_async()
        deadline = time.monotonic() + self.timeout
        while rclpy.ok() and not result_future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        if not result_future.done():
            self.get_logger().error('Navigation timeout; cancelling goal')
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=5.0)
            self._stop()
            return False

        result = result_future.result()
        self._stop()
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            dx = self.odom.pose.pose.position.x - goal_x
            dy = self.odom.pose.pose.position.y - goal_y
            error = math.hypot(dx, dy)
            self.get_logger().info(
                f'PASS: Nav2 reached the relative goal; position error={error:.3f} m')
            return True
        self.get_logger().error(f'Navigation ended with action status {result.status}')
        return False


def main(args=None):
    rclpy.init(args=args)
    node = None
    success = False
    try:
        node = RelativeNavigator()
        success = node.run()
    except (KeyboardInterrupt, ValueError) as error:
        if node is not None:
            node.get_logger().error(str(error))
    finally:
        if node is not None:
            node._stop()
            node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(0 if success else 1)


if __name__ == '__main__':
    main()
