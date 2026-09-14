#!/usr/bin/env python3

import math
import os
import time

import rclpy
import yaml
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node


class NamedPoseNavigator(Node):
    """Navigate to a validated, fixed arena pose expressed in odom."""

    def __init__(self):
        super().__init__('tiago_named_pose_navigator')
        self.declare_parameter('pose_name', 'shelf_overview')
        self.declare_parameter('poses_file', '')
        self.declare_parameter('timeout', 90.0)
        self.pose_name = str(self.get_parameter('pose_name').value)
        self.timeout = float(self.get_parameter('timeout').value)
        poses_file = str(self.get_parameter('poses_file').value)
        if not poses_file:
            poses_file = os.path.join(
                get_package_share_directory('erc_tiago_integration'),
                'config', 'named_poses.yaml')
        with open(poses_file, 'r', encoding='utf-8') as stream:
            document = yaml.safe_load(stream) or {}
        poses = document.get('poses', {})
        if self.pose_name not in poses:
            choices = ', '.join(sorted(poses))
            raise ValueError(
                f'Unknown pose_name {self.pose_name!r}; available poses: {choices}')
        self.target = dict(poses[self.pose_name])
        angle = float(document['arena_start_yaw'])
        ax, ay = float(self.target['x']), float(self.target['y'])
        self.target['x'] = math.cos(angle)*ax + math.sin(angle)*ay
        self.target['y'] = -math.sin(angle)*ax + math.cos(angle)*ay
        self.target['yaw'] = float(self.target['yaw']) - angle
        for key in ('x', 'y', 'yaw'):
            if key not in self.target or not isinstance(self.target[key], (int, float)):
                raise ValueError(f'Pose {self.pose_name!r} requires numeric {key}')
        if abs(float(self.target['x'])) > 5.0 or abs(float(self.target['y'])) > 4.0:
            raise ValueError('Named pose is outside the safe arena envelope')

        self.safety_fault = ''
        self.odom = None
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.stop_publisher = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        self.client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

    def _on_odom(self, message):
        self.odom = message

    def _stop(self):
        for _ in range(5):
            self.stop_publisher.publish(Twist())
            rclpy.spin_once(self, timeout_sec=0.02)

    def run(self):
        deadline = time.monotonic() + 10.0
        while rclpy.ok() and self.odom is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.odom is None:
            self.get_logger().error('No /odom received; refusing to navigate')
            return False
        if not self.client.wait_for_server(timeout_sec=15.0):
            self.get_logger().error('Nav2 /navigate_to_pose action is unavailable')
            return False

        x = float(self.target['x'])
        y = float(self.target['y'])
        yaw = float(self.target['yaw'])
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'odom'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = x
        goal.pose.pose.position.y = y
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.get_logger().info(
            f'Sending named pose {self.pose_name}: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}')

        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        if not send_future.done():
            self.get_logger().error('Goal acknowledgement timed out; no success claimed')
            return False
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Nav2 rejected the named pose')
            self._stop()
            return False

        result_future = goal_handle.get_result_async()
        deadline = time.monotonic() + self.timeout
        while rclpy.ok() and not result_future.done() and not self.safety_fault and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not result_future.done():
            self.get_logger().error('Navigation timeout; cancelling goal')
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=5.0)
            self._stop()
            return False

        status = result_future.result().status
        self._stop()
        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(f'Navigation ended with action status {status}')
            return False
        dx = self.odom.pose.pose.position.x - x
        dy = self.odom.pose.pose.position.y - y
        q = self.odom.pose.pose.orientation
        actual_yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        yaw_error = abs(math.atan2(math.sin(actual_yaw-yaw), math.cos(actual_yaw-yaw)))
        if math.hypot(dx, dy) > 0.12 or yaw_error > 0.15:
            self.get_logger().error(f'Final pose outside tolerance: xy={math.hypot(dx,dy):.3f}, yaw={yaw_error:.3f}')
            return False
        self.get_logger().info(
            f'PASS: reached {self.pose_name}; position error={math.hypot(dx, dy):.3f} m')
        return True


def main(args=None):
    rclpy.init(args=args)
    node = None
    success = False
    try:
        node = NamedPoseNavigator()
        success = node.run()
    except (KeyboardInterrupt, OSError, ValueError, yaml.YAMLError) as error:
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
