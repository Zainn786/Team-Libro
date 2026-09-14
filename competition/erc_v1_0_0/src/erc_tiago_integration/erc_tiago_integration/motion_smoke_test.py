#!/usr/bin/env python3

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


class MotionSmokeTest(Node):
    """Move the TIAGo a short distance and verify the result using odometry."""

    def __init__(self):
        super().__init__('tiago_motion_smoke_test')
        self.declare_parameter('distance', 0.20)
        self.declare_parameter('speed', 0.10)
        self.declare_parameter('odom_timeout', 10.0)
        self.declare_parameter('motion_timeout', 10.0)

        self.distance = float(self.get_parameter('distance').value)
        self.speed = float(self.get_parameter('speed').value)
        self.odom_timeout = float(self.get_parameter('odom_timeout').value)
        self.motion_timeout = float(self.get_parameter('motion_timeout').value)

        if not 0.0 < self.distance <= 0.50:
            raise ValueError('distance must be in the range (0.0, 0.50] metres')
        if not 0.0 < self.speed <= 0.20:
            raise ValueError('speed must be in the range (0.0, 0.20] m/s')

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            Odometry, '/odom', self._on_odom, 10)
        self.start_position = None
        self.latest_position = None

    def _on_odom(self, message):
        position = message.pose.pose.position
        self.latest_position = (position.x, position.y)
        if self.start_position is None:
            self.start_position = self.latest_position

    def stop(self):
        command = Twist()
        for _ in range(5):
            self.publisher.publish(command)
            rclpy.spin_once(self, timeout_sec=0.02)

    def run(self):
        self.get_logger().info('Waiting for /odom...')
        deadline = time.monotonic() + self.odom_timeout
        while rclpy.ok() and self.start_position is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.start_position is None:
            self.get_logger().error('No odometry received; refusing to move')
            return False

        self.get_logger().info(
            f'Moving at {self.speed:.2f} m/s for at most {self.distance:.2f} m')
        command = Twist()
        command.linear.x = self.speed
        deadline = time.monotonic() + self.motion_timeout

        while rclpy.ok() and time.monotonic() < deadline:
            self.publisher.publish(command)
            rclpy.spin_once(self, timeout_sec=0.05)
            dx = self.latest_position[0] - self.start_position[0]
            dy = self.latest_position[1] - self.start_position[1]
            travelled = math.hypot(dx, dy)
            if travelled >= self.distance:
                self.stop()
                self.get_logger().info(
                    f'PASS: TIAGo moved {travelled:.3f} m and stopped')
                return True

        self.stop()
        self.get_logger().error(
            'Motion timed out before odometry reached the target distance')
        return False


def main(args=None):
    rclpy.init(args=args)
    node = None
    success = False
    try:
        node = MotionSmokeTest()
        success = node.run()
    except (KeyboardInterrupt, ValueError) as error:
        if node is not None:
            node.get_logger().error(str(error))
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(0 if success else 1)


if __name__ == '__main__':
    main()
