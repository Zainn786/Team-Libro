"""Run every stage of the red-bin detector on one live RGB-D frame and report why a blob fails.

Usage (inside erc_sim, with the head camera facing the table):
    python3 probe_bin_detection.py [tilt]
"""
import sys
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

CAMERA = '/head_front_camera/head_front_camera'
frames = {}


def main():
    rclpy.init()
    node = Node('probe_bin_detection')
    node.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
    bridge = CvBridge()
    node.create_subscription(Image, CAMERA + '/color/image_raw',
                             lambda m: frames.update(rgb=bridge.imgmsg_to_cv2(m, 'bgr8')),
                             qos_profile_sensor_data)
    node.create_subscription(Image, CAMERA + '/depth/image_rect_raw',
                             lambda m: frames.update(depth=bridge.imgmsg_to_cv2(m, 'passthrough')),
                             qos_profile_sensor_data)
    node.create_subscription(CameraInfo, CAMERA + '/depth/camera_info',
                             lambda m: frames.update(info=m), qos_profile_sensor_data)
    end = time.monotonic() + 20
    while time.monotonic() < end and not all(k in frames for k in ('rgb', 'depth', 'info')):
        rclpy.spin_once(node, timeout_sec=.1)
    if not all(k in frames for k in ('rgb', 'depth', 'info')):
        print('no RGB-D frame received'); return
    image, depth = frames['rgb'], frames['depth'].astype(np.float32)
    if depth.max() > 100:  # millimetres
        depth = depth / 1000.
    cv2.imwrite('/tmp/probe_bin_rgb.png', image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = (cv2.inRange(hsv, (0, 100, 45), (12, 255, 255)) |
            cv2.inRange(hsv, (168, 100, 45), (179, 255, 255)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cv2.imwrite('/tmp/probe_bin_mask.png', mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    print(f'frame {image.shape[1]}x{image.shape[0]}, red contours: {len(contours)}')
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        reasons = []
        if h < 18: reasons.append(f'h {h}<18')
        if w < 30: reasons.append(f'w {w}<30')
        if area < max(250, image_area * .002): reasons.append(f'area {area:.0f}<{max(250, image_area*.002):.0f}')
        if h and w / h < 1.25: reasons.append(f'aspect {w/h:.2f}<1.25')
        if w * h and area / (w * h) < .35: reasons.append(f'fill {area/(w*h):.2f}<.35')
        patch = depth[y + h // 4:y + max(h // 4 + 1, 3 * h // 4), x + w // 4:x + max(w // 4 + 1, 3 * w // 4)]
        values = patch[np.isfinite(patch) & (patch > .2) & (patch < 8.)]
        if values.size < 3:
            depth_note = f'depth: only {values.size} valid px -> REJECT'
        else:
            spread = float(np.quantile(values, .9) - np.quantile(values, .1))
            depth_note = (f'depth median {np.median(values):.3f} m, p90-p10 spread {spread:.3f} m'
                          + (' -> REJECT (>0.20)' if spread > .20 else ' -> ok'))
        verdict = 'SHAPE FAIL: ' + ', '.join(reasons) if reasons else 'shape ok'
        print(f'  bbox=({x},{y},{w},{h}) area={area:.0f} | {verdict} | {depth_note}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
