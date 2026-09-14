"""Timestamped RGB-D observations and scoring evidence from the live camera."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Int32
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformListener, TransformException

from .vision import ShelfVision, robust_depth

CAMERA = '/head_front_camera/head_front_camera'


class LivePerception:
    def __init__(self, node, templates, evidence_dir):
        self.node = node
        self.vision = ShelfVision(templates)
        self.output = Path(evidence_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.bridge = CvBridge()
        self.rgb = self.depth = self.info = self.depth_info = None
        self.rgb_msg = self.depth_msg = None
        self.received = 0.
        self.frames = {}
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, node)
        self.column_pub = node.create_publisher(Int32, '/erc/shelf_column_identification', 10)
        self.row_pub = node.create_publisher(Int32, '/erc/shelf_row_identification', 10)
        node.create_subscription(Image, CAMERA+'/color/image_raw', self.on_rgb, qos_profile_sensor_data)
        node.create_subscription(Image, CAMERA+'/depth/image_rect_raw', self.on_depth, qos_profile_sensor_data)
        node.create_subscription(CameraInfo, CAMERA+'/color/camera_info', self.on_info, qos_profile_sensor_data)
        node.create_subscription(CameraInfo, CAMERA+'/depth/camera_info', self.on_depth_info, qos_profile_sensor_data)

    def on_info(self, msg):
        self.info = msg

    def on_depth_info(self,msg):
        self.depth_info=msg

    def on_rgb(self, msg):
        try:
            self.rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.rgb_msg = msg
            stamp = Time.from_msg(msg.header.stamp).nanoseconds
            self.frames[stamp] = self.rgb.copy()
            while len(self.frames) > 60:
                self.frames.pop(next(iter(self.frames)))
            self.received = time.monotonic()
        except Exception as exc:
            self.node.get_logger().warning(f'Invalid RGB: {exc}')

    def on_depth(self, msg):
        try:
            data = self.bridge.imgmsg_to_cv2(msg, 'passthrough').astype(np.float32)
            if msg.encoding == '16UC1':
                data *= .001
            elif msg.encoding != '32FC1':
                return
            self.depth, self.depth_msg = data, msg
        except Exception as exc:
            self.node.get_logger().warning(f'Invalid depth: {exc}')

    def ready(self):
        if any(x is None for x in (self.rgb, self.depth, self.info, self.depth_info, self.rgb_msg, self.depth_msg)):
            return False
        dt = abs((Time.from_msg(self.rgb_msg.header.stamp)-Time.from_msg(self.depth_msg.header.stamp)).nanoseconds)*1e-9
        return (time.monotonic()-self.received < 2. and dt < .12
                and self.rgb.shape[:2] == self.depth.shape[:2]
                and self.depth_info.k[0] > 0 and self.depth_info.k[4] > 0
                and np.allclose(self.info.k,self.depth_info.k,atol=1e-6))

    def point(self, bbox):
        if not self.ready():
            return None
        d = robust_depth(self.depth, bbox)
        if d is None:
            return None
        x,y,w,h = bbox
        u,v = x+w/2, y+h/2
        point = PointStamped()
        point.header = self.depth_msg.header
        point.point.x = (u-self.depth_info.k[2])*d/self.depth_info.k[0]
        point.point.y = (v-self.depth_info.k[5])*d/self.depth_info.k[4]
        point.point.z = d
        try:
            try:
                transform = self.tf.lookup_transform('odom', point.header.frame_id,
                                                     Time.from_msg(point.header.stamp))
            except TransformException:
                transform = self.tf.lookup_transform('odom', point.header.frame_id, Time())
            p = do_transform_point(point, transform).point
            return [p.x,p.y,p.z]
        except TransformException:
            return None

    def observe(self, kind, target, column_point=None):
        if not self.ready():
            return []
        if kind == 'column':
            detections = self.vision.numbers(self.rgb)
        elif kind == 'book':
            detections = self.vision.books(self.rgb)
        elif kind == 'bin':
            detections = self.vision.bins(self.rgb)
        else:
            raise ValueError(f'Unknown observation kind: {kind}')
        found = []
        for item in detections:
            if item.get('number' if kind == 'column' else 'colour') != target:
                continue
            point = self.point(item['bbox'])
            if point is None:
                continue
            if kind == 'column' and not 2.0 < point[2] < 2.5:
                continue
            if kind == 'book':
                if column_point is None or np.linalg.norm(np.array(point[:2])-np.array(column_point[:2])) > .65:
                    continue
                # Nominal fixed shelf heights classify an observed RGB-D book, not
                # its randomized colour or column. Active rows are numbered 1..4.
                heights = np.array([1.595,1.265,.935,.605])
                index = int(np.argmin(abs(heights-point[2])))
                if abs(heights[index]-point[2]) > .14:
                    continue
                item['row'] = index+1
            if kind == 'bin' and not .35 < point[2] < 1.8:
                continue
            item['point'] = point
            item['stamp_ns'] = Time.from_msg(self.rgb_msg.header.stamp).nanoseconds
            found.append(item)
        return found

    def save(self, kind, detection, target):
        if detection['stamp_ns'] not in self.frames:
            raise RuntimeError('Detection image has expired; re-observe before saving evidence')
        image = self.frames[detection['stamp_ns']].copy()
        x,y,w,h = detection['bbox']
        if kind == 'column':
            # A numeral's height is shorter than its 0.3 m marker plate. Save the
            # observed marker plus a separate full-column evidence frame later.
            label = f'shelf marker {target}'
        elif kind == 'book':
            label = f'{target} book, row {detection["row"]}'
        else:
            label = 'red collection bin'
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
        now = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
        cv2.putText(image,label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,0),3)
        cv2.putText(image,label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        cv2.putText(image,f'{now} sim_ns={detection["stamp_ns"]}',(8,image.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX,.37,(0,0,0),2)
        file = self.output/f'{kind}_{detection["stamp_ns"]}.png'
        if not cv2.imwrite(str(file), image):
            raise OSError(f'Could not save {file}')
        file.with_suffix('.json').write_text(json.dumps(detection,indent=2))
        if kind == 'column':
            self.column_pub.publish(Int32(data=int(target)))
        elif kind == 'book':
            self.row_pub.publish(Int32(data=int(detection['row'])))
        return str(file)
