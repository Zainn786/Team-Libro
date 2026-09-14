"""Project official collision geometry into Nav2 and stop for moving/stale arms."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point32, Polygon, Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener, TransformException


def convex_hull(points):
    points=sorted(set(map(tuple,points)))
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in points:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
        lower.append(p)
    upper=[]
    for p in reversed(points):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
        upper.append(p)
    return lower[:-1]+upper[:-1]


def rotate(points,q):
    vector=np.array([q.x,q.y,q.z])
    return points + 2*np.cross(vector,np.cross(vector,points)+q.w*points)


class FootprintGuard(Node):
    def __init__(self):
        super().__init__('libro_footprint_guard')
        share=Path(get_package_share_directory('erc_tiago_integration'))
        model=json.loads((share/'config/collision_corners.json').read_text())
        urdf=Path(get_package_share_directory('erc_description'))/'urdf/tiago_pro.urdf'
        if hashlib.sha256(urdf.read_bytes()).hexdigest()!=model['urdf_sha256']:
            raise RuntimeError('Robot geometry changed; regenerate collision corners before driving')
        self.links={k:np.array(v) for k,v in model['links'].items()}
        self.tf=Buffer();self.listener=TransformListener(self.tf,self)
        self.local=self.create_publisher(Polygon,'/local_costmap/footprint',10)
        self.global_=self.create_publisher(Polygon,'/global_costmap/footprint',10)
        self.output=self.create_publisher(Twist,'/cmd_vel',10)
        self.create_subscription(Twist,'/cmd_vel_collision',self.command,10)
        self.create_subscription(JointState,'/joint_states',self.joints,qos_profile_sensor_data)
        self.last_geometry=0.;self.last_joints=0.;self.last_cmd=0.;self.moving=True
        self.previous=None
        self.create_timer(.5,self.geometry)
        self.create_timer(.05,self.watchdog)

    def joints(self,msg):
        expected={f'arm_{side}_{i}_joint' for side in ('left','right') for i in range(1,8)}|{'torso_lift_joint'}
        joints=dict(zip(msg.name,msg.position))
        velocities=dict(zip(msg.name,msg.velocity))
        if not expected.issubset(joints) or not all(np.isfinite(joints[k]) for k in expected):
            self.moving=True
            return
        self.moving=not expected.issubset(velocities) or any(abs(velocities[k])>.03 for k in expected)
        self.last_joints=time.monotonic()

    def geometry(self):
        points=[]
        try:
            for link,corners in self.links.items():
                transform=self.tf.lookup_transform('base_footprint',link,Time()).transform
                moved=rotate(corners,transform.rotation)+np.array([transform.translation.x,transform.translation.y,transform.translation.z])
                # Expand bounds by 3 cm to include mesh approximation and tracking error.
                for dx,dy in ((-.03,-.03),(-.03,.03),(.03,-.03),(.03,.03)):
                    points.extend(moved[:,:2]+[dx,dy])
        except TransformException:
            return
        hull=convex_hull(np.round(points,4))
        polygon=Polygon(points=[Point32(x=float(x),y=float(y),z=0.) for x,y in hull])
        self.local.publish(polygon);self.global_.publish(polygon)
        self.last_geometry=time.monotonic()

    def ready(self):
        now=time.monotonic()
        return now-self.last_geometry<2. and now-self.last_joints<1. and not self.moving

    def command(self,msg):
        self.last_cmd=time.monotonic()
        self.output.publish(msg if self.ready() else Twist())

    def watchdog(self):
        if not self.ready() or time.monotonic()-self.last_cmd>.5:
            self.output.publish(Twist())


def main(args=None):
    rclpy.init(args=args);node=FootprintGuard()
    try:rclpy.spin(node)
    finally:node.output.publish(Twist());node.destroy_node();rclpy.shutdown()
