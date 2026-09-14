"""Simulator-only route acceptance; physical columns are NOT randomized labels."""
import json
import math
import os
import time
from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import qos_profile_sensor_data
from ros_gz_interfaces.msg import Contacts
from nav_msgs.msg import Path as NavPath
from .navigate_named import NamedPoseNavigator


def main(args=None):
    rclpy.init(args=args)
    node=NamedPoseNavigator()
    node.declare_parameter('report_file','navigation_results.json')
    node.declare_parameter('route',['shelf_column_1','shelf_column_2','shelf_column_3',
                                    'shelf_column_4','shelf_column_5','delivery','start'])
    document=yaml.safe_load((Path(get_package_share_directory('erc_tiago_integration'))/'config/named_poses.yaml').read_text())
    collisions=[]
    paths=[]
    def contact(msg):
        for c in msg.contacts:
            pair=[c.collision1.name,c.collision2.name]
            if any(any(word in name.lower() for word in ('shelf','book','table','collection_bin')) for name in pair):
                node.safety_fault = 'Unintended contact detected'
                if not collisions or collisions[-1]['pair']!=pair:
                    collisions.append({'pair':pair,'sim_ns':node.get_clock().now().nanoseconds})
    node.create_subscription(Contacts,'/contacts',contact,qos_profile_sensor_data)
    node.create_subscription(NavPath,'/plan',lambda m:paths.append([[p.pose.position.x,p.pose.position.y] for p in m.poses]),10)
    results=[]
    success=True
    try:
        for name in node.get_parameter('route').value:
            p=document['poses'][name];a=document['arena_start_yaw']
            node.pose_name=name
            node.target={'x':math.cos(a)*p['x']+math.sin(a)*p['y'],
                         'y':-math.sin(a)*p['x']+math.cos(a)*p['y'], 'yaw':p['yaw']-a}
            start_wall=time.monotonic();start_sim=node.get_clock().now().nanoseconds
            before=len(collisions)
            passed=node.run()
            item={'pose':name,'passed':passed,'wall_seconds':time.monotonic()-start_wall,
                  'sim_seconds':(node.get_clock().now().nanoseconds-start_sim)*1e-9,
                  'contact_observations':len(collisions)-before}
            results.append(item)
            node.get_logger().info(json.dumps(item))
            success=success and passed and len(collisions)==0
            if not passed or collisions:
                break
    finally:
        node._stop()
        output=Path(node.get_parameter('report_file').value)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps({'passed':success,'routes':results,'contacts':collisions,
                                      'planned_paths':paths,'scope':'empty-handed navigation only'},indent=2))
        node.destroy_node();rclpy.shutdown()
    raise SystemExit(0 if success else 1)
