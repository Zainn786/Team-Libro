"""Inspect all 20 books in the already-running official simulation."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share=get_package_share_directory('erc_tiago_integration')
    return LaunchDescription([
        DeclareLaunchArgument('evidence_dir',default_value=os.path.join(os.getcwd(),'erc_images')),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(share,'launch','navigation.launch.py'))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(share,'launch','manipulation.launch.py'))),
        Node(package='erc_tiago_integration',executable='inspect_books',name='libro_book_inspection',
             parameters=[{'use_sim_time':True,'shelf_column_number':1,'book_colour':'red',
                          'evidence_dir':LaunchConfiguration('evidence_dir')}],output='screen'),
    ])
