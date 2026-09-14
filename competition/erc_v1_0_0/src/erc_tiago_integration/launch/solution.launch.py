"""ERC evaluator entry point; start the official simulation separately first."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_trial(context):
    column = int(LaunchConfiguration('shelf_column_number').perform(context))
    colour = LaunchConfiguration('book_colour').perform(context)
    if column not in range(1,6) or colour not in ('red','blue','green','yellow'):
        raise ValueError('Expected shelf_column_number:=1..5 and book_colour:=red|blue|green|yellow')
    share=get_package_share_directory('erc_tiago_integration')
    return [
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(share,'launch','manipulation.launch.py'))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(share,'launch','navigation.launch.py'))),
        Node(package='erc_tiago_integration',executable='solution',name='libro_trial',
             parameters=[{'use_sim_time': True,'shelf_column_number':column,'book_colour':colour,
                          'evidence_dir':LaunchConfiguration('evidence_dir'),
                          'stage':LaunchConfiguration('stage')}],output='screen'),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('shelf_column_number'),
        DeclareLaunchArgument('book_colour'),
        DeclareLaunchArgument('evidence_dir',default_value=os.path.join(os.getcwd(),'erc_images')),
        DeclareLaunchArgument('stage',default_value='full',choices=['full','perception']),
        OpaqueFunction(function=launch_trial),
    ])
