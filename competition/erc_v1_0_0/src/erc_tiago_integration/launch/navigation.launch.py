"""Launch a dual-laser holonomic Nav2 stack for the ERC TIAGo Pro."""

import copy
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction, IncludeLaunchDescription
from launch_ros.actions import Node, SetRemap
from launch.launch_description_sources import PythonLaunchDescriptionSource


def _make_tiago_params():
    nav2_share = get_package_share_directory('nav2_bringup')
    source = os.path.join(nav2_share, 'params', 'nav2_params.yaml')
    with open(source, 'r', encoding='utf-8') as stream:
        params = yaml.safe_load(stream)

    bt = params['bt_navigator']['ros__parameters']
    bt.update({
        'global_frame': 'odom',
        'robot_base_frame': 'base_footprint',
        'odom_topic': '/odom',
    })

    controller = params['controller_server']['ros__parameters']
    controller.update({
        'odom_topic': '/odom',
        'controller_frequency': 10.0,
        'min_x_velocity_threshold': 0.001,
        'min_y_velocity_threshold': 0.001,
        'min_theta_velocity_threshold': 0.001,
    })
    dwb = controller['FollowPath']
    dwb.update({
        'min_vel_x': -0.15,
        'max_vel_x': 0.25,
        'min_vel_y': -0.15,
        'max_vel_y': 0.15,
        'max_vel_theta': 0.50,
        'max_speed_xy': 0.25,
        'acc_lim_x': 0.35,
        'acc_lim_y': 0.35,
        'acc_lim_theta': 0.75,
        'decel_lim_x': -0.35,
        'decel_lim_y': -0.35,
        'decel_lim_theta': -0.75,
        'vx_samples': 9,
        'vy_samples': 9,
        'vtheta_samples': 20,
        'sim_time': 1.5,
        'trans_stopped_velocity': 0.02,
        'xy_goal_tolerance': 0.08,
        'critics': [
            'RotateToGoal', 'Oscillation', 'ObstacleFootprint', 'PathDist', 'GoalDist',
        ],
        'ObstacleFootprint.scale': 0.05,
        'RotateToGoal.scale': 32.0,
        'RotateToGoal.slowing_factor': 5.0,
        'RotateToGoal.lookahead_time': -1.0,
        'PathDist.scale': 24.0,
        'GoalDist.scale': 32.0,
    })
    controller['progress_checker']['required_movement_radius'] = 0.10
    controller['progress_checker']['movement_time_allowance'] = 15.0
    controller['general_goal_checker']['xy_goal_tolerance'] = 0.08
    controller['general_goal_checker']['yaw_goal_tolerance'] = 0.10

    footprint = '[[1.06, 0.58], [1.06, -0.58], [-0.44, -0.58], [-0.44, 0.58]]'
    obstacle = {
        'plugin': 'nav2_costmap_2d::ObstacleLayer',
        'enabled': True,
        'observation_sources': 'front_scan rear_scan',
        'front_scan': {
            'topic': '/scan_front_raw',
            'data_type': 'LaserScan',
            'clearing': True,
            'marking': True,
            'obstacle_max_range': 8.0,
            'raytrace_max_range': 10.0,
            'max_obstacle_height': 2.0,
        },
    }

    obstacle['rear_scan'] = dict(obstacle['front_scan'], topic='/scan_rear_raw')

    for costmap_name, size in [('local_costmap', 5), ('global_costmap', 20)]:
        costmap = params[costmap_name][costmap_name]['ros__parameters']
        costmap.update({
            'global_frame': 'odom',
            'robot_base_frame': 'base_footprint',
            'rolling_window': True,
            'width': size,
            'height': size,
            'resolution': 0.05,
            'track_unknown_space': False,
            'footprint': footprint,
            'footprint_padding': 0.02,
            'plugins': ['obstacle_layer', 'inflation_layer'],
            'obstacle_layer': copy.deepcopy(obstacle),
            'always_send_full_costmap': True,
        })
        costmap['inflation_layer'].update({
            'inflation_radius': 0.70,
            'cost_scaling_factor': 3.0,
        })

    planner = params['planner_server']['ros__parameters']
    planner['expected_planner_frequency'] = 2.0
    planner['GridBased']['allow_unknown'] = True

    behavior = params['behavior_server']['ros__parameters']
    behavior['behavior_plugins'] = ['backup', 'wait']
    behavior.update({
        'global_frame': 'odom',
        'robot_base_frame': 'base_footprint',
        'max_rotational_vel': 0.5,
        'min_rotational_vel': 0.1,
        'rotational_acc_lim': 0.75,
    })

    velocity = params['velocity_smoother']['ros__parameters']
    velocity.update({
        'odom_topic': '/odom',
        'max_velocity': [0.25, 0.15, 0.5],
        'min_velocity': [-0.15, -0.15, -0.5],
        'max_accel': [0.35, 0.35, 0.75],
        'max_decel': [-0.35, -0.35, -0.75],
    })

    bt['default_nav_to_pose_bt_xml'] = os.path.join(
        get_package_share_directory('erc_tiago_integration'),
        'behavior_trees', 'navigate.xml')
    bt['default_nav_through_poses_bt_xml'] = os.path.join(
        get_package_share_directory('erc_tiago_integration'),
        'behavior_trees', 'navigate_through.xml')
    params['collision_monitor'] = {'ros__parameters': {
        'use_sim_time': True, 'base_frame_id': 'base_footprint',
        'odom_frame_id': 'odom', 'cmd_vel_in_topic': 'cmd_vel_smoothed',
        'cmd_vel_out_topic': 'cmd_vel_collision', 'transform_tolerance': 0.5,
        'source_timeout': 1.0, 'base_shift_correction': True,
        'stop_pub_timeout': 2.0, 'polygons': ['Stop', 'FootprintApproach'],
        'FootprintApproach': {'type':'polygon', 'action_type':'approach',
            'footprint_topic':'/local_costmap/published_footprint',
            'time_before_collision':1.0, 'simulation_time_step':0.1,
            'max_points':3, 'visualize':False},
        'Stop': {'type': 'polygon', 'points': [0.44,0.33,0.44,-0.33,-0.44,-0.33,-0.44,0.33],
                 'action_type': 'stop', 'max_points': 3, 'visualize': True,
                 'polygon_pub_topic': 'safety_stop_polygon'},
        'observation_sources': ['front', 'rear'],
        'front': {'type': 'scan', 'topic': '/scan_front_raw'},
        'rear': {'type': 'scan', 'topic': '/scan_rear_raw'},
    }}

    output = tempfile.NamedTemporaryFile(
        mode='w', prefix='erc_tiago_nav2_', suffix='.yaml', delete=False)
    yaml.safe_dump(params, output)
    output.close()
    return output.name


def generate_launch_description():
    # Explicit ownership of each command topic prevents upstream remap precedence
    # from letting the smoother bypass collision monitoring.
    params_file = _make_tiago_params()
    servers = [
        ('nav2_controller', 'controller_server'),
        ('nav2_smoother', 'smoother_server'),
        ('nav2_planner', 'planner_server'),
        ('nav2_behaviors', 'behavior_server'),
        ('nav2_bt_navigator', 'bt_navigator'),
        ('nav2_waypoint_follower', 'waypoint_follower'),
        ('nav2_velocity_smoother', 'velocity_smoother'),
        ('nav2_collision_monitor', 'collision_monitor'),
    ]
    nodes = []
    for package, name in servers:
        remappings = []
        if name in ('controller_server', 'behavior_server', 'velocity_smoother'):
            remappings = [('cmd_vel', 'cmd_vel_nav')]
        nodes.append(Node(package=package, executable=name, name=name,
                          parameters=[params_file, {'use_sim_time': True}],
                          remappings=remappings, output='screen'))
    nodes.append(Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_navigation', parameters=[{
            'use_sim_time': True, 'autostart': True,
            'node_names': [name for _, name in servers]}], output='screen'))
    nodes.append(Node(package='erc_tiago_integration', executable='footprint_guard',
                      parameters=[{'use_sim_time':True}], output='screen'))
    return LaunchDescription(nodes)
