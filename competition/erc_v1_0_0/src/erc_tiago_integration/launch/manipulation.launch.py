"""Team-owned MoveIt configuration for the unmodified official TIAGo Pro URDF."""
import os
import xml.etree.ElementTree as ET
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    description = open(os.path.join(get_package_share_directory('erc_description'),
                                    'urdf','tiago_pro.urdf'),encoding='utf-8').read()
    robot=ET.fromstring(description)
    semantic=ET.Element('robot',name=robot.get('name'))
    arm=ET.SubElement(semantic,'group',name='left_arm')
    ET.SubElement(arm,'chain',base_link='torso_lift_link',tip_link='gripper_left_grasping_link')
    whole=ET.SubElement(semantic,'group',name='left_arm_torso')
    ET.SubElement(whole,'chain',base_link='base_footprint',tip_link='gripper_left_grasping_link')
    right=ET.SubElement(semantic,'group',name='right_arm')
    ET.SubElement(right,'chain',base_link='torso_lift_link',tip_link='gripper_right_grasping_link')
    hand=ET.SubElement(semantic,'group',name='left_gripper')
    ET.SubElement(hand,'joint',name='gripper_left_finger_joint')
    ET.SubElement(semantic,'end_effector',name='left_gripper',parent_link='arm_left_tool_link',group='left_gripper',parent_group='left_arm')
    # Directly connected links intentionally share a joint boundary. All other
    # collision pairs remain enabled until checked with the official geometry.
    for joint in robot.findall('joint'):
        ET.SubElement(semantic,'disable_collisions',link1=joint.find('parent').get('link'),
                      link2=joint.find('child').get('link'),reason='Adjacent')
    for side in ('left','right'):
        mechanical_pairs = [('torso_base_link', f'arm_{side}_1_link'),
            ('torso_lift_link', f'arm_{side}_1_link'),
            (f'arm_{side}_7_link',f'gripper_{side}_base_link')]
        mechanical_pairs += [(f'arm_{side}_{index}_link',f'arm_{side}_{index+1}_link')
                             for index in range(1,7)]
        for finger in ('left','right'):
            mechanical_pairs += [(f'gripper_{side}_base_link',f'gripper_{side}_inner_finger_{finger}_link'),
                (f'gripper_{side}_outer_finger_{finger}_link',f'gripper_{side}_fingertip_{finger}_link')]
        for first,second in mechanical_pairs:
            ET.SubElement(semantic,'disable_collisions',link1=first,link2=second,reason='Mechanical joint overlap in official model')
    ET.SubElement(semantic,'disable_collisions',link1='head_2_link',
                  link2='head_front_camera_link',reason='Fixed camera housing overlap')
    ET.SubElement(semantic,'disable_collisions',link1='torso_lift_link',
                  link2='head_1_link',reason='Fixed head mount overlap')
    kinematics={group:{'kinematics_solver':'kdl_kinematics_plugin/KDLKinematicsPlugin',
                       'kinematics_solver_search_resolution':.005,
                       'kinematics_solver_timeout':.15} for group in ('left_arm','left_arm_torso','right_arm')}
    limits={}
    for joint in robot.findall('joint'):
        if joint.get('type') in ('revolute','prismatic') and joint.find('mimic') is None:
            declared=float(joint.find('limit').get('velocity'))
            limits[joint.get('name')]={'has_velocity_limits':True,
                                      'max_velocity':min(1.,declared),
                                      'has_acceleration_limits':True,'max_acceleration':1.}
    controllers={}
    for name,joints in [('arm_left_controller',[f'arm_left_{i}_joint' for i in range(1,8)]),
                        ('arm_right_controller',[f'arm_right_{i}_joint' for i in range(1,8)]),
                        ('torso_controller',['torso_lift_joint'])]:
        controllers[name]={'type':'FollowJointTrajectory','action_ns':'follow_joint_trajectory',
                           'default':True,'joints':joints}
    parameters={'use_sim_time':True,'robot_description':description,
                'robot_description_semantic':ET.tostring(semantic,encoding='unicode'),
                'robot_description_kinematics':kinematics,
                'robot_description_planning':{'joint_limits':limits},
                'planning_pipelines':['ompl'],'default_planning_pipeline':'ompl',
                'ompl':{'planning_plugin':'ompl_interface/OMPLPlanner',
                        'request_adapters':'default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints',
                        'start_state_max_bounds_error':.1,
                        'planner_configs':{'RRTConnectkConfigDefault':{'type':'geometric::RRTConnect','range':0.}},
                        **{g:{'planner_configs':['RRTConnectkConfigDefault']} for g in kinematics}},
                'moveit_controller_manager':'moveit_simple_controller_manager/MoveItSimpleControllerManager',
                'moveit_simple_controller_manager':{'controller_names':list(controllers),**controllers},
                'trajectory_execution':{'allowed_execution_duration_scaling':2.,'allowed_goal_duration_margin':3.,'allowed_start_tolerance':.05},
                'publish_robot_description_semantic':True,
                'publish_planning_scene':True,'publish_geometry_updates':True,
                'publish_state_updates':True,'publish_transforms_updates':True}
    return LaunchDescription([Node(package='moveit_ros_move_group',executable='move_group',
                                   parameters=[parameters],output='screen')])
