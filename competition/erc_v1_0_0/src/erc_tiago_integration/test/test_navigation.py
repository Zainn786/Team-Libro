import importlib.util
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]


def test_generated_humble_config_has_consistent_goal_and_sensor_contract():
    spec=importlib.util.spec_from_file_location('nav_launch',ROOT/'launch/navigation.launch.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=Path(module._make_tiago_params())
    try:
        params=yaml.safe_load(path.read_text())
    finally:
        path.unlink()
    ctrl=params['controller_server']['ros__parameters']
    assert ctrl['FollowPath']['xy_goal_tolerance']==ctrl['general_goal_checker']['xy_goal_tolerance']
    assert 'RotateToGoal' in ctrl['FollowPath']['critics']
    assert 'ObstacleFootprint' in ctrl['FollowPath']['critics']
    assert ctrl['FollowPath']['max_vel_y']>0
    for key in ('global_costmap','local_costmap'):
        obstacle=params[key][key]['ros__parameters']['obstacle_layer']
        assert set(obstacle['observation_sources'].split())=={'front_scan','rear_scan'}
    safety=params['collision_monitor']['ros__parameters']
    assert safety['cmd_vel_in_topic']=='cmd_vel_smoothed'
    assert safety['cmd_vel_out_topic']=='cmd_vel_collision'
    assert safety['source_timeout']<=1.
    for key in ('default_nav_to_pose_bt_xml','default_nav_through_poses_bt_xml'):
        assert '<Spin' not in Path(params['bt_navigator']['ros__parameters'][key]).read_text()
