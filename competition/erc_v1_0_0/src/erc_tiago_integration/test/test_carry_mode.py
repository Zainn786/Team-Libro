from erc_tiago_integration.solution import Trial


def test_carry_mode_lowers_every_speed_and_acceleration_limit():
    normal, carrying = Trial.motion_limits(False), Trial.motion_limits(True)
    assert normal.keys() == carrying.keys()
    for node in normal:
        assert normal[node].keys() == carrying[node].keys()
        for name, value in normal[node].items():
            slow = carrying[node][name]
            values, slows = (value, slow) if isinstance(value, list) else ([value], [slow])
            for v, s in zip(values, slows):
                assert abs(s) < abs(v), (node, name)
                assert (s < 0) == (v < 0), (node, name)


def test_leaving_carry_mode_restores_the_launch_limits():
    normal = Trial.motion_limits(False)
    assert normal['velocity_smoother']['max_velocity'] == [.25, .15, .5]
    assert normal['controller_server']['FollowPath.acc_lim_x'] == .35
