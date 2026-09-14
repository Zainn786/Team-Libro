from types import SimpleNamespace

import pytest
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from erc_tiago_integration.solution import Trial


def stow_node(reject_plan=False, fail_execution=False):
    calls=[]
    trajectory=RobotTrajectory()
    trajectory.joint_trajectory.points=[JointTrajectoryPoint(positions=[0.]*7)]

    def action(client,goal,timeout):
        calls.append(goal)
        if isinstance(goal,MoveGroup.Goal):
            if reject_plan and len(calls)==1:
                raise RuntimeError('Action failed: status=6')
            return SimpleNamespace(error_code=SimpleNamespace(val=1),
                                   planned_trajectory=trajectory)
        if fail_execution:
            raise RuntimeError('Action failed: status=6')
        return SimpleNamespace(error_code=SimpleNamespace(val=1))

    node=SimpleNamespace(move=object(),manipulator=SimpleNamespace(execute=object()),
                         action=action,event=lambda *args,**kwargs:None)
    return node,calls


def test_stow_replans_before_executing_and_folds_wrist_first():
    node,calls=stow_node(reject_plan=True)
    Trial.stow(node,'left')
    plans=[goal for goal in calls if isinstance(goal,MoveGroup.Goal)]
    assert len(plans)==4
    assert len(calls)==7
    assert all(goal.planning_options.plan_only for goal in plans)
    wrist=plans[2].request.goal_constraints[0].joint_constraints
    assert wrist[1].position==0.
    assert wrist[5].position==-1.2
    assert plans[3].request.goal_constraints[0].joint_constraints[1].position==-1.83


def test_stow_does_not_retry_execution_failure():
    node,calls=stow_node(fail_execution=True)
    with pytest.raises(RuntimeError,match='Action failed'):
        Trial.stow(node,'left')
    assert len(calls)==2
