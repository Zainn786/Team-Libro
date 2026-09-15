"""Collision-checked one-arm primitives; task success requires sensor verification."""
import math
import struct
import time
from pathlib import Path
import numpy as np
from ament_index_python.packages import get_package_share_directory
from control_msgs.action import FollowJointTrajectory
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, Pose
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (CollisionObject, Constraints, PositionConstraint,
                             OrientationConstraint, BoundingVolume, AttachedCollisionObject,
                             AllowedCollisionEntry)
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPlanningScene
from moveit_msgs.msg import LinkPadding, PlanningSceneComponents
from rclpy.action import ActionClient
from rclpy.time import Time
from sensor_msgs.msg import JointState
from shape_msgs.msg import Mesh, MeshTriangle, SolidPrimitive
from tf2_geometry_msgs import do_transform_pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class UnilateralBookContact(RuntimeError):
    def __init__(self,side):
        super().__init__(f'Sustained {side}-finger contact requires lateral alignment')
        self.side=side


class NoBookContact(RuntimeError):
    pass



class InsertionContact(RuntimeError):
    """Insertion stopped on the first gripper contact with the target book."""
    def __init__(self,side):
        super().__init__(f'Insertion stopped on {side or "unlocated"} gripper contact '
                         'with the target book')
        self.side=side


class GripperJammed(RuntimeError):
    """The finger joint did not move after a close command, with no book contact."""


class ActionInterrupted(RuntimeError):
    """An executing action was cancelled because a caller condition fired."""


class Manipulator:
    """One-arm primitives driving the unmodified PAL Pro gripper.

    Every constant below is measured from the organizer-supplied
    ``fingertip.stl`` and ``tiago_pro.urdf``; nothing here depends on a
    modified robot model, world, or Gazebo plugin.
    """

    BOOK_DEPTH = .160                    # into the shelf
    BOOK_HEIGHT = .250                   # standing on the shelf floor
    # Distance from base_footprint to the grasp frame at the seated pose.  The
    # base is placed from this rather than from the book's cover so that
    # retuning the insertion depth never changes how far the arm has to reach.
    BASE_GRASP_REACH = .571
    # Lateral search used when the jaws close on nothing or touch on one side.
    # A 30 mm book inside a 59.72 mm jaw leaves +/-14.9 mm of centring margin,
    # and the base arrives with up to ~0.1 rad of yaw error, which is ~13 mm of
    # lateral offset at this reach before any RGB-D error is counted.
    # Pre-grasp stand-off, measured from the fingertip's leading edge to the
    # cover. The base sits BASE_GRASP_REACH from the grasp frame, so a longer
    # stand-off pulls the wrist back towards the chassis: at .29 the wrist sits
    # only .236 m from base_footprint, which folds the forearm over the mobile
    # base and torso and makes the pose unplannable at the upper rows. Low rows
    # have to come in close for the same reason.
    LOW_ROW_HEIGHT = 1.1
    LOW_ROW_STANDOFF = .04
    HIGH_ROW_STANDOFF = .15
    ALIGNMENT_STEP = .012
    ALIGNMENT_SPAN = .048
    GRASP_ALIGNMENT_ATTEMPTS = 10
    CARTESIAN_PLAN_ATTEMPTS = 4
    GRIP_SQUEEZE = .008
    GRIP_MAX_WIDTH = .034
    # Tilt allowed along the carry and placement paths while holding a book.
    CARRY_TILT_TOLERANCE = .3
    # Slowdown for straight-line moves while holding a book. The pinch only
    # survives gentle motion: at a slowdown of 2 the carry drove arm joints to
    # 0.46 rad/s and the book slipped within 4 s. Every motion that kept it ran
    # slower -- extraction at 4 (<=0.25 rad/s) and the orientation-constrained
    # carry at velocity scaling 0.2 -- so carrying is slower than both.
    CARRY_TIME_SCALE = 6.
    EXTRACTION_TIME_SCALE = 4.
    # Placement reach, measured with the book attached and held level (roll pi,
    # yaw 0) at torso 0.30: collision-free IK solved 4/4 from 0.55 to 0.80 m
    # forward at z=1.20, 0/4 at 0.85 m, and 0/4 at any distance at z=1.10.
    # Rotating the book 90 deg about vertical cut the reach to 0.65-0.70 m.
    PLACE_TORSO = .30
    PLACE_MAX_REACH = .80
    PLACE_HEIGHT = 1.20
    # Transport grasp-frame position in base_footprint. From the top row a
    # straight line to z=1.05 stopped at 72-88% (the descent runs the wrist
    # along the shelf front and past its reach); from z=1.20 up both a direct
    # line and an across-then-down line planned completely.
    CARRY_POSITION = (.38,.28,1.20)
    # Any collision-free part of the final clearance move is worth executing.
    FINAL_CLEARANCE_MIN_FRACTION = .3
    # Opening commanded before an insertion, deliberately below the 0.070 m
    # joint limit. Commanding the limit itself lets Gazebo settle the finger
    # joint a few nanometres past it (0.07000000000001880), after which the left
    # gripper is locked for good: in a gripper-only test, four open/close cycles
    # to 0.065 all moved, while the very first open to 0.070 jammed the joint and
    # every later command, including closing, left it where it was. At 0.065 the
    # pads span 55.8 mm, keeping 12.9 mm of centring margin each side of a 30 mm
    # book (still wider than ALIGNMENT_STEP). Kept separate from the geometric
    # reference OPEN_FINGERTIP_GAP_JOINT so none of the jaw-span maths changes.
    GRIPPER_OPEN_COMMAND = .065
    # The gripper's JointTrajectoryController has no goal constraints, so it
    # reports success when the trajectory time elapses even if the joint never
    # moved. A live trial lost every attempt to a left finger joint frozen at
    # 0.07000000000000005 while the controller logged "Goal reached, success!".
    # If the jaws have not travelled this far, with no finger contact, this long
    # (sim time) after a close command, the joint is jammed.
    GRIPPER_JAM_WINDOW_NS = 4_000_000_000
    GRIPPER_JAM_MIN_TRAVEL = .002
    # Direction to move the jaws, in the odom-x lateral offset, when one finger
    # touches the book. Measured in Gazebo with the base squared to the shelf:
    # gripper_left_fingertip_left_link sits at base y=+0.006 m and the right one
    # at y=-0.077 m, and facing -pi/2 puts base +y along odom +x. A left-finger
    # touch therefore means the book lies on the +x side of the jaw centre. The
    # previous mapping had both signs inverted, so every correction stepped the
    # jaws away from the book and the sweep kept shoving it along the row.
    FINGER_SIDE_OFFSET_SIGN = {'left': 1., 'right': -1.}
    # 5 cm exceeds the ~45 mm wheel-odometry drift measured in trials. The
    # forearm is included: unpadded, MoveIt rated arm_left_4_link resting against
    # the real shelf end frame as collision-free. Probed at the tightest case
    # (bottom row, outer column, 102 mm insertion) the pre-grasp and grasp poses
    # stayed solvable 4/4 at this padding while the real contact was flagged.
    UPPER_ARM_PADDING = .05
    PADDED_ARM_LINKS = ('arm_left_1_link','arm_left_2_link','arm_left_3_link','arm_left_4_link')
    # A book that has shifted further than this is no longer standing where it
    # was seen, and further insertions would only push it deeper or topple it.
    BOOK_DISPLACED_LIMIT = .12
    # A height change larger than this, with the plane fix agreeing to within
    # REOBSERVE_PLANAR_AGREEMENT, is treated as arm occlusion, not movement.
    REOBSERVE_HEIGHT_JUMP = .04
    REOBSERVE_PLANAR_AGREEMENT = .015

    @classmethod
    def alignment_sweep(cls):
        """Candidate lateral offsets, nearest first, alternating either side."""
        steps=int(round(cls.ALIGNMENT_SPAN/cls.ALIGNMENT_STEP))
        return [sign*index*cls.ALIGNMENT_STEP
                for index in range(1,steps+1) for sign in (1,-1)]
    # Forward kinematics of the shipped finger linkage, evaluated in
    # gripper_left_base_link.  gripper_left_finger_joint is prismatic over
    # [-.001, .070]; the inner/outer/fingertip revolutes mimic it at -/+8.28,
    # which leaves the fingertip meshes parallel across the whole stroke.
    #
    # The fingertip mesh is not a flat jaw.  Along the insertion axis it
    # presents two raised contact pads, 2.0-7.0 mm and 37.0-42.0 mm behind the
    # leading edge, separated by a recess.  Those pads, not the mesh bounding
    # box, set the usable opening.
    OPEN_FINGERTIP_GAP_JOINT = .070      # upper limit of gripper_left_finger_joint
    OPEN_FINGERTIP_GAP = .05972          # pad-to-pad span at that limit
    FINGERTIP_GAP_PER_METRE = .78286     # d(gap)/d(finger joint), both jaws
    OPEN_FINGERTIP_LEADING_EDGE = .011859  # fingertip tip ahead of the grasp frame
    NEAR_PAD_DEPTH = .007                # far edge of the leading contact pad
    FAR_PAD_DEPTH = .042                 # far edge of the trailing contact pad
    # Where the two pads' contact centre sits behind the leading edge.
    PAD_CONTACT_CENTRE_DEPTH = (NEAR_PAD_DEPTH + FAR_PAD_DEPTH) / 2 - .0025
    # Distance from the fingertip leading edge back to the palm between the
    # jaws (gripper_left_base_link collision mesh, max z 0.1107 vs the edge at
    # 0.1690). Inserting deeper drives the palm into the cover, and the palm has
    # no contact sensor: a 102 mm insertion silently shoved a book 112 mm deeper
    # into the shelf before any fingertip contact stopped the arm.
    GRIPPER_THROAT_DEPTH = .0584
    PALM_CLEARANCE = .006
    # Deepest seat that keeps the palm off the cover, with both pads on it. A
    # pinch at the book's centre of mass (80 mm in) is out of this gripper's reach.
    FINGERTIP_BOOK_OVERLAP = GRIPPER_THROAT_DEPTH - PALM_CLEARANCE
    # Insertion depths tried in order; each keeps both contact pads on the cover.
    INSERTION_DEPTHS = (FINGERTIP_BOOK_OVERLAP, .045)

    @classmethod
    def insertion_depths(cls,start):
        """Depths not deeper than start, deepest first (a reduction persists across retries)."""
        return [depth for depth in cls.INSERTION_DEPTHS if depth <= start + 1e-9]

    @classmethod
    def finger_joint_for_width(cls, width):
        """Finger-joint position whose pad-to-pad span equals ``width``."""
        return cls.OPEN_FINGERTIP_GAP_JOINT - (
            cls.OPEN_FINGERTIP_GAP - float(width)) / cls.FINGERTIP_GAP_PER_METRE

    @classmethod
    def width_for_finger_joint(cls, position):
        """Pad-to-pad span at a measured finger-joint position."""
        return cls.OPEN_FINGERTIP_GAP - (
            cls.OPEN_FINGERTIP_GAP_JOINT - float(position)) * cls.FINGERTIP_GAP_PER_METRE

    def gripper_gap(self):
        """Current pad-to-pad span, or None before the first joint state."""
        if self.gripper_position is None:
            return None
        return max(0.,self.width_for_finger_joint(self.gripper_position))

    def __init__(self, trial):
        self.node=trial
        self.apply=trial.create_client(ApplyPlanningScene,'/apply_planning_scene')
        self.cartesian=trial.create_client(GetCartesianPath,'/compute_cartesian_path')
        self.scene=trial.create_client(GetPlanningScene,'/get_planning_scene')
        self.execute=ActionClient(trial,ExecuteTrajectory,'/execute_trajectory')
        self.gripper=ActionClient(trial,FollowJointTrajectory,
                                  '/gripper_left_controller/follow_joint_trajectory')
        self.torso=ActionClient(trial,FollowJointTrajectory,
                                '/torso_controller/follow_joint_trajectory')
        self.tip='gripper_left_grasping_link'
        self.gripper_position=None
        self.torso_position=None
        self.head_positions={}
        self.finger_contacts={}
        self.hold_goal=None
        trial.create_subscription(JointState,'/joint_states',self.on_joints,10)

    def on_joints(self,message):
        for joint in ('head_1_joint','head_2_joint'):
            if joint in message.name:
                self.head_positions[joint]=message.position[message.name.index(joint)]
        if 'gripper_left_finger_joint' in message.name:
            self.gripper_position=message.position[message.name.index('gripper_left_finger_joint')]
        if 'torso_lift_joint' in message.name:
            self.torso_position=message.position[message.name.index('torso_lift_joint')]

    def call(self,client,request):
        if not client.wait_for_service(timeout_sec=15.):
            raise RuntimeError('MoveIt service unavailable')
        future=client.call_async(request)
        if not self.node.wait(future.done,30.):
            raise RuntimeError('MoveIt service timeout')
        return future.result()

    def pose(self,point,yaw,roll=math.pi,pitch=0.):
        p=Pose()
        p.position=Point(x=float(point[0]),y=float(point[1]),z=float(point[2]))
        cr,sr=math.cos(roll/2),math.sin(roll/2)
        cp,sp=math.cos(pitch/2),math.sin(pitch/2)
        cy,sy=math.cos(yaw/2),math.sin(yaw/2)
        p.orientation.x=sr*cp*cy-cr*sp*sy
        p.orientation.y=cr*sp*cy+sr*cp*sy
        p.orientation.z=cr*cp*sy-sr*sp*cy
        p.orientation.w=cr*cp*cy+sr*sp*sy
        tf=self.node.vision.tf.lookup_transform('base_footprint','odom',Time())
        return do_transform_pose(p,tf)

    def go(self,pose,path_tilt_tolerance=None):
        last_error=None
        # OMPL occasionally returns a path that becomes invalid during time
        # parameterization.  A bounded replan is safe because MoveIt refuses the
        # invalid trajectory before execution and a new sample can use a clear
        # elbow configuration.
        for attempt in range(3):
            goal=MoveGroup.Goal()
            goal.request.group_name='left_arm'
            goal.request.allowed_planning_time=20. if path_tilt_tolerance is None else 30.
            goal.request.num_planning_attempts=5
            goal.request.max_velocity_scaling_factor=.2
            goal.request.max_acceleration_scaling_factor=.2
            goal.request.start_state.is_diff=True
            pc=PositionConstraint();pc.header.frame_id='base_footprint';pc.link_name=self.tip;pc.weight=1.
            pc.constraint_region=BoundingVolume(primitives=[SolidPrimitive(type=SolidPrimitive.SPHERE,dimensions=[.008])],primitive_poses=[pose])
            oc=OrientationConstraint();oc.header.frame_id='base_footprint';oc.link_name=self.tip;oc.orientation=pose.orientation
            oc.absolute_x_axis_tolerance=.08;oc.absolute_y_axis_tolerance=.08;oc.absolute_z_axis_tolerance=.08;oc.weight=1.
            goal.request.goal_constraints=[Constraints(position_constraints=[pc],orientation_constraints=[oc])]
            if path_tilt_tolerance is not None:
                # Hold the grasp frame level for the whole path, not just at the
                # goal. Every carry/place pose points the gripper roll pi, which
                # puts the link z axis straight down: turning about it is a turn
                # about the vertical and cannot tip the book, so z stays free
                # while x/y tilt is bounded.
                keep=OrientationConstraint();keep.header.frame_id='base_footprint'
                keep.link_name=self.tip;keep.orientation=pose.orientation;keep.weight=1.
                keep.absolute_x_axis_tolerance=path_tilt_tolerance
                keep.absolute_y_axis_tolerance=path_tilt_tolerance
                keep.absolute_z_axis_tolerance=math.pi
                goal.request.path_constraints=Constraints(orientation_constraints=[keep])
            goal.planning_options.planning_scene_diff.is_diff=True
            goal.planning_options.planning_scene_diff.robot_state.is_diff=True
            try:
                result=self.node.action(self.node.move,goal,600.)
            except RuntimeError as error:
                last_error=error
                if attempt == 2:
                    raise
                self.node.event('ARM_REPLANNING',attempt=attempt+1,reason=str(error))
                continue
            if result.error_code.val==1:
                return
            last_error=RuntimeError(f'Arm plan/execute failed: {result.error_code.val}')
            if attempt < 2:
                self.node.event('ARM_REPLANNING',attempt=attempt+1,
                                reason=str(last_error))
        raise last_error

    def straight(self,poses,time_scale=4.,min_fraction=.999,interrupt=None):
        request=GetCartesianPath.Request()
        request.header.frame_id='base_footprint';request.group_name='left_arm'
        request.link_name=self.tip;request.start_state.is_diff=True
        request.waypoints=poses;request.max_step=.005;request.jump_threshold=0.;request.avoid_collisions=True
        # A short path is often transient: KDL solves each 5 mm step by random
        # restarts, and the start state can still be settling after the
        # preceding motion. Replaying the identical request from a live trial
        # failure (5.5% of the path) returned a complete path 10 times out of
        # 10. Nothing is executed until a path is accepted, so retrying costs
        # only planning time; a genuine obstruction still fails every attempt.
        for attempt in range(self.CARTESIAN_PLAN_ATTEMPTS):
            result=self.call(self.cartesian,request)
            if result.error_code.val==1 and result.fraction >= min_fraction:
                break
            self.node.event('CARTESIAN_REPLANNING',attempt=attempt+1,
                            fraction=float(result.fraction))
            time.sleep(.5)
        else:
            raise RuntimeError(f'Incomplete collision-free Cartesian path: {result.fraction:.3f}')
        self.validate_cartesian_joints(result.solution.joint_trajectory)
        if not result.solution.joint_trajectory.points:
            raise RuntimeError('Empty Cartesian trajectory')
        # Humble Cartesian trajectories may lack timing. Refuse unparameterized
        # motion instead of streaming an instantaneous joint jump.
        end=result.solution.joint_trajectory.points[-1].time_from_start
        if end.sec==0 and end.nanosec==0:
            raise RuntimeError('Cartesian trajectory requires time parameterization')
        self.slow_cartesian(result.solution.joint_trajectory,time_scale)
        goal=ExecuteTrajectory.Goal(trajectory=result.solution)
        if interrupt is None:
            response=self.node.action(self.execute,goal,600.)
        else:
            response=self.node.action(self.execute,goal,600.,interrupt=interrupt)
        if response.error_code.val!=1:
            raise RuntimeError(f'Cartesian execution failed: {response.error_code.val}')

    @staticmethod
    def slow_cartesian(trajectory,time_scale=4.):
        for point in trajectory.points:
            duration=int(time_scale*(point.time_from_start.sec*1_000_000_000+
                                     point.time_from_start.nanosec))
            point.time_from_start.sec,point.time_from_start.nanosec=divmod(duration,1_000_000_000)
            point.velocities=[velocity/time_scale for velocity in point.velocities]
            point.accelerations=[acceleration/time_scale**2 for acceleration in point.accelerations]

    @staticmethod
    def validate_cartesian_joints(trajectory):
        if not trajectory.points or not trajectory.joint_names:
            raise RuntimeError('Empty Cartesian trajectory')
        positions=np.array([point.positions for point in trajectory.points],dtype=float)
        if positions.shape != (len(trajectory.points),len(trajectory.joint_names)) or not np.isfinite(positions).all():
            raise RuntimeError('Invalid Cartesian joint positions')
        if len(positions)>1 and np.max(np.abs(np.diff(positions,axis=0))) > .12:
            raise RuntimeError('Cartesian trajectory contains a joint jump above 0.12 radians')

    def command_gripper(self,opening):
        if not 0.<=opening<=.07:
            raise ValueError('Unsafe gripper opening')
        goal=FollowJointTrajectory.Goal()
        goal.trajectory.joint_names=['gripper_left_finger_joint']
        point=JointTrajectoryPoint(positions=[float(opening)])
        point.time_from_start.sec=2
        goal.trajectory.points=[point]
        if opening == 0.:
            self.close_on_book(goal)
            return
        self.node.action(self.gripper,goal,120.)
        self.hold_goal=None
        if not self.node.wait(lambda:self.gripper_position is not None and
                              abs(self.gripper_position-opening)<.012,30.):
            raise RuntimeError(f'Gripper did not reach commanded opening {opening:.3f}')

    def record_finger_contact(self,pair):
        stamp=self.node.get_clock().now().nanoseconds
        recorded=False
        for side in ('left','right'):
            if any(any(f'gripper_left_{segment}_finger_{side}_link' in name
                       for segment in ('inner','outer')) or
                   f'gripper_left_fingertip_{side}_link' in name for name in pair):
                self.finger_contacts[side]=stamp

    def bilateral_contact(self):
        stamp=self.node.get_clock().now().nanoseconds
        return all(0 <= stamp-self.finger_contacts.get(side,-10**18) <= 1_000_000_000
                   for side in ('left','right'))

    def book_retained(self):
        """Possession is proven only by fresh gripper/book contact reports."""
        return time.monotonic()-self.node.last_grasp_contact < 1.5

    def close_on_book(self,goal):
        self.finger_contacts.clear()
        initial_opening=self.gripper_position
        if initial_opening is None:
            raise RuntimeError('Gripper position unavailable before closure')
        future=self.gripper.send_goal_async(goal)
        if not self.node.wait(future.done,15.):
            future.add_done_callback(lambda completed:completed.result().cancel_goal_async()
                                     if completed.result() and completed.result().accepted else None)
            raise RuntimeError('Gripper closure acknowledgement timeout')
        handle=future.result()
        if not handle.accepted:
            raise RuntimeError('Gripper closure rejected')
        result=handle.get_result_async()
        goal_sent_ns=self.node.get_clock().now().nanoseconds
        stable_since=None
        unilateral_since=None
        unilateral_side=None

        def secured():
            nonlocal stable_since,unilateral_since,unilateral_side
            if self.node.abort_reason:
                raise RuntimeError(self.node.abort_reason)
            now_ns=self.node.get_clock().now().nanoseconds
            touching=any(0 <= now_ns-seen <= 1_000_000_000 for seen in self.finger_contacts.values())
            if (self.gripper_position is not None and not touching and
                    now_ns-goal_sent_ns >= self.GRIPPER_JAM_WINDOW_NS and
                    initial_opening-self.gripper_position < self.GRIPPER_JAM_MIN_TRAVEL):
                self.node.event('GRIPPER_JAMMED',position=float(self.gripper_position),
                                commanded=0.,waited_s=(now_ns-goal_sent_ns)/1e9)
                raise GripperJammed(
                    f'Gripper finger joint stuck at {self.gripper_position:.4f} after a close '
                    'command; the controller reports success without moving the joint')
            if result.done():
                response=result.result()
                if response.status != GoalStatus.STATUS_SUCCEEDED or response.result.error_code != 0:
                    raise RuntimeError('Gripper closure controller failed')
                if self.gripper_position is not None and self.gripper_position < .005:
                    raise NoBookContact('Gripper closed without contacting the target book')
            stamp=self.node.get_clock().now().nanoseconds
            recent=[side for side in ('left','right')
                    if 0 <= stamp-self.finger_contacts.get(side,-10**18) <= 1_000_000_000]
            contact_confirmed=len(recent)==2
            if len(recent)==1:
                stable_since=None
                if unilateral_side != recent[0]:
                    unilateral_side=recent[0]
                    unilateral_since=stamp
                elif stamp-unilateral_since >= 1_500_000_000:
                    raise UnilateralBookContact(unilateral_side)
                return False
            unilateral_since=None;unilateral_side=None
            if not contact_confirmed:
                stable_since=None
                return False
            if stable_since is None:
                stable_since=stamp
            return stamp-stable_since >= 100_000_000

        try:
            if not self.node.wait(secured,20.):
                raise NoBookContact('No sustained two-finger target contact')
        except Exception:
            handle.cancel_goal_async()
            raise
        opening=self.gripper_position
        if opening is None:
            handle.cancel_goal_async()
            raise RuntimeError('Gripper position unavailable at confirmed contact')
        cancel=handle.cancel_goal_async()
        self.node.wait(cancel.done,5.)
        width=self.width_for_finger_joint(opening)
        if width > self.GRIP_MAX_WIDTH:
            # The jaws stopped far wider than a 30 mm book: a fingertip is on a
            # corner or edge. Such a grip (36.8 mm) slipped on the first pull,
            # so reopen and let the alignment loop try again.
            self.node.event('GRIP_ON_EDGE',measured_width=width,contact_opening=opening)
            raise NoBookContact(f'Jaws stopped at {width*1000:.1f} mm: gripping a book edge, not its faces')
        # Command past the measured contact position. The book blocks that
        # travel, so the residual command becomes the squeeze that holds it.
        # A 5 mm squeeze held through extraction but let the book slip under
        # carry and drive accelerations.
        hold_opening=max(0.,opening-self.GRIP_SQUEEZE)
        hold=FollowJointTrajectory.Goal()
        hold.trajectory.joint_names=['gripper_left_finger_joint']
        point=JointTrajectoryPoint(positions=[hold_opening])
        point.time_from_start.nanosec=500_000_000
        hold.trajectory.points=[point]
        future=self.gripper.send_goal_async(hold)
        if not self.node.wait(future.done,15.):
            raise RuntimeError('Gripper preload acknowledgement timeout')
        preload_handle=future.result()
        if not preload_handle.accepted:
            raise RuntimeError('Gripper preload rejected')
        if not self.node.wait(self.bilateral_contact,5.):
            preload_handle.cancel_goal_async()
            raise RuntimeError('Two-finger contact was lost while holding the measured opening')
        self.hold_goal=preload_handle
        self.node.event('GRIP_CONFIRMED',gripper_position=self.gripper_position,
                        contact_opening=opening,hold_opening=hold_opening,
                        measured_book_width=self.width_for_finger_joint(opening),
                        finger_contacts=dict(self.finger_contacts))

    def command_torso(self,height):
        goal=FollowJointTrajectory.Goal()
        goal.trajectory.joint_names=['torso_lift_joint']
        point=JointTrajectoryPoint(positions=[float(height)])
        point.time_from_start.sec=4
        goal.trajectory.points=[point]
        self.node.action(self.torso,goal,300.)
        if not self.node.wait(lambda:self.torso_position is not None and
                              abs(self.torso_position-height)<.015,45.):
            raise RuntimeError(f'Torso did not reach commanded height {height:.3f}')

    def attach_book(self):
        request=ApplyPlanningScene.Request();request.scene.is_diff=True
        attached=AttachedCollisionObject();attached.link_name=self.tip
        attached.object.header.frame_id=self.tip;attached.object.id='carried_book'
        # Thickness is taken from the measured grip rather than assumed, so the
        # planning-scene box matches whichever book release is loaded.  A
        # degenerate measurement would shrink the box and let MoveIt plan the
        # carried book through the shelf, so clamp it to a usable floor.
        width=max(self.gripper_gap() or 0.,.02)
        shape=SolidPrimitive(type=SolidPrimitive.BOX,
                             dimensions=[self.BOOK_DEPTH,width,self.BOOK_HEIGHT])
        attached.object.primitives=[shape];attached.object.primitive_poses=[Pose()]
        attached.object.primitive_poses[0].orientation.w=1.
        attached.object.operation=CollisionObject.ADD
        attached.touch_links=[self.tip,'gripper_left_base_link',
                              'gripper_left_base_finger_left_link',
                              'gripper_left_base_finger_right_link',
                              'gripper_left_outer_finger_left_link',
                              'gripper_left_outer_finger_right_link',
                              'gripper_left_fingertip_left_link',
                              'gripper_left_fingertip_right_link',
                              'gripper_left_inner_finger_left_link',
                              'gripper_left_inner_finger_right_link',
                              'gripper_left_screw_left_link',
                              'gripper_left_screw_right_link']
        request.scene.robot_state.attached_collision_objects=[attached]
        request.scene.robot_state.is_diff=True
        fingertips={'gripper_left_fingertip_left_link','gripper_left_fingertip_right_link'}
        self.configure_matrix(request,
                              {('arena_shelf','carried_book')}|
                              {('arena_shelf',link) for link in fingertips},
                              {'arena_shelf','carried_book'}|fingertips)
        if not self.call(self.apply,request).success:
            raise RuntimeError('Could not attach carried book to planning scene')
        self.node.event('BOOK_ATTACHED_TO_PLANNING_SCENE',target=self.node.target_book_token)

    def detach_book(self,remove_world=False):
        request=ApplyPlanningScene.Request();request.scene.is_diff=True
        attached=AttachedCollisionObject();attached.link_name=self.tip
        attached.object.id='carried_book';attached.object.operation=CollisionObject.REMOVE
        request.scene.robot_state.attached_collision_objects=[attached]
        request.scene.robot_state.is_diff=True
        if not self.call(self.apply,request).success:
            raise RuntimeError('Could not detach carried book from planning scene')
        if remove_world:
            request=ApplyPlanningScene.Request();request.scene.is_diff=True
            request.scene.world.collision_objects=[CollisionObject(id='carried_book',operation=CollisionObject.REMOVE)]
            if not self.call(self.apply,request).success:
                raise RuntimeError('Could not remove stale carried-book geometry')

    def allow_shelf_fingertips(self,enabled):
        request=ApplyPlanningScene.Request();request.scene.is_diff=True
        fingertips={'gripper_left_fingertip_left_link','gripper_left_fingertip_right_link'}
        extra={('arena_shelf','carried_book')}
        if enabled:
            extra|={('arena_shelf',link) for link in fingertips}
        self.configure_matrix(request,extra,{'arena_shelf','carried_book'}|fingertips)
        if not self.call(self.apply,request).success:
            raise RuntimeError('Could not update shelf fingertip clearance')

    # The mobile base is one rigid assembly. Each wheel hangs off its own
    # suspension link rather than off base_link, so wheel-against-shell pairs
    # are not adjacent and are permanently in contact in the official meshes.
    # Leaving them enabled makes every whole-robot collision check fail, which
    # aborts planning even though single-group checks still report valid.
    BASE_ASSEMBLY_LINKS = (
        'base_footprint','base_link','base_dock_link','base_imu_link',
        'base_front_laser_link','base_rear_laser_link','virtual_base_laser_link',
        'base_antenna_left_link','base_antenna_right_link',
        'suspension_front_left_link','suspension_front_right_link',
        'suspension_rear_left_link','suspension_rear_right_link',
        'wheel_front_left_link','wheel_front_right_link',
        'wheel_rear_left_link','wheel_rear_right_link')

    @staticmethod
    def mechanical_pairs():
        pairs={('torso_base_link','torso_lift_link'),
               ('torso_lift_link','head_1_link'),
               ('head_2_link','head_front_camera_link'),
               ('base_link','torso_base_link'),
               ('base_link','torso_fixed_column_link')}
        assembly=Manipulator.BASE_ASSEMBLY_LINKS
        pairs.update({(first,second)
                      for index,first in enumerate(assembly)
                      for second in assembly[index+1:]})
        for side in ('left','right'):
            pairs.update({('torso_base_link',f'arm_{side}_1_link'),
                          ('torso_lift_link',f'arm_{side}_1_link'),
                          (f'arm_{side}_7_link',f'gripper_{side}_base_link')})
            pairs.update({(f'arm_{side}_{index}_link',f'arm_{side}_{index+1}_link')
                          for index in range(1,7)})
            for finger in ('left','right'):
                pairs.update({(f'gripper_{side}_base_link',f'gripper_{side}_base_finger_{finger}_link'),
                              (f'gripper_{side}_base_link',f'gripper_{side}_outer_finger_{finger}_link'),
                              (f'gripper_{side}_base_link',f'gripper_{side}_inner_finger_{finger}_link'),
                              (f'gripper_{side}_base_link',f'gripper_{side}_fingertip_{finger}_link'),
                              (f'gripper_{side}_base_finger_{finger}_link',f'gripper_{side}_screw_{finger}_link'),
                              (f'gripper_{side}_base_finger_{finger}_link',f'gripper_{side}_inner_finger_{finger}_link'),
                              (f'gripper_{side}_inner_finger_{finger}_link',f'gripper_{side}_fingertip_{finger}_link'),
                              (f'gripper_{side}_outer_finger_{finger}_link',f'gripper_{side}_fingertip_{finger}_link')})
            gripper_links=[f'gripper_{side}_base_link']+[
                f'gripper_{side}_{segment}_{finger}_link'
                for finger in ('left','right')
                for segment in ('base_finger','inner_finger','outer_finger','fingertip','screw')]
            pairs.update({(first,second) for index,first in enumerate(gripper_links)
                          for second in gripper_links[index+1:]})
        return pairs

    def live_matrix(self):
        """Current allowed-collision matrix, or None if the scene is unavailable.

        MoveIt seeds this from the SRDF with every pair its self-collision
        analysis found to be permanently or by-default in contact -- wheels,
        casters, the torso column, the base. Those entries are not reproduced
        anywhere in this package.
        """
        scene=getattr(self,'scene',None)
        if scene is None:
            return None
        if not scene.service_is_ready() and not scene.wait_for_service(timeout_sec=5.):
            return None
        request=GetPlanningScene.Request()
        request.components.components=PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        future=scene.call_async(request)
        if not self.node.wait(future.done,10.):
            return None
        result=future.result()
        return result.scene.allowed_collision_matrix if result else None

    def configure_matrix(self,request,extra_pairs=None,extra_names=None):
        """Add allowances to the live matrix instead of replacing it.

        Publishing a freshly built matrix drops every pair it does not mention,
        including MoveIt's SRDF defaults. The arm group still validates on its
        own, so IK and a single-state check both look healthy, but whole-robot
        checks -- which is what the Cartesian planner runs -- then report a
        permanent base/wheel self-collision and every path returns fraction 0.
        """
        pairs=self.mechanical_pairs()|set(extra_pairs or ())
        names=sorted({link for pair in pairs for link in pair}|set(extra_names or ()))
        matrix=self.live_matrix()
        if matrix is None or not matrix.entry_names:
            # No live scene to extend (offline tests, or the service is not up
            # yet): fall back to publishing just the pairs we know about.
            base_names,base_values=[],[]
        else:
            base_names=list(matrix.entry_names)
            base_values=[list(entry.enabled) for entry in matrix.entry_values]
        index={name:position for position,name in enumerate(base_names)}
        for name in names:
            if name not in index:
                index[name]=len(base_names)
                base_names.append(name)
                for row in base_values:
                    row.append(False)
                base_values.append([False]*len(base_names))
        # Rows added before the last append are short; pad them all to square.
        for row in base_values:
            row.extend([False]*(len(base_names)-len(row)))
        for first,second in pairs:
            base_values[index[first]][index[second]]=True
            base_values[index[second]][index[first]]=True
        for name in names:
            base_values[index[name]][index[name]]=True
        request.scene.allowed_collision_matrix.entry_names=base_names
        request.scene.allowed_collision_matrix.entry_values=[
            AllowedCollisionEntry(enabled=row) for row in base_values]

    def grasp(self,book_point):
        self.command_torso(self.torso_height_for_row(book_point[2]))
        self.update_fixed_scene()
        pre,grasp,pull,_=self.shelf_grasp_points(book_point)
        self.command_gripper(self.GRIPPER_OPEN_COMMAND)
        self.allow_shelf_fingertips(True)
        pre_pose=self.pose(pre,-math.pi/2)
        grasp_pose=self.pose(grasp,-math.pi/2)
        pull_pose=self.pose(pull,-math.pi/2)
        self.go(pre_pose)
        lateral_offset=0.
        tried_offsets={lateral_offset}
        depth=self.INSERTION_DEPTHS[0]
        for attempt in range(self.GRASP_ALIGNMENT_ATTEMPTS):
            try:
                # Deepest first: the pinch over the centre of mass is what stops
                # the book pivoting out. At row 3 with the torso down that line
                # was unplannable (10% with collisions, 19% without, gripper base
                # into the shelf), so fall back to shallower seats that still put
                # both pads on the cover.
                for depth in self.insertion_depths(depth):
                    _,grasp,pull,_=self.shelf_grasp_points(book_point,overlap=depth)
                    grasp[0]+=lateral_offset;pull[0]+=lateral_offset
                    grasp_pose=self.pose(grasp,-math.pi/2)
                    try:
                        self.insert_to(grasp_pose)
                        break
                    except RuntimeError as error:
                        if ('Incomplete collision-free Cartesian path' not in str(error) or
                                depth == self.INSERTION_DEPTHS[-1]):
                            raise
                        self.node.event('INSERTION_DEPTH_REDUCED',from_depth=depth,reason=str(error))
                self.command_gripper(0.)
                break
            except (UnilateralBookContact,NoBookContact,InsertionContact) as contact:
                if attempt == self.GRASP_ALIGNMENT_ATTEMPTS-1:
                    raise
                self.command_gripper(self.GRIPPER_OPEN_COMMAND)
                self.straight([pre_pose])
                # Look again before trying a different offset. Each failed
                # closure drives the open jaws a full insertion depth past the
                # cover, which can shove the book out of its row; sweeping
                # blindly after that just rams a book that is no longer there.
                # A fresh observation either corrects the aim or reports the
                # displacement, and re-observing is far cheaper than a miss.
                reobserved=self.reobserve_book(book_point)
                if reobserved is not None:
                    book_point=reobserved
                    pre,grasp,pull,_=self.shelf_grasp_points(book_point)
                    pre[0]+=lateral_offset;grasp[0]+=lateral_offset
                    pre_pose=self.pose(pre,-math.pi/2)
                    grasp_pose=self.pose(grasp,-math.pi/2)
                    pull_pose=self.pose(pull,-math.pi/2)
                side=getattr(contact,'side',None)
                if side in self.FINGER_SIDE_OFFSET_SIGN:
                    # One finger reached the book, so its side is known: step
                    # the jaws towards it until both pads bear.
                    target_offset=(lateral_offset+
                                   self.FINGER_SIDE_OFFSET_SIGN[side]*self.ALIGNMENT_STEP)
                else:
                    # Nothing located the book: sweep outwards in alternating
                    # steps across the jaw centring margin.
                    target_offset=next((candidate for candidate in self.alignment_sweep()
                                        if candidate not in tried_offsets),None)
                    if target_offset is None:
                        raise
                    side='none'
                correction=target_offset-lateral_offset
                lateral_offset=target_offset
                tried_offsets.add(lateral_offset)
                pre[0]+=correction
                grasp[0]+=correction
                pre_pose=self.pose(pre,-math.pi/2)
                grasp_pose=self.pose(grasp,-math.pi/2)
                # Shift across at the stand-off; the next pass inserts afresh
                # and again stops on first contact.
                self.straight([pre_pose])
                self.node.event('GRASP_LATERAL_REALIGNED',side=side,
                                cause=type(contact).__name__,
                                offset=float(lateral_offset),
                                correction=float(grasp[0]-book_point[0]))
        self.attach_book()
        extraction=[]
        for distance,height in ((.08,0.),(.16,.0125)):
            point=grasp.copy();point[1]+=distance;point[2]+=height
            extraction.append(point)
        extraction.append(pull)
        for index,point in enumerate(extraction,1):
            # While the book is still between the shelf boards the withdrawal
            # has to stay on the shelf normal, so those segments must complete
            # in full. The last segment only adds clearance and lift once a
            # full book depth is already out, and `carry` replans from wherever
            # it ends. At the bottom row that pull folds the forearm into the
            # torso (arm_left_4..6 against torso_base_link stopped it at 63.6%
            # in every retry), so it is best-effort: run what plans, else skip.
            final=index == len(extraction)
            if final:
                try:
                    self.straight([self.pose(point,-math.pi/2)],
                                  min_fraction=self.FINAL_CLEARANCE_MIN_FRACTION)
                except RuntimeError as error:
                    self.node.event('BOOK_FINAL_CLEARANCE_SKIPPED',reason=str(error))
            else:
                self.straight([self.pose(point,-math.pi/2)],min_fraction=.999)
            if not self.node.wait(self.book_retained,20.):
                raise RuntimeError(f'Target book was not retained during extraction segment {index}')
            self.node.event('BOOK_EXTRACTION_SEGMENT',segment=index)
        self.node.event('BOOK_CLEAR_OF_SHELF')
        self.allow_shelf_fingertips(False)
        self.node.event('BOOK_GRASPED',gripper_position=self.gripper_position,
                        contacts=len(self.node.grasp_contacts))

    @staticmethod
    def shelf_grasp_points(book_point,overlap=None):
        centre=np.array(book_point,dtype=float)
        # RGB-D supplies the visible front cover in odom (including wheel-odom
        # drift), plus the lateral centre and the row height.  Books are
        # randomized within each cell, so this measurement is the only source
        # for the lateral and depth coordinates.
        front=centre[1]
        grasp=centre.copy()
        # Seat the grasp frame so the fingertip's leading edge sits
        # FINGERTIP_BOOK_OVERLAP past the front cover.  Both raised contact
        # pads then bear on the book; a shallower seat closes the jaws on the
        # air ahead of it, and the book is pushed rather than pinched.
        depth=Manipulator.FINGERTIP_BOOK_OVERLAP if overlap is None else overlap
        grasp[1]=front+Manipulator.OPEN_FINGERTIP_LEADING_EDGE-depth
        # Stand-off is defined by the gap between the fingertip's leading edge
        # and the cover, so it stays fixed if the insertion depth is retuned.
        # At the two lower rows a long stand-back puts the wrist above the
        # mobile base footprint and forces the forearm through the base/head,
        # so those rows approach from close in.
        pre=grasp.copy()
        pre[1]=front+Manipulator.OPEN_FINGERTIP_LEADING_EDGE+(
            Manipulator.LOW_ROW_STANDOFF if centre[2] < Manipulator.LOW_ROW_HEIGHT
            else Manipulator.HIGH_ROW_STANDOFF)
        # Withdraw one full book depth plus 20 mm before lifting, measured from
        # the front cover rather than from the grasp frame.
        # The book is clear of the shelf once it has moved one full depth, so
        # the last extraction waypoint only lifts, straight up from there.
        # Pulling further back swung the forearm out into the shelf's end frame
        # at an outer column, and folded it into the torso at the bottom row.
        pull=grasp.copy();pull[1]=grasp[1]+Manipulator.BOOK_DEPTH;pull[2]+=.0375
        lift=pull.copy();lift[2]+=.03
        return pre,grasp,pull,lift

    @staticmethod
    def torso_height_for_row(height):
        """Torso lift that keeps the grasp and pre-grasp poses solvable.

        Measured against the shipped model with the base squared up to the
        shelf: with the torso fully down, a bottom-row target is reachable at
        the seated grasp depth but NOT at any shallower stand-off -- IK fails
        from every seed, with collision checking disabled, so the approach pose
        is genuinely outside the arm's envelope rather than merely awkward.
        Lifting the torso to .15 m makes every stand-off solvable there. The
        middle rows sit near shoulder height with the torso down and are left
        alone, since that configuration has grasped a book successfully.
        """
        if height > 1.35:
            return .3
        if height > 1.05:
            return .15
        if height > .8:
            return 0.
        return .15

    @staticmethod
    def shelf_base_pose(book_point):
        """Base (x, y) that puts the grasp frame at the demonstrated arm reach.

        The shelf front is parallel to odom X with its outward normal along +Y,
        so the base squares up to the shelf for every column.  A line-of-sight
        approach instead becomes strongly diagonal at the outer columns, which
        leaves the arm target lateral to the shoulder and the lower rows
        unreachable.
        """
        _,grasp,_,_=Manipulator.shelf_grasp_points(book_point)
        return float(book_point[0]),float(grasp[1]+Manipulator.BASE_GRASP_REACH)

    def insert_to(self,pose):
        """Insert along the shelf normal, stopping on first contact with the book.

        With the jaws aimed near the edge of their centring margin, a fingertip
        meets the cover face-on. Completing the move then pushes the book back
        along the shelf (a live trial displaced one 60 mm deeper and 139 mm
        sideways) and every later attempt aims at empty space. Cancelling the
        trajectory at the first touch keeps the book in place and reports which
        finger hit, which is exactly the lateral correction needed.
        """
        started=time.monotonic()
        stamp=self.node.get_clock().now().nanoseconds
        self.finger_contacts.clear()
        try:
            self.straight([pose],min_fraction=.84,
                          interrupt=lambda:self.node.last_grasp_contact > started)
        except ActionInterrupted:
            sides=[side for side in ('left','right')
                   if self.finger_contacts.get(side,-1) >= stamp]
            side=sides[0] if len(sides)==1 else None
            self.node.event('INSERTION_CONTACT',side=side or 'unlocated')
            raise InsertionContact(side)

    def reobserve_book(self,book_point):
        """Re-measure the target from the current pose, or None if unavailable.

        Raises if the book is clearly no longer on its row, so a displaced book
        ends the attempt instead of being pushed further into the shelf.
        """
        observer=getattr(self.node,'reobserve_target',None)
        if observer is None:
            return None
        observed=observer(book_point)
        if observed is None:
            return None
        observed=[float(v) for v in observed]
        planar=float(np.linalg.norm(np.array(observed[:2])-np.array(book_point[:2])))
        vertical=abs(observed[2]-float(book_point[2]))
        # After a failed closure the arm sits at the pre-grasp pose, directly in
        # front of the cover, and hides part of the book from the head camera.
        # The colour blob then shrinks from one end and its centroid height
        # jumps while the horizontal fix barely changes: a live trial re-measured
        # an undisturbed book 90 mm lower with under 1 mm of lateral change.
        # A book that has really been knocked moves in the plane too, so a
        # height-only jump keeps the original height.
        occluded=vertical > self.REOBSERVE_HEIGHT_JUMP and planar < self.REOBSERVE_PLANAR_AGREEMENT
        if occluded:
            observed[2]=float(book_point[2])
        moved=float(np.linalg.norm(np.array(observed)-np.array(book_point)))
        self.node.event('BOOK_REOBSERVED',point=observed,moved=moved,
                        planar_shift=planar,height_shift=vertical,
                        height_rejected_as_occlusion=occluded)
        if moved > self.BOOK_DISPLACED_LIMIT:
            raise RuntimeError(
                f'Target book moved {moved:.3f} m from where it was grasped; '
                'aborting instead of pushing it further')
        return observed

    def carry(self):
        # Lift the torso before the arm moves. The carry and placement reach
        # was measured at this height; with the book already clear of the
        # shelf the lift only raises it vertically.
        self.command_torso(self.PLACE_TORSO)
        self.update_fixed_scene()
        # An unconstrained plan to the transport pose matched the grasp
        # orientation only at its two ends; mid-path the wrist rotated, the
        # pinched 300 g book twisted out of the pads and landed on the floor.
        target=self._base_pose(np.array(self.CARRY_POSITION),0.)
        # A straight Cartesian line holds the grasp orientation exactly and
        # plans in well under a second. The orientation-constrained OMPL search
        # is the fallback: from the top row it needed more than 30 s and
        # returned "Unable to solve the planning problem" on its first attempt.
        try:
            self.straight([target],time_scale=self.CARRY_TIME_SCALE)
        except RuntimeError as error:
            self.node.event('CARRY_STRAIGHT_LINE_UNAVAILABLE',reason=str(error))
            self.go(target,path_tilt_tolerance=self.CARRY_TILT_TOLERANCE)
        if not self.node.wait(self.book_retained,20.):
            since=time.monotonic()-self.node.last_grasp_contact
            raise RuntimeError('Target book was not retained in transport pose '
                               f'(last finger contact {since:.1f} s ago)')
        self.node.event('BOOK_STOWED_FOR_TRANSPORT')

    def place(self,bin_point):
        self.update_fixed_scene()
        tf=self.node.vision.tf.lookup_transform('base_footprint','odom',Time())
        target=do_transform_pose(self._point_pose(bin_point),tf).position
        centre=self.place_position(target.x,target.y)
        yaw=math.atan2(centre[1],centre[0])
        pose=self._base_pose(centre,yaw)
        # Carry and placement share a height, so this is a level horizontal
        # reach: a straight line holds the book exactly as carried.
        try:
            self.straight([pose],time_scale=self.CARRY_TIME_SCALE)
        except RuntimeError as error:
            self.node.event('PLACE_STRAIGHT_LINE_UNAVAILABLE',reason=str(error))
            self.go(pose,path_tilt_tolerance=self.CARRY_TILT_TOLERANCE)
        if not self.node.wait(self.book_retained,20.):
            raise RuntimeError('Target book was not retained above collection bin')
        self.command_gripper(self.GRIPPER_OPEN_COMMAND)
        self.detach_book()
        if not self.node.wait(lambda:bool(self.node.bin_contacts),45.):
            raise RuntimeError('Released target book was not detected in collection bin')
        self.node.event('BOOK_PLACED',bin_contacts=len(self.node.bin_contacts))

    @classmethod
    def place_position(cls,x,y):
        """Grasp-frame target over the bin, clamped to the measured arm reach."""
        distance=math.hypot(x,y)
        scale=min(1.,cls.PLACE_MAX_REACH/distance) if distance > 0 else 1.
        return np.array([x*scale,y*scale,cls.PLACE_HEIGHT],dtype=float)

    @staticmethod
    def _point_pose(point):
        pose=Pose();pose.position=Point(x=float(point[0]),y=float(point[1]),z=float(point[2]))
        pose.orientation.w=1.
        return pose

    @staticmethod
    def _base_pose(point,yaw):
        pose=Pose();pose.position=Point(x=float(point[0]),y=float(point[1]),z=float(point[2]))
        pose.orientation.x=math.cos(yaw/2)
        pose.orientation.y=math.sin(yaw/2)
        pose.orientation.z=0.
        pose.orientation.w=0.
        return pose

    def update_fixed_scene(self):
        """Known fixed arena geometry only; randomized books are never read here.

        Convert into the current base frame before EACH arm operation after driving.
        The simulator's published odom axes differ from arena axes by pi/2.
        """
        share=Path(get_package_share_directory('erc_description'))
        request=ApplyPlanningScene.Request();request.scene.is_diff=True
        self.configure_matrix(request)
        for model,position in [('shelf',(3.,0.,1.1)),('table',(-1.,0.,.7)),
                               ('collection_bin',(-1.,0.,1.3))]:
            path=share/'models'/model/'meshes'/f'erc_base_{model}.STL'
            data=path.read_bytes();count=struct.unpack('<I',data[80:84])[0]
            dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')])
            if len(data)!=84+50*count:
                raise ValueError(f'Unexpected arena mesh format: {path}')
            vertices=np.frombuffer(data[84:],dtype=dtype)['vertices'].reshape(-1,3)
            mesh=Mesh(vertices=[Point(x=float(x),y=float(y),z=float(z)) for x,y,z in vertices],
                      triangles=[MeshTriangle(vertex_indices=[3*i,3*i+1,3*i+2]) for i in range(count)])
            # Arena RPY=(pi/2,0,-pi/2); then arena->odom yaw=-pi/2.
            pose=Pose();pose.position=Point(x=position[1],y=-position[0],z=position[2])
            pose.orientation.x=0.;pose.orientation.y=-math.sqrt(.5)
            pose.orientation.z=-math.sqrt(.5);pose.orientation.w=0.
            tf=self.node.vision.tf.lookup_transform('base_footprint','odom',Time())
            pose=do_transform_pose(pose,tf)
            obj=CollisionObject();obj.header.frame_id='base_footprint';obj.id='arena_'+model
            obj.meshes=[mesh];obj.mesh_poses=[pose];obj.operation=CollisionObject.ADD
            request.scene.world.collision_objects.append(obj)
        # The arena meshes are placed from wheel odometry, which drifted ~45 mm
        # from Gazebo truth in trials, and MoveIt plans with zero clearance by
        # default. A path it rated collision-free still drove arm_left_3_link
        # into the real shelf. Pad the upper arm for world collisions only
        # (MoveIt checks self-collision unpadded); the wrist and gripper links
        # stay exact because they have to work inside the shelf row.
        request.scene.link_padding=[LinkPadding(link_name=link,padding=self.UPPER_ARM_PADDING)
                                    for link in self.PADDED_ARM_LINKS]
        if not self.call(self.apply,request).success:
            raise RuntimeError('Could not update fixed arena collision scene')
