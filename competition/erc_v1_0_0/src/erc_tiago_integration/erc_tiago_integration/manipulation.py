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
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath
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


class Manipulator:
    def __init__(self, trial):
        self.node=trial
        self.apply=trial.create_client(ApplyPlanningScene,'/apply_planning_scene')
        self.cartesian=trial.create_client(GetCartesianPath,'/compute_cartesian_path')
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

    def go(self,pose):
        goal=MoveGroup.Goal()
        goal.request.group_name='left_arm'
        goal.request.allowed_planning_time=15.
        goal.request.num_planning_attempts=5
        goal.request.max_velocity_scaling_factor=.2
        goal.request.max_acceleration_scaling_factor=.2
        goal.request.start_state.is_diff=True
        pc=PositionConstraint();pc.header.frame_id='base_footprint';pc.link_name=self.tip;pc.weight=1.
        pc.constraint_region=BoundingVolume(primitives=[SolidPrimitive(type=SolidPrimitive.SPHERE,dimensions=[.008])],primitive_poses=[pose])
        oc=OrientationConstraint();oc.header.frame_id='base_footprint';oc.link_name=self.tip;oc.orientation=pose.orientation
        oc.absolute_x_axis_tolerance=.08;oc.absolute_y_axis_tolerance=.08;oc.absolute_z_axis_tolerance=.08;oc.weight=1.
        goal.request.goal_constraints=[Constraints(position_constraints=[pc],orientation_constraints=[oc])]
        goal.planning_options.planning_scene_diff.is_diff=True
        goal.planning_options.planning_scene_diff.robot_state.is_diff=True
        result=self.node.action(self.node.move,goal,600.)
        if result.error_code.val!=1:
            raise RuntimeError(f'Arm plan/execute failed: {result.error_code.val}')

    def straight(self,poses):
        request=GetCartesianPath.Request()
        request.header.frame_id='base_footprint';request.group_name='left_arm'
        request.link_name=self.tip;request.start_state.is_diff=True
        request.waypoints=poses;request.max_step=.005;request.jump_threshold=0.;request.avoid_collisions=True
        result=self.call(self.cartesian,request)
        if result.error_code.val!=1 or result.fraction < .999:
            raise RuntimeError(f'Incomplete collision-free Cartesian path: {result.fraction:.3f}')
        self.validate_cartesian_joints(result.solution.joint_trajectory)
        if not result.solution.joint_trajectory.points:
            raise RuntimeError('Empty Cartesian trajectory')
        # Humble Cartesian trajectories may lack timing. Refuse unparameterized
        # motion instead of streaming an instantaneous joint jump.
        end=result.solution.joint_trajectory.points[-1].time_from_start
        if end.sec==0 and end.nanosec==0:
            raise RuntimeError('Cartesian trajectory requires time parameterization')
        self.slow_cartesian(result.solution.joint_trajectory)
        goal=ExecuteTrajectory.Goal(trajectory=result.solution)
        response=self.node.action(self.execute,goal,600.)
        if response.error_code.val!=1:
            raise RuntimeError(f'Cartesian execution failed: {response.error_code.val}')

    @staticmethod
    def slow_cartesian(trajectory):
        for point in trajectory.points:
            duration=4*(point.time_from_start.sec*1_000_000_000+point.time_from_start.nanosec)
            point.time_from_start.sec,point.time_from_start.nanosec=divmod(duration,1_000_000_000)
            point.velocities=[velocity/4. for velocity in point.velocities]
            point.accelerations=[acceleration/16. for acceleration in point.accelerations]

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
        for side in ('left','right'):
            if any(any(f'gripper_left_{segment}_finger_{side}_link' in name
                       for segment in ('inner','outer')) or
                   f'gripper_left_fingertip_{side}_link' in name for name in pair):
                self.finger_contacts[side]=stamp

    def bilateral_contact(self):
        stamp=self.node.get_clock().now().nanoseconds
        return all(0 <= stamp-self.finger_contacts.get(side,-10**18) <= 1_000_000_000
                   for side in ('left','right'))

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
        stable_since=None
        unilateral_since=None
        unilateral_side=None

        def secured():
            nonlocal stable_since,unilateral_since,unilateral_side
            if self.node.abort_reason:
                raise RuntimeError(self.node.abort_reason)
            if result.done():
                response=result.result()
                if response.status != GoalStatus.STATUS_SUCCEEDED or response.result.error_code != 0:
                    raise RuntimeError('Gripper closure controller failed')
                if self.gripper_position is not None and self.gripper_position < .005:
                    raise NoBookContact('Gripper closed without contacting the target book')
            stamp=self.node.get_clock().now().nanoseconds
            closed_enough=(self.gripper_position is not None and
                           initial_opening-self.gripper_position >= .005)
            recent=[side for side in ('left','right')
                    if 0 <= stamp-self.finger_contacts.get(side,-10**18) <= 1_000_000_000]
            if closed_enough and len(recent)==1:
                stable_since=None
                if unilateral_side != recent[0]:
                    unilateral_side=recent[0]
                    unilateral_since=stamp
                elif stamp-unilateral_since >= 300_000_000:
                    raise UnilateralBookContact(unilateral_side)
                return False
            unilateral_since=None;unilateral_side=None
            if not closed_enough or len(recent)!=2:
                stable_since=None
                return False
            if stable_since is None:
                stable_since=stamp
            return stamp-stable_since >= 100_000_000

        try:
            if not self.node.wait(secured,120.):
                raise RuntimeError('No sustained two-finger target contact; extraction refused')
        except Exception:
            handle.cancel_goal_async()
            raise
        opening=self.gripper_position
        if opening is None:
            handle.cancel_goal_async()
            raise RuntimeError('Gripper position unavailable at confirmed contact')
        cancel=handle.cancel_goal_async()
        self.node.wait(cancel.done,5.)
        hold_opening=max(.05,opening-.005)
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
        shape=SolidPrimitive(type=SolidPrimitive.BOX,dimensions=[.16,.06,.25])
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

    @staticmethod
    def mechanical_pairs():
        pairs={('torso_base_link','torso_lift_link'),
               ('torso_lift_link','head_1_link'),
               ('head_2_link','head_front_camera_link')}
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
                              (f'gripper_{side}_base_finger_{finger}_link',f'gripper_{side}_screw_{finger}_link'),
                              (f'gripper_{side}_base_finger_{finger}_link',f'gripper_{side}_inner_finger_{finger}_link'),
                              (f'gripper_{side}_inner_finger_{finger}_link',f'gripper_{side}_fingertip_{finger}_link'),
                              (f'gripper_{side}_outer_finger_{finger}_link',f'gripper_{side}_fingertip_{finger}_link')})
        return pairs

    def configure_matrix(self,request,extra_pairs=None,extra_names=None):
        pairs=self.mechanical_pairs()|set(extra_pairs or ())
        names=sorted({link for pair in pairs for link in pair}|set(extra_names or ()))
        allowed=pairs|{(second,first) for first,second in pairs}|{(name,name) for name in names}
        request.scene.allowed_collision_matrix.entry_names=names
        request.scene.allowed_collision_matrix.entry_values=[AllowedCollisionEntry(
            enabled=[(first,second) in allowed for second in names]) for first in names]

    def grasp(self,book_point):
        self.command_torso(.3 if book_point[2] > 1.35 else .15 if book_point[2] > 1.05 else 0.)
        self.update_fixed_scene()
        pre,grasp,pull,lift=self.shelf_grasp_points(book_point)
        self.command_gripper(.07)
        self.allow_shelf_fingertips(True)
        pre_pose=self.pose(pre,-math.pi/2)
        grasp_pose=self.pose(grasp,-math.pi/2)
        pull_pose=self.pose(pull,-math.pi/2)
        lift_pose=self.pose(lift,-math.pi/2)
        self.go(pre_pose)
        self.straight([grasp_pose])
        for attempt in range(4):
            try:
                self.command_gripper(0.)
                break
            except (UnilateralBookContact,NoBookContact) as contact:
                if attempt==3:
                    raise
                self.command_gripper(.07)
                self.straight([pre_pose])
                if isinstance(contact,UnilateralBookContact):
                    correction=-.012 if contact.side=='left' else .012
                    side=contact.side
                else:
                    correction=(.012,-.024,.036)[attempt]
                    side='none'
                pre[0]+=correction
                grasp[0]+=correction
                pre_pose=self.pose(pre,-math.pi/2)
                grasp_pose=self.pose(grasp,-math.pi/2)
                self.straight([pre_pose,grasp_pose])
                self.node.event('GRASP_LATERAL_REALIGNED',side=side,
                                correction=float(grasp[0]-book_point[0]))
        self.attach_book()
        extraction=[]
        for distance,height in ((.08,0.),(.16,.0125)):
            point=grasp.copy();point[1]+=distance;point[2]+=height
            extraction.append(point)
        extraction.append(pull)
        for index,point in enumerate(extraction,1):
            self.straight([self.pose(point,-math.pi/2)])
            if not self.node.wait(lambda:time.monotonic()-self.node.last_grasp_contact<1.5,20.):
                raise RuntimeError(f'Target book was not retained during extraction segment {index}')
            self.node.event('BOOK_EXTRACTION_SEGMENT',segment=index)
            if index < len(extraction):
                if self.gripper_position is None:
                    raise RuntimeError('Gripper position unavailable during extraction preload')
                preload=max(.05,self.gripper_position-.003)
                if preload < self.gripper_position-.001:
                    self.command_gripper(preload)
                if not self.node.wait(self.bilateral_contact,5.):
                    raise RuntimeError('Bilateral grip lost while increasing extraction preload')
        self.node.event('BOOK_CLEAR_OF_SHELF')
        self.allow_shelf_fingertips(False)
        self.straight([lift_pose])
        if not self.node.wait(lambda:time.monotonic()-self.node.last_grasp_contact<1.5,20.):
            raise RuntimeError('Target book was not retained during lift')
        self.node.event('BOOK_GRASPED',gripper_position=self.gripper_position,
                        contacts=len(self.node.grasp_contacts))

    @staticmethod
    def shelf_grasp_points(book_point):
        centre=np.array(book_point,dtype=float)
        pre=centre.copy();pre[1]+=.30
        # The first grip catches the front corners. A short shelf-supported
        # pull exposes enough depth to reseat on the side faces without driving
        # the wrist into the shelf.
        grasp=centre.copy();grasp[1]-=.14
        pull=centre.copy();pull[1]+=.10;pull[2]+=.025
        lift=pull.copy();lift[2]+=.05
        return pre,grasp,pull,lift

    def carry(self):
        self.update_fixed_scene()
        self.go(self._base_pose(np.array([.38,.28,1.05]),0.))
        if not self.node.wait(lambda:time.monotonic()-self.node.last_grasp_contact<1.5,20.):
            raise RuntimeError('Target book was not retained in transport pose')
        self.node.event('BOOK_STOWED_FOR_TRANSPORT')

    def place(self,bin_point):
        self.update_fixed_scene()
        tf=self.node.vision.tf.lookup_transform('base_footprint','odom',Time())
        target=do_transform_pose(self._point_pose(bin_point),tf).position
        centre=np.array([target.x,target.y,max(target.z+.28,.85)],dtype=float)
        yaw=math.atan2(centre[1],centre[0])
        self.go(self._base_pose(centre,yaw))
        if not self.node.wait(lambda:time.monotonic()-self.node.last_grasp_contact<1.5,20.):
            raise RuntimeError('Target book was not retained above collection bin')
        self.command_gripper(.065)
        self.detach_book()
        if not self.node.wait(lambda:bool(self.node.bin_contacts),45.):
            raise RuntimeError('Released target book was not detected in collection bin')
        self.node.event('BOOK_PLACED',bin_contacts=len(self.node.bin_contacts))

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
        if not self.call(self.apply,request).success:
            raise RuntimeError('Could not update fixed arena collision scene')
