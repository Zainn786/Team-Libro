"""Team Libro trial entry point. Every transition requires measured evidence."""
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint
from nav_msgs.msg import Odometry, Path as NavPath
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose
from std_msgs.msg import String
from ros_gz_interfaces.msg import Contacts
from trajectory_msgs.msg import JointTrajectoryPoint

from .perception import LivePerception
from .manipulation import Manipulator


class Trial(Node):
    def __init__(self):
        super().__init__('libro_trial')
        self.declare_parameter('shelf_column_number', 0)
        self.declare_parameter('book_colour', '')
        self.declare_parameter('evidence_dir', os.path.join(os.getcwd(),'erc_images'))
        self.declare_parameter('stage', 'full')
        self.column = self.get_parameter('shelf_column_number').value
        self.colour = self.get_parameter('book_colour').value
        self.stage = self.get_parameter('stage').value
        if self.column not in range(1,6) or self.colour not in ('red','blue','green','yellow'):
            raise ValueError('shelf_column_number must be 1..5; book_colour red/blue/green/yellow')
        if self.stage not in ('full','perception'):
            raise ValueError('stage must be full or perception')
        share = Path(get_package_share_directory('erc_tiago_integration'))
        self.vision = LivePerception(self, share/'templates', self.get_parameter('evidence_dir').value)
        self.manipulator = Manipulator(self)
        self.odom = None
        self.odom_received = 0.
        self.executed_path = []
        self.planned_paths = []
        self.create_subscription(Odometry, '/odom', self.on_odom, qos_profile_sensor_data)
        self.create_subscription(NavPath, '/plan', self.on_plan, 10)
        self.nav = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.move = ActionClient(self, MoveGroup, '/move_action')
        self.head = ActionClient(self, FollowJointTrajectory, '/head_controller/follow_joint_trajectory')
        self.arm_left = ActionClient(self, FollowJointTrajectory,
                                     '/arm_left_controller/follow_joint_trajectory')
        self.arm_right = ActionClient(self, FollowJointTrajectory,
                                      '/arm_right_controller/follow_joint_trajectory')
        self.stop = self.create_publisher(Twist, '/cmd_vel_nav', 10)
        self.status = self.create_publisher(String, '/libro/trial_status', 10)
        self.events = []
        self.started = time.monotonic()
        self.abort_reason = ''
        self.contact_events = []
        self.grasp_contacts = []
        self.bin_contacts = []
        self.last_grasp_contact = -math.inf
        self.contact_phase = 'travel'
        self.target_book_token = ''
        self.completed = False
        self.create_subscription(Contacts,'/contacts',self.on_contacts,qos_profile_sensor_data)
        self.create_subscription(Contacts,'/bin_contacts',self.on_bin_contacts,qos_profile_sensor_data)
        self.start_pose = None
        self.active_goal = None

    def on_bin_contacts(self,message):
        if self.contact_phase != 'place' or not self.target_book_token:
            return
        for contact in message.contacts:
            pair=[contact.collision1.name,contact.collision2.name]
            if any(self.target_book_token in name for name in pair) and any('collection_bin' in name for name in pair):
                self.bin_contacts.append(pair)

    def on_contacts(self, message):
        for contact in message.contacts:
            pair=[contact.collision1.name,contact.collision2.name]
            joined=' '.join(pair).lower()
            near_shelf_links=('arm_left_5_link','arm_left_6_link','arm_left_7_link','gripper_left')
            if (self.contact_phase == 'grasp' and 'erc_shelf' in joined and
                    any(link in joined for link in near_shelf_links)):
                continue
            if self.target_book_token and self.target_book_token in joined:
                if 'gripper_left' in joined and self.contact_phase in ('grasp','carry','place'):
                    self.grasp_contacts.append(pair)
                    self.last_grasp_contact=time.monotonic()
                    self.manipulator.record_finger_contact(pair)
                    continue
                if 'collection_bin' in joined and self.contact_phase == 'place':
                    self.bin_contacts.append(pair)
                    continue
                if self.contact_phase in ('grasp','carry','place'):
                    continue
            if any(any(word in name.lower() for word in ('shelf','book','table','collection_bin')) for name in pair):
                if not self.abort_reason:
                    self.abort_reason='Unintended contact during travel/inspection'
                    self.contact_events.append(pair)
                    self.event('CONTACT_ABORT',pair=pair)
                    if self.active_goal is not None:
                        self.active_goal.cancel_goal_async()
                    self.stop.publish(Twist())

    def on_odom(self, message):
        self.odom, self.odom_received = message, time.monotonic()
        p = message.pose.pose.position
        if not self.executed_path or math.hypot(p.x-self.executed_path[-1][0],p.y-self.executed_path[-1][1]) > .025:
            self.executed_path.append([p.x,p.y])

    def on_plan(self, message):
        self.planned_paths.append([[p.pose.position.x,p.pose.position.y] for p in message.poses])
        self.planned_paths = self.planned_paths[-100:]

    def event(self, state, **data):
        item = {'state':state, 'wall_seconds':time.monotonic()-self.started,
                'sim_ns':self.get_clock().now().nanoseconds, **data}
        self.events.append(item)
        self.status.publish(String(data=json.dumps(item)))
        self.get_logger().info(json.dumps(item))

    def wait(self, predicate, timeout):
        deadline = time.monotonic()+timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=.05)
            if predicate():
                return True
        return False

    def action(self, client, goal, timeout):
        if self.abort_reason:
            raise RuntimeError(self.abort_reason)
        if not client.wait_for_server(timeout_sec=20.):
            raise RuntimeError(f'Action server unavailable: {client._action_name}')
        future = client.send_goal_async(goal)
        if not self.wait(future.done, 15.):
            # Retain a callback to cancel any late acceptance.
            future.add_done_callback(lambda f: f.result().cancel_goal_async() if f.result() and f.result().accepted else None)
            raise RuntimeError('Action acknowledgement timeout')
        handle = future.result()
        if not handle.accepted:
            raise RuntimeError('Action rejected')
        self.active_goal = handle
        result = handle.get_result_async()
        if not self.wait(result.done, timeout):
            cancel = handle.cancel_goal_async()
            self.wait(cancel.done, 5.)
            self.wait(result.done, 5.)
            raise RuntimeError('Action timeout; cancelled')
        self.active_goal = None
        wrapped = result.result()
        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(f'Action failed: status={wrapped.status}')
        if hasattr(wrapped.result,'error_code') and isinstance(wrapped.result.error_code,int) and wrapped.result.error_code != 0:
            raise RuntimeError(f'Trajectory error: {wrapped.result.error_code}')
        return wrapped.result

    def stow(self, side):
        fold=[0.,0.,0.,-2.2,0.,0.,0.]
        wrist_fold=[0.,0.,0.,-2.2,0.,-1.2,0.]
        home={'left':[.36,-1.83,.47,-2.35,0.,-1.2,0.],
              'right':[-.36,-1.83,-.47,-2.35,0.,-1.2,0.]}[side]
        for positions in (fold,wrist_fold,home):
            goal=MoveGroup.Goal()
            goal.request.group_name=side+'_arm'
            goal.request.allowed_planning_time=10.
            goal.request.num_planning_attempts=5
            goal.request.max_velocity_scaling_factor=.6
            goal.request.max_acceleration_scaling_factor=.6
            goal.request.start_state.is_diff=True
            goal.request.goal_constraints=[Constraints(joint_constraints=[
                JointConstraint(joint_name=f'arm_{side}_{i}_joint',position=positions[i-1],
                                tolerance_above=.01,tolerance_below=.01,weight=1.) for i in range(1,8)])]
            goal.planning_options.planning_scene_diff.is_diff=True
            goal.planning_options.planning_scene_diff.robot_state.is_diff=True
            goal.planning_options.plan_only=True
            for attempt in range(3):
                try:
                    result=self.action(self.move,goal,60.)
                except RuntimeError as error:
                    if str(error) != f'Action failed: status={GoalStatus.STATUS_ABORTED}' or attempt == 2:
                        raise
                    self.event('STOW_REPLANNING',side=side,attempt=attempt+1)
                    continue
                if result.error_code.val != 1:
                    raise RuntimeError(f'{side} stow planning failed: {result.error_code.val}')
                break
            if not result.planned_trajectory.joint_trajectory.points:
                raise RuntimeError(f'{side} stow returned an empty trajectory')
            execution=self.action(self.manipulator.execute,
                                  ExecuteTrajectory.Goal(trajectory=result.planned_trajectory),600.)
            if execution.error_code.val != 1:
                raise RuntimeError(f'{side} stow execution failed: {execution.error_code.val}')
        self.event('ARM_STOWED',side=side)

    def point_head(self, pan, tilt):
        if not math.isfinite(pan) or not math.isfinite(tilt) or not -1.0472 <= tilt <= .34907:
            raise ValueError('Head target is outside the camera tilt limits')
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ['head_1_joint','head_2_joint']
        point = JointTrajectoryPoint(positions=[float(pan),float(tilt)])
        point.time_from_start.sec = 2
        goal.trajectory.points = [point]
        self.action(self.head, goal, 120.)
        if not self.wait(lambda:all(abs(self.manipulator.head_positions.get(joint,math.inf)-target)<.04
                                   for joint,target in zip(goal.trajectory.joint_names,(pan,tilt))),30.):
            raise RuntimeError(f'Camera head did not reach requested position: {self.manipulator.head_positions}')
        stamp = self.get_clock().now().nanoseconds
        if not self.wait(lambda:self.vision.ready() and self.vision.rgb_msg.header.stamp.sec*10**9+self.vision.rgb_msg.header.stamp.nanosec > stamp+300000000,10.):
            raise RuntimeError('No fresh camera after head movement')

    def navigate(self, x, y, yaw):
        if self.odom is None or time.monotonic()-self.odom_received > 2.:
            raise RuntimeError('Odometry unavailable or stale')
        if not all(math.isfinite(v) for v in (x,y,yaw)):
            raise ValueError('Non-finite navigation goal')
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'odom'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x),float(y)
        goal.pose.pose.orientation.z,goal.pose.pose.orientation.w = math.sin(yaw/2),math.cos(yaw/2)
        self.event('NAVIGATING', target=[x,y,yaw])
        self.action(self.nav,goal,900.)
        if not self.wait(lambda: time.monotonic()-self.odom_received < .5,3.):
            raise RuntimeError('No fresh final odometry')
        p,q = self.odom.pose.pose.position,self.odom.pose.pose.orientation
        actual_yaw = math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        xy = math.hypot(p.x-x,p.y-y)
        dyaw = abs(math.atan2(math.sin(actual_yaw-yaw),math.cos(actual_yaw-yaw)))
        if xy > .12 or dyaw > .15:
            raise RuntimeError(f'Navigation final error: {xy:.3f}m, {dyaw:.3f}rad')
        self.event('ARRIVED',xy_error=xy,yaw_error=dyaw)

    def find(self, kind, target, column_point=None, timeout=12.):
        observations = []
        last_stamp = -1
        deadline = time.monotonic()+timeout
        while rclpy.ok() and time.monotonic()<deadline:
            rclpy.spin_once(self,timeout_sec=.05)
            candidates = self.vision.observe(kind,target,column_point)
            if len(candidates) != 1:
                observations=[]
                continue
            item = candidates[0]
            if item['stamp_ns'] == last_stamp:
                continue
            last_stamp = item['stamp_ns']
            if observations and np.linalg.norm(np.array(item['point'])-observations[-1]['point']) > .06:
                observations=[]
            observations.append(item)
            if len(observations)>=3:
                return item
        return None

    def aim_at_point(self,point):
        if not self.wait(self.vision.ready,15.):
            raise RuntimeError('Camera unavailable for target alignment')
        transform=self.vision.tf.lookup_transform(self.vision.depth_msg.header.frame_id,'odom',Time())
        target=do_transform_pose(self.manipulator._point_pose(point),transform).position
        tilt=self.manipulator.head_positions.get('head_2_joint')
        if tilt is None or target.z <= 0.:
            raise RuntimeError('Cannot aim camera at the selected target')
        self.point_head(0.,float(np.clip(tilt-math.atan2(target.y,math.hypot(target.x,target.z)),-1.,.3)))

    def refine_book(self,book,column_point):
        self.aim_at_point(book['point'])
        observed=self.find('book',self.colour,column_point,timeout=20.)
        if observed is None or observed['row'] != book['row'] or np.linalg.norm(np.array(observed['point'])-book['point']) > .12:
            raise RuntimeError('Selected book could not be verified at the grasp position')
        evidence=self.vision.save('book',observed,self.colour)
        self.event('BOOK_REALIGNED',detection=observed,evidence=evidence)
        return observed

    def reobserve_target(self,book_point):
        """Fresh RGB-D fix on the target book from the current base pose.

        Used between failed grasp attempts. Returns None when the book cannot
        be seen again, so the caller keeps its previous estimate rather than
        treating a missed frame as a displacement.
        """
        try:
            self.aim_at_point(book_point)
            observed=self.find('book',self.colour,book_point,timeout=12.)
        except RuntimeError:
            return None
        return observed['point'] if observed else None

    def run(self):
        self.event('STARTING', team='Libro',shelf=self.column,colour=self.colour,stage=self.stage)
        if not self.wait(lambda:self.odom is not None and self.vision.ready(),45.):
            raise RuntimeError('RGB-D, TF or odometry startup timeout')
        if not self.wait(lambda:self.get_clock().now().nanoseconds>=10_000_000_000,90.):
            raise RuntimeError('Simulation did not advance through controller startup')
        required=(self.nav,self.move,self.head,self.arm_left,self.arm_right,
                  self.manipulator.gripper,self.manipulator.torso)
        if not all(client.wait_for_server(timeout_sec=45.) for client in required):
            raise RuntimeError('Navigation, MoveIt, arm, head or gripper controller unavailable')
        if not self.wait(lambda:self.odom is not None and
                         time.monotonic()-self.odom_received<.5,3.):
            raise RuntimeError('Robot state did not settle after controller startup')
        p,q = self.odom.pose.pose.position,self.odom.pose.pose.orientation
        yaw = math.atan2(2*q.w*q.z,1-2*q.z*q.z)
        self.start_pose = [p.x,p.y,yaw]
        # Park the unused right arm; only the left arm will manipulate books.
        self.manipulator.update_fixed_scene()
        self.stow('left')
        self.stow('right')
        # Search with the camera from the measured start pose. No assumed shelf
        # number-to-column map or Gazebo object state enters target selection.
        column = None
        for heading in (yaw-math.pi/2,yaw,yaw+math.pi/2,yaw+math.pi):
            self.navigate(p.x,p.y,heading)
            for pan in (0., .4, -.4, .8, -.8, 1.15, -1.15):
                self.point_head(pan,.25)
                column = self.find('column',self.column,timeout=3.)
                if column:
                    break
            if column:
                break
        if not column:
            raise RuntimeError('Target shelf number not uniquely identified from live RGB-D')
        image = self.vision.save('column',column,self.column)
        self.event('SHELF_IDENTIFIED',detection=column,evidence=image)
        cp = column['point']
        # Approach along the current line of sight, with conservative stand-off.
        pos = self.odom.pose.pose.position
        direction = np.array(cp[:2])-np.array([pos.x,pos.y])
        direction /= np.linalg.norm(direction)
        target = np.array(cp[:2])-1.5*direction
        facing = math.atan2(direction[1],direction[0])
        self.navigate(*target,facing)
        book=None
        for tilt in (0.,-.35,-.65,.25):
            self.point_head(0.,tilt)
            book=self.find('book',self.colour,cp)
            if book:
                break
        if not book:
            raise RuntimeError('Target book not uniquely identified in selected column')
        image=self.vision.save('book',book,self.colour)
        self.event('BOOK_IDENTIFIED',detection=book,evidence=image)
        if self.stage == 'perception':
            self.event('PERCEPTION_REHEARSAL_COMPLETE',competition_trial_complete=False)
            return
        physical_column=int(np.clip(3-round(cp[0]),1,5))
        self.target_book_token=f'book_col_{physical_column}_row_{book["row"]+1}_{self.colour}'
        self.event('TARGET_BOOK_LOCKED',physical_column=physical_column,
                   token=self.target_book_token)
        # Square up to the shelf normal so the manipulation geometry is
        # identical across all five columns.
        grasp_base=Manipulator.shelf_base_pose(book['point'])
        self.navigate(*grasp_base,-math.pi/2)
        # Re-observe from the final base position.  Books are randomized within
        # each shelf cell, so the live RGB-D coordinate is the only usable
        # source; a marker-relative constant causes systematic misses and can
        # knock the book out of its row.
        book=self.refine_book(book,cp)
        self.event('BOOK_GRASP_CALIBRATED',grasp_point=[float(v) for v in book['point']],
                   base_pose=list(grasp_base))
        self.contact_phase='grasp'
        self.manipulator.grasp(book['point'])
        self.manipulator.carry()
        self.deliver()

    def deliver(self):
        self.contact_phase='carry'
        self.navigate(*self.start_pose)
        bin_detection=None
        for heading in (self.start_pose[2]+math.pi/2,self.start_pose[2],
                        self.start_pose[2]-math.pi/2,self.start_pose[2]+math.pi):
            self.navigate(self.start_pose[0],self.start_pose[1],heading)
            for pan in (0.,.65,-.65):
                self.point_head(pan,-.35)
                bin_detection=self.find('bin','red',timeout=8.)
                if bin_detection:
                    break
            if bin_detection:
                break
        if not bin_detection:
            raise RuntimeError('Red collection bin not uniquely identified from live RGB-D')
        image=self.vision.save('bin',bin_detection,'red')
        self.event('BIN_IDENTIFIED',detection=bin_detection,evidence=image)
        pos=self.odom.pose.pose.position
        direction=np.array(bin_detection['point'][:2])-np.array([pos.x,pos.y])
        direction/=np.linalg.norm(direction)
        drop_base=np.array(bin_detection['point'][:2])-.85*direction
        self.navigate(*drop_base,math.atan2(direction[1],direction[0]))
        self.aim_at_point(bin_detection['point'])
        refreshed=self.find('bin','red',timeout=10.)
        if not refreshed:
            raise RuntimeError('Red collection bin lost at placement pose')
        self.contact_phase='place'
        self.manipulator.place(refreshed['point'])
        self.completed=True
        self.event('COMPLETED',competition_trial_complete=True)

    def finish(self):
        if self.active_goal is not None:
            future=self.active_goal.cancel_goal_async()
            self.wait(future.done,5.)
        for _ in range(5):
            self.stop.publish(Twist())
            rclpy.spin_once(self,timeout_sec=.03)
        report={'events':self.events,'executed_path':self.executed_path,
                'planned_paths':self.planned_paths,'contacts':self.contact_events,
                'competition_ready':self.completed}
        (self.vision.output/f'trial_{time.time_ns()}.json').write_text(json.dumps(report,indent=2))


def main(args=None):
    rclpy.init(args=args)
    node=None
    code=1
    try:
        node=Trial()
        node.run()
        code=0
    except (Exception,KeyboardInterrupt) as exc:
        if node is not None:
            node.event('FAILED',reason=str(exc),competition_trial_complete=False)
    finally:
        if node is not None:
            node.finish()
            node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(code)
