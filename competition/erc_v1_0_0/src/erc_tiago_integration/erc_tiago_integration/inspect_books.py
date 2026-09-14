"""Identify and visit every coloured book using live RGB-D, with auditable results."""
import json
import math
import time
import numpy as np
import rclpy
from std_msgs.msg import String
from .solution import Trial
from .catalogue import ObservationCatalogue, shelf_axes, approach_pose, assign_rows, COLOURS


class BookInspection(Trial):
    def __init__(self):
        super().__init__()
        self.declare_parameter('visit_books',True)
        self.inventory={}
        self.visits=[]
        self.inventory_pub=self.create_publisher(String,'/libro/book_inventory',10)

    def scan_markers(self,store,duration):
        end=time.monotonic()+duration
        while rclpy.ok() and time.monotonic()<end:
            rclpy.spin_once(self,timeout_sec=.03)
            if self.abort_reason:raise RuntimeError(self.abort_reason)
            for number in range(1,6):
                detections=self.vision.observe('column',number)
                if len(detections)==1:
                    store.add(number,detections[0])
            if len(store.confirmed())==5:
                return

    def scan_books(self,column,marker,tangent,normal,store,duration):
        end=time.monotonic()+duration
        while rclpy.ok() and time.monotonic()<end:
            rclpy.spin_once(self,timeout_sec=.03)
            if self.abort_reason:raise RuntimeError(self.abort_reason)
            if not self.vision.ready():
                continue
            candidates={colour:[] for colour in COLOURS}
            for item in self.vision.vision.books(self.vision.rgb):
                point=self.vision.point(item['bbox'])
                if point is None or not .35<point[2]<1.95:
                    continue
                offset=np.array(point[:2])-marker['point'][:2]
                if abs(np.dot(offset,tangent))>.43 or abs(np.dot(offset,normal))>.35:
                    continue
                item.update(point=point,shelf_column_number=column,
                            stamp_ns=self.vision.rgb_msg.header.stamp.sec*10**9+self.vision.rgb_msg.header.stamp.nanosec)
                candidates[item['colour']].append(item)
            for colour,items in candidates.items():
                if len(items)==1:
                    store.add(colour,items[0])
            if len(store.confirmed())==4:
                return

    def run(self):
        self.event('INSPECTION_START',team='Libro',expected_books=20)
        if not self.wait(lambda:self.odom is not None and self.vision.ready(),60.):
            raise RuntimeError('No fresh RGB-D and odometry')
        pos=self.odom.pose.pose.position
        q=self.odom.pose.pose.orientation
        yaw=math.atan2(2*q.w*q.z,1-2*q.z*q.z)
        self.start_pose=[pos.x,pos.y,yaw]
        self.stow('left');self.stow('right')
        markers=ObservationCatalogue()
        for heading in (yaw-math.pi/2,yaw,yaw+math.pi/2,yaw+math.pi):
            self.navigate(self.start_pose[0],self.start_pose[1],heading)
            for pan in (0.,.65,-.65):
                self.point_head(pan,.25)
                self.scan_markers(markers,8.)
                if len(markers.confirmed())==5:break
            if len(markers.confirmed())==5:break
        confirmed=markers.confirmed()
        self.event('MARKER_SCAN',identified=sorted(confirmed))
        if len(confirmed)!=5:
            raise RuntimeError(f'Only {len(confirmed)}/5 shelf markers confirmed visually')
        tangent,normal=shelf_axes(confirmed,self.start_pose)
        self.event('SHELF_GEOMETRY',normal=normal.tolist(),tangent=tangent.tolist(),markers=confirmed)
        # Sweep physical locations in order to reduce travel. The keys remain the
        # numbers actually recognized, independently of their randomized order.
        columns=sorted(confirmed,key=lambda k:np.dot(confirmed[k]['point'][:2],tangent))
        for column in columns:
            marker=confirmed[column]
            self.navigate(*approach_pose(marker,normal,1.45))
            books=ObservationCatalogue()
            for tilt in (0.,-.30,-.65,.25):
                self.point_head(0.,tilt)
                self.scan_books(column,marker,tangent,normal,books,8.)
                if len(books.confirmed())==4:break
            confirmed_books=assign_rows(books.confirmed())
            self.inventory[column]=confirmed_books
            self.inventory_pub.publish(String(data=json.dumps(self.inventory)))
            self.event('COLUMN_INVENTORY',shelf=column,books=confirmed_books)
            if self.get_parameter('visit_books').value:
                for colour in COLOURS:
                    target=confirmed_books[colour]
                    self.navigate(*approach_pose(target,normal))
                    reacquired=None
                    # Re-identify the requested colour at the goal. Arrival alone
                    # is not recorded as a successful visit to that book.
                    camera_z=1.15
                    tilt=float(np.clip(math.atan2(target['point'][2]-camera_z,1.1),-.8,.25))
                    self.point_head(0.,tilt)
                    verification=ObservationCatalogue()
                    self.scan_books(column,marker,tangent,normal,verification,5.)
                    reacquired=verification.confirmed().get(colour)
                    if reacquired is None or np.linalg.norm(np.array(reacquired['point'])-target['point'])>.12:
                        raise RuntimeError(f'Could not re-identify {colour} book at shelf {column} after arrival')
                    reacquired['row']=target['row']
                    evidence=self.vision.save('book',reacquired,colour)
                    item={'shelf_column_number':column,'colour':colour,'row':target['row'],
                          'point':reacquired['point'],'evidence':evidence,'passed':True}
                    self.visits.append(item)
                    self.event('BOOK_VISITED',**item)
            self.write_inventory()
        self.navigate(*self.start_pose)
        self.event('INSPECTION_COMPLETE',identified_books=sum(map(len,self.inventory.values())),
                   verified_visits=len(self.visits),competition_trial_complete=False)

    def write_inventory(self):
        (self.vision.output/'book_inspection.json').write_text(json.dumps({
            'inventory':self.inventory,'visits':self.visits,
            'all_20_identified':len(self.inventory)==5 and all(len(x)==4 for x in self.inventory.values()),
            'all_20_visited':len(self.visits)==20,'events':self.events},indent=2))

    def finish(self):
        self.write_inventory()
        super().finish()


def main(args=None):
    rclpy.init(args=args);node=None;code=1
    try:
        node=BookInspection();node.run();code=0
    except (Exception,KeyboardInterrupt) as exc:
        if node is not None:node.event('FAILED',reason=str(exc))
    finally:
        if node is not None:node.finish();node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(code)
