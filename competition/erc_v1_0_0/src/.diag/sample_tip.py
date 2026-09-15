"""Sample the real gripper tip against the true book pose while a trial runs."""
import math, subprocess, time
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

BOOK = 'book_col_5_row_2_red'
rclpy.init(); node = Node('sample_tip')
node.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
buf = Buffer(); TransformListener(buf, node)
joints = {}
node.create_subscription(JointState, '/joint_states', lambda m: joints.update(zip(m.name, m.position)), 10)

def truth():
    out = subprocess.run(['gz', 'topic', '-e', '-t', '/world/erc_world/pose/info', '-n', '1'],
                         capture_output=True, text=True, timeout=12).stdout.splitlines()
    poses = {}
    for i, line in enumerate(out):
        if line.startswith('  name: "'):
            name = line.split('"')[1]
            vals = []
            for l in out[i+1:i+14]:
                l = l.strip()
                if l[:2] in ('x:', 'y:', 'z:', 'w:'):
                    vals.append(float(l.split()[1]))
            poses[name] = vals  # [px,py,pz,qx,qy,qz,qw] (zeros omitted by gz are handled below)
    return poses

def fill(block):
    # gz omits zero-valued fields; re-read robustly instead of trusting order
    return block

end = time.monotonic() + 150
last_print = 0
while time.monotonic() < end:
    rclpy.spin_once(node, timeout_sec=.1)
    if time.monotonic() - last_print < 3.:
        continue
    last_print = time.monotonic()
    try:
        tip = buf.lookup_transform('base_footprint', 'gripper_left_grasping_link', Time()).transform.translation
    except Exception as e:
        print('tip TF unavailable'); continue
    q = joints.get('gripper_left_finger_joint', float('nan'))
    raw = subprocess.run(['bash', '-c',
        f'gz topic -e -t /world/erc_world/pose/info -n 1 | grep -A12 -E "^  name: \\"(tiago_pro|{BOOK})\\""'],
        capture_output=True, text=True, timeout=12).stdout
    blocks = {}; cur = None
    for line in raw.splitlines():
        if line.startswith('  name:'):
            cur = line.split('"')[1]; blocks[cur] = {'position': {}, 'orientation': {}}; sect = None
        elif cur and line.strip() in ('position {', 'orientation {'):
            sect = line.strip().split()[0]
        elif cur and sect and line.strip()[:2] in ('x:', 'y:', 'z:', 'w:'):
            k, v = line.strip().split(); blocks[cur][sect][k[0]] = float(v)
    if 'tiago_pro' not in blocks or BOOK not in blocks:
        print('truth unavailable'); continue
    r = blocks['tiago_pro']; b = blocks[BOOK]
    rx, ry = r['position'].get('x', 0.), r['position'].get('y', 0.)
    ryaw = 2*math.atan2(r['orientation'].get('z', 0.), r['orientation'].get('w', 1.))
    # tip in world
    wx = rx + math.cos(ryaw)*tip.x - math.sin(ryaw)*tip.y
    wy = ry + math.sin(ryaw)*tip.x + math.cos(ryaw)*tip.y
    wz = tip.z
    bx, by, bz = b['position'].get('x', 0.), b['position'].get('y', 0.), b['position'].get('z', 0.)
    front = bx - 0.08  # book depth .16 along world x, robot on -x side
    lead = 0.011859
    print(f'gripper q={q:.4f} (gap~{(0.05972-(0.07-q)*0.78286)*1000:.0f}mm) | '
          f'fingertip edge past cover={(wx+lead-front)*1000:+.0f}mm  lateral={(wy-by)*1000:+.0f}mm  '
          f'height={(wz-bz)*1000:+.0f}mm  | robot yaw={math.degrees(ryaw):+.1f}deg', flush=True)
node.destroy_node(); rclpy.shutdown()
