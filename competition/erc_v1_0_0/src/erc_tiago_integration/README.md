# Team Libro — ERC 2026 Phase 1

Work in progress against the official TIAGo Pro simulator. This package is **not yet
qualified as a complete competition solution**. Full trials fail explicitly until
physical grasp, transport and placement are validated. No simulated teleport or
mock grasp is counted as competition success.

## Official sources

- Simulator: https://github.com/dfl-rlab/erc_sim_2026
- Development source revision: `2aa5a4a7d19177e0bb6625ea5f2c37f7b32bd33a`.
- Phase 1 brief supplied by Team Libro: `Emirates_Robotics_Competition___Edition_4__Phase_1_.pdf`.
- The brief specifies `v1.0.0` (`ae4d1d908232ff8af5788d0fa8c8d727495514dd`).
  This is an outstanding release-compatibility gate. The newer simulator changes
  book thickness/friction and gripper controller naming. Do not claim v1.0.0
  validation using results from the development revision.

The simulator's operating system, ROS distribution, Gazebo and robot model are
unchanged. This team package runs in the official Humble/Harmonic image.

## Build and run

Place `erc_tiago_integration` inside the official workspace's `src/` directory.
Inside the official container:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select erc_tiago_integration
source install/setup.bash
# Terminal 1: official environment (must run first)
ros2 launch erc_bringup simulation.launch.py
# Terminal 2: required evaluator interface
ros2 launch erc_tiago_integration solution.launch.py shelf_column_number:=2 book_colour:=red
```

Only shelf numbers 1–5 and colours red, blue, green, yellow are accepted. The
`stage:=perception` option is a development rehearsal, **not a complete trial**.
The simulator must already be running; the solution does not spawn or reset it.

When launching through `docker exec`, invoke `/entrypoint.sh` explicitly: Docker
does not rerun the container entrypoint for exec commands. That script supplies
Gazebo's model and plugin search paths. For a separate server and GUI:

```bash
docker exec erc_sim /entrypoint.sh bash -c 'source /opt/erc_ws/install/setup.bash; ros2 launch erc_bringup simulation.launch.py headless:=true'
docker exec erc_sim /entrypoint.sh gz sim -g
```

Wait for advancing `/clock`, live RGB-D frames and controller readiness before
launching the solution. A running launch process alone does not prove the world
loaded successfully.

Live evidence defaults to `erc_images/` under the launch working directory. Launch
from the team repository root, or pass `evidence_dir:=/path/to/team/repo/erc_images`.
The output includes timestamped camera images, detection metadata, trial state
transitions and planned/executed paths. A marker-only image is not yet claimed as
meeting the rubric's full-column bounding-box requirement.

## Modules

- `navigation.launch.py`: dual-laser holonomic DWB navigation, polygon collision
  scoring, matched goal tolerances, bounded replanning without spin recovery.
- `footprint_guard.py`: projects the official collision meshes through live TF,
  updates both costmap footprints and blocks base commands during arm motion or
  stale geometry/joint data. The base, torso, arms and grippers all contribute.
- `vision.py`: live-image numeral template matching and HSV colour segmentation.
- `perception.py`: RGB-D/TF validation, repeated-measurement support and evidence.
  Depth backprojection uses depth intrinsics, timestamp and optical frame. The
  supplied simulator renders RGB and depth at the camera-link origin even though
  the advertised RGB optical frame has a 15 mm baseline. Using that RGB frame
  for depth points introduces a lateral grasp error. Physical cameras require
  calibrated depth-to-colour registration; matching image sizes alone is insufficient.
- `solution.py`: bounded action calls, measured arrival checks, vision-based target
  selection, failure reporting and cancellation.
- `manipulation.launch.py`: team-owned MoveIt planning configuration; this alone
  is not a validated grasp pipeline.
- `manipulation.py`: collision-checked Cartesian insertion/extraction with 5 mm
  sampling and a 0.12 radian maximum adjacent joint step. The relative jump filter
  is replaced by this absolute continuity check; incomplete paths still fail
  before execution. Insertion follows the fixed shelf normal, with the book
  re-observed from the final base position. Head aiming checks measured joint
  positions before accepting a fresh image. Closure requires sustained contact
  on both target fingertips, stops at measured contact, then maintains a small
  preload. The insertion depth accounts for the book's visible front face and
  the offset from MoveIt's grasp frame to the fingertip contact plane.
- `navigation_regression.py`: simulation-only routes to physical columns and back.
  These column coordinates are never used to resolve randomized shelf labels.

Command path:

```text
Nav2 controller/behaviors -> /cmd_vel_nav -> velocity_smoother
 -> /cmd_vel_smoothed -> collision_monitor -> /cmd_vel_collision
 -> footprint_guard -> /cmd_vel -> official Gazebo bridge
```

The simulator spawns at arena yaw pi/2 while MecanumDrive initializes odom yaw to
zero. Named regression routes explicitly convert arena coordinates into odometry.
The trial instead selects shelf and book targets from live camera observations.
There is currently no persistent-map localization: this is local, sensor-driven
navigation for the compact official arena. Physical deployment needs localization
and site calibration, with measured drift bounds.

## Validation

```bash
python3 -m pytest src/erc_tiago_integration/test -q
ros2 launch erc_tiago_integration navigation.launch.py
ros2 run erc_tiago_integration navigation_regression --ros-args \
  -p use_sim_time:=true -p timeout:=300.0 -p report_file:=navigation_results.json
```

Run navigation with no second navigation launch or teleoperation publisher. The
regression fails on an action error, out-of-tolerance arrival or observed contact
with shelves/books/table/bin. Contact observations are diagnostics, not a computed
judging score. Planned paths alone do not establish collision-free motion.

Collision model regeneration (only if the official supplied URDF changes):

```bash
python3 src/erc_tiago_integration/scripts/generate_collision_model.py src \
  src/erc_tiago_integration/config/collision_corners.json
```

This development tool needs NumPy and SciPy. Runtime geometry projection needs
NumPy only and rejects a URDF hash mismatch.

## Submission gates

- Validate the exact organizer-approved simulator release on a clean second machine.
- Complete and repeat live visual shelf and row identification with full rubric images.
- Validate single-arm physical grasp, possession, stowed transport, visual bin
  identification and gentle placement confirmed by `/bin_contacts`.
- Run at least five complete randomized trials and report failures as well as successes.
- Record an unedited, real-time video of at most five minutes, showing Team Libro,
  both required launch commands and a timer.
- Produce the report (maximum five pages excluding title), in the six required
  sections: architecture, perception, navigation, manipulation, results, limitations.
- Supply GitHub and YouTube links in the organizer's submission form.

The earlier Libro Jazzy reference robot's mock regression is not evidence for ERC.
