# Team Libro — Emirates Robotics Competition 2026

Team Libro's autonomous book-retrieval solution for the official TIAGo Pro simulator.
The submission workspace uses ROS 2 Humble and Gazebo Harmonic.

## Competition workspace

- [Official simulator workspace](competition/erc_v1_0_0/README.md)
- [Team Libro integration package](competition/erc_v1_0_0/src/erc_tiago_integration/README.md)
- [TIAGo Pro limits](competition/erc_v1_0_0/docs/tiago_pro_limits.pdf)

The workspace vendors the dependencies required by the supplied simulator so a clone
contains the complete competition source tree. Generated builds, test caches, local
trial evidence, and editor files are intentionally excluded.

## Build and run

```bash
cd competition/erc_v1_0_0
./docker/up.sh --build
./docker/attach.sh

# Inside the Humble container
colcon build --symlink-install
source install/setup.bash
ros2 launch erc_bringup simulation.launch.py
```

In a second attached terminal, start the Team Libro solution with an evaluator-provided
shelf column and book colour:

```bash
source install/setup.bash
ros2 launch erc_tiago_integration solution.launch.py \
  shelf_column_number:=2 book_colour:=red
```

See the integration-package README for validation commands, architecture notes, known
limitations, and the remaining submission gates. This is competition work in progress;
full randomized grasp-and-delivery readiness must be demonstrated in the official Humble
environment before submission.
