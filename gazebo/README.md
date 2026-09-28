# gazebo/ — Custom Gazebo Classic assets and simulation evaluation

This repo does NOT contain PX4-Autopilot itself (clone it separately); it only keeps
the **custom-made** Gazebo assets and evaluation tools of this research. After cloning
PX4, use `install_sim_assets.sh` (or `deploy_to_px4.sh` in the repo root) to copy the
assets into the PX4 directory tree.

## Contents

| Path | Content |
|------|---------|
| `worlds/obstacle_course.world` | 20x20 m narrow S-shaped corridor course (R/L) — paper evaluation world 1 |
| `worlds/military_airfield.world` | 50x50 m military camp / city-block map (R/L) — paper evaluation world 2 |
| `worlds/<other 10>.world` | Generator worlds (building/canyon/forest/maze/new_world1/pole_grid/rings/rubble/urban_city/warehouse), for generalization experiments and demos |
| `worlds/_*_gen.py`, `worlds/world_render*.py` | Generator and render scripts for each world (regenerate .world files or produce figures) |
| `models/iris_rplidar_depth/` | Iris drone model with 360° RPLiDAR + depth camera |
| `start_px4.sh` / `start_mavros.sh` | One-shot SITL+Gazebo and MAVROS launchers (installed into `$PX4/` root) |
| `run_ros_eval.py` | ROS simulation batch evaluation: method x world x course x episode, incremental CSV logging, resume support |
| `wsl_sitl_gui.sh` | Manual full-stack GUI launch (SITL/MAVROS/perception/navigation in one shot, Ctrl+C cleans up everything) |

The two paper evaluation worlds only `<include>` Gazebo base models
(`sun`/`ground_plane`); obstacles are inline SDF boxes with no external model
dependencies; obstacle height 8 m, cruise altitude z=4 m. For coordinate details see
[`../ros2_ws/src/PIPELINE_OVERVIEW.md`](../ros2_ws/src/PIPELINE_OVERVIEW.md).

## Install

```bash
cd uav-gs-drl-nav/gazebo
./install_sim_assets.sh                     # auto-detect PX4 location (/home/dev/PX4-Autopilot etc.)
./install_sim_assets.sh /path/to/PX4-Autopilot   # or specify explicitly
```

The script copies `worlds/*.world`, `models/iris_rplidar_depth/`, and the two start
scripts into the matching locations of the PX4 tree.

## Usage

```bash
# Batch evaluation (inside WSL container; CSV output goes to <repo>/drl/data/ and ~/ros_eval_results.csv by default)
python3 gazebo/run_ros_eval.py
python3 gazebo/run_ros_eval.py --methods fm2 --worlds obstacle_course --episodes 2   # smoke test
python3 gazebo/run_ros_eval.py --resume           # skip completed episodes and resume

# Manual GUI (inside a WSL interactive terminal)
bash gazebo/wsl_sitl_gui.sh [world] [method] [course]
#   world: urban_city(default) obstacle_course military_airfield new_world1 building ...
#   method: fm2(default) | astar | ppo | sac
#   course: R(default) | L
```

> When modifying worlds, edit the originals in this directory's `worlds/` (or edit
> `_*_gen.py` and regenerate), then run `install_sim_assets.sh` to sync — do not edit
> the copies inside the PX4 tree directly.
