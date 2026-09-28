# uav-gs-drl-nav

**UAV autonomous manoeuvre decision-making with global search + deep reinforcement learning** (reproducible research code)

> Companion code for the paper: *Autonomous manoeuvre decision-making for UAVs in unknown environments
> by integrating global search with deep reinforcement learning*

## Method overview

Two-stage pipeline:

1. **2D grid RL** (`drl/`): lidar-style sector observations + a global-search guidance path (A\* or FM²)
   train a local manoeuvre policy. 2.5D navigation: altitude fixed at z=4 m; action space is
   forward linear velocity + yaw rate.
2. **ROS 2 / PX4 SITL deployment and evaluation** (`ros2_ws/` + `gazebo/`): the policy is exported to
   TorchScript and run through the MAVROS offboard interface, then batch-evaluated in Gazebo
   Classic worlds.

Method variants (all trained 10M steps, seeds 42/43/44, reported as mean±std):

| Variant | Guidance field | Training script | Offline eval script |
|---------|----------------|-----------------|---------------------|
| **FM²+PPO (main method)** | FM² velocity field | `run_train_fm2_ppo.py` | `run_eval_fm2_ppo.py` |
| A\*+PPO | A\* grid shortest path | `run_train_astar_ppo.py` | `run_eval_astar_ppo.py` |
| A\*+SAC | A\* grid shortest path | `run_train_astar_sac.py` | (training-time `--do_eval` output) |
| PPO-only (ablation) | none | `run_train_ppo_plain.py` | `run_eval_ppo_plain.py` |
| SAC-only (ablation) | none | `run_train_sac_plain.py` | `run_eval_sac_plain.py` |

The planner itself has a separate static benchmark: `planner_compare.py`
(path length / minimum clearance / planning time).

## Directory layout

```
uav-gs-drl-nav/
├── drl/                  # 2D grid RL: envs, training, offline eval, policy export
│   ├── nav_env_astar.py / nav_env_fm2.py / nav_env_plain.py   # the three envs
│   ├── fastmarch_field.py     # FM² field computation (shared by training and ROS side)
│   ├── train_*.py       # training entry points for the 5 method variants
│   ├── eval_*.py        # offline eval (weighted score; see script header)
│   ├── policy_exporter.py # export TorchScript policy + observation-normalization JSON
│   ├── planner_compare.py
│   └── requirements.txt # pinned training venv (torch 2.9.1 + cu128)
├── ros2_ws/src/          # ROS 2 (colcon) workspace sources
│   ├── px4_nav_perception/       # perception: point cloud -> 2D occupancy grid
│   ├── px4_nav_planner/          # goal publisher
│   ├── px4_rl_offboard_mavros/   # offboard control + policy inference (final policies in models/)
│   └── PIPELINE_OVERVIEW.md      # node/topic/frame documentation
├── gazebo/                 # Gazebo Classic assets and simulation evaluation
│   ├── worlds/          # 12 worlds (.world files + generator scripts)
│   ├── models/iris_rplidar_depth/   # Iris model with 360° RPLiDAR + depth camera
│   ├── run_ros_eval.py  # batch ROS simulation eval (success rate / collisions / timeouts)
│   └── wsl_sitl_gui.sh  # manual GUI full-stack launch
├── setup/               # environment setup (WSL2 + Docker + PX4, run in numbered order)
└── deploy_to_px4.sh     # deploy this repo into the PX4-Autopilot tree
```

## Quick start

### 0. Prerequisites

WSL2 + Ubuntu, Docker, git. PX4-Autopilot itself is **not in this repo**; clone the official
repository separately (this work is based on **v1.16.0-rc1**).

### 1. Environment setup (inside host WSL, run in numbered order)

```bash
cd ~/uav-gs-drl-nav/setup
bash 00_wsl_docker_check.sh        # WSL2/Docker self-check
bash 01_clone_and_build_px4.sh     # clone and build PX4 SITL (use clone_px4.sh on CN networks)
bash 02_deploy_repo.sh             # this repo -> ~/PX4-Autopilot (calls deploy_to_px4.sh)
bash 03_px4docker_alias.sh         # install the px4docker alias
bash 04_container_runtime_deps.sh  # MAVROS/torch/scipy inside the container (run in container)
bash 05_build_ros2_ws.sh           # colcon build of ros2_ws (run in container)
bash 06_rl_venv.sh                 # drl/.venv training environment (run in container)
bash 99_verify.sh                  # full-stack self-check
```

### 2. RL training (inside the container, `drl/` directory)

```bash
px4docker                              # enter the container
cd /home/dev/PX4-Autopilot/rl && source .venv/bin/activate

# main method FM²+PPO (10M steps; run seeds 42/43/44 via --seed)
python run_train_fm2_ppo.py --total_timesteps 10000000 \
  --target_kl 0.1 --n_epochs 10 --log_interval 1

# baselines / ablations (same protocol)
python run_train_astar_ppo.py  --total_timesteps 10000000   # A*+PPO
python run_train_astar_sac.py  --total_timesteps 10000000 --save_best --do_eval   # A*+SAC
python run_train_ppo_plain.py   --total_timesteps 10000000   # PPO-only
python run_train_sac_plain.py   --total_timesteps 10000000   # SAC-only
```

### 3. Offline evaluation and export

```bash
# offline eval (weighted score; PYTHONHASHSEED=0 fixes the scenario-seed protocol)
PYTHONUTF8=1 PYTHONHASHSEED=0 python run_eval_fm2_ppo.py \
  --model ./models_ppo_2d/<run_dir>/ppo_uav2d_final.zip \
  --vecnorm ./models_ppo_2d/<run_dir>/vecnormalize.pkl --runs 100

# export TorchScript policy -> ROS deployment
python policy_exporter.py --algo ppo \
  --model ./models_ppo_2d/<run_dir>/ppo_uav2d_final.zip \
  --vecnorm ./models_ppo_2d/<run_dir>/vecnormalize.pkl \
  --policy_filename policy_fm2_ts.pt --norm_filename obs_norm_fm2.json \
  --output_dir ./export
# copy the exported files into ros2_ws/src/px4_rl_offboard_mavros/models/ and re-deploy
```

> `ros2_ws/.../models/` already ships **4 final policies** exported after 10M steps
> (`policy_fm2_ts.pt` / `policy_hybrid_ts.pt` (A\*+SAC) / `policy_ppo_ts.pt` / `policy_sac_ts.pt`
> plus the matching `obs_norm_*.json`), so you can **skip training and run ROS simulation
> directly**.

### 4. ROS / PX4 simulation evaluation (inside the container)

```bash
# batch eval: method (fm2/astar_ppo/ppo/sac) x world (obstacle_course/military_airfield)
#             x course (R/L) x 8 episodes; incremental CSV logging, resumable
python3 gazebo/run_ros_eval.py
python3 gazebo/run_ros_eval.py --methods fm2 --worlds obstacle_course --episodes 2   # smoke test

# manual GUI full stack (12 maps x 4 methods)
bash gazebo/wsl_sitl_gui.sh [map] [method] [course]    # default: urban_city fm2 R
```

`nav_mode` (astar/ppo/sac/fm2) and goals are configured in
`ros2_ws/src/px4_rl_offboard_mavros/config/params.yaml`
(each method has its own `params_*.yaml`) and `px4_nav_planner/config/goals.yaml`.
See [`ros2_ws/src/PIPELINE_OVERVIEW.md`](ros2_ws/src/PIPELINE_OVERVIEW.md) for the pipeline
structure and coordinate frames.

## Version baseline

| Component | Version |
|-----------|---------|
| PX4-Autopilot | v1.16.0-rc1 (official repo, cloned separately) |
| ROS 2 | Humble (`osrf/ros:humble-desktop-full` container) |
| Python training env | see `drl/requirements.txt` (gymnasium 1.2.3 / SB3 2.7.1 / torch 2.9.1+cu128) |
| Simulation | Gazebo Classic (bundled with PX4 SITL) |

## Naming map from the original workspace (astar_ppo_hybrid-main)

This repo is a cleaned-up, renamed subset of the research workspace `astar_ppo_hybrid-main`:

| Original file | Current file |
|---------------|--------------|
| `drl/env_2d_nav.py` | `drl/nav_env_astar.py` |
| `drl/env_2d_nav_ppo_only.py` | `drl/nav_env_plain.py` |
| `drl/env_2d_nav_fm2.py` | `drl/nav_env_fm2.py` |
| `drl/train_ppo_2d.py` | `drl/run_train_astar_ppo.py` |
| `drl/train_sac_2d.py` | `drl/run_train_astar_sac.py` |
| `drl/train_ppo_2d_ppo_only.py` | `drl/run_train_ppo_plain.py` |
| `drl/train_sac_2d_ppo_only.py` | `drl/run_train_sac_plain.py` |
| `drl/train_ppo_2d_fm2.py` | `drl/run_train_fm2_ppo.py` |
| `drl/eval_ppo_offline.py` | `drl/run_eval_astar_ppo.py` |
| `drl/eval_ppo_only_offline.py` | `drl/run_eval_ppo_plain.py` |
| `drl/eval_sac_only_offline.py` | `drl/run_eval_sac_plain.py` |
| `drl/eval_fm2_offline.py` | `drl/run_eval_fm2_ppo.py` |

Training logs, model checkpoints, figures, and paper-writing material are not tracked;
the full original workspace is backed up at
`Scientific_Reports/astar_ppo_hybrid-main_backup_2026-09-27/`.
