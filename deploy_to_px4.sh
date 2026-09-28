#!/usr/bin/env bash
# deploy_to_px4.sh
# Deploy this repo (uav-gs-drl-nav) code/assets into the cloned PX4-Autopilot tree:
#   - ros2_ws/src         -> $PX4/ros2_ws/src        (ROS2 packages, mirrored)
#   - drl/ sources (.py/requirements.txt) -> $PX4/drl (training code, add-only, never delete)
#   - gazebo/ assets        -> PX4 Gazebo worlds/models + start_px4.sh/start_mavros.sh
#
# docker (px4docker) only mounts ~/PX4-Autopilot, so this script must sync the
# repo into the PX4 tree before building/running.
#
# Usage:
#   ./deploy_to_px4.sh [PX4_DIR]
#   PX4_DIR=/path/to/PX4-Autopilot ./deploy_to_px4.sh
# With no argument/env var, common locations are auto-detected.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- locate the PX4 directory ---
PX4="${1:-${PX4_DIR:-}}"
if [[ -z "$PX4" ]]; then
  for cand in /home/dev/PX4-Autopilot "$HOME/PX4-Autopilot" ./PX4-Autopilot ../PX4-Autopilot; do
    [[ -d "$cand/Tools/simulation/gazebo-classic/sitl_gazebo-classic" ]] && { PX4="$cand"; break; }
  done
fi
if [[ -z "$PX4" || ! -d "$PX4/Tools/simulation/gazebo-classic/sitl_gazebo-classic" ]]; then
  echo "[ERR] PX4-Autopilot tree not found. Usage: $0 /path/to/PX4-Autopilot"; exit 1
fi
echo "[INFO] PX4 = $PX4"

have_rsync() { command -v rsync >/dev/null 2>&1; }

# --- 1) ros2_ws/src (ROS2 packages): mirror sync ---
mkdir -p "$PX4/ros2_ws/src"
if have_rsync; then
  rsync -a --delete --exclude="__pycache__/" --exclude="*.pyc" \
    "$REPO/ros2_ws/src/" "$PX4/ros2_ws/src/"
else
  cp -r "$REPO/ros2_ws/src/." "$PX4/ros2_ws/src/"
fi
echo "[OK] ros2_ws/src deployed"

# --- 2) drl/ sources (keep venv/models/logs; nothing deleted) ---
mkdir -p "$PX4/drl"
for f in env_guided_astar.py env_unguided.py env_guided_fm2.py fm2_field.py \
         curve_monitor_callback.py \
         train_astar_ppo.py train_astar_sac.py train_ppo_only.py train_sac_only.py \
         train_fm2_ppo.py \
         eval_astar_ppo.py eval_ppo_only.py eval_sac_only.py eval_fm2_ppo.py \
         export_policy.py bench_fm2_vs_astar.py requirements.txt; do
  [[ -f "$REPO/drl/$f" ]] && cp -f "$REPO/drl/$f" "$PX4/drl/$f"
done
echo "[OK] drl sources deployed (venv/models/logs untouched)"

# --- 3) gazebo assets (worlds/models/launch scripts) ---
bash "$REPO/gazebo/install_sim_assets.sh" "$PX4"

echo
echo "[DONE] Deployment complete. Next step:"
echo "  In container: cd $PX4/ros2_ws && colcon build --symlink-install && source install/setup.bash"
