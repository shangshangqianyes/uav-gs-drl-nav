#!/usr/bin/env python3
"""RL Offboard: obs(lidar+goal_rel_body) -> policy -> velocity setpoint (local/world ENU).
A single velocity setpoint stream handles OFFBOARD/ARM acquisition + homing + RL control (single-flow design)."""

from __future__ import annotations

import heapq
import json
import math
import threading
import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data

try:
    from scipy import ndimage as _ndimage
    from scipy.interpolate import splprep as _splprep, splev as _splev
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

# FM2 (Fast Marching Square) guidance field -- in-package copy of drl/fm2_field.py (pure numpy, no external deps).
# Used only for nav_mode=fm2. If the import fails, only that mode is disabled (other modes unaffected).
try:
    from .fm2_field import fm2_path as _fm2_path_fn
    _FM2_OK = True
except ImportError:
    _FM2_OK = False

from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from nav_msgs.msg import Path as NavPath
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, CommandBool

from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose_stamped


def quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def sanitize_lidar(arr: np.ndarray, max_range: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.where(np.isnan(arr) | np.isinf(arr), max_range, arr)
    return np.clip(arr, 0.0, max_range).astype(np.float32)


def downsample_lidar(lidar: np.ndarray, target: int, mode: str = "min") -> np.ndarray:
    n = len(lidar)
    if n == target:
        return lidar.copy().astype(np.float32)
    if target > n:
        idx = np.linspace(0, n - 1, target).astype(int)
        return lidar[idx].astype(np.float32)
    indices = np.array_split(np.arange(n), target)
    if mode == "min":
        return np.array([float(np.min(lidar[i])) for i in indices], dtype=np.float32)
    if mode == "avg":
        return np.array([float(np.mean(lidar[i])) for i in indices], dtype=np.float32)
    raise ValueError(f"downsample_mode must be 'min' or 'avg', got {mode}")


class RLOffboardNode(Node):
    def __init__(self):
        super().__init__("rl_offboard_node")

        # Params (declare + get in one pass). See config/params.yaml for rationale.
        params = [
            ("mavros_state_topic", "/mavros/state"),
            ("mavros_odom_topic", "/mavros/local_position/odom"),
            ("mavros_setpoint_topic", "/mavros/setpoint_velocity/cmd_vel_unstamped"),
            ("scan_topic", "/scan"),
            ("goal_topic", "/planner/goal"),
            ("homing_done_topic", "/planner/homing_done"),
            ("wait_for_homing", True),
            # Single-flow homing: rl_offboard_node flies to home_xyz itself, then switches to goal.
            # course_goal_publisher no longer publishes position setpoints (avoids velocity-stream conflicts).
            ("home_xyz", [5.0, 0.0, 4.0]),
            ("spawn_enu", [0.0, 0.0, 0.83]),
            ("home_tolerance", 0.3),
            ("home_timeout_sec", 20.0),
            ("setpoint_rate_hz", 20.0),
            ("pre_setpoint_sec", 2.0),
            ("policy_path", ""),
            ("obs_norm_path", ""),
            ("obs_beams", 64),
            ("max_range", 10.0),
            ("world_size", 50.0),
            ("downsample_mode", "min"),
            ("odom_timeout_sec", 0.3),
            ("scan_timeout_sec", 0.5),
            ("goal_radius", 1.5),
            ("nav_mode", "hybrid"),
            ("grid_topic", "/local_grid"),
            ("occ_threshold", 50),
            ("allow_diag", True),
            ("replan_period", 1.0),
            ("planner_max_yaw_rate", 1.0),
            ("wp_advance_search_window", 10),
            ("rl_front_deg", 90.0),
            ("lidar_range_min_override", 0.35),
            ("emergency_brake_dist", 0.0),
            ("emergency_brake_deg", 30.0),
            ("enable_altitude_hold", True),
            ("altitude_hold_margin", 0.1),
            ("altitude_hold_gain", 0.4),
            ("mode_retry_sec", 1.0),
            ("arm_retry_sec", 1.0),
            ("astar_min_inflation_cells",      1),
            ("shortcut_los_clearance_cells",   1),
            ("hybrid_lookahead_dist_m",        1.0),
            ("smooth_bspline_enabled",         True),
            ("smooth_bspline_s",               0.0),
            ("smooth_bspline_output_points",   50),
            ("smooth_bspline_skip_if_few_wps", 3),
            # FM2 (nav_mode=fm2): same parameters as the drl/env_guided_fm2.py training env.
            # cell_size reuses the /local_grid resolution (_grid_res) as-is (0.2m, close to training's 0.25m).
            ("fm2_alpha",                      1.0),
            ("fm2_beta",                       1.0),
        ]
        for name, default in params:
            self.declare_parameter(name, default)
        for name, _ in params:
            setattr(self, name, self.get_parameter(name).value)
        # Disable smoothing if scipy is not installed
        if not _SCIPY_OK:
            self.smooth_bspline_enabled = False
            self.get_logger().warn("scipy missing: B-spline smoothing disabled.")

        # Per-nav_mode default model files (when policy_path/obs_norm_path are unset).
        #   hybrid -> policy_hybrid_ts.pt / obs_norm_hybrid.json
        #   ppo    -> policy_ppo_ts.pt    / obs_norm_ppo.json
        #   astar  -> no policy needed
        policy_path = self.policy_path or ""
        obs_norm_path = self.obs_norm_path or ""
        if self.nav_mode != "astar" and (not policy_path or not obs_norm_path):
            if self.nav_mode == "hybrid":
                default_policy  = "policy_hybrid_ts.pt"
                default_obs_norm = "obs_norm_hybrid.json"
            elif self.nav_mode == "fm2":
                default_policy  = "policy_fm2_ts.pt"
                default_obs_norm = "obs_norm_fm2.json"
            else:  # "ppo"
                default_policy  = "policy_ppo_ts.pt"
                default_obs_norm = "obs_norm_ppo.json"
            try:
                from ament_index_python.packages import get_package_share_directory
                pkg = Path(get_package_share_directory("px4_rl_offboard_mavros"))
            except Exception:
                pkg = Path(__file__).resolve().parents[2]
            policy_path   = policy_path   or str(pkg / "models" / default_policy)
            obs_norm_path = obs_norm_path or str(pkg / "models" / default_obs_norm)

        # Load obs_norm + policy (not needed for astar mode)
        self.obs_dim = self.obs_beams + 2
        self.obs_mean = np.zeros(self.obs_dim, dtype=np.float32)
        self.obs_std  = np.ones(self.obs_dim, dtype=np.float32)
        self.clip_obs = 10.0
        self.v_min = 0.3
        self.v_max = 0.8
        self.max_yaw_rate = 1.0
        self.policy = None
        self._torch = None
        if self.nav_mode != "astar":
            with open(obs_norm_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            obs_cfg = cfg["obs"]
            self.obs_dim = int(obs_cfg.get("obs_dim", self.obs_beams + 2))
            if self.obs_dim != self.obs_beams + 2:
                raise ValueError(f"obs_dim mismatch: {self.obs_dim} != {self.obs_beams + 2}")
            self.obs_mean = np.array(obs_cfg["mean"], dtype=np.float32)
            self.obs_std  = np.array(obs_cfg["std"],  dtype=np.float32)
            self.clip_obs = float(obs_cfg.get("clip_obs", 10.0))
            act = cfg["action"]["scale"]
            self.v_min = float(act["v_min"])
            self.v_max = float(act["v_max"])
            self.max_yaw_rate = float(act["max_yaw_rate"])
            import torch
            self._torch = torch
            self.policy = torch.jit.load(policy_path, map_location="cpu")
            self.policy.eval()

        # State
        self.mav_state: Optional[State] = None
        self.odom: Optional[Odometry] = None
        self.scan: Optional[LaserScan] = None
        self.goal_xy: Optional[np.ndarray] = None   # ENU (same frame as odom)
        # Sensor timestamps use the sim clock (avoids wall-clock false positives when Gazebo slows down)
        self.t_last_odom = self.get_clock().now()
        self.t_last_scan = self.get_clock().now()
        self._pre_setpoint_count = 0
        self._pre_setpoint_needed = int(max(0, self.pre_setpoint_sec) * self.setpoint_rate_hz)
        self._t_last_mode_req = 0.0
        self._t_last_arm_req = 0.0
        self._homing_done = not self.wait_for_homing
        self._goal_reached = False

        # Single-flow homing state: fly to home_xyz (spawn-relative ENU) first, then switch to goal.
        # With wait_for_homing=False the HOMING phase is skipped and we go straight to NAVIGATING.
        try:
            sx, sy, sz = float(self.spawn_enu[0]), float(self.spawn_enu[1]), float(self.spawn_enu[2])
        except (TypeError, IndexError, ValueError):
            sx, sy, sz = 0.0, 0.0, 0.0
        try:
            hx, hy, hz = float(self.home_xyz[0]), float(self.home_xyz[1]), float(self.home_xyz[2])
        except (TypeError, IndexError, ValueError):
            hx, hy, hz = 0.0, 0.0, 2.5
        # home_xy / home_z: in the odom local frame (same coordinate system as goal_xy = spawn-relative ENU)
        self.home_xy: Optional[np.ndarray] = np.array(
            [hx - sx, hy - sy], dtype=np.float32
        ) if self.wait_for_homing else None
        self.home_z: float = float(hz - sz)
        # _phase: "HOMING" | "NAVIGATING". During HOMING the control target is home_xy; during NAVIGATING, goal_xy.
        self._phase: str = "HOMING" if self.wait_for_homing else "NAVIGATING"
        self._homing_start_time: Optional[float] = None
        self._homing_logged_start = False
        self._homing_logged_done = False
        self._scan_warn_last = 0.0   # throttle for scan-missing diagnostic logs
        self._emergency_brake_active = False  # log only on emergency-brake entry
        self._cached_ranges: Optional[np.ndarray] = None  # cached scan ranges (avoid re-creating)
        self._brake_count = 0                               # consecutive brake-tick counter
        self._brake_rotate_left: Optional[bool] = None      # decided once on brake entry, fixed until release
        self._path_xy: List[Tuple[float, float]] = []
        self._path_wp_idx = 0
        self._have_grid = False
        self._grid_data: Optional[np.ndarray] = None
        self._grid_ox = self._grid_oy = 0.0
        self._grid_res = 0.1
        self._grid_w = self._grid_h = 0
        self._hold_z: Optional[float] = None  # Altitude hold reference
        self._plan_s_ij: Optional[Tuple[int, int]] = None
        self._plan_g_ij: Optional[Tuple[int, int]] = None
        self._t_last_debug_log = 0.0

        # QoS -- /scan must explicitly match the Gazebo RPLidar's qos_profile_sensor_data (no messages on mismatch)
        q_sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
        q_scan = qos_profile_sensor_data
        q_goal = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1,
                           durability=DurabilityPolicy.TRANSIENT_LOCAL)
        q_setpoint = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=10)

        # Callback groups -- tick/replan/sensor run on separate threads to avoid A* blocking.
        # tick and replan are MutuallyExclusive (single execution each); sensor is Reentrant (store-only, concurrent OK).
        self._cb_group_tick   = MutuallyExclusiveCallbackGroup()
        self._cb_group_replan = MutuallyExclusiveCallbackGroup()
        self._cb_group_sensor = ReentrantCallbackGroup()

        # _path_lock: prevents tick<->replan races on _path_xy/_path_wp_idx (RLock allows re-entry from the same thread).
        self._path_lock = threading.RLock()
        # _grid_lock: guarantees atomic grid update/read between _cb_grid(sensor) and _plan_path(replan).
        self._grid_lock = threading.Lock()

        # Subs
        self.create_subscription(State, self.mavros_state_topic, self._cb_state, q_sensor,
                                 callback_group=self._cb_group_sensor)
        self.create_subscription(Odometry, self.mavros_odom_topic, self._cb_odom, q_sensor,
                                 callback_group=self._cb_group_sensor)
        self.create_subscription(LaserScan, self.scan_topic, self._cb_scan, q_scan,
                                 callback_group=self._cb_group_sensor)
        self.create_subscription(PoseStamped, self.goal_topic, self._cb_goal, q_goal,
                                 callback_group=self._cb_group_sensor)
        if self.nav_mode in ("hybrid", "astar", "fm2"):
            self.create_subscription(OccupancyGrid, self.grid_topic, self._cb_grid, q_sensor,
                                     callback_group=self._cb_group_sensor)
        if self.wait_for_homing:
            self.create_subscription(Bool, self.homing_done_topic, self._cb_homing_done, q_goal,
                                     callback_group=self._cb_group_sensor)

        # Pub
        self._pub = self.create_publisher(Twist, self.mavros_setpoint_topic, q_setpoint)
        self._path_pub = self.create_publisher(NavPath, '/planner/path', 10)
        self._cli_mode = self.create_client(SetMode, "/mavros/set_mode")
        self._cli_arm = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.create_timer(1.0 / self.setpoint_rate_hz, self._tick,
                          callback_group=self._cb_group_tick)
        if self.nav_mode in ("hybrid", "astar", "fm2"):
            self.create_timer(self.replan_period, self._tick_replan,
                              callback_group=self._cb_group_replan)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            f"RL Offboard node started (single velocity stream). odom={self.mavros_odom_topic} "
            f"scan={self.scan_topic} goal={self.goal_topic} nav_mode={self.nav_mode} "
            f"phase={self._phase} rate={self.setpoint_rate_hz}Hz"
        )

    def _cb_state(self, msg: State):
        self.mav_state = msg

    def _cb_odom(self, msg: Odometry):
        self.odom = msg
        self.t_last_odom = self.get_clock().now()

    def _cb_scan(self, msg: LaserScan):
        self.scan = msg
        self.t_last_scan = self.get_clock().now()
        # Cache scan ranges once -> shared by _dist_in_sector/_build_obs (self-detection handled via lidar_range_min_override)
        self._cached_ranges = np.array(msg.ranges, dtype=np.float32)

    def _cb_goal(self, msg: PoseStamped):
        # Transform /planner/goal into the odom frame
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom',
                msg.header.frame_id,
                rclpy.time.Time()
            )
            transformed_pose = do_transform_pose_stamped(msg, transform)
            self.goal_xy = np.array(
                [transformed_pose.pose.position.x, transformed_pose.pose.position.y],
                dtype=np.float32
            )
            if self.nav_mode in ("hybrid", "astar", "fm2"):
                self._plan_path()
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed for goal: {e}. Ignore this goal update.")
            return

    def _cb_grid(self, msg: OccupancyGrid):
        new_h   = msg.info.height
        new_w   = msg.info.width
        new_res = msg.info.resolution
        new_ox  = msg.info.origin.position.x
        new_oy  = msg.info.origin.position.y
        new_data = np.array(msg.data, dtype=np.int16).reshape(new_h, new_w)
        with self._grid_lock:
            self._grid_h    = new_h
            self._grid_w    = new_w
            self._grid_res  = new_res
            self._grid_ox   = new_ox
            self._grid_oy   = new_oy
            self._grid_data = new_data
            self._have_grid = True
        self.get_logger().debug(
            f"[GRID] Received: {self._grid_w}x{self._grid_h} cells, res={self._grid_res}m, "
            f"origin=({self._grid_ox:.1f},{self._grid_oy:.1f}), "
            f"bounds x:[{self._grid_ox:.1f}, {self._grid_ox + self._grid_w*self._grid_res:.1f}] "
            f"y:[{self._grid_oy:.1f}, {self._grid_oy + self._grid_h*self._grid_res:.1f}]"
        )

    def _cb_homing_done(self, msg: Bool):
        # In the single-flow model, homing is performed by rl_offboard_node itself (owned by the internal phase machine).
        # External homing_done signals are ignored (course_goal_publisher may send them for compatibility, but phase transitions
        # occur only on home arrival/timeout). Conflict prevention: external signals must not bypass internal homing.
        if msg.data and self._phase == "HOMING":
            self.get_logger().debug(
                "External homing_done=True received but ignored (single-stream: homing owned internally)."
            )

    def _finish_homing(self, reason: str) -> None:
        """HOMING -> NAVIGATING transition. _hold_z is fixed to the current altitude (altitude hold reference)."""
        if self._phase != "HOMING":
            return
        self._phase = "NAVIGATING"
        self._homing_done = True
        if self.odom is not None and self._hold_z is None:
            self._hold_z = float(self.odom.pose.pose.position.z)
        if not self._homing_logged_done:
            self._homing_logged_done = True
            self.get_logger().info(f"Homing done ({reason}). Switching to navigation toward goal.")

    def _active_goal_xy(self) -> Optional[np.ndarray]:
        """Active goal for the current phase (HOMING->home_xy, NAVIGATING->goal_xy)."""
        if self._phase == "HOMING":
            return self.home_xy
        return self.goal_xy

    def _active_goal_z(self) -> float:
        """Active altitude goal for the current phase (HOMING->home_z, NAVIGATING->_hold_z or home_z)."""
        if self._phase == "HOMING":
            return self.home_z
        # NAVIGATING: prefer _hold_z (altitude hold reference); fall back to home_z if unset.
        return self._hold_z if self._hold_z is not None else self.home_z

    def _publish(self, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0, yaw_rate: float = 0.0):
        """vx, vy: velocities in the local/world (ENU) frame. Published to cmd_vel_unstamped."""
        vx = 0.0 if not math.isfinite(vx) else max(-self.v_max, min(self.v_max, float(vx)))
        vy = 0.0 if not math.isfinite(vy) else max(-self.v_max, min(self.v_max, float(vy)))
        vz = 0.0 if not math.isfinite(vz) else max(-0.5, min(0.5, float(vz)))
        # yaw_rate clamp: planner_max_yaw_rate for astar, max_yaw_rate for hybrid/ppo (protects training distribution)
        if self.nav_mode == "astar":
            yaw_limit = float(getattr(self, 'planner_max_yaw_rate', self.max_yaw_rate))
        else:
            yaw_limit = self.max_yaw_rate
        yaw_rate = 0.0 if not math.isfinite(yaw_rate) else max(-yaw_limit, min(yaw_limit, float(yaw_rate)))
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = vx, vy, vz
        msg.angular.z = yaw_rate
        self._pub.publish(msg)

    def _sensors_ok(self) -> bool:
        """Freshness check for odom/scan/goal (based on sim clock)."""
        now = self.get_clock().now()
        odom_age = (now - self.t_last_odom).nanoseconds / 1e9
        scan_age = (now - self.t_last_scan).nanoseconds / 1e9

        if self.odom is None or odom_age > self.odom_timeout_sec:
            return False
        if self.scan is None or scan_age > self.scan_timeout_sec:
            if self.allow_diag:
                wall_now = time.time()
                if (wall_now - self._scan_warn_last) > 5.0:
                    self._scan_warn_last = wall_now
                    if self.scan is None:
                        self.get_logger().warn(
                            f"[SCAN] {self.scan_topic} never received. "
                            "Possible QoS mismatch: check publisher QoS with ros2 topic info /scan -v and "
                            "verify it matches qos_profile_sensor_data (BEST_EFFORT+VOLATILE)."
                        )
                    else:
                        self.get_logger().warn(
                            f"[SCAN] {self.scan_topic} timeout (>{self.scan_timeout_sec}s, age={scan_age:.2f}s). "
                            "Gazebo load or temporary scan interruption. QoS is OK."
                        )
            return False
        if self._active_goal_xy() is None:
            return False
        return True

    def _build_obs(self) -> Optional[np.ndarray]:
        """Build the observation vector: lidar(64) + goal_rel(body frame, 2) = obs_dim(66).
        Uses home_xy as the target in HOMING phase, goal_xy in NAVIGATING (single-flow homing support)."""
        goal_xy = self._active_goal_xy()
        if self.odom is None or self.scan is None or goal_xy is None:
            return None
        # Prefer the scan cache (created once in the scan callback)
        if self._cached_ranges is not None and len(self._cached_ranges) == len(self.scan.ranges):
            raw = self._cached_ranges.copy()
        else:
            raw = np.array(self.scan.ranges, dtype=np.float32)
        rmin = float(getattr(self.scan, "range_min", 0.0))
        override = float(getattr(self, "lidar_range_min_override", 0.35) or 0.0)
        # Apply the self-detection override for RL observations (emergency brake skips it with use_override=False)
        if override > rmin:
            rmin = override
        raw = np.where(np.isfinite(raw) & (raw > rmin), raw, self.max_range)
        ranges = sanitize_lidar(raw, self.max_range)
        if len(ranges) != self.obs_beams:
            ranges = downsample_lidar(ranges, self.obs_beams, self.downsample_mode)
        lidar = np.clip(ranges / self.max_range, 0.0, 1.0) * 2.0 - 1.0
        px, py = float(self.odom.pose.pose.position.x), float(self.odom.pose.pose.position.y)
        yaw = quat_to_yaw(self.odom.pose.pose.orientation)

        # scan-odom dead reckoning correction: extrapolate position/yaw by the timestamp difference -> improves goal_rel accuracy
        if self.scan is not None and self.scan.header.stamp.sec != 0:
            scan_t = rclpy.time.Time.from_msg(self.scan.header.stamp).nanoseconds / 1e9
            odom_t = rclpy.time.Time.from_msg(self.odom.header.stamp).nanoseconds / 1e9
            dt = scan_t - odom_t
            if abs(dt) > 0.01:
                vx_b = self.odom.twist.twist.linear.x
                vy_b = self.odom.twist.twist.linear.y
                wz   = self.odom.twist.twist.angular.z
                vx_w = vx_b * math.cos(yaw) - vy_b * math.sin(yaw)
                vy_w = vx_b * math.sin(yaw) + vy_b * math.cos(yaw)
                px  += vx_w * dt
                py  += vy_w * dt
                yaw += wz   * dt

        # goal_rel target: a distance-based lookahead point on the path for hybrid, the final goal for ppo.
        # norm_scale is fixed to the training value 3.0 for hybrid (preserves input distribution); ppo uses world_size normalization.
        LOOKAHEAD_NORM_SCALE = 3.0
        lookahead_dist = float(getattr(self, "hybrid_lookahead_dist_m", 1.0))
        if self.nav_mode in ("hybrid", "fm2") and self._path_xy:
            wp = self._path_lookahead_wp_at_dist(px, py, lookahead_dist)
            target_xy = np.array(wp, dtype=np.float32) if wp else goal_xy
            norm_scale = LOOKAHEAD_NORM_SCALE
        else:
            target_xy = goal_xy
            norm_scale = max(self.world_size, 1e-6)

        dx_world = (target_xy - np.array([px, py], dtype=np.float32)) / norm_scale
        c, s = math.cos(-yaw), math.sin(-yaw)
        dx_body = c * dx_world[0] - s * dx_world[1]
        dy_body = s * dx_world[0] + c * dx_world[1]
        goal_rel = np.clip(np.array([dx_body, dy_body], dtype=np.float32), -1.0, 1.0)
        obs = np.concatenate([lidar.astype(np.float32), goal_rel])
        return obs if obs.shape[0] == self.obs_dim else None

    def _infer(self, obs: np.ndarray) -> np.ndarray:
        obs_norm = np.clip((obs - self.obs_mean) / self.obs_std, -self.clip_obs, self.clip_obs).astype(np.float32)
        if np.any(np.isnan(obs_norm)) or np.any(np.isinf(obs_norm)):
            return np.zeros(2, dtype=np.float32)
        with self._torch.no_grad():
            t = self._torch.from_numpy(obs_norm).float().unsqueeze(0)
            out = self.policy(t).detach().cpu().numpy().flatten().astype(np.float32)
        if out.shape[0] < 2 or np.any(np.isnan(out)) or np.any(np.isinf(out)):
            return np.zeros(2, dtype=np.float32)
        return np.clip(out, -1.0, 1.0)

    def _action_to_vel(self, action: np.ndarray) -> Tuple[float, float]:
        v_cmd = self.v_min + (action[0] + 1.0) * 0.5 * (self.v_max - self.v_min)
        yaw_rate = action[1] * self.max_yaw_rate
        return float(v_cmd), float(yaw_rate)

    def _compute_cmd_vel(self, v_cmd: float, yaw: float) -> Tuple[float, float]:
        """Convert heading-direction speed to local/world (ENU) vx/vy."""
        vx = v_cmd * math.cos(yaw)
        vy = v_cmd * math.sin(yaw)
        return vx, vy

    def _vz_correction(self) -> float:
        if not self.enable_altitude_hold or self.odom is None:
            return 0.0
        # Per-phase altitude target: HOMING->home_z (e.g. 4m), NAVIGATING->_hold_z (fixed at home arrival).
        target = self._active_goal_z()
        if target is None:
            return 0.0
        err = target - float(self.odom.pose.pose.position.z)
        if abs(err) <= self.altitude_hold_margin:
            return 0.0
        return max(-0.5, min(0.5, self.altitude_hold_gain * err))

    def _dist_in_sector(self, low_deg: float, high_deg: float, agg: str, use_override: bool = True) -> float:
        """Aggregate LiDAR distances over a body-frame angular sector (deg). agg: min|median|mean.
        If use_override=False, lidar_range_min_override is not applied to rmin (e.g. emergency brake)."""
        if self.scan is None or not self.scan.ranges:
            return self.max_range
        if self._cached_ranges is not None and len(self._cached_ranges) == len(self.scan.ranges):
            ranges = self._cached_ranges.copy()
        else:
            ranges = np.array(self.scan.ranges, dtype=np.float32)
        rmin = float(getattr(self.scan, "range_min", 0.0))
        override = float(getattr(self, "lidar_range_min_override", 0.35) or 0.0)
        if use_override and override > rmin:
            rmin = override
        ranges = np.where(np.isfinite(ranges) & (ranges > rmin), ranges, self.max_range)
        ang_min = float(self.scan.angle_min)
        inc = float(self.scan.angle_increment)
        # Compare in [0, 2*pi) space (avoids wrap-around where the atan2 (-pi,pi] range misses sectors wider than 180deg)
        _TWO_PI = 2.0 * math.pi
        low_pos  = math.radians(low_deg)  % _TWO_PI
        high_pos = math.radians(high_deg) % _TWO_PI
        vals: List[float] = []
        for i in range(len(ranges)):
            ang_pos = (ang_min + i * inc) % _TWO_PI
            if low_pos <= high_pos:
                in_sector = low_pos <= ang_pos <= high_pos
            else:
                # Sector crosses the 0/2*pi boundary (e.g. 350deg~10deg)
                in_sector = ang_pos >= low_pos or ang_pos <= high_pos
            if in_sector:
                vals.append(float(ranges[i]))
        if not vals:
            return self.max_range
        a = agg.lower()
        if a == "median":
            return float(np.median(np.array(vals, dtype=np.float32)))
        if a == "mean":
            return float(np.mean(np.array(vals, dtype=np.float32)))
        return float(min(vals))

    def _xy_to_ij(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        if not self._have_grid:
            return None
        j = int(math.floor((x - self._grid_ox) / self._grid_res))
        i = int(math.floor((y - self._grid_oy) / self._grid_res))
        if 0 <= i < self._grid_h and 0 <= j < self._grid_w:
            return (i, j)
        return None

    def _ij_to_xy(self, i: int, j: int) -> Tuple[float, float]:
        x = self._grid_ox + (j + 0.5) * self._grid_res
        y = self._grid_oy + (i + 0.5) * self._grid_res
        return (x, y)

    def _is_occ(self, i: int, j: int) -> bool:
        if i < 0 or i >= self._grid_h or j < 0 or j >= self._grid_w:
            return True
        return self._grid_data[i, j] >= self.occ_threshold

    def _is_plan_blocked(self, i: int, j: int, grid: Optional[np.ndarray] = None) -> bool:
        """A* cell blocked check (occ>=threshold). start/goal cells are exempt.
        grid: self._grid_data if None. Pass an independent grid for concurrent calls to avoid races."""
        if i < 0 or i >= self._grid_h or j < 0 or j >= self._grid_w:
            return True
        g = grid if grid is not None else self._grid_data
        val = int(g[i, j])
        if val >= self.occ_threshold:
            return True
        if self._plan_s_ij is not None and (i, j) == self._plan_s_ij:
            return False
        if self._plan_g_ij is not None and (i, j) == self._plan_g_ij:
            return False
        return False

    @staticmethod
    def _h_octile(i: int, j: int, gi: int, gj: int) -> float:
        dx, dy = abs(i - gi), abs(j - gj)
        return max(dx, dy) + (math.sqrt(2) - 1.0) * min(dx, dy)

    def _astar_neighbors(self, i: int, j: int, grid: Optional[np.ndarray] = None):
        steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if self.allow_diag:
            steps += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for di, dj in steps:
            ni, nj = i + di, j + dj
            if 0 <= ni < self._grid_h and 0 <= nj < self._grid_w and not self._is_plan_blocked(ni, nj, grid):
                yield ni, nj, math.hypot(di, dj)

    def _astar(
        self,
        start_ij: Tuple[int, int],
        goal_ij: Tuple[int, int],
        grid: Optional[np.ndarray] = None,
    ) -> Optional[List[Tuple[int, int]]]:
        """A* path search. grid=None uses self._grid_data (pass an independent grid for thread safety)."""
        si, sj = start_ij
        gi, gj = goal_ij
        if (si, sj) == (gi, gj):
            return [(si, sj)]
        openq = []
        heapq.heappush(openq, (self._h_octile(si, sj, gi, gj), 0.0, (si, sj)))
        gscore = {(si, sj): 0.0}
        parent = {(si, sj): None}
        visited = set()
        while openq:
            _, gc, (i, j) = heapq.heappop(openq)
            if (i, j) in visited:
                continue
            visited.add((i, j))
            if (i, j) == (gi, gj):
                break
            for ni, nj, step in self._astar_neighbors(i, j, grid):
                ng = gc + step
                if ng < gscore.get((ni, nj), 1e18):
                    gscore[(ni, nj)] = ng
                    parent[(ni, nj)] = (i, j)
                    heapq.heappush(openq, (ng + self._h_octile(ni, nj, gi, gj), ng, (ni, nj)))
        if (gi, gj) not in parent:
            return None
        path = []
        node = (gi, gj)
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path

    # -- Path post-processing (LOS shortcut + inflation + B-spline smoothing) ----------

    def _bresenham_cells(self, i0: int, j0: int, i1: int, j1: int):
        """Yield all (i, j) grid cells on the Bresenham line segment."""
        di = abs(i1 - i0)
        dj = abs(j1 - j0)
        si = 1 if i0 < i1 else -1
        sj = 1 if j0 < j1 else -1
        err = di - dj
        i, j = i0, j0
        while True:
            yield i, j
            if i == i1 and j == j1:
                break
            e2 = 2 * err
            if e2 > -dj:
                err -= dj
                i += si
            if e2 < di:
                err += di
                j += sj

    def _los_clear(
        self,
        x0: float, y0: float,
        x1: float, y1: float,
        grid: Optional[np.ndarray] = None,
    ) -> bool:
        """True if the segment between two world coordinates has no obstacles. grid=None uses raw self._grid_data."""
        if not self._have_grid:
            return True
        c0 = self._xy_to_ij(x0, y0)
        c1 = self._xy_to_ij(x1, y1)
        if c0 is None or c1 is None:
            return False
        g = grid if grid is not None else self._grid_data
        for ci, cj in self._bresenham_cells(c0[0], c0[1], c1[0], c1[1]):
            if ci < 0 or ci >= self._grid_h or cj < 0 or cj >= self._grid_w:
                return False
            if int(g[ci, cj]) >= self.occ_threshold:
                return False
        return True

    def _shortcut_path(self, path_xy: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Greedy LOS-based path shortcutting. If clearance>0, the LOS check uses an inflation margin (avoids wall proximity)."""
        if len(path_xy) <= 2:
            return list(path_xy)
        clearance = int(getattr(self, 'shortcut_los_clearance_cells', 1))
        los_grid = self._inflate_grid_temp(clearance) if clearance > 0 else None
        result = [path_xy[0]]
        i = 0
        while i < len(path_xy) - 1:
            # Try the farthest j first
            j = len(path_xy) - 1
            while j > i + 1:
                if self._los_clear(path_xy[i][0], path_xy[i][1],
                                   path_xy[j][0], path_xy[j][1],
                                   grid=los_grid):
                    break
                j -= 1
            result.append(path_xy[j])
            i = j
        return result

    def _inflate_grid_temp(self, radius_cells: int) -> np.ndarray:
        """Return a temp grid with obstacles inflated by radius_cells (original _grid_data unchanged)."""
        grid = self._grid_data  # capture the reference once (atomic in CPython)
        occ_mask = (grid >= self.occ_threshold)
        if radius_cells > 0:
            inflated = _ndimage.binary_dilation(occ_mask, iterations=radius_cells)
        else:
            inflated = occ_mask
        result = grid.copy()
        result[inflated & ~occ_mask] = int(self.occ_threshold)
        return result

    def _astar_with_inflation(
        self,
        start_ij: Tuple[int, int],
        goal_ij: Tuple[int, int],
        radius_cells: int,
    ) -> Optional[List[Tuple[int, int]]]:
        """Run A* on the inflated grid (start/goal cells are force-cleared)."""
        inflated = self._inflate_grid_temp(radius_cells)
        # Force-clear the start/goal cells in case they became blocked by inflation
        si, sj = start_ij
        gi, gj = goal_ij
        inflated[si, sj] = min(inflated[si, sj], int(self.occ_threshold) - 1)
        inflated[gi, gj] = min(inflated[gi, gj], int(self.occ_threshold) - 1)
        return self._astar(start_ij, goal_ij, grid=inflated)

    def _smooth_path_bspline(
        self,
        path_xy: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """Smooth the path with a B-spline. Returns the original path if any output point lies inside an obstacle."""
        if not _SCIPY_OK or len(path_xy) < 4:
            return path_xy
        try:
            xs = [p[0] for p in path_xy]
            ys = [p[1] for p in path_xy]
            s = float(getattr(self, 'smooth_bspline_s', 0.0))
            n_out = int(getattr(self, 'smooth_bspline_output_points', 50))
            tck, _ = _splprep([xs, ys], s=s, k=min(3, len(path_xy) - 1))
            u_fine = np.linspace(0.0, 1.0, n_out)
            out_x, out_y = _splev(u_fine, tck)
            result = list(zip(out_x.tolist(), out_y.tolist()))
            if self._have_grid:
                min_r = int(getattr(self, 'astar_min_inflation_cells', 1))
                if min_r > 0:
                    inflated_grid = self._inflate_grid_temp(min_r)
                    for (rx, ry) in result:
                        ij = self._xy_to_ij(rx, ry)
                        if ij is not None and inflated_grid[ij[0], ij[1]] >= self.occ_threshold:
                            return path_xy
                else:
                    for (rx, ry) in result:
                        ij = self._xy_to_ij(rx, ry)
                        if ij is not None and self._is_occ(*ij):
                            return path_xy
            result[-1] = path_xy[-1]
            return result
        except Exception as e:
            self.get_logger().warn(f"B-spline smoothing failed ({type(e).__name__}: {e}), returning original path.")
            return path_xy

    def _publish_path_msg(self) -> None:
        """Publish the current _path_xy as nav_msgs/Path."""
        if not self._path_xy:
            return
        msg = NavPath()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        z = self._active_goal_z()
        for (x, y) in self._path_xy:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = float(z)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self._path_pub.publish(msg)

    # -- Path planning --------------------------------------------------------

    def _plan_path(self) -> None:
        """A* path planning. Falls back to a straight line (goal) if no grid or on failure.
        In HOMING phase only a straight flight to home (avoids complex path planning before the target switch)."""
        goal_xy = self._active_goal_xy()
        if self.odom is None or goal_xy is None:
            return
        px = float(self.odom.pose.pose.position.x)
        py = float(self.odom.pose.pose.position.y)
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        if self._phase == "HOMING":
            # HOMING: straight setpoint to home only (skip A* path planning, for a fast phase transition)
            with self._path_lock:
                self._path_xy = [(gx, gy)]
                self._path_wp_idx = 0
            self._publish_path_msg()
            return
        if self._have_grid:
            s_ij = self._xy_to_ij(px, py)
            g_ij = self._xy_to_ij(gx, gy)
            self.get_logger().debug(
                f"[ASTAR] start_xy=({px:.2f},{py:.2f}) -> ij={s_ij}, "
                f"goal_xy=({gx:.2f},{gy:.2f}) -> ij={g_ij}"
            )
            if not s_ij:
                if self.allow_diag:
                    self.get_logger().warn(
                        f"Path fallback (straight): start ({px:.1f},{py:.1f}) outside grid "
                        f"[bounds x:[{self._grid_ox:.1f},{self._grid_ox+self._grid_w*self._grid_res:.1f}] "
                        f"y:[{self._grid_oy:.1f},{self._grid_oy+self._grid_h*self._grid_res:.1f}]]"
                    )
            elif not g_ij:
                if self.allow_diag:
                    self.get_logger().warn(
                        f"Path fallback (straight): goal ({gx:.1f},{gy:.1f}) outside grid "
                        f"[bounds x:[{self._grid_ox:.1f},{self._grid_ox+self._grid_w*self._grid_res:.1f}] "
                        f"y:[{self._grid_oy:.1f},{self._grid_oy+self._grid_h*self._grid_res:.1f}]]"
                    )
            else:
                # Even if start/goal is occupied due to inflation, force-clear and try A* (secures a detour path)
                si, sj = s_ij
                gi, gj = g_ij
                if self._is_occ(*s_ij):
                    if self.allow_diag:
                        val = int(self._grid_data[s_ij])
                        self.get_logger().warn(
                            f"A* start cell occupied (ij={s_ij}, val={val}): "
                            f"force-clearing start for A* (inflation artifact)"
                        )
                if self._is_occ(*g_ij):
                    if self.allow_diag:
                        val = int(self._grid_data[g_ij])
                        self.get_logger().warn(
                            f"A* goal cell occupied (ij={g_ij}, val={val}): "
                            f"force-clearing goal for A* (inflation artifact)"
                        )
                self._plan_s_ij = s_ij
                self._plan_g_ij = g_ij
                try:
                    if self.nav_mode == "fm2":
                        # FM2 (the method proposed in this paper): same as the training env (env_guided_fm2),
                        # use the raw path without LOS shortcut / B-spline post-processing.
                        path_xy = self._fm2_plan_path(px, py, gx, gy)
                        if path_xy:
                            path_xy[-1] = (gx, gy)
                            with self._path_lock:
                                self._path_xy = path_xy
                                self._path_wp_idx = 0
                            self.get_logger().info(
                                f"FM2 succeeded: {len(path_xy)} wps, "
                                f"start_ij={s_ij}, goal_ij={g_ij}"
                            )
                            self._publish_path_msg()
                            return
                        if self.allow_diag:
                            self.get_logger().warn(
                                f"Path fallback (straight): FM2 failed (start={s_ij}, goal={g_ij})"
                            )
                    else:
                        # Gradual fallback min_r -> 0 guarantees a path even in narrow passages
                        min_r = int(getattr(self, 'astar_min_inflation_cells', 1))
                        path_ij = None
                        for r in range(min_r, -1, -1):
                            path_ij = self._astar_with_inflation(s_ij, g_ij, radius_cells=r)
                            if path_ij:
                                if r < min_r:
                                    self.get_logger().warn(
                                        f"A* min_inflation reduced {min_r} -> {r} (narrow passage)"
                                    )
                                break

                        if path_ij:
                            path_xy = [self._ij_to_xy(ci, cj) for ci, cj in path_ij]
                            path_xy = self._shortcut_path(path_xy)           # LOS shortcutting
                            skip_n = int(getattr(self, 'smooth_bspline_skip_if_few_wps', 3))
                            if self.smooth_bspline_enabled and len(path_xy) >= skip_n:
                                path_xy = self._smooth_path_bspline(path_xy)  # B-spline smoothing
                            # Fix the goal coordinates, then store (_path_lock prevents races with the tick thread)
                            path_xy[-1] = (gx, gy)
                            with self._path_lock:
                                self._path_xy = path_xy
                                self._path_wp_idx = 0
                            self.get_logger().info(
                                f"A* succeeded: {len(path_xy)} wps "
                                f"(smoothed={self.smooth_bspline_enabled}), "
                                f"start_ij={s_ij}, goal_ij={g_ij}"
                            )
                            self._publish_path_msg()
                            return
                        if self.allow_diag:
                            self.get_logger().warn(
                                f"Path fallback (straight): A* failed to find path (start={s_ij}, goal={g_ij})"
                            )
                finally:
                    self._plan_s_ij = None
                    self._plan_g_ij = None
        else:
            if self.allow_diag:
                self.get_logger().warn(
                    f"Path fallback (straight): no grid (_have_grid=False)"
                )
        # fallback: replace with a straight path (goal) (_path_lock prevents races with the tick thread)
        self.get_logger().warn("Path fallback (straight): using direct goal line.")
        with self._path_lock:
            self._path_xy = [(gx, gy)]
            self._path_wp_idx = 0
        self._publish_path_msg()

    def _fm2_plan_path(self, px: float, py: float, gx: float, gy: float) -> Optional[List[Tuple[float, float]]]:
        """FM2 (nav_mode=fm2) path generation. /local_grid -> velocity field W -> Eikonal T -> gradient descent path.

        Consistency with the drl/env_guided_fm2.py training env:
          - input grid: 1=obstacle (>= occ_threshold), cell_size=/local_grid resolution (_grid_res)
          - no LOS shortcut / B-spline post-processing (not used in training either)
          - on failure the [goal] fallback is handled inside fm2_path -> same outcome as the caller-side straight fallback
        Coordinates: fm2_field assumes the grid origin (lower-left) frame -> convert world->grid-local and back."""
        if not _FM2_OK:
            self.get_logger().error("Failed to load fm2_field module: nav_mode=fm2 unavailable")
            return None
        with self._grid_lock:
            occ = (self._grid_data >= self.occ_threshold).astype(np.uint8)
            ox, oy = float(self._grid_ox), float(self._grid_oy)
            res = float(self._grid_res)
        if occ.shape[0] != occ.shape[1]:
            self.get_logger().warn(f"FM2: non-square grid ({occ.shape}) -- skip")
            return None
        t0 = time.time()
        path_local = _fm2_path_fn(
            occ, (px - ox, py - oy), (gx - ox, gy - oy),
            cell_size=res, alpha=float(self.fm2_alpha), beta=float(self.fm2_beta),
        )
        dt_ms = (time.time() - t0) * 1000.0
        if dt_ms > 500.0:  # log only when exceeding half of the replan thread budget (2s)
            self.get_logger().warn(f"FM2 compute slow: {dt_ms:.0f} ms (grid {occ.shape[0]}x{occ.shape[1]})")
        return [(x + ox, y + oy) for (x, y) in path_local]

    def _tick_replan(self) -> None:
        """Periodic replanning (dedicated replan thread). Direct calls from the tick thread are forbidden (race risk)."""
        if not (self.nav_mode in ("hybrid", "astar", "fm2")
                and self.odom is not None
                and self.goal_xy is not None
                and not self._goal_reached):
            return
        self._plan_path()

    def _path_current_wp(self) -> Optional[Tuple[float, float]]:
        with self._path_lock:
            if not self._path_xy or self._path_wp_idx >= len(self._path_xy):
                return None
            return self._path_xy[self._path_wp_idx]

    def _path_lookahead_wp_at_dist(
        self, px: float, py: float, lookahead_dist: float
    ) -> Optional[Tuple[float, float]]:
        """Return the point lookahead_dist(m) ahead on the path via distance-based interpolation.
        Distance-based lookahead points a fixed distance ahead regardless of path wp count (avoids the step-count goal-jump problem)."""
        with self._path_lock:
            if not self._path_xy:
                return None
            path = self._path_xy
            n = len(path)
            start_idx = max(0, min(self._path_wp_idx, n - 1))

        # Skip already-passed start segments: if dot product<0 (angle>90deg), advance start_idx.
        # (blocks the bug where lookahead returns a point behind, causing a 180deg turn)
        while start_idx + 1 < n:
            ax, ay = path[start_idx]
            bx, by = path[start_idx + 1]
            if (ax - px) * (bx - ax) + (ay - py) * (by - ay) < 0.0:
                start_idx += 1
            else:
                break

        # Include the distance from the current position to path[start_idx] as the first segment
        prev_x, prev_y = px, py
        accum = 0.0
        for i in range(start_idx, n):
            wp_x, wp_y = path[i]
            seg = math.hypot(wp_x - prev_x, wp_y - prev_y)
            if seg < 1e-6:
                prev_x, prev_y = wp_x, wp_y
                continue
            if accum + seg >= lookahead_dist:
                t = (lookahead_dist - accum) / seg
                return (prev_x + t * (wp_x - prev_x), prev_y + t * (wp_y - prev_y))
            accum += seg
            prev_x, prev_y = wp_x, wp_y

        # If the remaining path is shorter than lookahead_dist, return one step ahead of start_idx (prevents stop deadlock)
        return path[min(n - 1, start_idx + 1)]

    def _path_advance_wp(self, px: float, py: float):
        """Update wp_idx by perpendicular projection onto the nearest path segment (robust to lateral deviation)."""
        with self._path_lock:
            if not self._path_xy or self._path_wp_idx >= len(self._path_xy):
                return
            n = len(self._path_xy)
            search_window = int(getattr(self, 'wp_advance_search_window', 10))
            search_end = min(n - 1, self._path_wp_idx + search_window)
            best_seg_idx = self._path_wp_idx
            best_t       = 0.0
            best_dist    = float('inf')
            for i in range(self._path_wp_idx, search_end):
                ax, ay = self._path_xy[i]
                bx, by = self._path_xy[i + 1]
                dx, dy = bx - ax, by - ay
                seg_len_sq = dx * dx + dy * dy
                if seg_len_sq < 1e-10:
                    continue
                t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
                t = max(0.0, min(1.0, t))
                cx = ax + t * dx
                cy = ay + t * dy
                d  = math.hypot(px - cx, py - cy)
                if d < best_dist:
                    best_dist    = d
                    best_seg_idx = i
                    best_t       = t
            if best_t > 0.5 and best_seg_idx + 1 < n:
                self._path_wp_idx = max(self._path_wp_idx, best_seg_idx + 1)
            else:
                self._path_wp_idx = max(self._path_wp_idx, best_seg_idx)

    def _tick_global(self, px: float, py: float, yaw: float, now: float) -> Tuple[float, float, float, float]:
        """Global planning: align heading toward the lookahead wp, then velocity toward the target (local/world ENU frame)."""
        self._path_advance_wp(px, py)
        wp = self._path_lookahead_wp_at_dist(px, py, self.hybrid_lookahead_dist_m)
        curr_wp = self._path_current_wp()

        if now - self._t_last_debug_log >= 1.0:
            self.get_logger().debug(
                f"[GLOBAL_WP] curr_idx={self._path_wp_idx}, "
                f"curr_wp={curr_wp}, lookahead_wp={wp}, "
                f"robot=({px:.2f},{py:.2f}), path_len={len(self._path_xy)}"
            )
            self._t_last_debug_log = now

        if wp is None:
            gx, gy = self.goal_xy[0] - px, self.goal_xy[1] - py
        else:
            gx, gy = wp[0] - px, wp[1] - py
        gnorm = math.hypot(gx, gy)
        if gnorm < 1e-4:
            return 0.0, 0.0, self._vz_correction(), 0.0
        target_yaw = math.atan2(gy, gx)
        yaw_err = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
        # Rotate proportionally to yaw error + slow down for larger errors (no alignment dead zone, natural cornering)
        planner_yaw_limit = float(getattr(self, 'planner_max_yaw_rate', self.max_yaw_rate))
        yaw_rate = max(-planner_yaw_limit, min(planner_yaw_limit, 2.0 * yaw_err))
        align_scale = max(0.0, 1.0 - abs(yaw_err) / (math.pi / 2))
        v_cmd = min(self.v_max, 0.8 * gnorm) * align_scale
        vx = v_cmd * math.cos(target_yaw)
        vy = v_cmd * math.sin(target_yaw)
        return vx, vy, self._vz_correction(), yaw_rate

    def _tick(self):
        """20Hz control loop: single velocity stream for OFFBOARD/ARM acquisition -> (HOMING->NAVIGATING) -> emergency brake -> per-nav_mode control.
        Keeps publishing velocity setpoints during homing (maintains PX4 OFFBOARD continuity, removes conflict with position plugin)."""
        now = time.time()
        if self.mav_state is None or not self.mav_state.connected:
            return
        # (Option A single flow) publish setpoints even during homing -- prevents stream interruption.
        # The old wait_for_homing stop gate was removed (course_goal_publisher no longer uses position setpoints).
        if self._pre_setpoint_count < self._pre_setpoint_needed:
            self._pre_setpoint_count += 1
            self._publish()
            return
        if self.mav_state.mode != "OFFBOARD":
            if now - self._t_last_mode_req > self.mode_retry_sec:
                self._t_last_mode_req = now
                if self._cli_mode.service_is_ready():
                    req = SetMode.Request()
                    req.base_mode, req.custom_mode = 0, "OFFBOARD"
                    self._cli_mode.call_async(req)
                    self.get_logger().info("Requested OFFBOARD.")
            self._publish()
            return
        if not self.mav_state.armed:
            if now - self._t_last_arm_req > self.arm_retry_sec:
                self._t_last_arm_req = now
                if self._cli_arm.service_is_ready():
                    req = CommandBool.Request()
                    req.value = True
                    self._cli_arm.call_async(req)
                    self.get_logger().info("Requested ARM.")
            self._publish()
            return

        if not self._sensors_ok():
            self._publish()
            return

        px = float(self.odom.pose.pose.position.x)
        py = float(self.odom.pose.pose.position.y)

        # HOMING phase: switch to NAVIGATING on reaching home_xy or on timeout.
        # _hold_z is fixed to the actual altitude at home arrival (subsequent altitude hold reference).
        if self._phase == "HOMING":
            if not self._homing_logged_start:
                self._homing_logged_start = True
                if self.home_xy is not None:
                    self.get_logger().info(
                        f"Homing to home_xy=({self.home_xy[0]:.2f},{self.home_xy[1]:.2f}) "
                        f"z={self.home_z:.2f} (tol={self.home_tolerance}m, timeout={self.home_timeout_sec}s)."
                    )
                if self._homing_start_time is None:
                    self._homing_start_time = now
            elif self._homing_start_time is None:
                self._homing_start_time = now
            if self.home_xy is not None:
                hdist = math.hypot(self.home_xy[0] - px, self.home_xy[1] - py)
                elapsed = now - self._homing_start_time
                if hdist <= self.home_tolerance:
                    self._finish_homing(reason=f"reached home (dist={hdist:.2f}m)")
                elif elapsed >= self.home_timeout_sec:
                    self._finish_homing(reason=f"home timeout ({elapsed:.1f}s, dist={hdist:.2f}m)")
            # Even if the phase just switched, continue this tick with the new goal (logic below)

        goal_xy = self._active_goal_xy()
        if goal_xy is None:
            self._publish()
            return
        dist = math.hypot(goal_xy[0] - px, goal_xy[1] - py)
        # Reaching the final goal is only meaningful in the NAVIGATING phase (HOMING arrival handled separately above).
        if self._phase == "NAVIGATING" and (self._goal_reached or dist < self.goal_radius):
            if not self._goal_reached:
                self._goal_reached = True
                self.get_logger().info(f"Goal reached (dist={dist:.2f}m). Complete stop.")
            self._publish(0.0, 0.0, 0.0, 0.0)
            return

        yaw = quat_to_yaw(self.odom.pose.pose.orientation)

        half = getattr(self, "rl_front_deg", 90.0) / 2.0

        # Emergency brake: if front min distance <= threshold, rotate + retreat. use_override filters body self-detection.
        d_front_min = self._dist_in_sector(-half, half, "min", use_override=True)
        if d_front_min <= self.emergency_brake_dist:
            d_left  = self._dist_in_sector(self.emergency_brake_deg, 90.0, "min", use_override=False)
            d_right = self._dist_in_sector(-90.0, -self.emergency_brake_deg, "min", use_override=False)
            self._brake_count += 1

            # Decide the rotation direction only once on entry, fixed until brake release (prevents direction oscillation)
            if self._brake_rotate_left is None:
                rotate_reason = "lidar"
                rotate_left = (d_left >= d_right)
                wp_la = self._path_lookahead_wp_at_dist(px, py, self.hybrid_lookahead_dist_m)
                if wp_la is not None:
                    wp_angle_world = math.atan2(wp_la[1] - py, wp_la[0] - px)
                    # Relative angle in body frame: + means left, - means right
                    angle_diff = (wp_angle_world - yaw + math.pi) % (2 * math.pi) - math.pi
                    rotate_left = (angle_diff >= 0.0)
                    rotate_reason = "astar"
                self._brake_rotate_left = rotate_left
            else:
                rotate_left = self._brake_rotate_left
                rotate_reason = "locked"

            yaw_rate = 0.0 if self._brake_count > 30 else (
                self.max_yaw_rate if rotate_left else -self.max_yaw_rate
            )
            if not self._emergency_brake_active:
                self._emergency_brake_active = True
                self.get_logger().info(
                    f"Emergency brake: d_front={d_front_min:.2f}m <= {self.emergency_brake_dist}, "
                    f"rotating {'left' if rotate_left else 'right'} (src={rotate_reason})"
                )
            d_rear = self._dist_in_sector(
                180.0 - self.emergency_brake_deg,
                180.0 + self.emergency_brake_deg, "min", use_override=False)
            retreat_speed = 0.4 if d_rear > 0.8 else 0.0
            vx_r = -retreat_speed * math.cos(yaw)
            vy_r = -retreat_speed * math.sin(yaw)
            self._publish(vx_r, vy_r, self._vz_correction(), yaw_rate)
            return
        self._emergency_brake_active = False
        self._brake_count = 0
        self._brake_rotate_left = None

        # Per-nav_mode control logic
        if self.nav_mode == "astar":
            vx, vy, vz, yaw_rate = self._tick_astar_mode(px, py, yaw, now)
        elif self.nav_mode == "ppo":
            vx, vy, vz, yaw_rate = self._tick_ppo_mode(yaw)
        else:  # "hybrid" / "fm2" (same control: PPO inference with path lookahead as obs)
            vx, vy, vz, yaw_rate = self._tick_hybrid_mode(px, py, yaw)

        self._publish(vx, vy, vz, yaw_rate)

    def _tick_astar_mode(self, px: float, py: float, yaw: float, now: float) -> Tuple[float, float, float, float]:
        """A*-only mode: use only the global planner (_tick_global). No PPO."""
        if not self._path_xy:
            self._plan_path()
        return self._tick_global(px, py, yaw, now)

    def _tick_ppo_mode(self, yaw: float) -> Tuple[float, float, float, float]:
        """PPO-only mode: infer toward the final goal every tick. Publish the PPO output unmodified."""
        obs = self._build_obs()
        if obs is None:
            return 0.0, 0.0, self._vz_correction(), 0.0
        action = self._infer(obs)
        v_cmd, yaw_rate = self._action_to_vel(action)
        vx, vy = self._compute_cmd_vel(v_cmd, yaw)
        return vx, vy, self._vz_correction(), yaw_rate

    def _tick_hybrid_mode(self, px: float, py: float, yaw: float) -> Tuple[float, float, float, float]:
        """Hybrid mode: PPO inference every tick with the A* path lookahead as obs -> v_cmd/yaw_rate.
        Motion is along the current heading; heading is controlled by the PPO yaw_rate (matches the training unicycle model)."""
        with self._path_lock:
            have_path = bool(self._path_xy)
        if have_path:
            self._path_advance_wp(px, py)
        else:
            self._plan_path()

        obs = self._build_obs()
        if obs is None:
            return 0.0, 0.0, self._vz_correction(), 0.0
        action = self._infer(obs)
        v_cmd, yaw_rate = self._action_to_vel(action)
        vx = v_cmd * math.cos(yaw)
        vy = v_cmd * math.sin(yaw)
        return vx, vy, self._vz_correction(), yaw_rate


def main(args=None):
    rclpy.init(args=args)
    node = RLOffboardNode()
    # Split tick / replan / sensor callback groups onto separate threads (thread count = callback group count)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
