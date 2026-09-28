#!/usr/bin/env python3
"""
2D UAV navigation env -- FM2 (Fast Marching Square) guided variant (2026-09-14).

The only difference from env_guided_astar.py: the guidance path source is switched from A* to FM2
(fm2_field.fm2_field_compute + extract_path):
occupancy grid -> velocity map W (clearance saturation alpha / exponent beta) -> Eikonal arrival-time field T -> gradient-descent path.
T depends only on (grid, goal), not on start -> cached per episode (reused on _fm2_t_key hit);
replan (2.0s period) only re-runs the gradient extraction (~40x speedup, bit-identical to no caching).
Observation structure (64 lidar + 2-D body-frame lookahead wp), C2 reward, curriculum, scenarios,
replan period (2.0s), lookahead mechanism (1.0m / norm_scale 3.0) are all preserved verbatim,
guaranteeing the comparison with the A*-PPO run varies only the single factor "planner".

Reward (official C2 model): collision -collision_penalty / BFS-progress w_progress*(prev-curr) /
success +success_reward. (time-pressure/proximity/spin/no_progress/timeout penalties removed -- zero weight in C2)
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import math
import numpy as np

from fm2_field import extract_path, fm2_field_compute

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


@dataclass
class RectObstacle:
    """Axis-aligned rectangle obstacle in 2D."""
    x: float  # center x
    y: float  # center y
    w: float  # width (x extent)
    h: float  # height (y extent)

    @property
    def xmin(self) -> float:
        return self.x - self.w / 2.0

    @property
    def xmax(self) -> float:
        return self.x + self.w / 2.0

    @property
    def ymin(self) -> float:
        return self.y - self.h / 2.0

    @property
    def ymax(self) -> float:
        return self.y + self.h / 2.0


# ENU obstacles (cx, cy, w, h) = SDF <box><size>. In env coordinates, shifted by COURSE_L_S_OFFSET_XY.
COURSE_L_S_ENU_BOXES: List[Tuple[float, float, float, float]] = [
    (-10.0, 1.0, 0.3, 18.0),  # s_outer_left
    (-6.0, 10.0, 8.0, 0.3),  # s_outer_top
    (-6.0, -10.0, 8.0, 0.3),  # s_outer_bottom
    (-4.0, 0.0, 0.3, 8.0),  # inner_s_v1
    (-8.0, 0.0, 0.3, 8.0),  # inner_s_v2
    (-6.0, 6.0, 0.3, 8.0),  # inner_s_v3
    (-6.0, -6.0, 0.3, 8.0),  # inner_s_v4
]
COURSE_L_S_OFFSET_XY: Tuple[float, float] = (12.0, 10.0)


class UAVNav2DEnv(gym.Env):
    """
    2D UAV navigation env (continuous position/action).

    Obs: lidar_ranges (obs_beams, downsample->[-1,1]) + goal_rel (dx,dy) body-frame.
         goal_rel is the 1.0m (hybrid_lookahead_dist_m) lookahead wp along the FM2 arrival-time field path (same interface as A*).
    Action: (a_v, a_yaw) in [-1,1] (C2: PPO controls both v and yaw).
            v_cmd=v_min+(a_v+1)/2*(v_max-v_min), yaw_rate_cmd=a_yaw*yaw_rate_max.
    Reward (C2): BFS progress + success bonus; terminal collision -> -collision_penalty.
    Episode ends: success(goal_radius) / collision / timeout(max_steps).
    """

    _LIDAR_RAY_STEP: float = 0.2   # ray marching step size (m)

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        world_size: float = 20.0,
        num_obstacles: int = 2,
        obstacle_size_range: Tuple[float, float] = (1.5, 3.0),
        num_beams: int = 360,          # for internal ray computation (matches ROS scan resolution)
        obs_beams: int = 64,           # number of downsampled beams used for observation
        downsample_mode: str = "min",  # "min" or "avg"
        fov: float = 2 * math.pi,      # 360 deg
        max_range: float = 10.0,       # r_FOV (rFOV in the paper)
        v_max: float = 0.8,            # max v_cmd speed (m/s)
        v_min: float = 0.3,            # min v_cmd speed (prevents stopping, guarantees forward progress toward wp within a replan period)
        yaw_rate_max: float = 1.0,     # max yaw_rate (rad/s)
        dt: float = 0.05,              # sim step time (s). Matches the 20Hz deployment setpoint
        goal_radius: float = 2.0,      # goal-reached threshold radius (curriculum start value, eases early learning)
        max_steps: int = 1800,         # episode step cap. At dt=0.05 the time horizon is ~90s
        use_curriculum: bool = True,   # enable curriculum learning
        curriculum_levels: Optional[List[Tuple[float, float, int]]] = None,
        collision_penalty: float = 500.0,  # collision penalty (aimed at reducing stress-test collisions)
        success_reward: float = 800.0,  # goal-reached reward (sparse anchor)
        w_progress: float = 50.0,        # +50 per meter closer to the goal (BFS-distance based)
        yaw_drift_std: float = 0.0,       # std of random yaw drift
        min_clearance: float = 1.0,    # minimum clearance for start/goal positions (m)
        eval_scenario: Optional[str] = None,  # "obstacle_density_*", "narrow_passage", "course_l_s"
        eval_sensor_noise: bool = False,     # eval-only: yaw_drift_std=0.05
        curriculum_verbose: bool = True,     # when False, omit curriculum progress logs (avoids duplicate output from eval/parallel envs)
        seed: Optional[int] = None,
        # --- A* path following (obs matches the deployment environment) ---
        lookahead_norm_scale: float = 3.0,   # goal_rel normalization scale (denominator) -- NOT the lookahead "distance". Matches deployment LOOKAHEAD_NORM_SCALE=3.0
        astar_inflation_cells: int = 1,      # (unused in the FM2 version, legacy kept for signature compatibility)
        use_astar_lookahead: bool = True,    # True: obs goal_rel = FM2 lookahead waypoint (hybrid). False: final goal/world_size (PPO-only baseline)
        hybrid_lookahead_dist_m: float = 1.0,  # actual obs lookahead distance (m). Matches deployment 1.0m (normalization denominator is lookahead_norm_scale=3.0)
        # --- FM2 velocity-field parameters (alpha/beta from Sensors 2022) ---
        fm2_alpha: float = 1.0,      # velocity-field saturation clearance (m): clearance >= alpha -> W=1
        fm2_beta: float = 1.0,       # velocity-field exponent shaping W=(clearance/alpha)^beta
        # --- Weights for reward ablation (default 0 = exactly the official C2 model) ---
        # Replays the Sec.30 cumulative ablation (FM2 version, 2026-09-16): for P3/P4/P5 only.
        w_no_progress: float = 0.0,       # A3-term: no-progress termination penalty
        w_no_progress_soft: float = 0.0,  # A3-soft: per-step penalty while stuck
        w_timeout: float = 0.0,           # A4: timeout termination penalty
        w_danger: float = 0.0,            # A6-d: proximity exponential penalty coefficient
        w_close: float = 0.0,             # A6-c: proximity linear penalty coefficient
        no_progress_steps: int = 150,     # A3 no-progress detection window (steps) -- historical value from Sec.30/13-4
        safe_dist: float = 0.9,           # proximity penalty threshold distance (m) -- 13-3 final value
        danger_decay: float = 0.2,        # exponential penalty decay -- 13-3
    ):
        super().__init__()
        self.world_size = float(world_size)
        self.num_obstacles = int(num_obstacles)
        self.obstacle_size_range = obstacle_size_range

        self.num_beams = int(num_beams)
        self.obs_beams = int(obs_beams)
        self.downsample_mode = str(downsample_mode).lower()
        if self.downsample_mode not in ("min", "avg"):
            raise ValueError(f"downsample_mode must be 'min' or 'avg', got {downsample_mode!r}")
        self.fov = float(fov)
        self.max_range = float(max_range)

        self.v_max = float(v_max)
        self.v_min = float(v_min)
        self.yaw_rate_max = float(yaw_rate_max)
        self.dt = float(dt)
        self.goal_radius_base = float(goal_radius)  # curriculum starting point
        self.goal_radius = float(goal_radius)       # current value (dynamically adjusted)
        self.max_steps = int(max_steps)

        # Curriculum learning
        self.use_curriculum = bool(use_curriculum)
        if curriculum_levels is None:
            # (goal_radius, min_clearance, num_obstacles)
            # num_obstacles: int (fixed) or tuple(lo, hi) -> rng.integers(lo, hi+1) applied each episode
            curriculum_levels = [
                (2.0, float(min_clearance), (0, 3)),    # stage 0: 0-3 random (basic navigation)
                (2.0, float(min_clearance), (4, 6)),    # stage 1: 4-6 random
                (1.5, float(min_clearance), 8),         # stage 2: 8 fixed
                (1.0, float(min_clearance), (8, 12)),   # stage 3: 8-12 random
                (0.5, float(min_clearance), (12, 20)),  # stage 4: 12-20 random
            ]
        self.curriculum_levels = [
            (float(r), float(c), n) for r, c, n in curriculum_levels
        ]
        if not self.curriculum_levels:
            self.curriculum_levels = [(self.goal_radius_base, float(min_clearance), int(num_obstacles))]
        self.curriculum_level_idx = 0
        self.curriculum_level_thresholds = [0.60, 0.70, 0.80, 0.90]

        self.collision_penalty = float(collision_penalty)
        self.success_reward = float(success_reward)
        self.w_progress = float(w_progress)
        # Reward ablation weights (default 0 -> bit-identical to the official C2 reward)
        self.w_no_progress = float(w_no_progress)
        self.w_no_progress_soft = float(w_no_progress_soft)
        self.w_timeout = float(w_timeout)
        self.w_danger = float(w_danger)
        self.w_close = float(w_close)
        self.no_progress_steps = int(no_progress_steps)
        self.safe_dist = float(safe_dist)
        self.danger_decay = float(danger_decay)
        self.yaw_drift_std = float(yaw_drift_std)
        self.min_clearance = float(min_clearance)

        # Eval only
        self.eval_scenario = eval_scenario
        self.eval_sensor_noise = bool(eval_sensor_noise)
        self.curriculum_verbose = bool(curriculum_verbose)
        if self.eval_sensor_noise:
            self.yaw_drift_std = 0.05
        if self.eval_scenario or self.eval_sensor_noise:
            self.use_curriculum = False  # curriculum disabled during eval

        self.rng = np.random.default_rng(seed)

        # Obs: lidar (N,) + goal_rel body (2,)
        obs_dim = self.obs_beams + 2
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        # Action: (a_v, a_yaw) in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # A* path following (consistent with deployment obs)
        self.lookahead_norm_scale = float(lookahead_norm_scale)  # goal_rel normalization denominator (not the lookahead distance)
        self.astar_inflation_cells = int(astar_inflation_cells)  # unused in the FM2 version (legacy compatibility)
        self.use_astar_lookahead = bool(use_astar_lookahead)  # hybrid vs PPO-only
        self.hybrid_lookahead_dist_m = float(hybrid_lookahead_dist_m)
        self.fm2_alpha = float(fm2_alpha)
        self.fm2_beta = float(fm2_beta)

        # State
        self.pos = np.zeros(2, dtype=np.float32)
        self.yaw = 0.0  # heading (rad)
        self.goal = np.zeros(2, dtype=np.float32)
        self.obstacles: List[RectObstacle] = []
        self.step_count = 0
        self.prev_goal_dist = 0.0
        self._bfs_dist_map: Optional[np.ndarray] = None
        self._bfs_cell_size: float = 0.25       # must be <=0.25 to guarantee coverage of the 0.3m course_l_s walls
        self._guide_path: Optional[List[Tuple[float, float]]] = None
        self._path_wp_idx: int = 0
        # FM2 T-field per-episode cache: key = (obstacle tuple, goal, cell_size); T is independent of start
        self._fm2_t_field: Optional[np.ndarray] = None
        self._fm2_t_key: Optional[tuple] = None
        self._fm2_stats: Dict[str, int] = {}  # fallback/rescue/snap/sweep_cap telemetry
        self._replan_interval: int = max(1, round(2.0 / self.dt))  # deployment replan_period=2.0s expressed in steps (auto-scales with dt)
        self._steps_since_replan: int = 0
        self.episode_reward = 0.0

        self.last_collision = False
        self.last_success = False

        # Curriculum tracking
        # A3 no-progress detection state (used only when w_no_progress* > 0)
        self._bfs_window: List[float] = []
        self._stuck = False
        self._stuck_streak = 0

        self._curriculum_episode_count = 0
        self._curriculum_success_count = 0
        self._curriculum_window = 100  # based on the most recent 100 episodes
        # Per-stage scenario mix ratios (the stress-scenario share grows with the curriculum stage).
        # Each tuple holds cumulative boundary probabilities: normal -> density_15 -> density_20 -> narrow -> course_l_s
        self._stage_mix_thresholds = [
            (1.00, 1.00, 1.00, 1.00, 1.00),  # stage 0: normal 100%
            (0.90, 0.90, 0.90, 0.95, 1.00),  # stage 1: normal 90%, narrow 5%, course_l_s 5%
            (0.70, 0.80, 0.80, 0.90, 1.00),  # stage 2: normal 70%, d15 10%, narrow 10%, course_l_s 10%
            (0.45, 0.60, 0.70, 0.85, 1.00),  # stage 3: normal 45%, d15 15%, d20 10%, narrow 15%, course_l_s 15%
            (0.05, 0.20, 0.55, 0.70, 1.00),  # stage 4: normal 5%, d15 15%, d20 35%, narrow 15%, course_l_s 30%
        ]

    # --- Reset & Step ---
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.last_collision = False
        self.last_success = False
        self._bfs_dist_map = None
        self._fm2_t_key = None  # invalidate the FM2 T-field cache (the field changes with grid+goal)
        self._bfs_window = []   # reset A3 no-progress window state
        self._stuck = False
        self._stuck_streak = 0
        self.episode_reward = 0.0

        # Curriculum: jump straight to the highest stage whose success-rate threshold is met
        if self.use_curriculum and self._curriculum_episode_count >= self._curriculum_window:
            success_rate = self._curriculum_success_count / max(self._curriculum_episode_count, 1)
            old_idx = self.curriculum_level_idx
            target_idx = old_idx
            for next_level in range(old_idx + 1, len(self.curriculum_levels)):
                thr_idx = next_level - 1
                if thr_idx < len(self.curriculum_level_thresholds) and success_rate >= self.curriculum_level_thresholds[thr_idx]:
                    target_idx = next_level
                else:
                    break
            if target_idx > old_idx:
                self.curriculum_level_idx = target_idx
                self._apply_curriculum_level()
                required_rate = self.curriculum_level_thresholds[target_idx - 1] if target_idx > 0 else 0.0
                if self.curriculum_verbose:
                    print(
                        f"[Curriculum] Success {success_rate*100:.1f}% (>= {required_rate*100:.0f}%) -> "
                        f"goal_radius={self.goal_radius:.2f}, min_clearance={self.min_clearance:.2f}"
                    )
            if target_idx > old_idx:
                self._curriculum_episode_count = 0
                self._curriculum_success_count = 0

        if self.use_curriculum:
            self._apply_curriculum_level()

        # Eval scenario or per-stage mix ratios
        if self.eval_scenario == "obstacle_density_15":
            self._reset_eval_obstacle_density(num_obstacles=15)
        elif self.eval_scenario == "obstacle_density_20":
            self._reset_eval_obstacle_density(num_obstacles=20)
        elif self.eval_scenario == "narrow_passage":
            self._reset_eval_narrow_passage()
        elif self.eval_scenario == "course_l_s":
            self._reset_course_l_s()
        elif self.use_curriculum:
            idx = min(self.curriculum_level_idx, len(self._stage_mix_thresholds) - 1)
            t = self._stage_mix_thresholds[idx]
            r = float(self.rng.random())
            if r < t[0]:      # normal
                self._reset_normal_scenario()
            elif r < t[1]:    # density_15
                self._reset_eval_obstacle_density(num_obstacles=15)
            elif r < t[2]:    # density_20
                self._reset_eval_obstacle_density(num_obstacles=20)
            elif r < t[3]:    # narrow_passage
                self._reset_eval_narrow_passage()
            else:             # course_l_s
                self._reset_course_l_s()
        else:
            self._reset_normal_scenario()

        # BFS distance map / FM2 path: once per episode (after goal & obstacles are fixed). FM2 skipped for PPO-only.
        self._bfs_dist_map = self._compute_bfs_dist_map(self._bfs_cell_size)
        if self.use_astar_lookahead:
            self._guide_path = self._compute_fm2_path(self.pos, self.goal, self._bfs_cell_size)
        else:
            self._guide_path = None
        self._path_wp_idx = 0
        self._steps_since_replan = 0

        self.yaw = self.rng.uniform(-math.pi, math.pi)

        obs = self._get_obs()
        info = self._get_info()
        info["episode_reward"] = float(self.episode_reward)
        return obs, info

    def _apply_curriculum_level(self) -> None:
        if not self.curriculum_levels:
            return
        radius, clearance, num_obstacles = self.curriculum_levels[self.curriculum_level_idx]
        self.goal_radius = float(radius)
        self.min_clearance = float(clearance)
        if isinstance(num_obstacles, tuple):  # (lo, hi): random in [lo, hi] each episode
            lo, hi = num_obstacles
            self.num_obstacles = int(self.rng.integers(lo, hi + 1))
        else:
            self.num_obstacles = int(num_obstacles)

    def _reset_eval_obstacle_density(self, num_obstacles: int) -> None:
        """Eval-only: Obstacle Density Stress Test. goal behind obstacle."""
        self.obstacles = []
        for _ in range(num_obstacles):
            w = float(self.rng.uniform(*self.obstacle_size_range))
            h = float(self.rng.uniform(*self.obstacle_size_range))
            x = float(self.rng.uniform(w / 2, self.world_size - w / 2))
            y = float(self.rng.uniform(h / 2, self.world_size - h / 2))
            self.obstacles.append(RectObstacle(x=x, y=y, w=w, h=h))

        self.pos = self._sample_free_position(min_clearance=self.min_clearance)

        if num_obstacles == 0:  # with no obstacles, the "goal behind obstacle" logic is unnecessary
            self.goal = self._sample_free_position(
                min_dist=10.0, ref=self.pos, min_clearance=self.min_clearance
            )
            self.prev_goal_dist = float(np.linalg.norm(self.goal - self.pos))
            return

        # place the goal right behind an obstacle (obstacle between start<->goal)
        for _ in range(500):
            idx = self.rng.integers(0, len(self.obstacles))
            obs = self.obstacles[idx]
            dir_vec = np.array([obs.x, obs.y]) - self.pos
            dist = float(np.linalg.norm(dir_vec))
            if dist < 1e-6:
                continue
            dir_vec = dir_vec / dist
            goal_offs = 2.0 + self.rng.uniform(0.5, 2.0)
            goal_cand = np.array([obs.x, obs.y], dtype=np.float32) + dir_vec * goal_offs
            goal_cand = np.clip(goal_cand, 1.0, self.world_size - 1.0)
            if not self._check_collision(goal_cand) and float(np.linalg.norm(goal_cand - self.pos)) >= 5.0:
                if self._is_reachable_bfs(self.pos, goal_cand, cell_size=0.3):
                    self.goal = goal_cand
                    break
        else:
            self.goal = self._sample_free_position(min_dist=10.0, ref=self.pos, min_clearance=self.min_clearance)
        self.prev_goal_dist = float(np.linalg.norm(self.goal - self.pos))

    def _reset_eval_narrow_passage(self) -> None:
        """Eval-only Narrow Passage: keep only an L-shaped corridor as free space and fill the rest
        with walls to form a closed tube -> reaching the goal requires a 90deg turn inside the corridor.
        Randomized over 4 orientations via flips."""
        W = self.world_size
        for _ in range(200):
            cw = float(self.rng.uniform(1.8, 2.4))     # corridor width (m)
            vx0 = float(self.rng.uniform(2.0, 3.5))    # vertical leg left x (pushes the corner toward the corner)
            hy0 = float(self.rng.uniform(2.0, 3.5))    # horizontal leg bottom y (pushes the corner toward the corner)
            vy1 = float(self.rng.uniform(16.5, 18.0))  # vertical leg top y (endpoint at the edge)
            hx1 = float(self.rng.uniform(16.5, 18.0))  # horizontal leg right x (endpoint at the edge)
            vx1, hy1 = vx0 + cw, hy0 + cw
            flip_x = bool(self.rng.integers(0, 2))
            flip_y = bool(self.rng.integers(0, 2))

            def fx(x: float) -> float:
                return W - x if flip_x else x

            def fy(y: float) -> float:
                return W - y if flip_y else y

            def box(xmin: float, xmax: float, ymin: float, ymax: float) -> RectObstacle:
                xs = sorted([fx(xmin), fx(xmax)])
                ys = sorted([fy(ymin), fy(ymax)])
                return RectObstacle(
                    x=(xs[0] + xs[1]) / 2.0, y=(ys[0] + ys[1]) / 2.0,
                    w=xs[1] - xs[0], h=ys[1] - ys[0],
                )

            # enclose everything except the L corridor with 5 walls to form a closed tube
            self.obstacles = [
                box(0.0, W, 0.0, hy0),      # bottom
                box(0.0, W, vy1, W),        # top
                box(0.0, vx0, hy0, vy1),    # left
                box(vx1, W, hy1, vy1),      # inner (right of vertical leg / above horizontal leg)
                box(hx1, W, hy0, hy1),      # blocking the right of the horizontal leg
            ]

            # corridor endpoints (start/goal)
            end_v = np.array([fx((vx0 + vx1) / 2.0), fy(vy1 - 0.9)], dtype=np.float32)
            end_h = np.array([fx(hx1 - 0.9), fy((hy0 + hy1) / 2.0)], dtype=np.float32)
            if self.rng.integers(0, 2):
                self.pos, self.goal = end_v.copy(), end_h.copy()
            else:
                self.pos, self.goal = end_h.copy(), end_v.copy()

            if self._check_collision(self.pos) or self._check_collision(self.goal):
                continue
            if self._is_reachable_bfs(self.pos, self.goal, cell_size=0.2):
                break
        self.prev_goal_dist = float(np.linalg.norm(self.goal - self.pos))

    def _reset_course_l_s(self) -> None:
        """Fixed placement of the Course L (S-course) from Gazebo obstacle_course_light.world in env coordinates.
        The goal lies in the free space at the southwest of the S-course -- only BFS-reachable pairs are accepted."""
        ox, oy = COURSE_L_S_OFFSET_XY
        self.obstacles = []
        for cx, cy, w, h in COURSE_L_S_ENU_BOXES:
            self.obstacles.append(RectObstacle(x=cx + ox, y=cy + oy, w=w, h=h))

        # 3-tier curriculum spawn (BFS distance per difficulty):
        #   ultra-easy 50%: goal in the same corridor, BFS ~3-8m
        #   medium 40%: near the v4 boundary; min_goal_dist=5.0 filters out short-distance spawns, dominated by 8-15m east of v4
        #   hard 10%: long range, BFS ~50m. Structurally hard to complete -- too large a share converges to no_progress and dilutes the signal -> kept low
        r = self.rng.random()
        if r < 0.50:
            spawn_x = (4.2, 5.7)
            spawn_y = (0.5, 5.5)
            min_goal_dist = 1.5
        elif r < 0.80:
            spawn_x = (10.0, 13.0)
            spawn_y = (2.5, 7.5)
            min_goal_dist = 5.0
        else:
            spawn_x = (12.0, 16.0)
            spawn_y = (14.0, 20.0)
            min_goal_dist = 10.0
        goal_x = (2.6, 4.0)
        goal_y = (1.0, 4.0)

        for _ in range(3000):
            self.pos = np.array(
                [
                    float(self.rng.uniform(*spawn_x)),
                    float(self.rng.uniform(*spawn_y)),
                ],
                dtype=np.float32,
            )
            self.goal = np.array(
                [
                    float(self.rng.uniform(*goal_x)),
                    float(self.rng.uniform(*goal_y)),
                ],
                dtype=np.float32,
            )
            if self._check_collision(self.pos) or self._check_collision(self.goal):
                continue
            if float(np.linalg.norm(self.goal - self.pos)) < min_goal_dist:
                continue
            if self.min_clearance > 0.0:
                if self._min_lidar_at(self.pos) < self.min_clearance or \
                        self._min_lidar_at(self.goal) < self.min_clearance:
                    continue
            if self._is_reachable_bfs(self.pos, self.goal, cell_size=0.2):
                self.prev_goal_dist = float(np.linalg.norm(self.goal - self.pos))
                return

        # fallback: representative corridor points verified manually
        self.pos = np.array([13.0, 11.0], dtype=np.float32)
        self.goal = np.array([4.8, 3.0], dtype=np.float32)
        if self._check_collision(self.pos):
            self.pos = np.array([12.5, 10.0], dtype=np.float32)
        if self._check_collision(self.goal):
            self.goal = np.array([5.2, 2.8], dtype=np.float32)
        if not self._is_reachable_bfs(self.pos, self.goal, cell_size=0.25):
            self.goal = np.array([4.0, 4.5], dtype=np.float32)
        self.prev_goal_dist = float(np.linalg.norm(self.goal - self.pos))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 2:
            raise ValueError(f"Action must have 2 elements (a_v, a_yaw) in [-1,1], got shape {action.shape}")
        action = np.clip(action, -1.0, 1.0)
        self.step_count += 1

        if self.yaw_drift_std > 0.0:
            self.yaw = self._wrap_angle(self.yaw + self.rng.normal(0.0, self.yaw_drift_std))

        # action [-1,1] -> (v_cmd, yaw_rate_cmd)
        a_v, a_yaw = float(action[0]), float(action[1])
        v_cmd = self.v_min + (a_v + 1.0) / 2.0 * (self.v_max - self.v_min)
        yaw_rate_cmd = a_yaw * self.yaw_rate_max
        self.yaw = self._wrap_angle(self.yaw + yaw_rate_cmd * self.dt)
        delta = (v_cmd * self.dt) * np.array(
            [math.cos(self.yaw), math.sin(self.yaw)], dtype=np.float32
        )

        prev_bfs_dist = self._get_bfs_dist(self.pos)
        raw_new_pos = self.pos + delta
        # segment sampling prevents "teleport" collisions
        collision, hit_pos, collision_kind = self._check_path_collision(self.pos, raw_new_pos, check_bounds=True)
        self.last_collision = collision

        if not collision:
            self.pos = np.clip(raw_new_pos, 0.0, self.world_size)
        elif hit_pos is not None:
            self.pos = hit_pos

        # time-based FM2 replan (2.0s, same period as A*). For PPO-only, both replan and advance are skipped.
        if self.use_astar_lookahead:
            self._steps_since_replan += 1
            if self._steps_since_replan >= self._replan_interval:
                self._guide_path = self._compute_fm2_path(self.pos, self.goal, self._bfs_cell_size)
                self._path_wp_idx = 0
                self._steps_since_replan = 0
            self._advance_wp()

        dist_to_goal = float(np.linalg.norm(self.goal - self.pos))
        curr_bfs_dist = self._get_bfs_dist(self.pos)
        success = dist_to_goal <= self.goal_radius
        self.last_success = success

        # A3 no-progress detection (active only when w_no_progress* > 0):
        # BFS progress < 1.0m within the window (no_progress_steps) -> stuck (soft-penalty zone),
        # 2 consecutive detections (streak>=2, i.e. no progress over twice the window width) -> terminate + terminal penalty.
        no_prog = False
        if (self.w_no_progress > 0.0 or self.w_no_progress_soft > 0.0) and not (collision or success):
            self._bfs_window.append(curr_bfs_dist)
            if len(self._bfs_window) > self.no_progress_steps:
                self._bfs_window.pop(0)
            if len(self._bfs_window) == self.no_progress_steps:
                if (self._bfs_window[0] - self._bfs_window[-1]) < 1.0:
                    self._stuck_streak += 1
                    self._stuck = True
                else:
                    self._stuck_streak = 0
                    self._stuck = False
            no_prog = bool(self.w_no_progress > 0.0 and self._stuck_streak >= 2)

        terminated = bool(collision or success or no_prog)
        truncated = bool(self.step_count >= self.max_steps)

        reward, lidar = self._compute_reward(
            prev_bfs_dist=prev_bfs_dist,
            curr_bfs_dist=curr_bfs_dist,
            collision=collision,
            success=success,
            no_prog_soft=self.w_no_progress_soft if self._stuck else 0.0,
        )
        if no_prog:
            reward -= self.w_no_progress
        if truncated and not (collision or success or no_prog) and self.w_timeout > 0.0:
            reward -= self.w_timeout

        obs = self._get_obs(lidar=lidar if not collision else None)
        info = self._get_info()
        info["collision_kind"] = collision_kind if collision else "none"
        if terminated:
            if collision:
                info["done_reason"] = "collision"
            elif success:
                info["done_reason"] = "reached"
            else:
                info["done_reason"] = "no_progress"
        elif truncated:
            info["done_reason"] = "timeout"

        if terminated or truncated:
            self._curriculum_episode_count += 1
            if success:
                self._curriculum_success_count += 1

        return obs, reward, terminated, truncated, info

    # --- FM2 path-following utilities ---
    def _compute_fm2_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        cell_size: float = 0.25,
    ) -> List[Tuple[float, float]]:
        """FM2 (Fast Marching Square) path. Falls back to [goal] on failure (same semantics as the A* version).

        Solves the Eikonal arrival-time field T over the velocity field W (slower near walls) and extracts
        the path by gradient descent -- unlike A*'s shortest wall-hugging paths, this yields a path that
        follows the middle of passages.
        T depends only on (grid, goal) -> cached within an episode; replan re-runs only the extraction.
        If the start cell is occupied in rasterization (T=inf), extract_path snaps to an adjacent free cell.
        The I/O contract (world-coordinate waypoint list) matches the A* version."""
        key = (
            tuple((o.x, o.y, o.w, o.h) for o in self.obstacles),
            float(goal[0]), float(goal[1]), float(cell_size),
        )
        if self._fm2_t_key != key or self._fm2_t_field is None:
            grid, _ = self._build_obstacle_grid(cell_size)
            self._fm2_t_field = fm2_field_compute(
                grid,
                (float(goal[0]), float(goal[1])),
                cell_size=cell_size,
                alpha=self.fm2_alpha,
                beta=self.fm2_beta,
                stats=self._fm2_stats,
            )
            self._fm2_t_key = key
        path = extract_path(
            self._fm2_t_field,
            (float(start[0]), float(start[1])),
            (float(goal[0]), float(goal[1])),
            cell_size=cell_size,
            world_size=self.world_size,
            stats=self._fm2_stats,
        )
        if path is None or len(path) < 2:
            n_fb = self._fm2_stats.get("fallback", 0) + 1
            self._fm2_stats["fallback"] = n_fb
            if n_fb <= 20 or n_fb % 100 == 0:  # surface silent fallbacks in the training log
                print(
                    f"[FM2] fallback #{n_fb} start=({float(start[0]):.2f},{float(start[1]):.2f}) "
                    f"goal=({float(goal[0]):.2f},{float(goal[1]):.2f}) stats={self._fm2_stats}",
                    flush=True,
                )
            return [(float(goal[0]), float(goal[1]))]
        return path

    def _get_lookahead_wp_at_dist(self, lookahead_dist: float) -> Tuple[float, float]:
        """Point lookahead_dist (m) ahead along the FM2 path (distance interpolation, same logic as the A* version).
        Falls back to the final goal when there is no path."""
        if not self._guide_path:
            return (float(self.goal[0]), float(self.goal[1]))
        path = self._guide_path
        n = len(path)
        px, py = float(self.pos[0]), float(self.pos[1])
        start_idx = max(0, min(self._path_wp_idx, n - 1))

        # advance past already-passed wps (blocks backward starting segments, same as deployment)
        while start_idx + 1 < n:
            ax, ay = path[start_idx]
            bx, by = path[start_idx + 1]
            if (ax - px) * (bx - ax) + (ay - py) * (by - ay) < 0.0:
                start_idx += 1
            else:
                break

        # accumulate distance and interpolate the point at lookahead_dist
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

        # if the path remainder is shorter than lookahead_dist, take one wp ahead (deadlock avoidance, same as deployment)
        return path[min(n - 1, start_idx + 1)]

    def _advance_wp(self) -> None:
        """Advance to the next wp when within 0.5m of the current one (same as deployment _path_advance_wp())."""
        if not self._guide_path or self._path_wp_idx >= len(self._guide_path) - 1:
            return
        wp = self._guide_path[self._path_wp_idx]
        if math.hypot(wp[0] - float(self.pos[0]), wp[1] - float(self.pos[1])) < 0.5:
            self._path_wp_idx += 1

    # --- BFS utilities ---
    def _build_obstacle_grid(self, cell_size: float) -> Tuple[np.ndarray, int]:
        """Obstacle occupancy grid (vectorized). Returns (grid[n,n], n)."""
        n = int(math.ceil(self.world_size / cell_size))
        grid = np.zeros((n, n), dtype=np.uint8)
        if not self.obstacles:
            return grid, n
        xs = (np.arange(n, dtype=np.float32) + 0.5) * cell_size
        ys = (np.arange(n, dtype=np.float32) + 0.5) * cell_size
        XX, YY = np.meshgrid(xs, ys)
        for obs in self.obstacles:
            grid |= ((XX >= obs.xmin) & (XX <= obs.xmax) &
                     (YY >= obs.ymin) & (YY <= obs.ymax))
        return grid, n

    def _compute_bfs_dist_map(self, cell_size: float = 0.5) -> np.ndarray:
        """Whole-grid shortest distance (m) via reverse BFS from the goal. Obstacle/unreachable cells are np.inf. Once per episode."""
        grid, n = self._build_obstacle_grid(cell_size)
        dist = np.full((n, n), np.inf, dtype=np.float32)
        gj = max(0, min(n - 1, int(self.goal[0] / cell_size)))
        gi = max(0, min(n - 1, int(self.goal[1] / cell_size)))
        if grid[gi, gj]:
            return dist
        dist[gi, gj] = 0.0
        q = deque([(gi, gj)])
        while q:
            i, j = q.popleft()
            d_next = dist[i, j] + cell_size
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n and not grid[ni, nj] and dist[ni, nj] == np.inf:
                    dist[ni, nj] = d_next
                    q.append((ni, nj))
        return dist

    def _get_bfs_dist(self, pos: np.ndarray) -> float:
        """BFS distance lookup. Falls back to Euclidean on unreachable cells (keeps the shaping signal)."""
        if self._bfs_dist_map is None:
            return float(np.linalg.norm(self.goal - pos))
        n = self._bfs_dist_map.shape[0]
        j = max(0, min(n - 1, int(pos[0] / self._bfs_cell_size)))
        i = max(0, min(n - 1, int(pos[1] / self._bfs_cell_size)))
        d = float(self._bfs_dist_map[i, j])
        return d if not math.isinf(d) else float(np.linalg.norm(self.goal - pos))

    def _is_reachable_bfs(self, start: np.ndarray, goal: np.ndarray, cell_size: float = 0.5) -> bool:
        """DFS check whether goal is reachable from start while avoiding obstacles."""
        if cell_size <= 0.0:
            return True
        grid, n = self._build_obstacle_grid(cell_size)

        def to_cell(p: np.ndarray) -> Tuple[int, int]:
            return (max(0, min(n - 1, int(p[1] / cell_size))),
                    max(0, min(n - 1, int(p[0] / cell_size))))

        si, sj = to_cell(start)
        gi, gj = to_cell(goal)
        if grid[si, sj] or grid[gi, gj]:
            return False

        visited = np.zeros((n, n), dtype=np.uint8)
        stack = [(si, sj)]
        visited[si, sj] = 1
        while stack:
            i, j = stack.pop()
            if i == gi and j == gj:
                return True
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n and not visited[ni, nj] and not grid[ni, nj]:
                    visited[ni, nj] = 1
                    stack.append((ni, nj))
        return False

    def _compute_reward(
        self,
        *,
        prev_bfs_dist: float,
        curr_bfs_dist: float,
        collision: bool,
        success: bool,
        no_prog_soft: float = 0.0,
    ) -> Tuple[float, Optional[np.ndarray]]:
        # C2: collision penalty + BFS-progress shaping + success bonus
        # (+ ablation-only: A3-soft stuck penalty / A6 proximity penalty. Default 0 -> never triggers)
        if collision:
            return -self.collision_penalty, None

        lidar = self._lidar_ranges(self.pos, self.yaw)
        lidar = self._sanitize_lidar(lidar)

        reward = self.w_progress * (prev_bfs_dist - curr_bfs_dist)  # BFS progress shaping
        if no_prog_soft > 0.0:
            reward -= no_prog_soft  # A3-soft: per-step while stuck
        if self.w_danger > 0.0 or self.w_close > 0.0:
            min_r = float(np.min(lidar))
            if min_r < self.safe_dist:
                gap = self.safe_dist - min_r
                if self.w_danger > 0.0:  # A6-d: exponential (13-3 final formula)
                    reward -= self.w_danger * (math.exp(gap / self.danger_decay) - 1.0)
                if self.w_close > 0.0:   # A6-c: linear
                    reward -= self.w_close * gap
        if success:
            reward += self.success_reward
        return reward, lidar

    # --- Observation (real-sensor friendly) ---
    def _get_obs(self, *, lidar: Optional[np.ndarray] = None) -> np.ndarray:
        if lidar is None:
            lidar = self._lidar_ranges(self.pos, self.yaw)
        lidar = self._downsample_lidar(lidar, self.obs_beams, self.downsample_mode)
        lidar_norm = (lidar / self.max_range).astype(np.float32)
        lidar_norm = lidar_norm * 2.0 - 1.0  # [0,1] -> [-1,1] for NN

        # goal_rel: body-frame target (dx, dy), normalized.
        # hybrid: FM2 1.0m lookahead wp, normalized by lookahead_norm_scale (3.0).
        # PPO-only: final goal, normalized by world_size.
        if self.use_astar_lookahead:
            wp = self._get_lookahead_wp_at_dist(self.hybrid_lookahead_dist_m)
            target = np.array(wp, dtype=np.float32)
            norm_scale = self.lookahead_norm_scale
        else:
            target = self.goal.astype(np.float32)
            norm_scale = self.world_size
        dx_world = (target - self.pos).astype(np.float32)
        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        dx_body = c * dx_world[0] - s * dx_world[1]
        dy_body = s * dx_world[0] + c * dx_world[1]
        goal_rel = np.array([dx_body, dy_body], dtype=np.float32) / max(norm_scale, 1e-6)
        goal_rel = np.clip(goal_rel, -1.0, 1.0).astype(np.float32)

        obs = np.concatenate([lidar_norm, goal_rel], axis=0).astype(np.float32)
        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self.step_count,
            "collision": self.last_collision,
            "success": self.last_success,
            "dist_to_goal": float(np.linalg.norm(self.goal - self.pos)),
        }

    # --- Obstacle generation & collision ---
    def _min_lidar_at(self, pos: np.ndarray) -> float:
        """Minimum omnidirectional lidar distance (m) at pos. Used for clearance checks."""
        return float(np.min(self._lidar_ranges(pos, 0.0)))

    def _reset_normal_scenario(self) -> None:
        """Curriculum normal episode: random obstacles + pos/goal sampling."""
        # 50% of stage 4 uses 0-5 obstacles -> include low densities in the training distribution to close the eval domain gap
        if self.curriculum_level_idx == 4 and self.rng.random() < 0.50:
            saved = self.num_obstacles
            self.num_obstacles = int(self.rng.integers(0, 6))
            self._spawn_obstacles()
            self.num_obstacles = saved
        else:
            self._spawn_obstacles()
        self.pos = self._sample_free_position(min_clearance=self.min_clearance)
        self.goal = self._sample_free_position(min_dist=5.0, ref=self.pos, min_clearance=self.min_clearance)
        self.prev_goal_dist = float(np.linalg.norm(self.goal - self.pos))

    def _spawn_obstacles(self) -> None:
        self.obstacles = []
        for _ in range(self.num_obstacles):
            w = float(self.rng.uniform(*self.obstacle_size_range))
            h = float(self.rng.uniform(*self.obstacle_size_range))
            x = float(self.rng.uniform(w/2, self.world_size - w/2))
            y = float(self.rng.uniform(h/2, self.world_size - h/2))
            self.obstacles.append(RectObstacle(x=x, y=y, w=w, h=h))

    def _sample_free_position(
        self,
        min_dist: float = 0.0,
        ref: Optional[np.ndarray] = None,
        min_clearance: float = 0.0,
    ) -> np.ndarray:
        for _ in range(2000):
            p = np.array(
                [self.rng.uniform(0.5, self.world_size - 0.5),
                 self.rng.uniform(0.5, self.world_size - 0.5)],
                dtype=np.float32
            )
            if self._check_collision(p):
                continue
            if ref is not None and float(np.linalg.norm(p - ref)) < min_dist:
                continue
            if min_clearance > 0.0 and self._min_lidar_at(p) < min_clearance:
                continue
            return p
        # fallback
        return np.array([1.0, 1.0], dtype=np.float32)

    def _check_collision(self, p: np.ndarray) -> bool:
        # Point-in-rectangle (UAV simplified to a point)
        x, y = float(p[0]), float(p[1])
        for obs in self.obstacles:
            if (obs.xmin <= x <= obs.xmax) and (obs.ymin <= y <= obs.ymax):
                return True
        return False

    def _check_path_collision(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        *,
        check_bounds: bool = True,
    ) -> Tuple[bool, Optional[np.ndarray], str]:
        """Sample the straight p0->p1 path and check obstacle/boundary collisions. Returns hit_pos on collision."""
        delta = p1 - p0
        dist = float(np.linalg.norm(delta))
        if dist <= 1e-6:
            if self._check_collision(p0):
                return True, p0.copy(), "obstacle"
            return False, None, "none"

        step = min(0.05, dist / 10.0)
        steps = max(1, int(math.ceil(dist / max(step, 1e-6))))
        for i in range(1, steps + 1):
            t = i / steps
            p = p0 + delta * t
            x, y = float(p[0]), float(p[1])
            if check_bounds and (x < 0.0 or x > self.world_size or y < 0.0 or y > self.world_size):
                return True, p.astype(np.float32), "boundary"
            if self._check_collision(p.astype(np.float32)):
                return True, p.astype(np.float32), "obstacle"
        return False, None, "none"

    # --- Lidar simulation (LaserScan-like ranges) ---
    def _lidar_ranges(self, p: np.ndarray, yaw: float) -> np.ndarray:
        """Ray-marching distances for num_beams rays (clipped to max_range, vectorized).
        Replaceable with ROS2 /scan.ranges at deployment."""
        ray_step = self._LIDAR_RAY_STEP
        max_steps = int(self.max_range / ray_step)

        angles = np.linspace(-self.fov/2, self.fov/2, self.num_beams, dtype=np.float32) + yaw
        dx = np.cos(angles)
        dy = np.sin(angles)

        # all (beam, step) coordinates: (num_beams, max_steps)
        ks = np.arange(1, max_steps + 1, dtype=np.float32)
        xs = float(p[0]) + dx[:, None] * (ks[None, :] * ray_step)
        ys = float(p[1]) + dy[:, None] * (ks[None, :] * ray_step)

        hit = (xs < 0.0) | (xs > self.world_size) | (ys < 0.0) | (ys > self.world_size)  # outside bounds
        for obs in self.obstacles:
            hit |= ((xs >= obs.xmin) & (xs <= obs.xmax) &
                    (ys >= obs.ymin) & (ys <= obs.ymax))

        # first hit distance per beam
        has_hit = hit.any(axis=1)
        first_idx = np.argmax(hit, axis=1)  # returns 0 when there is no hit -> masked via has_hit
        ranges = np.full(self.num_beams, self.max_range, dtype=np.float32)
        ranges[has_hit] = np.minimum((first_idx[has_hit] + 1) * ray_step, self.max_range)

        return ranges

    def _sanitize_lidar(self, lidar: np.ndarray) -> np.ndarray:
        arr = np.asarray(lidar, dtype=np.float32)
        nan_mask = np.isnan(arr)
        inf_mask = np.isinf(arr)
        arr = np.where(nan_mask | inf_mask, self.max_range, arr)
        arr = np.clip(arr, 0.0, self.max_range)
        return arr

    @staticmethod
    def _downsample_lidar(lidar: np.ndarray, target_beams: int, mode: str) -> np.ndarray:
        if target_beams <= 0:
            raise ValueError(f"target_beams must be > 0, got {target_beams}")
        if len(lidar) == target_beams:
            return lidar.copy()
        if target_beams > len(lidar):
            idx = np.linspace(0, len(lidar) - 1, target_beams).astype(int)
            return lidar[idx]

        # split into sectors and pool per sector (min: preserves the nearest obstacle)
        indices = np.array_split(np.arange(len(lidar)), target_beams)
        if mode == "min":
            return np.array([float(np.min(lidar[idx])) for idx in indices], dtype=lidar.dtype)
        if mode == "avg":
            return np.array([float(np.mean(lidar[idx])) for idx in indices], dtype=lidar.dtype)
        raise ValueError(f"Unknown downsample mode: {mode}")

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi
