#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_ros_eval.py — ROS/PX4 simulation success-rate batch experiment runner (2026-09)

Experiment matrix: method (fm2/astar_ppo/ppo/sac) x world (obstacle_course/military_airfield)
                   x course (R/L) x episodes (default 8) = one independent SITL session per run.
Fully unmanned (arm/homing/navigation are performed by rl_offboard_node itself).
Judgment: success (stay within 1.5 m of goal for 2 s) / collision (disarm mid-flight) / timeout.
Incremental CSV logging + resume after interruption (skip completed runs) + infra-error retries (3 per run).

Usage (inside WSL):
  python3 run_ros_eval.py                          # all 128 runs
  python3 run_ros_eval.py --methods fm2 --worlds obstacle_course --episodes 2   # smoke test
  python3 run_ros_eval.py --resume                 # skip CSV-completed runs and continue
"""
import argparse
import csv
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

# ---------------------------------------------------------------- config
PX4_DIR = os.path.expanduser("~/PX4-Autopilot")
WS_DIR = os.path.join(PX4_DIR, "ros2_ws")
BUILD_DIR = os.path.join(PX4_DIR, "build", "px4_sitl_default")
PX4_BIN = os.path.join(BUILD_DIR, "bin", "px4")
SITL_RUN = os.path.join(PX4_DIR, "Tools/simulation/gazebo-classic/sitl_run.sh")

SHARE_RL = os.path.join(WS_DIR, "install/px4_rl_offboard_mavros/share/px4_rl_offboard_mavros")
SHARE_PLANNER = os.path.join(WS_DIR, "install/px4_nav_planner/share/px4_nav_planner")
GOALS_YAML = os.path.join(SHARE_PLANNER, "config/goals.yaml")
PARAMS_YAML = os.path.join(SHARE_RL, "config/params.yaml")

LOG_ROOT = os.path.expanduser("~/ros_eval_logs")
CSV_PRIMARY = ("/mnt/d/学术/CODE/uav-gs-drl-nav"
               "/drl/data/ros_eval_results.csv")
CSV_MIRROR = os.path.expanduser("~/ros_eval_results.csv")

ROS_ENV_CMD = ("source /opt/ros/humble/setup.bash && "
               f"source {WS_DIR}/install/setup.bash && exec ")

SITL_ENV = {
    "HEADLESS": "0",
    # 2026-09-21: gzserver advertises 10.255.255.254 by default, which hangs gz CLI spawn
    "GAZEBO_IP": "127.0.0.1",
    "GAZEBO_MASTER_URI": "http://127.0.0.1:11345",
    "GAZEBO_PLUGIN_PATH": "/usr/lib/x86_64-linux-gnu/gazebo-11/plugins:/opt/ros/humble/lib",
    "GAZEBO_MODEL_PATH": (f"{PX4_DIR}/Tools/simulation/gazebo-classic/sitl_gazebo-classic/models:"
                          f"{os.path.expanduser('~/.gazebo/models')}:/usr/share/gazebo-11/models"),
    "GAZEBO_RESOURCE_PATH": (f"/usr/share/gazebo-11:{PX4_DIR}/Tools/simulation/"
                             "gazebo-classic/sitl_gazebo-classic"),
    "ROS_VERSION": "2",
}

# method -> launch argument mapping (rl_offboard.launch.py)
METHODS = {
    "fm2":       {"desc": "FM2+PPO (ours)", "overlay": "fm2"},
    "astar_ppo": {"desc": "A*+PPO hybrid",  "overlay": None},          # default node (hybrid)
    "ppo":       {"desc": "PPO only",        "rl_only": "true"},
    "sac":       {"desc": "SAC only",        "overlay": "sac"},
}

WORLDS = {
    "obstacle_course": dict(
        world_file="obstacle_course",          # file name under the sitl_run.sh worlds/ dir
        home=[0.0, 8.0, 4.0], goal_R=[8.0, -8.2, 4.0], goal_L=[-8.0, -8.2, 4.0],
        world_size=20.0, nav_timeout_s=180.0),
    "military_airfield": dict(
        world_file="military_airfield",
        home=[0.0, 0.0, 4.0], goal_R=[20.0, 20.0, 4.0], goal_L=[-18.0, -20.0, 4.0],
        world_size=50.0, nav_timeout_s=240.0),
    "urban_city": dict(                       # added 2026-09: urban city blocks (56m, street grid + plaza/park)
        world_file="urban_city",
        home=[5.0, 0.0, 4.0], goal_R=[-14.0, -14.0, 4.0], goal_L=[14.0, 14.0, 4.0],
        world_size=56.0, nav_timeout_s=240.0),
}

FINAL_RESULTS = {"success", "collision", "timeout"}   # not retryable
# ----------------------------------------------------------------


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ================================================================ YAML rewriting
def _rewrite_key(path, key, value_yaml):
    """Regex-replace one `key: [...]` / `key: 50.0` line in a yaml file (line-based, preserves comments)."""
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    new, n = re.subn(rf"(?m)^(\s*{key}:)\s*(\[[^]]*\]|[-\d.]+).*$",
                     lambda m: f"{m.group(1)} {value_yaml}", txt)
    if n != 1:
        raise RuntimeError(f"{path}: failed to replace '{key}' ({n} matches)")
    with open(path, "w", encoding="utf-8") as f:
        f.write(new)


def write_world_config(world):
    """Rewrite install-space yaml coordinates/world size before each episode
    (reverted on rebuild, so rewrite every run)."""
    w = WORLDS[world]
    vec = lambda v: "[" + ", ".join(f"{x}" for x in v) + "]"
    _rewrite_key(GOALS_YAML, "home_xyz", vec(w["home"]))
    _rewrite_key(GOALS_YAML, "goal_R", vec(w["goal_R"]))
    _rewrite_key(GOALS_YAML, "goal_L", vec(w["goal_L"]))
    _rewrite_key(PARAMS_YAML, "home_xyz", vec(w["home"]))
    _rewrite_key(PARAMS_YAML, "world_size", f"{w['world_size']}")


# ================================================================ process management
def sh(cmd, timeout=30):
    return subprocess.run(["/bin/bash", "-c", cmd], capture_output=True,
                          text=True, timeout=timeout)


def kill_tree(proc, grace=6.0):
    """Terminate a Popen group: SIGINT (clean ros2 launch shutdown) -> grace -> SIGKILL."""
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except Exception:
        proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait(timeout=5)


CLEAN_PATTERNS = ["gzserver", "gzclient", "bin/px4", "px4-[a-z_]+", "sitl_run.sh",
                  "mavros_node", "rl_offboard_node", "course_goal_pub", "grid_fuser",
                  "robot_state_pub", "rviz2", "static_transform_pub", "ros2 launch"]


def cleanup_environment():
    """Thoroughly clean leftover processes between runs (prevents timeout pipe hangs
    and orphaned gzserver/px4).
    Note: the px4 binary command line is '.../bin/px4 none iris...' — a 'px4-'
    pattern alone does not match it."""
    for pat in CLEAN_PATTERNS:
        subprocess.run(["/bin/bash", "-c", f"pkill -9 -f '[{pat[0]}]{pat[1:]}' || true"],
                       capture_output=True)
    for _ in range(15):
        r = sh("pgrep -x gzserver || pgrep -f bin/px4 || true")
        if not r.stdout.strip():
            break
        time.sleep(1)
    time.sleep(2.0)


def popen_ros(args, logfile, cwd=None, env_extra=None):
    """Run a ROS command under a bash+source environment, stdout goes to a log file."""
    lf = open(logfile, "ab", buffering=0)
    cmd = ROS_ENV_CMD + " ".join(args)
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    p = subprocess.Popen(["/bin/bash", "-c", cmd], stdout=lf, stderr=subprocess.STDOUT,
                         cwd=cwd, env=env, start_new_session=True)
    return p, lf


def start_sitl(world, logdir):
    wfile = WORLDS[world]["world_file"]
    cmd = (f"{SITL_RUN} {PX4_BIN} none iris_rplidar_depth {wfile} "
           f"{PX4_DIR} {BUILD_DIR}")
    lf = open(os.path.join(logdir, "sitl.log"), "ab", buffering=0)
    env = dict(os.environ)
    env.update(SITL_ENV)
    # avoid gz model --spawn hang: allocate a PTY (start_px4.sh approach)
    p = subprocess.Popen(["/usr/bin/script", "-qfc", cmd,
                          os.path.join(logdir, "sitl_pty.log")],
                         stdout=lf, stderr=subprocess.STDOUT,
                         cwd=PX4_DIR, env=env, start_new_session=True)
    return p, lf


def wait_px4_home(logdir, deadline_s=90):
    """Wait for 'home set' (GPS/EKF ready) in sitl.log + stabilization buffer.
    Skipping this and launching nav arms with an unconverged EKF -> Attitude failure/blind land."""
    path = os.path.join(logdir, "sitl.log")
    t0 = time.monotonic()
    while time.monotonic() - t0 < deadline_s:
        if os.path.exists(path) and sh(
                f"grep -aq 'home set' {path} && echo OK || true",
                timeout=5).stdout.strip():
            # EKF yaw (magnetometer) convergence buffer — attitude yaw alignment
            # still needs tens of seconds after 'Ready for takeoff'; arming early
            # causes attitude failure right after takeoff
            time.sleep(25.0)
            return True
        time.sleep(1.5)
    return False


def wait_udp_port(port, logdir, deadline_s=150):
    """Wait until the PX4 of this SITL session opens port 14580 (avoids confusion
    with leftover instances: cleanup runs first + sitl.log tail diagnostics every 30 s)."""
    t0 = time.monotonic()
    last_diag = 0.0
    while time.monotonic() - t0 < deadline_s:
        r = sh(f"ss -uln | grep -q ':{port} ' && echo OK || true")
        if r.stdout.strip():
            return True
        now = time.monotonic()
        if now - last_diag > 30:
            last_diag = now
            tail = sh(f"tail -c 400 {os.path.join(logdir, 'sitl.log')} "
                      f"| tr '\\n' ' ' | tail -c 200", timeout=5)
            log(f"  [sitl] waiting {now - t0:.0f}s ... {tail.stdout.strip()[:180]}")
        time.sleep(1.5)
    return False


# ================================================================ judgment monitor
def run_monitor(goal_timeout_s, nav_deadline_s, arm_deadline_s, home_xy):
    """Collect state and judge with a mini rclpy node. Returns (result, metrics).
    - success  : goal 2D distance < 1.5 m held for 2.0 s continuously
    - collision: disarm after takeoff (was armed + z>0.5) while not yet judged
    - timeout  : nav deadline exceeded
    """
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from mavros_msgs.msg import State
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    from geometry_msgs.msg import PoseStamped

    class Monitor(Node):
        def __init__(self):
            super().__init__("eval_monitor")
            q = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(State, "/mavros/state", self._cb_state, 10)
            self.create_subscription(Odometry, "/mavros/local_position/odom",
                                     self._cb_odom, q)
            self.create_subscription(LaserScan, "/scan", self._cb_scan, q)
            self.create_subscription(PoseStamped, "/planner/goal", self._cb_goal, 10)
            self.connected = False
            self.armed_ever = False
            self.armed = False
            self.scan_ok = False
            self.pos = None            # (x, y, z)
            self.goal = None
            self.min_scan = math.inf
            self.path_len = 0.0
            self.t_goal_in = None      # time of entering the goal radius
            self.result = None
            self.note = ""
            self.t_arm = None
            self.t_nav0 = None

        def _cb_state(self, m):
            self.connected = m.connected
            if m.armed:
                self.armed_ever = True
                self.armed = True
                if self.t_arm is None:
                    self.t_arm = time.monotonic()
            else:
                # disarm after takeoff = crash/landed (when success not yet judged)
                if self.armed_ever and self.pos and self.pos[2] > 0.5 \
                        and self.result is None:
                    self.result = "collision"
                    self.note = f"disarm at z={self.pos[2]:.2f}"
                self.armed = False

        def _cb_odom(self, m):
            p = m.pose.pose.position
            prev, self.pos = self.pos, (p.x, p.y, p.z)
            if prev and self.armed and self.pos[2] > 1.0:
                self.path_len += math.hypot(p.x - prev[0], p.y - prev[1])
            # homing done (NAVIGATING starts) approx: first entry within 1.0 m of home
            if self.t_nav0 is None and self.armed and self.pos[2] > 1.5 \
                    and math.hypot(p.x - home_xy[0], p.y - home_xy[1]) < 1.0:
                self.t_nav0 = time.monotonic()
            # goal-reached judgment
            if self.goal and self.armed:
                d = math.hypot(p.x - self.goal[0], p.y - self.goal[1])
                if d < 1.5:
                    now = time.monotonic()
                    if self.t_goal_in is None:
                        self.t_goal_in = now
                    elif now - self.t_goal_in >= 2.0 and self.result is None:
                        self.result = "success"
                        self.note = f"goal_dist={d:.2f}m"
                else:
                    self.t_goal_in = None

        def _cb_scan(self, m):
            self.scan_ok = True
            if self.armed:
                # below 0.30 m is self-detection noise (matches lidar_range_min_override=0.35)
                for r in m.ranges:
                    if 0.30 < r < math.inf and r < self.min_scan:
                        self.min_scan = r

        def _cb_goal(self, m):
            p = m.pose.position
            self.goal = (p.x, p.y)

    rclpy.init()
    node = Monitor()
    stop = threading.Event()

    def _spin():
        while not stop.is_set() and node.result is None:
            rclpy.spin_once(node, timeout_sec=0.2)

    spin = threading.Thread(target=_spin, daemon=True)
    spin.start()

    t_launch = time.monotonic()
    result, note, metrics = None, "", {}
    try:
        while node.result is None:
            el = time.monotonic() - t_launch
            if not node.connected and el > goal_timeout_s:
                result, note = "infra_error", "mavros not connected"
                break
            if not node.scan_ok and el > goal_timeout_s + 30:
                result, note = "infra_error", "no /scan"
                break
            if not node.armed_ever and el > arm_deadline_s:
                result, note = "infra_error", "never armed"
                break
            if el > nav_deadline_s:
                result, note = "timeout", f"nav deadline {nav_deadline_s:.0f}s"
                break
            time.sleep(0.5)
    finally:
        stop.set()
        if node.result is not None and result is None:
            result, note = node.result, node.note
        if result is None:
            result, note = "infra_error", "monitor aborted"
        metrics = dict(
            t_total=round(time.monotonic() - t_launch, 1),
            t_nav=round((node.t_nav0 and time.monotonic() - node.t_nav0) or
                        (node.t_arm and time.monotonic() - node.t_arm) or 0.0, 1),
            path_len=round(node.path_len, 2),
            min_scan=(round(node.min_scan, 3) if math.isfinite(node.min_scan) else ""),
            final_goal_dist=(node.goal and node.pos and
                             round(math.hypot(node.pos[0] - node.goal[0],
                                              node.pos[1] - node.goal[1]), 2) or ""),
        )
        node.destroy_node()
        rclpy.shutdown()
        spin.join(timeout=3)
    return result, note, metrics


# ================================================================ CSV
CSV_COLS = ["ts", "method", "world", "course", "ep", "attempt", "result",
            "t_total_s", "t_nav_s", "path_len_m", "min_scan_m",
            "final_goal_dist_m", "note"]


def load_csv(path):
    done, rows = set(), []
    if not os.path.exists(path):
        return done, rows
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
            if r["result"] in FINAL_RESULTS:
                done.add((r["method"], r["world"], r["course"], int(r["ep"])))
    return done, rows


def append_csv_rows(new_rows, superseded_keys=()):
    """Update CSV: remove superseded infra rows, rewrite + mirror copy
    (temp file guards against mid-write abort)."""
    for path in (CSV_PRIMARY, CSV_MIRROR):
        _, old = load_csv(path)
        keep = [r for r in old
                if (r["method"], r["world"], r["course"], int(r["ep"]))
                not in superseded_keys]
        keep.extend(new_rows)
        tmp = path + ".tmp"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS)
            w.writeheader()
            w.writerows(keep)
        os.replace(tmp, path)


# ================================================================ episode run
def run_episode(method, world, course, ep, attempt):
    w = WORLDS[world]
    tag = f"{method}_{world}_{course}_ep{ep}_a{attempt}"
    logdir = os.path.join(LOG_ROOT, tag)
    shutil.rmtree(logdir, ignore_errors=True)   # keep old experiment appends from mixing in
    os.makedirs(logdir, exist_ok=True)
    log(f"=== [{tag}] start ===")

    write_world_config(world)
    cleanup_environment()

    procs, lfs = [], []

    def launch(name, starter):
        p, lf = starter()
        procs.append(p)
        lfs.append(lf)
        log(f"  [{name}] pid={p.pid}")

    try:
        # 1) SITL (headless, PTY) -> wait for 14580
        launch("sitl", lambda: start_sitl(world, logdir))
        if not wait_udp_port(14580, logdir):
            return "infra_error", "sitl udp 14580 not ready", {}
        log("  [sitl] udp 14580 OK")

        # 2) MAVROS (ROS2 syntax, remote 14580)
        launch("mavros", lambda: popen_ros(
            ["ros2 run mavros mavros_node --ros-args "
             "-p fcu_url:=udp://:14540@127.0.0.1:14580 -p tgt_system:=1"],
            os.path.join(logdir, "mavros.log")))

        # 2.5) wait for PX4 home set (EKF/GPS ready) — prevents early arm
        if not wait_px4_home(logdir):
            return "infra_error", "px4 home set not detected", {}
        log("  [px4] home set OK (+8s buffer)")

        # 3) perception stack (rviz disabled)
        launch("percep", lambda: popen_ros(
            ["ros2 launch px4_nav_perception s3_grid_fuser.launch.py rviz:=false"],
            os.path.join(logdir, "percep.log")))

        # 4) navigation (per-method node selection + course)
        m = METHODS[method]
        nav_args = ["ros2 launch px4_rl_offboard_mavros rl_offboard.launch.py",
                    f"course:={course}"]
        if m.get("overlay") is not None:
            nav_args.append(f"overlay:={m['overlay']}")
        elif m.get("rl_only"):
            nav_args.append("rl_only:=true")
        launch("nav", lambda: popen_ros(nav_args, os.path.join(logdir, "nav.log")))

        # 5) judgment monitor (mavros connect 45s / arm 120s / homing+nav deadline)
        result, note, metrics = run_monitor(
            goal_timeout_s=45,
            arm_deadline_s=120,
            nav_deadline_s=40 + w["nav_timeout_s"],
            home_xy=(w["home"][0], w["home"][1]))
        log(f"  [result] {result} ({note}) {metrics}")
        return result, note, metrics
    finally:
        for p in reversed(procs):
            kill_tree(p)
        for lf in lfs:
            try:
                lf.close()
            except Exception:
                pass
        cleanup_environment()
        # save log folder space: truncate large sitl_pty.log
        try:
            pty_log = os.path.join(logdir, "sitl_pty.log")
            if os.path.exists(pty_log) and os.path.getsize(pty_log) > 50e6:
                open(pty_log, "w").close()
        except Exception:
            pass


def main():
    global CSV_PRIMARY
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS),
                    choices=list(METHODS))
    ap.add_argument("--worlds", nargs="+", default=list(WORLDS), choices=list(WORLDS))
    ap.add_argument("--courses", nargs="+", default=["R", "L"], choices=["R", "L"])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--resume", action="store_true", help="skip runs already completed in CSV")
    ap.add_argument("--csv", default=CSV_PRIMARY)
    args = ap.parse_args()
    CSV_PRIMARY = args.csv

    if not os.path.exists(PX4_BIN):
        sys.exit(f"px4 binary not found: {PX4_BIN}")

    done, _ = load_csv(CSV_PRIMARY)
    if done and not args.resume:
        log(f"WARNING: CSV already has {len(done)} completed runs — rerunning adds duplicate rows. "
            f"Use --resume to continue")

    total = len(args.methods) * len(args.worlds) * len(args.courses) * args.episodes
    log(f"Experiment start: {total}-run matrix "
        f"({', '.join(args.methods)} x {', '.join(args.worlds)} x "
        f"{'/'.join(args.courses)} x {args.episodes}ep)")

    n = 0
    t_start = time.time()
    for method in args.methods:
        for world in args.worlds:
            for course in args.courses:
                for ep in range(1, args.episodes + 1):
                    n += 1
                    key = (method, world, course, ep)
                    if key in done:
                        log(f"[{n}/{total}] {key} — done, skipped")
                        continue
                    attempt = 1
                    superseded = set()
                    while attempt <= args.max_retries + 1:
                        result, note, metrics = run_episode(method, world, course,
                                                            ep, attempt)
                        row = dict(ts=datetime.now().isoformat(timespec="seconds"),
                                   method=method, world=world, course=course, ep=ep,
                                   attempt=attempt, result=result,
                                   t_total_s=metrics.get("t_total", ""),
                                   t_nav_s=metrics.get("t_nav", ""),
                                   path_len_m=metrics.get("path_len", ""),
                                   min_scan_m=metrics.get("min_scan", ""),
                                   final_goal_dist_m=metrics.get("final_goal_dist", ""),
                                   note=note)
                        append_csv_rows([row], superseded)
                        superseded.add(key)
                        if result in FINAL_RESULTS:
                            done.add(key)
                            break
                        attempt += 1
                        log(f"  infra error -> retry {attempt}/{args.max_retries + 1}")
                    el = (time.time() - t_start) / 60
                    log(f"[{n}/{total}] progress: {el:.1f} min elapsed")

    # summary
    _, rows = load_csv(CSV_PRIMARY)
    log("\n===== summary (final rows only) =====")
    for method in args.methods:
        for world in args.worlds:
            sel = [r for r in rows if r["method"] == method and r["world"] == world
                   and r["result"] in FINAL_RESULTS]
            if not sel:
                continue
            s = sum(r["result"] == "success" for r in sel)
            c = sum(r["result"] == "collision" for r in sel)
            t = sum(r["result"] == "timeout" for r in sel)
            log(f"{method:10s} {world:18s}: {len(sel)} runs "
                f"success={s} collision={c} timeout={t} "
                f"({100 * s / len(sel):.0f}%)")
    log("done.")


if __name__ == "__main__":
    main()
