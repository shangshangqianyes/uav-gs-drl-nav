
import argparse
import math
import os
import time
from typing import List, Tuple

import numpy as np

from nav_env_astar import UAVNav2DEnv as AstarEnv
from nav_env_fm2 import UAVNav2DEnv as FM2Env

CELL = 0.25
SCENARIOS = ("obstacle_density_15", "obstacle_density_20", "narrow_passage", "course_l_s")


def path_length(path: List[Tuple[float, float]]) -> float:
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])))


def path_clearance(
    path: List[Tuple[float, float]],
    obstacles,
    world_size: float,
    sample_step: float = 0.05,
) -> Tuple[float, float]:
    """Sample every sample_step along the path; distance (min, mean) to the nearest obstacle-rectangle boundary / world boundary."""
    if len(path) < 2:
        return 0.0, 0.0
    pts: List[Tuple[float, float]] = []
    for a, b in zip(path, path[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        k = max(1, int(seg / sample_step))
        for t in range(k):
            s = t / k
            pts.append((a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])))
    dists = []
    for x, y in pts:
        d = min(x, y, world_size - x, world_size - y)  # world boundaries count as collision surfaces too
        for obs in obstacles:
            dx = max(obs.xmin - x, 0.0, x - obs.xmax)
            dy = max(obs.ymin - y, 0.0, y - obs.ymax)
            d = min(d, math.hypot(dx, dy))
        dists.append(d)
    return float(np.min(dists)), float(np.mean(dists))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=int, default=40, help="number of maps per scenario")
    ap.add_argument("--out_dir", type=str, default="./bench_out")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "planner_compare.csv")

    env_a = AstarEnv(eval_scenario=None)   # used only to obtain the A* planner
    env_f = FM2Env(eval_scenario=None)     # used only to obtain the FM2 planner
    sampler = AstarEnv(curriculum_verbose=False)  # scenario sampler (A* env, but map generation is identical to the FM2 env)

    rows = []
    header = "scenario,map_idx,planner,path_length,min_clear,mean_clear,runtime_ms,waypoints"
    print(header)
    png_done = set()

    for scenario in SCENARIOS:
        for k in range(args.maps):
            sampler.eval_scenario = scenario
            sampler.use_curriculum = False
            sampler.reset(seed=10_000 + k)
            # Planner envs must receive the sampler's map, otherwise they plan on an empty map!
            env_a.obstacles = sampler.obstacles
            env_f.obstacles = sampler.obstacles
            start = (float(sampler.pos[0]), float(sampler.pos[1]))
            goal = (float(sampler.goal[0]), float(sampler.goal[1]))

            paths = {}
            for name, env, fn in (
                ("astar", env_a, env_a._compute_astar_path),
                ("fm2", env_f, env_f._compute_fm2_path),
            ):
                t0 = time.perf_counter()
                p = fn(np.array(start), np.array(goal), CELL)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                ln = path_length(p)
                mn, mean = path_clearance(p, sampler.obstacles, sampler.world_size)
                rows.append((scenario, k, name, ln, mn, mean, dt_ms, len(p)))
                print(f"{scenario},{k},{name},{ln:.3f},{mn:.3f},{mean:.3f},{dt_ms:.1f},{len(p)}")
                paths[name] = p

            if scenario not in png_done:
                grid, _ = sampler._build_obstacle_grid(CELL)
                _save_overlay(png_done, out_dir, scenario, sampler, grid, paths)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(str(x) for x in r[:2]) + f",{r[2]},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f},{r[6]:.2f},{r[7]}\n")

    # Summary table
    print("\n=== Summary (mean over maps) ===")
    print(f"{'scenario':<22}{'planner':<8}{'length':>9}{'min_clr':>9}{'mean_clr':>10}{'ms':>8}")
    for scenario in SCENARIOS:
        for name in ("astar", "fm2"):
            sel = [r for r in rows if r[0] == scenario and r[2] == name]
            print(
                f"{scenario:<22}{name:<8}"
                f"{np.mean([r[3] for r in sel]):>9.3f}"
                f"{np.mean([r[4] for r in sel]):>9.3f}"
                f"{np.mean([r[5] for r in sel]):>10.3f}"
                f"{np.median([r[6] for r in sel]):>8.1f}"
            )
    print(f"\nCSV: {csv_path}")
    print(f"FM2 telemetry: {env_f._fm2_stats}")


def _save_overlay(png_done, out_dir, scenario, sampler, grid, paths) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, origin="lower", extent=(0, sampler.world_size, 0, sampler.world_size),
              cmap="Greys", alpha=0.6)
    for name, color in (("astar", "tab:red"), ("fm2", "tab:blue")):
        p = np.array(paths[name])
        ax.plot(p[:, 0], p[:, 1], "-o", ms=2, lw=1.5, color=color, label=name)
    ax.plot(*sampler.pos, "g*", ms=15, label="start")
    ax.plot(*sampler.goal, "r*", ms=15, label="goal")
    ax.set_title(f"{scenario}: A* (red) vs FM2 (blue)")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"bench_{scenario}.png"), dpi=150)
    plt.close(fig)
    png_done.add(scenario)


if __name__ == "__main__":
    main()
