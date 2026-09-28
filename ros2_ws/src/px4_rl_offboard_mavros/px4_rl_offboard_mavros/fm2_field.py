#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fm2_field.py -- FM2 (Fast Marching Square) guidance field, pure numpy (no scipy/numba).

Two-pass structure (Garrido et al. 2007; Gomez et al. 2013; Valero-Gomez et al. 2013 IEEE RAM):
  1) Velocity map W: occupancy grid -> chamfer clearance distance field (m, iterated to
     convergence) -> W = clip(clearance/alpha,0,1)**beta. Obstacle cells W=0, free cells
     floored at 0.02 (guards against S=1/W blow-up). Low speed near walls -> paths keep
     to corridor centers.
  2) Arrival-time field T: solve the Eikonal equation |gradT| = 1/W from the goal cell with
     Fast Sweeping (Jacobi parallel update, first-order upwind Godunov; same discrete
     solution as heap-based FMM, see Gomez 2019 IEEE Access survey).
     T depends only on (grid, goal), not on start -> callers may cache it per episode;
     replanning only re-runs extraction.

Path extraction: bilinear-interpolated gradient descent on T from start (obstacle /
unreachable cells are filled with a wall-penalty value so interpolation stays finite
everywhere). On stall, first run a discrete 8-neighbor argmin-T rescue, then fall back
to [goal] (same failure semantics as A*).

Machine-specific contract (nav_env_astar.py Sec.32): on this machine (i9-14900K +
numpy 2.x + py3.13) numpy scalar indexing in hot loops causes access violations, so
extraction converts fields to Python lists via .tolist() and does all bilinear reads
in pure float. The module has no mutable global state (safe for 8-process
SubprocVecEnv imports).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

# Telemetry counter keys (the stats dict is supplied by the caller; module is stateless)
STAT_CALLS = "calls"
STAT_FALLBACK = "fallback"          # full fallback to [goal]
STAT_RESCUE = "rescue"              # discrete rescue succeeded after stall
STAT_SWEEP_CAP = "sweep_cap"        # sweeps hit the cap (no convergence)
STAT_SNAP = "start_snap"            # start cell had T=inf and was snapped to a free cell


def _bump(stats: Optional[Dict[str, int]], key: str) -> None:
    if stats is not None:
        stats[key] = stats.get(key, 0) + 1


# ---------------------------------------------------------------------------
# 1) Clearance distance field (chamfer relaxation to convergence, 8-neighbor,
#    weights 1 / sqrt(2))
# ---------------------------------------------------------------------------
def chamfer_clearance(grid: np.ndarray, cell_size: float) -> np.ndarray:
    """Approximate Euclidean distance (m) from each free cell to the nearest obstacle
    cell. Obstacle cells = 0.

    Iterative chamfer relaxation (np.pad + slicing; roll disabled to avoid toroidal
    leakage across borders), exits when stable; capped at 2n rounds (worst-case
    diagonal propagation). Diagonal distances are overestimated by at most ~7.6%
    (monotone, conservative direction). Note: world borders are not rasterized as
    obstacles (consistent with A*/BFS); border cells are free space.
    """
    n = grid.shape[0]
    if int(grid.sum()) == 0:
        # no obstacles: clearance unbounded, return a large constant (W saturates to 1)
        return np.full((n, n), 1e3, dtype=np.float32)

    INF = np.float32(1e9)
    dist = np.where(grid > 0, np.float32(0.0), INF)
    c1 = np.float32(cell_size)
    c2 = np.float32(cell_size * 1.4142135623730951)
    neighbors = (
        (-1, 0, c1), (1, 0, c1), (0, -1, c1), (0, 1, c1),
        (-1, -1, c2), (-1, 1, c2), (1, -1, c2), (1, 1, c2),
    )
    for _ in range(2 * n):
        p = np.pad(dist, 1, constant_values=INF)
        best = dist
        for di, dj, w in neighbors:
            best = np.minimum(best, p[1 + di: 1 + di + n, 1 + dj: 1 + dj + n] + w)
        if np.array_equal(best, dist):
            break
        dist = best
    dist = np.where(dist >= INF, np.float32(0.0), dist)
    return dist


# ---------------------------------------------------------------------------
# 2) Velocity map W in [0,1]
# ---------------------------------------------------------------------------
def velocity_map(
    grid: np.ndarray,
    cell_size: float,
    alpha: float = 1.0,
    beta: float = 1.0,
    w_floor: float = 0.02,
) -> np.ndarray:
    """W = clip(clearance/alpha, 0, 1)**beta; obstacles W=0; free cells floored at w_floor.

    alpha: saturation clearance (m) -- cells with clearance >= alpha get speed 1
           (the alpha saturation parameter from Sensors 2022).
    beta:  exponent shaping (the beta from Sensors 2022).
    w_floor: defensive floor (S=1/W <= 50) so no parameter drift can produce inf.
    """
    clearance = chamfer_clearance(grid, cell_size)
    w = np.clip(clearance / max(float(alpha), 1e-6), 0.0, 1.0)
    if abs(beta - 1.0) > 1e-12:
        w = w ** float(beta)
    free = grid == 0
    w = np.where(free, np.maximum(w, float(w_floor)), 0.0).astype(np.float64)
    return w


# ---------------------------------------------------------------------------
# 3) Eikonal arrival-time field (Fast Sweeping, Jacobi parallel iteration,
#    fully vectorized)
# ---------------------------------------------------------------------------
def sweep_travel_time(
    w_map: np.ndarray,
    goal_cell: Tuple[int, int],
    cell_size: float,
    tol: float = 1e-4,
    max_sweeps: int = 400,
    stats: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """Solve |gradT| = 1/W from the goal cell. Returns T (n,n) float64; obstacle /
    unreachable cells = inf.

    Jacobi form: T <- min(T, upwind quadratic update); information propagates at
    least 1 cell per round. max_sweeps=400 covers 100 m-scale winding paths
    (measured convergence on these scenarios: 128-232 rounds).
    The 2D update is valid iff hS >= |a-b| (large root >= max(a,b) iff this holds);
    otherwise it degenerates to the 1D update.
    """
    n = w_map.shape[0]
    h = float(cell_size)
    free = w_map > 0.0
    slowness = np.where(free, 1.0 / np.maximum(w_map, 1e-9), np.inf)  # S = 1/F
    hs = h * slowness

    T = np.full((n, n), np.inf, dtype=np.float64)
    gi, gj = int(goal_cell[0]), int(goal_cell[1])
    T[gi, gj] = 0.0  # seed (caller guarantees the goal cell is free)

    for sweep in range(max_sweeps):
        p = np.pad(T, 1, constant_values=np.inf)
        a = np.minimum(p[:-2, 1:-1], p[2:, 1:-1])   # min of up/down (i direction)
        b = np.minimum(p[1:-1, :-2], p[1:-1, 2:])   # min of left/right (j direction)
        a_fin = np.isfinite(a)
        b_fin = np.isfinite(b)

        mn = np.minimum(a, b)
        single = mn + hs
        # Quadratic update (T-a)^2+(T-b)^2 = (hS)^2; the large root is valid iff
        # hS >= |a-b| (exact condition, replacing the looser disc>0 test: for
        # |a-b| in [hS, sqrt(2)*hS) disc>0 but the large root < max(a,b), invalid)
        with np.errstate(invalid="ignore"):
            quad_root = 0.5 * (a + b + np.sqrt(np.maximum(2.0 * hs * hs - (a - b) ** 2, 0.0)))
            quad_ok = a_fin & b_fin & (hs >= np.abs(a - b))
        cand = np.where(quad_ok, quad_root, single)
        cand = np.where(a_fin | b_fin, cand, np.inf)

        new_t = np.where(free, np.minimum(T, cand), T)
        fin = np.isfinite(new_t)
        if fin.any():
            old = np.where(fin, T, 0.0)
            change = float(np.max(np.abs(new_t[fin] - old[fin])))
        else:
            change = 0.0
        T = new_t
        if change < tol:
            break
    else:
        _bump(stats, STAT_SWEEP_CAP)  # hit the cap = not converged; visible in telemetry
    return T


# ---------------------------------------------------------------------------
# 4) Gradient-descent path extraction
#    Entry converts fields to Python lists (see Sec.32: numpy scalar-indexing hot
#    loops previously caused access violations on this machine)
# ---------------------------------------------------------------------------
def _bilinear_list(field: List[List[float]], x: float, y: float, cell_size: float) -> float:
    """Bilinear interpolation at world coordinates (x, y), pure Python floats.
    x -> column (j), y -> row (i).

    Clamp the *coordinates*, not the indices: if fj<0 and we clamped j0=0 while
    keeping the unclamped fractional part tj, out-of-border probes would alias
    to the same weights as interior points -> the central difference gx on the
    border column would be identically 0 (measured root cause of a 2-period
    oscillation fake-stall along the world border in the density scenario).
    """
    n = len(field)
    fj = x / cell_size - 0.5
    fi = y / cell_size - 0.5
    if fj < 0.0:
        fj = 0.0
    elif fj > n - 1.0:
        fj = n - 1.0
    if fi < 0.0:
        fi = 0.0
    elif fi > n - 1.0:
        fi = n - 1.0
    j0 = int(math.floor(fj))
    i0 = int(math.floor(fi))
    if j0 > n - 2:
        j0 = n - 2
    if i0 > n - 2:
        i0 = n - 2
    tj = fj - j0
    ti = fi - i0
    row0 = field[i0]
    row1 = field[i0 + 1]
    v00 = row0[j0]
    v01 = row0[j0 + 1]
    v10 = row1[j0]
    v11 = row1[j0 + 1]
    return (
        (1 - ti) * (1 - tj) * v00
        + (1 - ti) * tj * v01
        + ti * (1 - tj) * v10
        + ti * tj * v11
    )


def _discrete_rescue(
    t_list: List[List[float]],
    start: Tuple[float, float],
    goal_cell: Tuple[int, int],
    cell_size: float,
    max_steps: int = 4000,
) -> Optional[List[Tuple[float, float]]]:
    """Stall rescue: walk from the start cell to the goal cell along 8-neighbor
    argmin-T steps. Godunov causality guarantees every non-goal free cell has a
    strictly smaller neighbor, so this cannot loop. Returns None on failure
    (no smaller neighbor / out of bounds)."""
    n = len(t_list)
    j = max(0, min(n - 1, int(start[0] / cell_size)))
    i = max(0, min(n - 1, int(start[1] / cell_size)))
    gi, gj = goal_cell
    if not math.isfinite(t_list[i][j]):
        return None
    path: List[Tuple[float, float]] = []
    visited = set()
    for _ in range(max_steps):
        path.append(((j + 0.5) * cell_size, (i + 0.5) * cell_size))
        if i == gi and j == gj:
            return path
        t_cur = t_list[i][j]
        best = None
        best_t = t_cur
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n:
                    tv = t_list[ni][nj]
                    if tv < best_t:  # inf never participates (inf < inf is False)
                        best_t = tv
                        best = (ni, nj)
        if best is None or best in visited:
            return None
        visited.add(best)
        i, j = best
    return None


def _snap_nearest_finite(
    t_list: List[List[float]],
    x: float,
    y: float,
    cell_size: float,
    max_radius: int = 4,
) -> Optional[Tuple[int, int]]:
    """Finite-T cell (i,j) closest to (x,y) (min squared cell distance), expanding
    rings up to max_radius."""
    n = len(t_list)
    si = max(0, min(n - 1, int(y / cell_size)))
    sj = max(0, min(n - 1, int(x / cell_size)))
    if math.isfinite(t_list[si][sj]):
        return (si, sj)
    for radius in range(1, max_radius + 1):
        best = None
        best_d = math.inf
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                ni, nj = si + di, sj + dj
                if 0 <= ni < n and 0 <= nj < n and math.isfinite(t_list[ni][nj]):
                    d = di * di + dj * dj
                    if d < best_d:
                        best_d = d
                        best = (ni, nj)
        if best is not None:
            return best
    return None


def extract_path(
    travel_time: np.ndarray,
    start: Tuple[float, float],
    goal: Tuple[float, float],
    cell_size: float,
    world_size: float,
    step: float | None = None,
    max_iters: int = 8000,
    stall_window: int = 10,
    stats: Optional[Dict[str, int]] = None,
) -> Optional[List[Tuple[float, float]]]:
    """Gradient descent on -gradT from start to goal. Returns world-coordinate
    waypoints (last point is the exact goal).

    Two gradient sources:
      - If the +-eps difference probes' covered neighborhood (4x4 cells) is fully
        finite -> bilinear gradient (smooth region);
      - If the neighborhood touches a wall (always the case inside a 0.25 m
        single-cell crack) -> Godunov upwind difference gradient on the current
        cell. Causality guarantees a non-goal finite cell has at least one axial
        neighbor strictly smaller, so the gradient is nonzero and points along
        the crack axis; this avoids wall_penalty (~t_max+1000) being sampled by
        the difference probes and drowning the descent direction into along-wall
        / along-border oscillation (measured root cause of fake stalls in the
        density scenario).
    |gradT| = S >= 1, so each step normally lowers T by ~step; stall_window
    consecutive steps without decrease = truly stuck. Then first check goal-cell
    arrival (cell diagonal 0.354 m may exceed the arrival threshold), then run a
    full discrete argmin-T rescue (snapping to the nearest finite cell first if
    start has T=inf). Returns None if everything fails (caller falls back to
    [goal]).
    """
    finite = np.isfinite(travel_time)
    if not finite.any():
        return None
    n = travel_time.shape[0]
    gi = max(0, min(n - 1, int(goal[1] / cell_size)))
    gj = max(0, min(n - 1, int(goal[0] / cell_size)))

    t_max = float(travel_time[finite].max())
    wall_penalty = t_max + 1000.0  # penalty for obstacle/unreachable cells: interpolation stays finite, gradient points away from walls
    g_list = np.where(finite, travel_time, wall_penalty).tolist()  # Sec.32 workaround
    t_list = travel_time.tolist()

    # start cell T=inf (replan start cell rasterized as occupied) -> snap to nearest finite cell
    sx, sy = float(start[0]), float(start[1])
    si = max(0, min(n - 1, int(sy / cell_size)))
    sj = max(0, min(n - 1, int(sx / cell_size)))
    if not math.isfinite(t_list[si][sj]):
        snapped = _snap_nearest_finite(t_list, sx, sy, cell_size)
        if snapped is None:
            return None
        _bump(stats, STAT_SNAP)
        sx = (snapped[1] + 0.5) * cell_size
        sy = (snapped[0] + 0.5) * cell_size

    if step is None:
        step = 0.5 * cell_size
    eps = 0.5 * cell_size

    px, py = sx, sy
    gx, gy = float(goal[0]), float(goal[1])
    path: List[Tuple[float, float]] = [(px, py)]
    since_record = 0.0
    last_t = _bilinear_list(g_list, px, py, cell_size)
    stall = 0

    for _ in range(max_iters):
        ci = max(0, min(n - 1, int(py / cell_size)))
        cj = max(0, min(n - 1, int(px / cell_size)))
        if ci == gi and cj == gj:  # inside goal cell: finish directly (diagonal 0.354 m may exceed arrival threshold)
            path.append((gx, gy))
            return path

        # Neighborhood (i0-1..i0+2, j0-1..j0+2) fully covers the +-eps probe
        # stencil: any wall contact -> upwind mode (otherwise probes sample
        # wall_penalty ~ t_max+1000 and the gradient is drowned into along-crack
        # oscillation -- measured fake-stall root cause #2 in the density scenario)
        fj = px / cell_size - 0.5
        fi = py / cell_size - 0.5
        j0 = max(0, min(n - 2, int(math.floor(fj))))
        i0 = max(0, min(n - 2, int(math.floor(fi))))
        nbhd_fin = True
        for ii in range(max(0, i0 - 1), min(n, i0 + 3)):
            row = t_list[ii]
            for jj in range(max(0, j0 - 1), min(n, j0 + 3)):
                if not math.isfinite(row[jj]):
                    nbhd_fin = False
                    break
            if not nbhd_fin:
                break

        if nbhd_fin:
            gx_ = (_bilinear_list(g_list, px + eps, py, cell_size)
                   - _bilinear_list(g_list, px - eps, py, cell_size)) / (2.0 * eps)
            gy_ = (_bilinear_list(g_list, px, py + eps, cell_size)
                   - _bilinear_list(g_list, px, py - eps, cell_size)) / (2.0 * eps)
        else:
            t_cur = t_list[ci][cj]
            if math.isfinite(t_cur):
                # Godunov upwind: take the smaller axial neighbor per axis; gradient
                # points opposite to the direction of increasing T
                tl = t_list[ci][cj - 1] if cj > 0 else math.inf
                tr = t_list[ci][cj + 1] if cj < n - 1 else math.inf
                ax = min(tl, tr)
                gx_ = 0.0
                if math.isfinite(ax):
                    gx_ = max(t_cur - ax, 0.0) / cell_size
                    if tr < tl:
                        gx_ = -gx_
                td = t_list[ci - 1][cj] if ci > 0 else math.inf
                tu = t_list[ci + 1][cj] if ci < n - 1 else math.inf
                ay = min(td, tu)
                gy_ = 0.0
                if math.isfinite(ay):
                    gy_ = max(t_cur - ay, 0.0) / cell_size
                    if tu < td:
                        gy_ = -gy_
            else:
                # position landed in a wall cell (interpolation drift): move toward
                # the nearest finite cell
                best = _snap_nearest_finite(t_list, px, py, cell_size, max_radius=1)
                if best is None:
                    break
                gx_ = float(best[1] - cj)
                gy_ = float(best[0] - ci)

        nrm = math.hypot(gx_, gy_)
        if nrm < 1e-9:
            break  # theoretically unreachable (upwind gradient of a non-goal finite cell is nonzero); defensive
        px -= step * gx_ / nrm
        py -= step * gy_ / nrm
        px = max(0.0, min(world_size, px))
        py = max(0.0, min(world_size, py))

        # Progress check uses the same mode as the gradient: bilinear value in
        # smooth regions, current-cell value near walls
        if nbhd_fin:
            t_now = _bilinear_list(g_list, px, py, cell_size)
        else:
            tv = t_list[max(0, min(n - 1, int(py / cell_size)))][max(0, min(n - 1, int(px / cell_size)))]
            t_now = tv if math.isfinite(tv) else last_t + 1.0  # drifted into a wall cell counts as no progress
        if t_now < last_t - 1e-4:  # net-decrease threshold tolerates float noise
            stall = 0
            last_t = t_now
        else:
            stall += 1
            if stall >= stall_window:
                break  # truly stuck -> discrete rescue

        since_record += step
        if since_record >= cell_size:
            path.append((px, py))
            since_record = 0.0

        if math.hypot(px - gx, py - gy) <= cell_size:
            path.append((gx, gy))  # last waypoint = exact goal (matches A* last-waypoint semantics)
            return path

    if math.hypot(px - gx, py - gy) <= 2.0 * cell_size:
        path.append((gx, gy))
        return path

    # Stalled -> (snap first if in a wall cell) -> full discrete argmin-T rescue
    snap2 = _snap_nearest_finite(t_list, px, py, cell_size)
    if snap2 is None:
        return None
    rescue = _discrete_rescue(
        t_list,
        ((snap2[1] + 0.5) * cell_size, (snap2[0] + 0.5) * cell_size),
        (gi, gj),
        cell_size,
    )
    if rescue:
        _bump(stats, STAT_RESCUE)
        path.extend(rescue)
        path.append((gx, gy))
        return path
    return None


# ---------------------------------------------------------------------------
# 5) Top-level API: field computation (cacheable) + one-shot convenience wrapper
# ---------------------------------------------------------------------------
def fm2_field_compute(
    grid: np.ndarray,
    goal: Tuple[float, float],
    cell_size: float = 0.25,
    alpha: float = 1.0,
    beta: float = 1.0,
    stats: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """Compute the arrival-time field T (depends only on grid and goal, not start,
    so it can be cached per episode).

    grid: (n,n) uint8, 1=obstacle (output of env._build_obstacle_grid).
    The goal cell is forced free (mirrors A*'s grid[gi,gj]=0; order: force-free
    before clearance).
    """
    n = grid.shape[0]
    grid = grid.copy()
    gj = max(0, min(n - 1, int(goal[0] / cell_size)))
    gi = max(0, min(n - 1, int(goal[1] / cell_size)))
    grid[gi, gj] = 0

    w_map = velocity_map(grid, cell_size, alpha=alpha, beta=beta)
    return sweep_travel_time(w_map, (gi, gj), cell_size, stats=stats)


def fm2_path(
    grid: np.ndarray,
    start: Tuple[float, float],
    goal: Tuple[float, float],
    cell_size: float = 0.25,
    alpha: float = 1.0,
    beta: float = 1.0,
    world_size: float | None = None,
    stats: Optional[Dict[str, int]] = None,
) -> List[Tuple[float, float]]:
    """One-shot FM2 guidance path (world-coordinate waypoints, last = exact goal).
    Falls back to [goal] on failure (same as A*).

    Callers that cache T per episode (fm2_field_compute + extract_path) must not
    use this function to recompute.
    """
    _bump(stats, STAT_CALLS)
    n = grid.shape[0]
    if world_size is None:
        world_size = n * cell_size

    grid = grid.copy()
    sj = max(0, min(n - 1, int(start[0] / cell_size)))
    si = max(0, min(n - 1, int(start[1] / cell_size)))
    grid[si, sj] = 0  # start cell also forced free (one-shot semantics; in cached mode extract_path snaps instead)

    travel = fm2_field_compute(grid, goal, cell_size=cell_size, alpha=alpha, beta=beta, stats=stats)
    path = extract_path(travel, start, goal, cell_size, world_size, stats=stats)
    if path is None or len(path) < 2:
        _bump(stats, STAT_FALLBACK)
        return [(float(goal[0]), float(goal[1]))]
    return path


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Synthetic vertical wall with a gap: verify (a) goes through the gap
    # (b) determinism (c) runtime (d) telemetry
    import time

    cs = 0.25
    n = 80
    g = np.zeros((n, n), dtype=np.uint8)
    g[:, 40] = 1
    g[56:60, 40] = 0
    start = (5.0, 10.0)
    goal = (15.0, 10.0)

    stats: Dict[str, int] = {}
    t0 = time.perf_counter()
    p1 = fm2_path(g, start, goal, cell_size=cs, stats=stats)
    t1 = time.perf_counter()
    p2 = fm2_path(g, start, goal, cell_size=cs)
    assert p1 == p2, "must be deterministic"
    xs = np.array([q[0] for q in p1])
    ys = np.array([q[1] for q in p1])
    near_wall = np.abs(xs - 10.0) < 1.0
    assert ys[near_wall].min() > 13.0, "path should pass through the gap"
    assert abs(p1[-1][0] - goal[0]) < 1e-9 and abs(p1[-1][1] - goal[1]) < 1e-9, "last point = exact goal"

    # Cache semantics: reuse T with different starts; must be bit-identical to one-shot
    T = fm2_field_compute(g, goal, cell_size=cs)
    p3 = extract_path(T, start, goal, cs, n * cs)
    assert p3 == p1[:-1] or p3 == p1, f"cache semantics should match: {len(p3)} vs {len(p1)}"

    print(f"self-test OK: {len(p1)} waypoints, {(t1-t0)*1000:.1f} ms/call, stats={stats}")
