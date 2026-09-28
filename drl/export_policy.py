

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Tuple

# Prevent UnicodeEncodeError when printing on non-UTF-8 terminals such as the Windows GBK console
# (encoding is kept; only unrepresentable characters are replaced with '?' -- no behavior change on UTF-8 terminals)
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(errors="replace")
        except Exception:
            pass

import numpy as np
import torch

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_guided_astar import UAVNav2DEnv


# -----------------------------
# Action-only wrapper
# SACPolicy.forward() returns only the action, while ActorCriticPolicy(PPO) returns
# an (action, value, log_prob) tuple -> branch on the runtime type
# (tracing fixes a single path, so this is TorchScript-safe)
# -----------------------------
class ActionOnlyWrapper(torch.nn.Module):
    """Wraps an SB3 policy to return only the action. deterministic=True fixed. For TorchScript tracing."""

    def __init__(self, policy: torch.nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.policy(obs, deterministic=True)
        if isinstance(out, tuple):
            return out[0]
        return out


def make_env(args: argparse.Namespace) -> DummyVecEnv:
    """
    The env should be created with the same parameters as training.
    (fallback handling in case a constructor signature mismatch raises TypeError)
    """
    def _make() -> UAVNav2DEnv:
        # Pass args to match training as closely as possible (adjust to env_guided_astar.py __init__ signature if needed)
        kwargs: Dict[str, Any] = dict(
            obs_beams=args.obs_beams,
            max_range=args.max_range,
            world_size=args.world_size,
            v_min=args.v_min,
            v_max=args.v_max,
            yaw_rate_max=args.yaw_rate_max,
        )
        try:
            return UAVNav2DEnv(**kwargs)
        except TypeError:
            # fallback: minimal args only
            return UAVNav2DEnv(obs_beams=args.obs_beams, max_range=args.max_range)

    return DummyVecEnv([_make])


def export_norm_json(
    out_dir: str,
    args: argparse.Namespace,
    mean: np.ndarray,
    std: np.ndarray,
    clip_obs: float,
    epsilon: float,
    obs_dim: int,
    action_dim: int,
    filename: str = "obs_norm_hybrid.json",
) -> str:
    """Save obs_norm.json"""

    # obs layout (body frame, no heading):
    #   [ lidar_0..lidar_{N-1}, goal_rel_x, goal_rel_y ]  (66 dim)
    # env norm:
    #   lidar: r/max_range*2 - 1  -> [-1,1]
    #   goal_rel: R(-yaw) @ (goal-pos) / world_size  (body frame, x=forward, y=left)
    obs_ordering = [
        {
            "name": "lidar",
            "indices": f"0:{args.obs_beams}",
            "unit": "m",
            "raw_range": [0.0, float(args.max_range)],
            "env_norm": "lidar/max_range*2-1 -> [-1,1]",
        },
        {
            "name": "goal_rel",
            "indices": f"{args.obs_beams}:{args.obs_beams+2}",
            "unit": "m/m",
            "raw_range": "[-1,1] (scaled)",
            "env_norm": "R(-yaw)@(goal-pos)/world_size (body frame)",
        },
    ]

    norm_dict = {
        "obs": {
            "obs_dim": int(obs_dim),
            "ordering": obs_ordering,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "clip_obs": float(clip_obs),
            "epsilon": float(epsilon),
            "note": "Apply env_norm first, then VecNormalize: (obs-mean)/std, then clip to [-clip_obs, clip_obs].",
        },
        "action": {
            "action_dim": int(action_dim),
            "action_range": [-1.0, 1.0],
            "action_type": "velocity",
            "scale": {
                "v_min": float(args.v_min),
                "v_max": float(args.v_max),
                "max_vxy": float(args.v_max),
                "max_yaw_rate": float(args.yaw_rate_max),
            },
            "formula": {
                "v_cmd": "v_min + (a_v + 1) / 2 * (v_max - v_min)",
                "yaw_rate_cmd": "a_yaw * max_yaw_rate",
            },
        },
        "env_params": {
            "obs_beams": int(args.obs_beams),
            "max_range": float(args.max_range),
            "world_size": float(args.world_size),
        },
    }

    norm_path = os.path.join(out_dir, filename)
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(norm_dict, f, indent=2, ensure_ascii=False)

    return norm_path


def env_norm_example(raw_obs: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """
    "Example" env_norm for checking the deployment pipeline.
    Must match the observation construction in env_guided_astar.py.
    obs: lidar (n) + goal_rel body (2) = n+2
    """
    n = args.obs_beams
    assert raw_obs.shape[0] == n + 2

    lidar = raw_obs[:n]
    goal = raw_obs[n:n+2]

    # lidar: [0,max_range] -> [-1,1]
    lidar_n = np.clip(lidar, 0.0, args.max_range) / float(args.max_range)
    lidar_n = lidar_n * 2.0 - 1.0

    # goal_rel body: already R(-yaw)@(goal-pos)/world_size, just clip
    goal_n = np.clip(goal, -1.0, 1.0)

    return np.concatenate([lidar_n, goal_n]).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description="Export SAC policy for ROS2/PX4 deployment")
    p.add_argument("--algo", type=str, default="sac", choices=["sac", "ppo"],
                   help="Algorithm to export: sac (main training script train_astar_sac.py, default) / ppo (baseline train_astar_ppo.py)")
    p.add_argument("--model", type=str, required=True, help="Model path (.zip may be omitted; use the artifact matching the algorithm)")
    p.add_argument("--vecnorm", type=str, required=True, help="VecNormalize path (file saved via VecNormalize.save())")
    p.add_argument("--output_dir", type=str, default="./export", help="Output directory")
    p.add_argument("--norm_filename", type=str, default="obs_norm_hybrid.json",
                   help="Output filename for obs/action normalization JSON")
    p.add_argument("--policy_filename", type=str, default="policy_hybrid_ts.pt",
                   help="Output filename for TorchScript policy (.pt)")

    # env spec (must match training)
    p.add_argument("--obs_beams", type=int, default=64, help="Must match training env")
    p.add_argument("--max_range", type=float, default=10.0, help="Lidar max range (m)")
    p.add_argument("--world_size", type=float, default=20.0, help="World size for goal_rel scale")

    # action spec (must match training)
    p.add_argument("--v_min", type=float, default=0.3, help="Min velocity (m/s)")
    p.add_argument("--v_max", type=float, default=0.8, help="Max velocity (m/s)")
    p.add_argument("--yaw_rate_max", type=float, default=1.0, help="Max yaw rate (rad/s)")

    # verification options
    p.add_argument("--verify_full_pipeline", action="store_true",
                   help="Also verify the example pipeline raw->env_norm->vecnorm->policy (meaningful only if env_norm matches the real one)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- (A) create dummy env (same parameters as training recommended) ----
    dummy_env = make_env(args)

    # ---- (B) load VecNormalize (robust method) ----
    vecnorm = VecNormalize.load(args.vecnorm, dummy_env)
    vecnorm.training = False
    vecnorm.norm_reward = False

    obs_rms = vecnorm.obs_rms
    mean = np.array(obs_rms.mean, dtype=np.float64)
    var = np.array(obs_rms.var, dtype=np.float64)
    epsilon = float(getattr(vecnorm, "epsilon", 1e-8))
    std = np.sqrt(var + epsilon)
    clip_obs = float(getattr(vecnorm, "clip_obs", 10.0))

    obs_dim = int(mean.shape[0])

    # ---- (C) load model (SAC by default; PPO baseline with --algo ppo) ----
    load_path = args.model.rstrip("/")
    if load_path.lower().endswith(".zip"):
        load_path = load_path[:-4]

    algo_cls = SAC if args.algo == "sac" else PPO
    model = algo_cls.load(load_path, env=dummy_env)

    # do not hardcode action_dim
    action_dim = int(np.prod(model.action_space.shape))
    model_obs_dim = int(np.prod(model.observation_space.shape))

    # ---- (D) mandatory consistency checks ----
    if model_obs_dim != obs_dim:
        raise ValueError(f"obs_dim mismatch: model={model_obs_dim}, vecnorm={obs_dim}")

    # verify obs layout (lidar N + goal_rel 2) (body frame, no heading)
    expected_obs_dim = int(args.obs_beams + 2)
    if obs_dim != expected_obs_dim:
        raise ValueError(
            f"Unexpected obs_dim={obs_dim}. Expected beams+2={expected_obs_dim}. "
            f"(env observation layout / VecNormalize file may differ from training)"
        )

    # ---- (E) save obs_norm.json ----
    norm_path = export_norm_json(
        out_dir=args.output_dir,
        args=args,
        mean=mean.astype(np.float64),
        std=std.astype(np.float64),
        clip_obs=clip_obs,
        epsilon=epsilon,
        obs_dim=obs_dim,
        action_dim=action_dim,
        filename=args.norm_filename,
    )
    print(f"Saved: {norm_path}")

    # ---- (F) TorchScript export ----
    model.policy.set_training_mode(False)
    policy = model.policy
    wrapper = ActionOnlyWrapper(policy).eval()

    dummy_input = torch.zeros(1, obs_dim, dtype=torch.float32)
    with torch.no_grad():
        scripted = torch.jit.trace(wrapper, dummy_input)

    policy_path = os.path.join(args.output_dir, args.policy_filename)
    scripted.save(policy_path)
    print(f"Saved: {policy_path}")

    test_obs = torch.randn(1, obs_dim, dtype=torch.float32) * 0.5
    with torch.no_grad():
        orig_action = wrapper(test_obs)
        ts_action = scripted(test_obs)
    diff = (orig_action - ts_action).abs().max().item()
    print(f"Verification-1 (wrapper vs TorchScript): max diff = {diff:.8f} (should be ~0)")

    if args.verify_full_pipeline:
        n = args.obs_beams
        raw_lidar = np.random.uniform(0.0, args.max_range, size=(n,)).astype(np.float32)
        raw_goal = np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)
        raw_obs = np.concatenate([raw_lidar, raw_goal], axis=0).astype(np.float32)

        # (1) env_norm (example) -> (2) VecNormalize normalize -> clip -> (3) policy
        obs_env = env_norm_example(raw_obs, args)
        obs_vn = (obs_env - mean.astype(np.float32)) / std.astype(np.float32)
        obs_vn = np.clip(obs_vn, -clip_obs, clip_obs).astype(np.float32)

        obs_t = torch.from_numpy(obs_vn).unsqueeze(0)
        with torch.no_grad():
            a_wrap = wrapper(obs_t)
            a_ts = scripted(obs_t)
        diff2 = (a_wrap - a_ts).abs().max().item()



if __name__ == "__main__":
    main()
