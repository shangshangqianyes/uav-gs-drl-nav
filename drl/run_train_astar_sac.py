#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_train_astar_sac.py

SB3 SAC training + (optional) periodic/final evaluation + best model saving.
Main training script: the run_train_astar_ppo.py skeleton ported to SAC (max-entropy off-policy).

Differences from the PPO version (run_train_astar_ppo.py, now the baseline):
- EntCoefScheduleCallback removed entirely: replaced by SAC's automatic entropy temperature (target_entropy="auto")
- PPO-only elements removed: n_steps/n_epochs/clip_range/target_kl/gae_lambda/vf_coef/max_grad_norm/log_std_init,
  and the clip_range FloatSchedule override on resume (the lr_schedule override is kept: SAC.train() reads
  lr from self.lr_schedule, so it is needed for the --learning_rate resume override)
- VecNormalize norm_reward=False: off-policy Q learning uses raw rewards
- n_envs default 8 -> 4, SubprocVecEnv -> DummyVecEnv: SAC updates gradients every step, so subprocess
  gains little (Windows spawn cost), and env internals are easier to access for eval/debugging
- EpisodeStatsCallback V/A/Q logging removed (rollout_buffer is PPO-only).
  Instead, SB3 automatically logs SAC actor_loss/critic_loss/ent_coef under train/*

Curriculum check (2026-09-07): the curriculum in nav_env_astar.py can only "promote"
(100-episode window, thresholds 60/70/80/90%, no demotion) and starts at stage 0 (0~3 obstacles).
Random-action collision failures during learning_starts (10k) only keep the curriculum at level 0
and never demote it, so curriculum gating during the random phase is unnecessary (confirmed).

Usage:
  python3 run_train_astar_sac.py --do_eval --eval_runs 300 --eval_obstacles 0 5 10
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

# On non-UTF-8 terminals (e.g. Windows GBK console), keep Korean output from dying with
# UnicodeEncodeError (encoding is kept; only unrepresentable chars become '?' -- UTF-8 terminals unchanged)
for _s in (sys.stdout, sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(errors="replace")
        except Exception:
            pass

import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed, FloatSchedule, update_learning_rate

from nav_env_astar import UAVNav2DEnv
from tb_curve_monitor import CurveMonitorCallback


# --- Config ---
@dataclass
class TrainConfig:
    # SAC
    n_envs: int = 4
    total_timesteps: int = 6_000_000
    learning_rate: float = 1e-4
    buffer_size: int = 300_000
    learning_starts: int = 10_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.9983  # same as PPO (0.99^(1/6)). Preserves real-time discount horizon (~30s) at dt=0.05
    train_freq: int = 1
    gradient_steps: int = 1
    # target_entropy="auto" (SAC default): automatic entropy temperature <- replaces PPO EntCoefSchedule

    # Misc
    seed: int = 42
    verbose: int = 1

    # Logging/Save
    log_dir: str = "./logs_sac_2d"
    save_dir: str = "./models_sac_2d"
    checkpoint_freq_timesteps: int = 500_000  # in global timesteps


# --- Evaluation ---
@dataclass
class EvalMetrics:
    runs: int
    success: int
    collision: int
    no_progress: int
    timeout: int
    avg_steps: float
    avg_distance: float

    @property
    def success_rate(self) -> float:
        return self.success / max(self.runs, 1)

    @property
    def collision_rate(self) -> float:
        return self.collision / max(self.runs, 1)

    @property
    def no_progress_rate(self) -> float:
        return self.no_progress / max(self.runs, 1)

    @property
    def timeout_rate(self) -> float:
        return self.timeout / max(self.runs, 1)


def make_single_env(
    cfg: TrainConfig,
    num_obstacles: int,
    seed: int,
    eval_scenario: Optional[str] = None,
    eval_sensor_noise: bool = False,
) -> UAVNav2DEnv:
    env_kwargs = dict(
        num_obstacles=num_obstacles,
        seed=seed,
        eval_scenario=eval_scenario,
        eval_sensor_noise=eval_sensor_noise,
        use_curriculum=False,      # no curriculum in eval env: num_obstacles/goal_radius fixed
        curriculum_verbose=False,
    )
    env = UAVNav2DEnv(**env_kwargs)
    return env


def evaluate_model(
    model: SAC,
    cfg: TrainConfig,
    runs: int,
    num_obstacles: int,
    seed: int = 0,
    deterministic: bool = True,
    vecnorm_path: Optional[str] = None,
    print_every: int = 0,
    eval_scenario: Optional[str] = None,
    eval_sensor_noise: bool = False,
    print_label: Optional[str] = None,
) -> EvalMetrics:
    base_env = make_single_env(
        cfg,
        num_obstacles=num_obstacles,
        seed=seed,
        eval_scenario=eval_scenario,
        eval_sensor_noise=eval_sensor_noise,
    )
    if vecnorm_path:
        env = DummyVecEnv([lambda: base_env])
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        unwrapped = base_env  # for .pos access
    else:
        env = base_env
        unwrapped = base_env

    success = 0
    collision = 0
    no_progress = 0
    timeout = 0
    steps_list: List[int] = []
    dist_list: List[float] = []

    # VecEnv (DummyVecEnv) auto-resets when episode ends, so calling reset() each episode
    # would double-count: 300 episodes -> 600 resets. Only call reset() for first episode.
    use_vecenv = vecnorm_path is not None

    for ep in range(runs):
        if ep == 0 or not use_vecenv:
            # VecNormalize.reset() does not support seed; VecEnv.reset() may return (obs,) or (obs, info)
            try:
                result = env.reset(seed=seed + ep)
            except TypeError:
                result = env.reset()
            if isinstance(result, tuple) and len(result) >= 2:
                obs, info = result[0], result[1]
            else:
                obs = result
                info = {}

        done = False
        ep_steps = 0
        traveled = 0.0

        prev_pos = unwrapped.pos.copy()

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            step_result = env.step(action)
            # VecEnv returns (obs, rewards, dones, infos); Gymnasium returns (obs, reward, terminated, truncated, info)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = bool(terminated or truncated)
            else:
                obs, rewards, dones, infos = step_result
                done = bool(dones[0])
                info = infos[0] if infos else {}

            ep_steps += 1
            traveled += float(np.linalg.norm(unwrapped.pos - prev_pos))
            prev_pos = unwrapped.pos.copy()

        reason = info.get("done_reason", "")
        if reason == "reached":
            success += 1
        elif reason == "collision":
            collision += 1
        elif reason == "no_progress":
            no_progress += 1
        else:
            timeout += 1

        steps_list.append(ep_steps)
        dist_list.append(traveled)

        # Periodic stats print (last episode excluded: it duplicates the caller summary)
        if print_every > 0 and (ep + 1) % print_every == 0 and (ep + 1) < runs:
            total = ep + 1
            reached_rate = success / total
            collision_rate = collision / total
            no_progress_rate = no_progress / total
            timeout_rate = timeout / total
            if print_label is not None:
                label = print_label
            elif eval_scenario:
                label = eval_scenario
            elif eval_sensor_noise:
                label = "sensor_noise"
            else:
                label = f"obs={num_obstacles}"
            print(
                f"[Eval {label}] episodes={total} "
                f"reached={success} ({reached_rate*100:.2f}%) "
                f"collision={collision} ({collision_rate*100:.2f}%) "
                f"no_progress={no_progress} ({no_progress_rate*100:.2f}%) "
                f"timeout={timeout} ({timeout_rate*100:.2f}%)"
            )

    metrics = EvalMetrics(
        runs=runs,
        success=success,
        collision=collision,
        no_progress=no_progress,
        timeout=timeout,
        avg_steps=float(np.mean(steps_list)) if steps_list else 0.0,
        avg_distance=float(np.mean(dist_list)) if dist_list else 0.0,
    )
    env.close()
    return metrics


def print_eval_summary(scenario: str, m: EvalMetrics, prefix: str = "") -> None:
    print(
        f"{prefix}[Eval {scenario}] "
        f"success={m.success}/{m.runs} ({m.success_rate*100:.2f}%) "
        f"collision={m.collision} ({m.collision_rate*100:.2f}%) "
        f"no_progress={m.no_progress} ({m.no_progress_rate*100:.2f}%) "
        f"timeout={m.timeout} ({m.timeout_rate*100:.2f}%) "
        f"avg_steps={m.avg_steps:.1f} avg_dist={m.avg_distance:.2f}"
    )


# --- Callbacks ---
class EpisodeStatsCallback(BaseCallback):
    """
    Episode outcome stats logging.
    (The PPO version's V/A/Q logging was removed: it depends on rollout_buffer.
    SAC actor/critic loss and ent_coef are automatically logged by SB3 under train/*)
    """

    def __init__(self, print_every_episodes: int, verbose: int = 1):
        super().__init__(verbose=verbose)
        self.print_every_episodes = int(print_every_episodes)
        self.ep_total = 0
        self.ep_success = 0
        self.ep_collision = 0
        self.ep_timeout = 0
        self.ep_no_progress = 0
        self._last_print_total = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)
        if dones is None or infos is None:
            return True

        for done, info in zip(dones, infos):
            if not done:
                continue
            self.ep_total += 1
            reason = info.get("done_reason", "")
            if reason == "reached":
                self.ep_success += 1
            elif reason == "collision":
                self.ep_collision += 1
            elif reason == "no_progress":
                self.ep_no_progress += 1
            else:
                self.ep_timeout += 1  # reached max_steps

        if (self.ep_total - self._last_print_total) >= self.print_every_episodes:
            self._last_print_total = self.ep_total
            total = max(self.ep_total, 1)
            reached_rate = self.ep_success / total
            collision_rate = self.ep_collision / total
            no_progress_rate = self.ep_no_progress / total
            timeout_rate = self.ep_timeout / total
            # Also record to TensorBoard (stdout-only text cannot be reused except by parsing)
            self.logger.record("episode/success_rate", reached_rate)
            self.logger.record("episode/collision_rate", collision_rate)
            self.logger.record("episode/no_progress_rate", no_progress_rate)
            self.logger.record("episode/timeout_rate", timeout_rate)
            self.logger.record("episode/count", self.ep_total)
            if self.verbose:
                print(
                    f"[episodes={self.ep_total}] "
                    f"reached={self.ep_success} ({reached_rate*100:.2f}%) "
                    f"collision={self.ep_collision} ({collision_rate*100:.2f}%) "
                    f"no_progress={self.ep_no_progress} ({no_progress_rate*100:.2f}%) "
                    f"timeout={self.ep_timeout} ({timeout_rate*100:.2f}%)"
                )
        return True


# best_model weights (sum 1.0): obs + density + narrow + Gazebo obstacle_course_light Course L(S)
# (same criteria as the PPO version -- keeps SAC/PPO comparable)
EVAL_SCORE_WEIGHTS = {
    "obs_0": 0.08,
    "obs_5": 0.08,
    "obs_10": 0.08,
    "obstacle_density_15": 0.21,
    "obstacle_density_20": 0.21,
    "narrow_passage": 0.16,
    "course_l_s": 0.18,
}


class PeriodicEvalCallback(BaseCallback):
    """
    Periodic evaluation during training + (optional) best_model saving.

    - eval_freq: in timesteps (global timesteps)
    - eval_runs: number of evaluation episodes
    - save_best: save best_model.zip by weighted score (obs + stress + course_l_s)
      score = EVAL_SCORE_WEIGHTS composite (matches TensorBoard eval/*_success_rate)
    - based on model.predict(), so it is algorithm-agnostic (shared by PPO/SAC)
    """

    def __init__(
        self,
        cfg: TrainConfig,
        eval_freq: int,
        eval_runs: int,
        eval_obstacles: List[int],
        save_path: str,
        save_best: bool = True,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.cfg = cfg
        self.eval_freq = int(eval_freq)
        self.eval_runs = int(eval_runs)
        self.eval_obstacles = [int(x) for x in eval_obstacles]
        self.save_path = save_path
        self.save_best = bool(save_best)
        self.deterministic = bool(deterministic)

        self._last_eval_step = 0
        self.best_score = -np.inf

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True

        if (self.num_timesteps - self._last_eval_step) < self.eval_freq:
            return True

        self._last_eval_step = self.num_timesteps

        vecnorm_path = os.path.join(self.save_path, "vecnormalize.pkl")
        os.makedirs(self.save_path, exist_ok=True)
        self.model.env.save(vecnorm_path)

        seed_base = self.cfg.seed + 10_000 + self.num_timesteps
        scores_dict: Dict[str, float] = {}

        # obs=0, 5, 10
        for n_obs in (0, 5, 10):
            m = evaluate_model(
                model=self.model,
                cfg=self.cfg,
                runs=self.eval_runs,
                num_obstacles=n_obs,
                seed=seed_base + n_obs,
                deterministic=self.deterministic,
                vecnorm_path=vecnorm_path,
                print_every=0,
            )
            key = f"obs_{n_obs}"
            scores_dict[key] = float(m.success_rate)
            if self.verbose:
                print_eval_summary(f"obs={n_obs}", m, prefix=f"[t={self.num_timesteps}] ")
            self.logger.record(f"eval/{key}_success_rate", scores_dict[key])
            self.logger.record(f"eval/{key}_collision_rate", float(m.collision_rate))

        # stress test + Gazebo S course (eval map)
        for scenario in ("obstacle_density_15", "obstacle_density_20", "narrow_passage", "course_l_s"):
            m = evaluate_model(
                model=self.model,
                cfg=self.cfg,
                runs=self.eval_runs,
                num_obstacles=0,
                seed=seed_base + hash(scenario) % 10000,
                deterministic=self.deterministic,
                vecnorm_path=vecnorm_path,
                print_every=0,
                eval_scenario=scenario,
                print_label=scenario,
            )
            scores_dict[scenario] = float(m.success_rate)
            if self.verbose:
                print_eval_summary(scenario, m, prefix=f"[t={self.num_timesteps}] ")
            self.logger.record(f"eval/{scenario}_success_rate", scores_dict[scenario])

        # Weighted score
        weighted_score = sum(
            EVAL_SCORE_WEIGHTS.get(k, 0.0) * v for k, v in scores_dict.items()
        )
        self.logger.record("eval/weighted_score", weighted_score)
        self.logger.record("eval/mean_success_rate", float(np.mean(list(scores_dict.values()))))  # for compatibility

        if self.save_best and weighted_score > self.best_score:
            self.best_score = weighted_score
            best_path = os.path.join(self.save_path, "best_model.zip")
            self.model.save(best_path)
            if self.verbose:
                print(f"[t={self.num_timesteps}] New best weighted_score={weighted_score:.4f} -> saved: {best_path}")

        return True


# --- Final evaluation helper ---
def _run_final_evaluation(
    model: SAC,
    cfg: TrainConfig,
    args: argparse.Namespace,
    vecnorm_path: str,
) -> None:
    """Evaluate all scenarios after training and save best_training."""
    eval_seed = cfg.seed + 7777
    scores_dict: Dict[str, float] = {}

    if args.do_eval:
        print("\n=== Final Evaluation ===")
    for n_obs in (0, 5, 10):
        m = evaluate_model(
            model=model,
            cfg=cfg,
            runs=int(args.eval_runs),
            num_obstacles=int(n_obs),
            seed=eval_seed + int(n_obs),
            deterministic=True,
            vecnorm_path=vecnorm_path,
        )
        scores_dict[f"obs_{n_obs}"] = float(m.success_rate)
        if args.do_eval:
            print_eval_summary(f"obs={n_obs}", m)

    if args.do_eval:
        print("\n=== Stress Test Evaluation ===")
    for scenario in ("obstacle_density_15", "obstacle_density_20", "narrow_passage", "course_l_s"):
        m = evaluate_model(
            model=model,
            cfg=cfg,
            runs=int(args.eval_runs),
            num_obstacles=0,
            seed=eval_seed + hash(scenario) % 10000,
            deterministic=True,
            vecnorm_path=vecnorm_path,
            eval_scenario=scenario,
            print_every=0,
            print_label=scenario,
        )
        scores_dict[scenario] = float(m.success_rate)
        if args.do_eval:
            print_eval_summary(scenario, m)

    weighted_score = sum(EVAL_SCORE_WEIGHTS.get(k, 0.0) * v for k, v in scores_dict.items())

    best_training_path = os.path.join(cfg.save_dir, "sac_uav2d_best_training.zip")
    meta_path = os.path.join(cfg.save_dir, "sac_uav2d_best_training_meta.json")
    prev_best = -np.inf
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                prev_best = float(meta.get("weighted_score", -np.inf))
        except (json.JSONDecodeError, KeyError):
            pass

    if weighted_score > prev_best:
        model.save(best_training_path)
        with open(meta_path, "w") as f:
            json.dump({"weighted_score": weighted_score, "scores": scores_dict}, f, indent=2)
        print(f"New best training result (weighted_score={weighted_score:.4f}) -> saved: {best_training_path}")
    else:
        print(f"Training result (weighted_score={weighted_score:.4f}) <= prev best ({prev_best:.4f}), best_training not updated")

    # sensor_noise is not part of the score, so only run it when do_eval
    if args.do_eval:
        m = evaluate_model(
            model=model,
            cfg=cfg,
            runs=int(args.eval_runs),
            num_obstacles=2,
            seed=cfg.seed + 8888,
            deterministic=True,
            vecnorm_path=vecnorm_path,
            eval_sensor_noise=True,
            print_label="sensor_noise",
        )
        print_eval_summary("sensor_noise", m)
        print("=== End Evaluation ===")


# --- Env factory (train) ---
def make_env(cfg: TrainConfig, rank: int, seed: int):
    def _init():
        env = UAVNav2DEnv(
            seed=seed + rank,
            curriculum_verbose=(rank == 0),  # only rank 0 prints curriculum logs (avoids duplicates)
        )
        return Monitor(env)

    return _init


# --- Main ---
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # train core
    p.add_argument("--total_timesteps", type=int, default=None)
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning_rate", type=float, default=None, help="learning rate (lower it for fine-tuning on resume)")
    p.add_argument("--buffer_size", type=int, default=None, help="replay buffer size")
    p.add_argument("--learning_starts", type=int, default=None, help="initial steps of random actions to fill the buffer")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--tau", type=float, default=None, help="target network soft update coefficient")
    p.add_argument("--train_freq", type=int, default=None, help="update every N steps")
    p.add_argument("--gradient_steps", type=int, default=None, help="gradient steps per update")
    p.add_argument("--verbose", type=int, default=1, help="0=minimal logging, 1=default")
    p.add_argument("--log_interval", type=int, default=10, help="train log print interval")

    # eval after train
    p.add_argument("--do_eval", action="store_true", help="run eval automatically after training")
    p.add_argument("--eval_runs", type=int, default=300, help="number of evaluation episodes")
    p.add_argument("--eval_obstacles", type=int, nargs="+", default=[0, 5, 10], help="list of obstacle counts for evaluation")

    # periodic eval during training
    p.add_argument("--eval_freq", type=int, default=0, help="periodic eval interval in timesteps during training (0 disables)")
    p.add_argument("--save_best", action="store_true", help="save best_model at periodic eval")
    p.add_argument("--resume_model", type=str, default=None, help="Resume SAC model path")
    p.add_argument("--vecnorm_path", type=str, default=None, help="VecNormalize stats path")
    p.add_argument(
        "--additional_timesteps",
        type=int,
        default=None,
        help="(with resume_model) number of extra steps to train. Continues from the previous run.",
    )
    p.add_argument("--replay_buffer", type=str, default=None,
                   help="replay buffer path to load on resume (default: auto-detect <resume_model>_replay_buffer.pkl)")

    # Output directory override. best_model/best_training/checkpoint/vecnormalize are all saved under save_dir
    p.add_argument("--log_dir", type=str, default=None, help="TensorBoard log directory (default: ./logs_sac_2d)")
    p.add_argument("--save_dir", type=str, default=None, help="model/checkpoint save directory (default: ./models_sac_2d)")

    # Fixed-point-count curve logging (run 32): separate TB dir, writes only episode_reward/episode_length scalars
    p.add_argument("--curve_log_dir", type=str, default=None,
                   help="fixed-point curve TB dir (only reward/ep_len scalars; disabled if not set)")
    p.add_argument("--curve_points", type=int, default=2000, help="target curve point count (default 2000, window mode only)")
    p.add_argument("--curve_per_episode", action="store_true",
                   help="per-episode curve logging: write 1 point per finished episode (~25-30k points for 10M), no window averaging")

    return p.parse_args()


def main():
    args = parse_args()

    cfg = TrainConfig()
    if args.total_timesteps is not None:
        cfg.total_timesteps = int(args.total_timesteps)
    if args.learning_rate is not None:
        cfg.learning_rate = float(args.learning_rate)
    if args.buffer_size is not None:
        cfg.buffer_size = int(args.buffer_size)
    if args.learning_starts is not None:
        cfg.learning_starts = int(args.learning_starts)
    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.tau is not None:
        cfg.tau = float(args.tau)
    if args.train_freq is not None:
        cfg.train_freq = int(args.train_freq)
    if args.gradient_steps is not None:
        cfg.gradient_steps = int(args.gradient_steps)
    cfg.n_envs = int(args.n_envs)
    cfg.seed = int(args.seed)

    if args.log_dir is not None:
        cfg.log_dir = str(args.log_dir)
    if args.save_dir is not None:
        cfg.save_dir = str(args.save_dir)

    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.save_dir, exist_ok=True)

    set_random_seed(cfg.seed)

    if args.additional_timesteps is not None and not args.resume_model:
        raise ValueError("--additional_timesteps requires --resume_model")

    if args.resume_model and not args.vecnorm_path:
        args.vecnorm_path = os.path.join(cfg.save_dir, "vecnormalize.pkl")
        # Resuming without training stats silently shifts the obs scale -- failing loudly is safer
        if not os.path.isfile(args.vecnorm_path):
            raise FileNotFoundError(
                f"VecNormalize stats not found: {args.vecnorm_path}\n"
                f"Resume requires training-time obs stats. Options:\n"
                f"  1) pass --vecnorm_path explicitly (e.g. a checkpoint's "
                f"sac_uav2d_<N>_steps_vecnormalize.pkl, or vecnormalize.pkl from a finished run),\n"
                f"  2) retrain from scratch."
            )

    # SAC updates gradients every step -> use DummyVecEnv (subprocess gains little, Windows spawn cost)
    env = DummyVecEnv([make_env(cfg, i, cfg.seed) for i in range(cfg.n_envs)])

    if args.vecnorm_path:
        env = VecNormalize.load(args.vecnorm_path, env)
        env.training = True
        env.norm_reward = False  # SAC: learn Q on raw rewards
    else:
        # off-policy standard: normalize obs only, no reward normalization
        env = VecNormalize(env, norm_obs=True, norm_reward=False)

    # Divide checkpoint save_freq because VecEnv counts steps differently
    if cfg.n_envs > 1:
        save_freq = max(int(cfg.checkpoint_freq_timesteps // cfg.n_envs), 1)
    else:
        save_freq = int(cfg.checkpoint_freq_timesteps)

    total_ts = int(args.additional_timesteps) if args.additional_timesteps is not None else cfg.total_timesteps
    cb_list: List[BaseCallback] = [
        EpisodeStatsCallback(print_every_episodes=500, verbose=1),
        CheckpointCallback(
            save_freq=save_freq,
            save_path=cfg.save_dir,
            name_prefix="sac_uav2d",
            # off-policy: also save the replay buffer with each checkpoint -> no experience loss on resume
            # (300kx66d buffer ~ 170MB each; clean up old checkpoints periodically)
            save_replay_buffer=True,
            save_vecnormalize=True,  # stats for resuming an interrupted eval_freq=0 run (..._steps_vecnormalize.pkl)
        ),
    ]

    if args.curve_log_dir:
        cb_list.append(
            CurveMonitorCallback(
                log_dir=args.curve_log_dir,
                total_timesteps=total_ts,
                n_points=int(args.curve_points),
                per_episode=bool(args.curve_per_episode),
            )
        )

    if args.eval_freq and args.eval_freq > 0:
        cb_list.append(
            PeriodicEvalCallback(
                cfg=cfg,
                eval_freq=int(args.eval_freq),
                eval_runs=int(args.eval_runs),
                eval_obstacles=[int(x) for x in args.eval_obstacles],
                save_path=cfg.save_dir,
                save_best=bool(args.save_best),
                deterministic=True,
                verbose=1,
            )
        )

    callbacks = CallbackList(cb_list)

    # Asymmetric net_arch: critic (qf=[256,256,128]) wider than policy (pi=[64,64])
    # -> Q-estimate precision (carries the PPO version's vf asymmetry idea over to SAC qf)
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[256, 256, 128]))

    if args.resume_model:
        # SB3 load() adds .zip internally; strip it to avoid "path.zip.zip"
        load_path = args.resume_model.rstrip("/")
        if load_path.lower().endswith(".zip"):
            load_path = load_path[:-4]
        model = SAC.load(load_path, env=env, tensorboard_log=cfg.log_dir, verbose=cfg.verbose)
        print(f"Resumed SAC model from: {args.resume_model}")
        if args.learning_rate is not None:
            new_lr = float(args.learning_rate)
            model.learning_rate = new_lr
            model.lr_schedule = FloatSchedule(new_lr)
            for optimizer in (
                model.actor.optimizer,
                model.critic.optimizer,
                getattr(model, "ent_coef_optimizer", None),
            ):
                if optimizer is not None:
                    # SB3 2.7.1: update_learning_rate(optimizer, learning_rate) -- learning_rate is required
                    update_learning_rate(optimizer, new_lr)
            print(f"Learning rate overridden to {new_lr}")
        # replay buffer: explicit path > auto-detect <resume_model stem>_replay_buffer.pkl
        rb_path = args.replay_buffer
        if rb_path is None:
            rb_path = load_path + "_replay_buffer.pkl"
        if os.path.isfile(rb_path):
            model.load_replay_buffer(rb_path)
            # With an inherited buffer the initial random phase is unneeded -> resume learning immediately
            model.learning_starts = 0
            print(f"Loaded replay buffer from: {rb_path} ({model.replay_buffer.size()} transitions), learning_starts=0")
        else:
            print(f"Replay buffer not found at: {rb_path} (starting with empty buffer, learning_starts kept)")
        if args.learning_starts is not None:
            model.learning_starts = int(args.learning_starts)
            print(f"learning_starts overridden to {model.learning_starts}")
    else:
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=cfg.learning_rate,
            buffer_size=cfg.buffer_size,
            learning_starts=cfg.learning_starts,
            batch_size=cfg.batch_size,
            tau=cfg.tau,
            gamma=cfg.gamma,
            train_freq=cfg.train_freq,
            gradient_steps=cfg.gradient_steps,
            target_entropy="auto",  # automatic replacement for the PPO EntCoefScheduleCallback
            verbose=cfg.verbose,
            tensorboard_log=cfg.log_dir,
            seed=cfg.seed,
            policy_kwargs=policy_kwargs,
        )

    print("=== Training start ===")
    total_display = (
        args.additional_timesteps
        if args.additional_timesteps is not None
        else cfg.total_timesteps
    )
    print(
        f"n_envs={cfg.n_envs}, total_timesteps={total_display}, "
        f"buffer_size={cfg.buffer_size}, batch_size={cfg.batch_size}, "
        f"learning_starts={cfg.learning_starts}, tau={cfg.tau}, gamma={cfg.gamma}"
    )
    if args.eval_freq and args.eval_freq > 0:
        print(f"Periodic eval: freq={args.eval_freq} runs={args.eval_runs} obstacles={args.eval_obstacles} save_best={args.save_best}")
    else:
        print("Periodic eval: OFF")

    reset_ts = True  # reset num_timesteps -> progress/eval timing stays correct
    if args.additional_timesteps is not None:
        print(f"Resume mode: training {total_ts} steps (num_timesteps reset)")

    model.learn(
        total_timesteps=total_ts,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=reset_ts,
        log_interval=args.log_interval,
    )

    vecnorm_path = os.path.join(cfg.save_dir, "vecnormalize.pkl")
    env.save(vecnorm_path)
    final_path = os.path.join(cfg.save_dir, "sac_uav2d_final.zip")
    model.save(final_path)
    # off-policy: also save the replay buffer for resume support (300k x 66d float32 ~ 170MB)
    replay_buffer_path = os.path.join(cfg.save_dir, "sac_uav2d_final_replay_buffer.pkl")
    model.save_replay_buffer(replay_buffer_path)
    print(f"Saved final model to: {final_path}")
    print(f"Saved replay buffer to: {replay_buffer_path}")
    print(f"Saved VecNormalize stats to: {vecnorm_path}")

    if args.do_eval:
        _run_final_evaluation(model, cfg, args, vecnorm_path)
    else:
        print("Final eval skipped (no --do_eval). Final numbers can be produced offline later.")


if __name__ == "__main__":
    main()
