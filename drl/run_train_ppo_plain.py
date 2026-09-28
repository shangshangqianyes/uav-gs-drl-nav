#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_train_ppo_plain.py

PPO-only ablation variant of run_train_astar_ppo.py. Training logic/hyperparameters/callbacks/eval
are identical to the parent; the only difference is that the env is imported from
nav_env_plain.UAVNav2DEnv (baseline with the A* lookahead hybrid removed). Purpose: compare
hybrid (A*+PPO) vs ppo_only.

SB3 PPO training + automatic periodic/final evaluation + best model saving. Q estimates are
logged as Q=V+A by definition.

Usage example:
  python3 run_train_ppo_plain.py --do_eval --eval_runs 300 --eval_obstacles 0 5 10
"""

import os
import json
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed, FloatSchedule

from nav_env_plain import UAVNav2DEnv
from tb_curve_monitor import CurveMonitorCallback
import torch


# Config
@dataclass
class TrainConfig:
    # PPO
    n_envs: int = 8
    total_timesteps: int = 6_000_000
    learning_rate: float = 1e-4
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.9983  # 0.99^(1/6). Preserves the real-time discount horizon (~30s) at dt=0.05
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.03  # initial ent_coef (the AdaptiveEntropy callback adjusts it dynamically during training)
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Misc
    seed: int = 42
    verbose: int = 1

    # Logging/Save
    log_dir: str = "./logs_ppo_2d"
    save_dir: str = "./models_ppo_2d"
    checkpoint_freq_timesteps: int = 500_000  # in global timesteps


# Utility: evaluation
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
        use_curriculum=False,      # eval: num_obstacles/goal_radius fixed
        curriculum_verbose=False,
    )
    env = UAVNav2DEnv(**env_kwargs)
    return env


def evaluate_model(
    model: PPO,
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

        # periodic stats output (the last episode is excluded -- the caller's summary prints it)
        if print_every > 0 and (ep + 1) % print_every == 0 and (ep + 1) < runs:
            total = ep + 1
            reached_rate = success / total
            collision_rate = collision / total
            no_progress_rate = no_progress / total
            timeout_rate = timeout / total
            # scenario/sensor_noise are independent of the obstacle count -> use a meaningful label
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


# Callbacks
class EntCoefScheduleCallback(BaseCallback):
    """
    Adaptive ent_coef control:
    - base_coef: ent_coef in the normal state
    - boost_coef: raised value when entropy is insufficient
    - low_thr: trigger boost when entropy_loss is higher than this (entropy too low)
    - high_thr: return to base when entropy_loss is lower than this (entropy sufficient)
    - cooldown: hold the state for at least N rollouts after a boost/return (prevents oscillation)
    - ceil_thr/floor_coef: entropy ceiling guard. When entropy_loss < ceil_thr (over-exploration),
      hard-cap ent_coef to floor_coef regardless of boost/cooldown state, blocking std runaway.
    - fixed_ent_coef: when set, ignore the adaptive logic and keep the fixed value (--ent_coef arg)
    """

    def __init__(
        self,
        total_timesteps: int,           # unused (kept for backward compatibility)
        base_coef: float = 0.004,
        boost_coef: float = 0.03,
        low_thr: float = -0.6,
        high_thr: float = -0.9,
        cooldown: int = 10,
        ceil_thr: float = -2.5,
        floor_coef: float = 0.001,
        fixed_ent_coef: Optional[float] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.total_timesteps = int(total_timesteps)
        self.base_coef = base_coef
        self.boost_coef = boost_coef
        self.low_thr = low_thr
        self.high_thr = high_thr
        self.cooldown = cooldown
        self.ceil_thr = ceil_thr
        self.floor_coef = floor_coef
        self.fixed_ent_coef = fixed_ent_coef
        self._boosted = False
        self._capped = False
        self._cooldown_count = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # keep the fixed value when --ent_coef is given
        if self.fixed_ent_coef is not None:
            self.model.ent_coef = self.fixed_ent_coef
            return

        entropy_loss = self.logger.name_to_value.get("train/entropy_loss")
        if entropy_loss is None:
            return  # first rollout has no record yet

        # ceiling guard (highest priority): when std is excessive (entropy_loss < ceil_thr),
        # hard-cap to floor regardless of boost/cooldown. Blocks runaways that even base cannot catch.
        if entropy_loss < self.ceil_thr:
            if not self._capped and self.verbose:
                print(f"[AdaptiveEnt] entropy_loss={entropy_loss:.3f} < ceil {self.ceil_thr}"
                      f" -> ent_coef hard-cap {self.floor_coef} (std runaway guard)")
            self.model.ent_coef = self.floor_coef
            self._capped = True
            self._boosted = False
            return
        if self._capped:
            # guard released (entropy back to normal) -> return to base, then resume toggle logic
            self._capped = False
            self.model.ent_coef = self.base_coef
            self._cooldown_count = self.cooldown
            if self.verbose:
                print(f"[AdaptiveEnt] entropy_loss={entropy_loss:.3f} >= ceil {self.ceil_thr}"
                      f" -> ent_coef restore base {self.base_coef} (guard released)")
            return

        if self._cooldown_count > 0:
            self._cooldown_count -= 1
            return

        if not self._boosted and entropy_loss > self.low_thr:
            # entropy insufficient -> raise ent_coef
            self.model.ent_coef = self.boost_coef
            self._boosted = True
            self._cooldown_count = self.cooldown
            if self.verbose:
                print(f"[AdaptiveEnt] entropy_loss={entropy_loss:.3f} > {self.low_thr}"
                      f" -> ent_coef raise {self.boost_coef} (cooldown={self.cooldown})")
        elif self._boosted and entropy_loss < self.high_thr:
            # entropy recovered -> restore ent_coef
            self.model.ent_coef = self.base_coef
            self._boosted = False
            self._cooldown_count = self.cooldown
            if self.verbose:
                print(f"[AdaptiveEnt] entropy_loss={entropy_loss:.3f} < {self.high_thr}"
                      f" -> ent_coef lower {self.base_coef} (cooldown={self.cooldown})")


class EpisodeStatsCallback(BaseCallback):
    """
    Episode outcome statistics + V/A/Q estimate logging.
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
            # also record to TensorBoard (stdout-only logs cannot be reused beyond parsing)
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

    def _on_rollout_end(self) -> None:
        # V/A/Q estimate logging (Q=V+A)
        try:
            rb = self.model.rollout_buffer
            adv = rb.advantages
            val = rb.values
            self.logger.record("debug/adv_mean", float(np.mean(adv)))
            self.logger.record("debug/v_mean", float(np.mean(val)))
            self.logger.record("debug/q_est_mean", float(np.mean(val + adv)))
        except Exception:
            pass


# best_model weights (sum 1.0): obs + density + narrow + Gazebo Course L (S-course)
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
    save_best uses the EVAL_SCORE_WEIGHTS weighted score (obs + stress + course_l_s).
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

        # weighted score
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


# Final evaluation helper
def _run_final_evaluation(
    model: PPO,
    cfg: TrainConfig,
    args: argparse.Namespace,
    vecnorm_path: str,
) -> None:
    """Full-scenario evaluation after training + best_training saving."""
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

    best_training_path = os.path.join(cfg.save_dir, "ppo_uav2d_best_training.zip")
    meta_path = os.path.join(cfg.save_dir, "ppo_uav2d_best_training_meta.json")
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

    # sensor_noise is not part of the score -> only run additionally with do_eval
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


# Env factory (train)
def make_env(cfg: TrainConfig, rank: int, seed: int):
    def _init():
        env = UAVNav2DEnv(
            seed=seed + rank,
            curriculum_verbose=(rank == 0),  # only rank 0 logs (avoids 8x duplication)
        )
        return Monitor(env)

    return _init


# Main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # train core
    p.add_argument("--total_timesteps", type=int, default=None)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning_rate", type=float, default=None, help="learning rate (lower it for fine-tuning on additional training)")
    p.add_argument("--clip_range", type=float, default=None, help="PPO clip_range (conservative fine-tune mode: 0.1~0.15)")
    p.add_argument("--target_kl", type=float, default=None, help="PPO target_kl, KL early stop (fine-tune: ~0.01)")
    p.add_argument("--n_epochs", type=int, default=None, help="PPO n_epochs (reduce if training becomes unstable)")
    p.add_argument("--verbose", type=int, default=1, help="0=minimal logging, 1=default")
    p.add_argument("--log_interval", type=int, default=10, help="debug/rollout/time/train log period (per iteration, default 10)")

    # eval after train
    p.add_argument("--do_eval", action="store_true", help="run eval automatically after training")
    p.add_argument("--eval_runs", type=int, default=300, help="number of evaluation episodes")
    p.add_argument("--eval_obstacles", type=int, nargs="+", default=[0, 5, 10], help="list of obstacle counts for evaluation")

    # periodic eval during training
    p.add_argument("--eval_freq", type=int, default=0, help="periodic eval interval in timesteps during training (0 disables)")
    p.add_argument("--save_best", action="store_true", help="save best_model from periodic evaluation")
    p.add_argument("--resume_model", type=str, default=None, help="Resume PPO model path")
    p.add_argument("--vecnorm_path", type=str, default=None, help="VecNormalize stats path")
    p.add_argument(
        "--additional_timesteps",
        type=int,
        default=None,
        help="(with resume_model) number of additional steps to train. Continues from the previous run.",
    )
    p.add_argument("--ent_coef", type=float, default=None, help="fixed ent_coef (when set, the adaptive logic is ignored)")
    # Asymmetric base/boost: the hold-phase base is lowered (0.004) to prevent std blow-up at dt=0.05,
    # while the push-phase boost (0.03) is kept for fast convergence.
    p.add_argument("--ent_base_coef", type=float, default=0.004, help="adaptive ent_coef default (entropy sufficient, hold)")
    p.add_argument("--ent_boost_coef", type=float, default=0.03, help="adaptive ent_coef raised value (entropy insufficient, push)")
    p.add_argument("--ent_low_thr", type=float, default=-0.6, help="trigger boost when entropy_loss is higher than this")
    p.add_argument("--ent_high_thr", type=float, default=-0.9, help="return to base when entropy_loss is lower than this")
    p.add_argument("--ent_cooldown", type=int, default=10, help="minimum number of rollouts to hold after boost/return")
    # entropy ceiling guard: when std still blows up at base (over-exploration), hard-cap to floor to block runaway.
    p.add_argument("--ent_ceil_thr", type=float, default=-2.5, help="when entropy_loss is lower than this (~std>0.65, over-exploration), hard-cap ent_coef to floor (ignoring boost/cooldown)")
    p.add_argument("--ent_floor_coef", type=float, default=0.001, help="forced ent_coef when the ceiling guard triggers (blocks std runaway)")

    # output directory override: best_model/best_training/checkpoint/vecnormalize all follow save_dir
    p.add_argument("--log_dir", type=str, default=None, help="TensorBoard log directory (default: ./logs_ppo_2d)")
    p.add_argument("--save_dir", type=str, default=None, help="model/checkpoint save directory (default: ./models_ppo_2d)")

    # Fixed-point-count curve logging (same as run_train_astar_sac.py Sec.32): a separate TB directory writing only episode_reward/episode_length
    p.add_argument("--curve_log_dir", type=str, default=None,
                   help="TB directory for fixed-point-count curves (only the reward/ep_len scalars; disabled if unset)")
    p.add_argument("--curve_points", type=int, default=2000, help="target number of curve points (default 2000, window mode only)")
    p.add_argument("--curve_per_episode", action="store_true",
                   help="per-episode curve logging: 1 point per finished episode (~25-30k points over 10M), no window averaging")

    return p.parse_args()


def main():
    args = parse_args()

    cfg = TrainConfig()
    if args.total_timesteps is not None:
        cfg.total_timesteps = int(args.total_timesteps)
    if args.learning_rate is not None:
        cfg.learning_rate = float(args.learning_rate)
    if args.clip_range is not None:
        cfg.clip_range = float(args.clip_range)
    if args.n_epochs is not None:
        cfg.n_epochs = int(args.n_epochs)
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

    # Vec env (train)
    if cfg.n_envs > 1:
        env = SubprocVecEnv([make_env(cfg, i, cfg.seed) for i in range(cfg.n_envs)])
    else:
        env = DummyVecEnv([make_env(cfg, 0, cfg.seed)])

    if args.vecnorm_path:
        env = VecNormalize.load(args.vecnorm_path, env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    # checkpoint save_freq: with Subproc it is in per-env steps, so divide by n_envs
    if cfg.n_envs > 1:
        save_freq = max(int(cfg.checkpoint_freq_timesteps // cfg.n_envs), 1)
    else:
        save_freq = int(cfg.checkpoint_freq_timesteps)

    # (Optional) dt-proportional ent_coef scaling hook. ent_coef was tuned at dt=0.3, so at smaller dt
    # the per-step advantage (std-decrease) weakens while ent_coef (std-increase) stays fixed -- a std divergence risk.
    # If needed, shrink init/base/boost via _ent_scale (toggle logic unchanged). Currently 1.0 to
    # reproduce the adopted model.
    _ENT_REF_DT = 0.3  # dt reference for ent_coef tuning
    _env_dt = float(UAVNav2DEnv().dt)
    _ent_scale = 1.0  # dt-scale not applied (with sqrt scaling it would be (_env_dt/_ENT_REF_DT)**0.5)
    if abs(_ent_scale - 1.0) > 1e-9:
        cfg.ent_coef *= _ent_scale
        args.ent_base_coef = float(args.ent_base_coef) * _ent_scale
        args.ent_boost_coef = float(args.ent_boost_coef) * _ent_scale
        print(f"[ent dt-scale] dt={_env_dt:g} (ref {_ENT_REF_DT:g}) -> x{_ent_scale:.3f}: "
              f"init={cfg.ent_coef:.5f} base={args.ent_base_coef:.5f} boost={args.ent_boost_coef:.5f}")

    total_ts = int(args.additional_timesteps) if args.additional_timesteps is not None else cfg.total_timesteps
    cb_list: List[BaseCallback] = [
        EntCoefScheduleCallback(
            total_timesteps=total_ts,
            base_coef=float(args.ent_base_coef),
            boost_coef=float(args.ent_boost_coef),
            low_thr=float(args.ent_low_thr),
            high_thr=float(args.ent_high_thr),
            cooldown=int(args.ent_cooldown),
            ceil_thr=float(args.ent_ceil_thr),
            floor_coef=float(args.ent_floor_coef),
            fixed_ent_coef=args.ent_coef,
            verbose=1,
        ),
        EpisodeStatsCallback(print_every_episodes=500, verbose=1),
        CheckpointCallback(
            save_freq=save_freq,
            save_path=cfg.save_dir,
            name_prefix="ppo_uav2d",
            save_replay_buffer=False,
            save_vecnormalize=True,  # store vecnorm per checkpoint -> prevents obs statistics drift when resuming after interruption
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

    # periodic eval callback (optional)
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

    # Asymmetric net_arch: vf wider than pi (pi=[64,64], vf=[256,256,128]) -> improves explained_variance
    policy_kwargs = dict(
        log_std_init=-2.0,
        net_arch=dict(pi=[64, 64], vf=[256, 256, 128]),
    )

    if args.resume_model:
        # SB3 load() adds .zip internally; strip it to avoid "path.zip.zip"
        load_path = args.resume_model.rstrip("/")
        if load_path.lower().endswith(".zip"):
            load_path = load_path[:-4]
        model = PPO.load(load_path, env=env, tensorboard_log=cfg.log_dir, verbose=cfg.verbose)
        print(f"Resumed PPO model from: {args.resume_model}")
        if args.learning_rate is not None:
            new_lr = float(args.learning_rate)
            model.learning_rate = new_lr
            model.lr_schedule = FloatSchedule(new_lr)  # overwrite the loaded lr_schedule
            print(f"Learning rate overridden to {new_lr}")
        if args.clip_range is not None:
            model.clip_range = FloatSchedule(float(args.clip_range))
            print(f"clip_range overridden to {args.clip_range}")
        if args.target_kl is not None:
            model.target_kl = float(args.target_kl)
            print(f"target_kl overridden to {args.target_kl}")
        if args.n_epochs is not None:
            model.n_epochs = int(args.n_epochs)
            print(f"n_epochs overridden to {args.n_epochs}")
        if args.ent_coef is not None:
            model.ent_coef = float(args.ent_coef)
            print(f"ent_coef overridden to {args.ent_coef}")
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            target_kl=args.target_kl if args.target_kl is not None else None,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
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
    cr_val = args.clip_range if args.clip_range is not None else cfg.clip_range
    extra = []
    if args.target_kl is not None:
        extra.append(f"target_kl={args.target_kl}")
    if args.n_epochs is not None:
        extra.append(f"n_epochs={args.n_epochs}")
    extra_str = ", " + ", ".join(extra) if extra else ""
    print(
        f"n_envs={cfg.n_envs}, total_timesteps={total_display}, "
        f"n_steps={cfg.n_steps}, batch_size={cfg.batch_size}, clip_range={cr_val}{extra_str}"
    )
    if args.eval_freq and args.eval_freq > 0:
        print(f"Periodic eval: freq={args.eval_freq} runs={args.eval_runs} obstacles={args.eval_obstacles} save_best={args.save_best}")
    else:
        print("Periodic eval: OFF")

    reset_ts = True  # reset num_timesteps -> progress/eval timing stay correct
    if args.additional_timesteps is not None:
        print(f"Resume mode: training {total_ts} steps (num_timesteps reset)")

    model.learn(
        total_timesteps=total_ts,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=reset_ts,
        log_interval=args.log_interval,
    )

    # save VecNormalize stats and final model
    vecnorm_path = os.path.join(cfg.save_dir, "vecnormalize.pkl")
    env.save(vecnorm_path)
    final_path = os.path.join(cfg.save_dir, "ppo_uav2d_final.zip")
    model.save(final_path)
    print(f"Saved final model to: {final_path}")
    print(f"Saved VecNormalize stats to: {vecnorm_path}")

    if args.do_eval:
        _run_final_evaluation(model, cfg, args, vecnorm_path)
    else:
        print("Final eval skipped (no --do_eval). Final numbers can be produced offline later.")


if __name__ == "__main__":
    main()
