#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_fm2_ppo.py

FM2 variant of eval_astar_ppo.py (2026-09-14): scenarios/seeds/weighting are identical
(seed base +7777 / +8888, EVAL_SCORE_WEIGHTS); only the import comes from train_fm2_ppo,
so evaluate_model builds the FM2-guided environment. Directly comparable with A*-PPO's 0.9167.

Usage (from the drl/ directory):
  PYTHONUTF8=1 PYTHONHASHSEED=0 python eval_fm2_ppo.py \
      --model ./models_ppo_2d/FM2_C2_dt05_s42_noeval/ppo_uav2d_final.zip \
      --vecnorm ./models_ppo_2d/FM2_C2_dt05_s42_noeval/vecnormalize.pkl --runs 100
"""

import argparse
import os
import sys

# Eval protocol contract: scenario seeds use the built-in hash(scenario); PYTHONHASHSEED=0
# is required for bit-exact consistency with the historical A*-PPO evaluations. If unset
# at startup, automatically re-exec itself with PYTHONHASHSEED=0.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable, os.path.abspath(__file__), *sys.argv[1:]])

from stable_baselines3 import PPO

from train_fm2_ppo import (
    EVAL_SCORE_WEIGHTS,
    TrainConfig,
    evaluate_model,
    print_eval_summary,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default="./models_ppo_2d/FM2_C2_dt05_s42_noeval/ppo_uav2d_final.zip")
    p.add_argument("--vecnorm", type=str,
                   default="./models_ppo_2d/FM2_C2_dt05_s42_noeval/vecnormalize.pkl")
    p.add_argument("--runs", type=int, default=100, help="episodes per scenario (historical runs used 100)")
    p.add_argument("--seed", type=int, default=42, help="align with training seed (42 -> eval seed +7777)")
    p.add_argument("--fm2_alpha", type=float, default=None,
                   help="FM2 alpha saturation gap (m); eval of an alpha-ablation model must match training, default 1.0")
    # Reward ablation weights (default None = env default; they must be set the same as
    # in training so done_reason/episode-length stats, e.g. no_progress termination, match)
    p.add_argument("--w_progress", type=float, default=None)
    p.add_argument("--success_reward", type=float, default=None)
    p.add_argument("--w_no_progress", type=float, default=None)
    p.add_argument("--w_no_progress_soft", type=float, default=None)
    p.add_argument("--w_timeout", type=float, default=None)
    p.add_argument("--w_danger", type=float, default=None)
    p.add_argument("--w_close", type=float, default=None)
    args = p.parse_args()

    cfg = TrainConfig()
    if args.fm2_alpha is not None:
        cfg.fm2_alpha = float(args.fm2_alpha)
        print(f"fm2_alpha = {cfg.fm2_alpha}")
    for key in ("w_progress", "success_reward", "w_no_progress", "w_no_progress_soft",
                "w_timeout", "w_danger", "w_close"):
        val = getattr(args, key)
        if val is not None:
            setattr(cfg, key, float(val))
            print(f"{key} = {val}")
    model = PPO.load(args.model, device="cpu")
    print(f"Loaded model: {args.model}")
    print(f"VecNormalize: {args.vecnorm}")
    print(f"runs/scenario={args.runs}, deterministic=True")

    eval_seed = args.seed + 7777
    scores = {}

    print("\n=== Offline Evaluation (final model) ===")
    for n_obs in (0, 5, 10):
        m = evaluate_model(
            model=model, cfg=cfg, runs=args.runs, num_obstacles=n_obs,
            seed=eval_seed + n_obs, deterministic=True, vecnorm_path=args.vecnorm,
        )
        scores[f"obs_{n_obs}"] = float(m.success_rate)
        print_eval_summary(f"obs={n_obs}", m)

    print("\n=== Stress Test Evaluation ===")
    for scenario in ("obstacle_density_15", "obstacle_density_20", "narrow_passage", "course_l_s"):
        m = evaluate_model(
            model=model, cfg=cfg, runs=args.runs, num_obstacles=0,
            seed=eval_seed + hash(scenario) % 10000, deterministic=True,
            vecnorm_path=args.vecnorm, eval_scenario=scenario, print_label=scenario,
        )
        scores[scenario] = float(m.success_rate)
        print_eval_summary(scenario, m)

    weighted = sum(EVAL_SCORE_WEIGHTS.get(k, 0.0) * v for k, v in scores.items())
    print(f"\nweighted_score={weighted:.4f} (weights: {EVAL_SCORE_WEIGHTS})")

    m = evaluate_model(
        model=model, cfg=cfg, runs=args.runs, num_obstacles=2,
        seed=args.seed + 8888, deterministic=True, vecnorm_path=args.vecnorm,
        eval_sensor_noise=True, print_label="sensor_noise",
    )
    print_eval_summary("sensor_noise", m)
    print("=== End Evaluation ===")


if __name__ == "__main__":
    main()
