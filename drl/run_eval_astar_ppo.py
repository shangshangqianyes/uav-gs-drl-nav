#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_eval_astar_ppo.py

Offline eval after training (2026-09-12): reuses run_train_astar_ppo.evaluate_model,
with scenarios/seeds/weighting identical to the training-time PeriodicEvalCallback + final eval:
  - obs = 0/5/10 (runs episodes each, seed = base+7777+n_obs)
  - obstacle_density_15/20, narrow_passage, course_l_s (seed = base+7777+hash%10000)
  - sensor_noise (excluded from weighting, seed = base+8888, num_obstacles=2)
  - weighted_score = weighted sum under EVAL_SCORE_WEIGHTS (comparable with historical runs)

Usage (from the drl/ directory):
  PYTHONUTF8=1 PYTHONHASHSEED=0 python run_eval_astar_ppo.py \
      --model ./models_ppo_2d/C2_dt05_v2_noeval/ppo_uav2d_final.zip \
      --vecnorm ./models_ppo_2d/C2_dt05_v2_noeval/vecnormalize.pkl --runs 100
(PYTHONHASHSEED=0 pins hash() so seeds stay identical across invocations)
"""

import argparse

from stable_baselines3 import PPO

from run_train_astar_ppo import (
    EVAL_SCORE_WEIGHTS,
    TrainConfig,
    evaluate_model,
    print_eval_summary,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default="./models_ppo_2d/C2_dt05_v2_noeval/ppo_uav2d_final.zip")
    p.add_argument("--vecnorm", type=str,
                   default="./models_ppo_2d/C2_dt05_v2_noeval/vecnormalize.pkl")
    p.add_argument("--runs", type=int, default=100, help="episodes per scenario (historical runs used 100)")
    p.add_argument("--seed", type=int, default=42, help="align with training seed (42 -> eval seed +7777)")
    args = p.parse_args()

    cfg = TrainConfig()
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
