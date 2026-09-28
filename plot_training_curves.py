# -*- coding: utf-8 -*-
"""Plot training-curve comparison of 4 RL algorithms (mean +/- std over 3 seeds).

Data: TensorBoard event files under result/episode_curves/<RUN>/.
Output: result/training_curves_comparison.png (dpi=200)
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

BASE = os.path.dirname(os.path.abspath(__file__))
CURVES = os.path.join(BASE, "result", "episode_curves")

# algorithm -> list of run folders (one per seed)
ALGOS = {
    "PPO":      ["PPO-ONLY-NOEVAL", "PPO-ONLY-NOEVAL-S43", "PPO-ONLY-NOEVAL-S44"],
    "SAC":      ["SAC-ONLY-NOEVAL", "SAC-ONLY-NOEVAL-S43", "SAC-ONLY-NOEVAL-S44"],
    "ASTAR-PPO": ["ASTAR-PPO-NOEVAL", "ASTAR-PPO-NOEVAL-S43", "ASTAR-PPO-NOEVAL-S44"],
    "FM2-PPO":  ["FM2-PPO-S42-NOEVAL", "FM2-PPO-NOEVAL-S43", "FM2-PPO-NOEVAL-S44"],
}

# validated categorical palette (fixed order)
COLORS = {
    "PPO": "#2a78d6",       # blue
    "SAC": "#eb6834",       # orange
    "ASTAR-PPO": "#1baf7a", # aqua
    "FM2-PPO": "#eda100",   # yellow
}

TAG = "episode_reward"
SMOOTH = 200  # moving-average window (episodes)


def load_run(run_dir):
    """Load episode_reward from all event files in a run dir, concatenated by time."""
    files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    values = []
    for f in files:
        ea = EventAccumulator(f)
        ea.Reload()
        if TAG in ea.Tags()["scalars"]:
            values.extend([e.value for e in ea.Scalars(TAG)])
    return np.asarray(values, dtype=float)


def smooth(y, w):
    if len(y) < w:
        w = max(1, len(y) // 10)
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="valid")


def main():
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for name, runs in ALGOS.items():
        smoothed = []
        for r in runs:
            y = load_run(os.path.join(CURVES, r))
            if len(y) == 0:
                print(f"WARNING: no data in {r}")
                continue
            smoothed.append(smooth(y, SMOOTH))
        # align seeds to common episode count
        n = min(len(s) for s in smoothed)
        arr = np.stack([s[:n] for s in smoothed])  # (n_seeds, n_episodes)
        x = np.arange(n)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        color = COLORS[name]
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=2, label=name)
        # direct end-of-line label (relief for low-contrast series)
        ax.annotate(name, xy=(x[-1], mean[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center",
                    fontsize=9, color="#0b0b0b")

    ax.set_xlabel("Episode", fontsize=11, color="#0b0b0b")
    ax.set_ylabel("Episode reward (smoothed)", fontsize=11, color="#0b0b0b")
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    ax.margins(x=0.02)
    # leave room for end labels
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] * 1.12)

    fig.tight_layout()
    out = os.path.join(BASE, "result", "training_curves_comparison.png")
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    print("saved:", out)


if __name__ == "__main__":
    main()
