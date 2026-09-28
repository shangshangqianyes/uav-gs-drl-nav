

from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


class CurveMonitorCallback(BaseCallback):
    """
    Every `interval` global steps, write the mean reward/length of episodes finished
    within the window to a separate TensorBoard directory (just two scalars, for
    clean comparison monitoring).

    Args:
        log_dir:         separate TB directory (e.g. ./logs_tb_monitor/ASTAR-SAC)
        total_timesteps: total steps of this learn() call (used to derive the record interval)
        n_points:        target number of points (default 2000, only used in window mode)
        flush_secs:      SummaryWriter flush period in seconds, for real-time monitoring
        per_episode:     when True, log per episode (1 point per finished episode, no averaging)
    """

    def __init__(
        self,
        log_dir: str,
        total_timesteps: int,
        n_points: int = 2000,
        flush_secs: int = 10,
        per_episode: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.log_dir = str(log_dir)
        self.total_timesteps = int(total_timesteps)
        self.n_points = int(n_points)
        self.interval = max(self.total_timesteps // self.n_points, 1)
        self.flush_secs = int(flush_secs)
        self.per_episode = bool(per_episode)

        self._writer: Optional[SummaryWriter] = None
        self._next_record = self.interval
        self._window_rewards: list = []
        self._window_lengths: list = []
        self._last_reward: float = float("nan")
        self._last_length: float = float("nan")
        self._n_written = 0

    def _on_training_start(self) -> None:
        self._writer = SummaryWriter(log_dir=self.log_dir, flush_secs=self.flush_secs)
        if self.per_episode:
            print(
                f"[CurveMonitor] per-episode logging episode_reward/episode_length "
                f"to {self.log_dir} (1 point per finished episode)"
            )
        else:
            print(
                f"[CurveMonitor] logging episode_reward/episode_length to {self.log_dir} "
                f"every {self.interval} steps ({self.n_points} points over {self.total_timesteps})"
            )

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)
        if dones is not None and infos is not None:
            for done, info in zip(dones, infos):
                if not done:
                    continue
                ep = info.get("episode")
                if ep is None:
                    continue
                if self.per_episode:
                    # Per-episode mode: write on episode end, x = current global step count
                    self._writer.add_scalar("episode_reward", float(ep["r"]), self.num_timesteps)
                    self._writer.add_scalar("episode_length", float(ep["l"]), self.num_timesteps)
                else:
                    self._window_rewards.append(float(ep["r"]))
                    self._window_lengths.append(float(ep["l"]))

        if not self.per_episode and self.num_timesteps >= self._next_record:
            self._record(self._next_record)
            self._next_record += self.interval
        return True

    def _record(self, step: int) -> None:
        if self._window_rewards:
            self._last_reward = float(np.mean(self._window_rewards))
            self._last_length = float(np.mean(self._window_lengths))
            self._window_rewards.clear()
            self._window_lengths.clear()
        # Empty window -> reuse the last value (NaN if none yet, leaving a gap in the curve)
        self._writer.add_scalar("episode_reward", self._last_reward, step)
        self._writer.add_scalar("episode_length", self._last_length, step)
        self._n_written += 1

    def _on_training_end(self) -> None:
        # Window mode: append a final record for the tail; per-episode mode needs no
        # catch-up, just close the writer
        if self._writer is not None:
            if not self.per_episode and self._window_rewards and self._n_written < self.n_points:
                self._record(self.total_timesteps)
            self._writer.flush()
            self._writer.close()
            self._writer = None
