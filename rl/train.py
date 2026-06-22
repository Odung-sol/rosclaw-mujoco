#!/usr/bin/env python3
"""Train a PPO policy to balance the Segway.

Run (from repo root):
    python -m rl.train --timesteps 300000

Saves:
    rl/models/ppo_segway.zip      trained policy
    rl/models/vecnormalize.pkl    obs-normalization stats (needed at eval)
    rl/models/training_curve.png  episode-reward-vs-timesteps plot
    rl/models/training_curve.csv  the same curve as data

The env is wrapped in VecNormalize (PPO trains far better with normalized,
unbounded observations). A small callback records the running mean episode
reward at each rollout so you can watch learning progress as a curve.
"""

import sys
import csv
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.utils import safe_mean  # noqa: E402

# ── Make the repo root (for `import rl.*`) importable when run as a script ────
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl.segway_env import SegwayBalanceEnv  # noqa: E402

MODELS_DIR = _REPO_ROOT / "rl" / "models"


class RewardCurveCallback(BaseCallback):
    """Record running mean episode reward at each rollout end (for the curve)."""

    def __init__(self):
        super().__init__()
        self.timesteps = []
        self.ep_rew_mean = []

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        buf = self.model.ep_info_buffer
        if buf and len(buf) > 0:
            self.timesteps.append(int(self.num_timesteps))
            self.ep_rew_mean.append(float(safe_mean([e["r"] for e in buf])))


def _save_curve(cb):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = MODELS_DIR / "training_curve.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timesteps", "ep_rew_mean"])
        w.writerows(zip(cb.timesteps, cb.ep_rew_mean))

    plt.figure(figsize=(7, 4))
    plt.plot(cb.timesteps, cb.ep_rew_mean, marker="o", ms=3)
    plt.xlabel("timesteps")
    plt.ylabel("episode reward (running mean)")
    plt.title("PPO Segway balancing — training curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "training_curve.png", dpi=120)
    return csv_path


def main():
    p = argparse.ArgumentParser(description="Train PPO to balance the Segway.")
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episode-steps", type=int, default=1000)
    p.add_argument("--out", default=str(MODELS_DIR / "ppo_segway.zip"))
    p.add_argument("--vecnorm", default=str(MODELS_DIR / "vecnormalize.pkl"))
    args = p.parse_args()

    def make_env():
        return Monitor(SegwayBalanceEnv(max_episode_steps=args.episode_steps))

    venv = DummyVecEnv([make_env])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO("MlpPolicy", venv, seed=args.seed, verbose=1)
    cb = RewardCurveCallback()
    model.learn(total_timesteps=args.timesteps, callback=cb)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    venv.save(args.vecnorm)
    csv_path = _save_curve(cb)

    print(f"\n[done] policy   -> {args.out}")
    print(f"[done] vecnorm  -> {args.vecnorm}")
    print(f"[done] curve    -> {MODELS_DIR / 'training_curve.png'}  (+ {csv_path.name})")
    if cb.ep_rew_mean:
        print(f"[done] ep_rew_mean: {cb.ep_rew_mean[0]:.1f} (start) "
              f"-> {cb.ep_rew_mean[-1]:.1f} (end)")
    venv.close()


if __name__ == "__main__":
    main()
