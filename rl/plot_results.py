#!/usr/bin/env python3
"""Generate the README result figures for the RL (PPO) controller.

Run (from repo root, in the MuJoCo env):  python -m rl.plot_results

Produces (committed, for the README):
    docs/rl_training_curve.png   ep_rew_mean vs timesteps (from training_curve.csv)
    docs/rl_vs_lqr.png           RL vs LQR: peak |theta| + settling across pitches

Reuses the existing training log and trained policy in rl/models/ — no retrain.
"""

import sys
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "mujoco_sim")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from segway_sim import SIM_DT  # noqa: E402
from lqr_controller import SegwayLQR  # noqa: E402
from rl.segway_env import SegwayBalanceEnv  # noqa: E402
from rl.metrics import compute_episode_metrics  # noqa: E402
from rl.evaluate import run_episode, make_lqr_action_fn, load_policy  # noqa: E402

DOCS = _REPO_ROOT / "docs"
MODELS = _REPO_ROOT / "rl" / "models"
PITCHES = [1.0, 2.0, 3.0]
RL_COLOR = "#185FA5"
LQR_COLOR = "#D85A30"
FAIL_COLOR = "#A32D2D"


def plot_training_curve():
    with open(MODELS / "training_curve.csv") as f:
        rows = list(csv.reader(f))[1:]
    ts = [int(r[0]) for r in rows]
    rw = [float(r[1]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, rw, color=RL_COLOR, lw=2)
    ax.fill_between(ts, rw, min(rw), color=RL_COLOR, alpha=0.08)
    ax.set_xlabel("timesteps")
    ax.set_ylabel("mean episode reward")
    ax.set_title("PPO Segway balancing — training curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = DOCS / "rl_training_curve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[curve]   {out}  ({len(ts)} pts, reward {rw[0]:.0f} -> {rw[-1]:.0f})")


def _collect_comparison():
    env = SegwayBalanceEnv(max_episode_steps=1000)
    dt = SIM_DT * env.frame_skip
    rl_fn, venv = load_policy(MODELS / "ppo_segway.zip", MODELS / "vecnormalize.pkl")
    lqr_fn = make_lqr_action_fn(SegwayLQR(torque_limit=env.max_torque), env.max_torque)

    res = {n: {"peak": [], "settle": [], "fell": []} for n in ("RL", "LQR")}
    for pitch in PITCHES:
        for name, fn in (("RL", rl_fn), ("LQR", lqr_fn)):
            th, ph, ta, fell = run_episode(env, fn, pitch, 1000)
            m = compute_episode_metrics(th, ph, ta, dt, fell=fell)
            res[name]["peak"].append(m["peak_theta_deg"])
            res[name]["settle"].append(m["settling_time_s"])
            res[name]["fell"].append(not m["survived"])
    env.close()
    venv.close()
    return res


def plot_comparison():
    res = _collect_comparison()
    x = np.arange(len(PITCHES))
    w = 0.38
    labels = [f"+{p:.0f}°" for p in PITCHES]
    fig, (axp, axs) = plt.subplots(1, 2, figsize=(10, 4))

    # peak |theta|
    axp.bar(x - w / 2, res["RL"]["peak"], w, label="RL (PPO)", color=RL_COLOR)
    axp.bar(x + w / 2, res["LQR"]["peak"], w, label="LQR", color=LQR_COLOR)
    axp.axhline(30, ls="--", color=FAIL_COLOR, lw=1)
    axp.text(-0.45, 30.6, "fall (30°)", color=FAIL_COLOR, fontsize=9, ha="left", va="bottom")
    for i, fell in enumerate(res["RL"]["fell"]):
        if fell:
            axp.text(i - w / 2, res["RL"]["peak"][i] + 0.7, "fell", ha="center",
                     color=FAIL_COLOR, fontsize=9)
    axp.set_xticks(x)
    axp.set_xticklabels(labels)
    axp.set_xlabel("initial pitch")
    axp.set_ylabel("peak |θ| (deg)")
    axp.set_title("Peak tilt (lower is better)")
    axp.legend()
    axp.grid(True, axis="y", alpha=0.3)

    # settling time (inf -> no bar, annotate)
    rl_settle = [v if np.isfinite(v) else 0.0 for v in res["RL"]["settle"]]
    lqr_settle = [v if np.isfinite(v) else 0.0 for v in res["LQR"]["settle"]]
    axs.bar(x - w / 2, rl_settle, w, label="RL (PPO)", color=RL_COLOR)
    axs.bar(x + w / 2, lqr_settle, w, label="LQR", color=LQR_COLOR)
    for i, v in enumerate(res["RL"]["settle"]):
        if not np.isfinite(v):
            axs.text(i - w / 2, 0.3, "n/a\n(fell)", ha="center", color=FAIL_COLOR, fontsize=8)
    axs.set_xticks(x)
    axs.set_xticklabels(labels)
    axs.set_xlabel("initial pitch")
    axs.set_ylabel("settling time to |θ|<0.5° (s)")
    axs.set_title("Settling time (lower is better)")
    axs.legend()
    axs.grid(True, axis="y", alpha=0.3)

    fig.suptitle("RL (PPO) vs LQR — balancing from an initial tilt")
    fig.tight_layout()
    out = DOCS / "rl_vs_lqr.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[compare] {out}")
    for name in ("RL", "LQR"):
        peaks = [round(v, 1) for v in res[name]["peak"]]
        settle = [round(v, 2) if np.isfinite(v) else "inf" for v in res[name]["settle"]]
        print(f"          {name}: peak={peaks}  settle={settle}")


def main():
    DOCS.mkdir(exist_ok=True)
    plot_training_curve()
    plot_comparison()


if __name__ == "__main__":
    main()
