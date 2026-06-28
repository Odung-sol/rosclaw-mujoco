#!/usr/bin/env python3
"""Generate the README result figures for the RL (PPO) controller.

Run (repo root, MuJoCo env):  python -m rl.plot_results

Produces:
    docs/rl_training_curve.png   ep_rew_mean vs timesteps (from training_curve.csv)
    docs/rl_vs_lqr.png           peak |theta| for RL vs the fixed-gain LQR vs a
                                 CARE-tuned LQR, on a step tilt and a 1N impulse

Reuses the existing training log + trained policy in rl/models/ (no retrain).
The CARE-tuned LQR is included to show that the fixed gains' big step-tilt
overshoot is a *tuning artifact*, not an LQR-vs-RL gap.
"""

import os
import sys
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "mujoco_sim")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from segway_sim import SegwaySimulation  # noqa: E402
from rl.policy_adapter import RLPolicy  # noqa: E402
from rl.lqr_tuning import make_care_lqr  # noqa: E402
from rl.benchmark import (  # noqa: E402
    make_controllers, roa_tilt_deg, roa_force_N, disturbance_response,
)

DOCS = _REPO_ROOT / "docs"
MODELS = _REPO_ROOT / "rl" / "models"
STEP_PITCHES = [1.0, 2.0, 3.0]
COL = {"RL (PPO)": "#185FA5", "LQR (fixed)": "#D85A30", "LQR (CARE-tuned)": "#1D9E75"}
FAIL_COLOR = "#A32D2D"


def plot_training_curve():
    with open(MODELS / "training_curve.csv") as f:
        rows = list(csv.reader(f))[1:]
    ts = [int(r[0]) for r in rows]
    rw = [float(r[1]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, rw, color="#185FA5", lw=2)
    ax.fill_between(ts, rw, min(rw), color="#185FA5", alpha=0.08)
    ax.set_xlabel("timesteps")
    ax.set_ylabel("mean episode reward")
    ax.set_title("PPO Segway balancing — training curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = DOCS / "rl_training_curve.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[curve]   {out}  (reward {rw[0]:.0f} -> {rw[-1]:.0f})")


def _step_peak(controller, pitch, secs=8.0):
    sim = SegwaySimulation(use_ros2=False, controller=controller)
    sim.reset(pitch_deg=pitch)
    peak = 0.0
    for _ in range(int(secs / 0.002)):
        s, _, _ = sim.step()
        peak = max(peak, abs(s[0]))
    failed = sim.failed
    sim.close()
    return np.degrees(peak), failed


def _impulse_peak(controller, force=1.0):
    sim = SegwaySimulation(use_ros2=False, controller=controller)
    sim.reset(pitch_deg=0.0)
    for _ in range(500):
        sim.step()
    sim.apply_disturbance(force_N=force, duration_s=0.3)
    peak = 0.0
    tend = sim.data.time + 0.3
    while sim.data.time < tend + 2.0:
        s, _, _ = sim.step()
        peak = max(peak, abs(s[0]))
    failed = sim.failed
    sim.close()
    return np.degrees(peak), failed


def plot_comparison():
    # segway.xml loads via a cwd-relative path; DOCS/MODELS are absolute.
    os.chdir(_REPO_ROOT / "mujoco_sim")
    m0 = SegwaySimulation(use_ros2=False)
    care = make_care_lqr(m0.model)
    m0.close()
    rl = RLPolicy.from_files(MODELS / "ppo_segway.zip", MODELS / "vecnormalize.pkl")
    controllers = [("RL (PPO)", rl), ("LQR (fixed)", None), ("LQR (CARE-tuned)", care)]
    names = [n for n, _ in controllers]

    step = {n: [] for n in names}
    step_fell = {n: [] for n in names}
    imp = {}
    for name, c in controllers:
        for p in STEP_PITCHES:
            pk, fell = _step_peak(c, p)
            step[name].append(pk)
            step_fell[name].append(fell)
        imp[name], _ = _impulse_peak(c)
        print(f"[data] {name:<18} step={[round(float(v), 1) for v in step[name]]} impulse={imp[name]:.2f}")

    fig, (axs, axi) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [2.2, 1]})

    ceil = 33.0  # clamp fallen bars (peak ~180°) so the chart stays readable
    x = np.arange(len(STEP_PITCHES))
    w = 0.26
    for j, name in enumerate(names):
        offs = (j - 1) * w
        heights = [min(v, ceil) for v in step[name]]
        axs.bar(x + offs, heights, w, label=name, color=COL[name])
        for i, fell in enumerate(step_fell[name]):
            if fell:
                axs.text(x[i] + offs, min(step[name][i], ceil) + 0.5, "fell", ha="center",
                         color=FAIL_COLOR, fontsize=8)
    axs.axhline(30, ls="--", color=FAIL_COLOR, lw=1)
    axs.text(-0.45, 30.5, "fall (30°)", color=FAIL_COLOR, fontsize=9, ha="left", va="bottom")
    axs.set_ylim(0, ceil + 4)
    axs.set_xticks(x)
    axs.set_xticklabels([f"+{p:.0f}°" for p in STEP_PITCHES])
    axs.set_xlabel("released from a step tilt")
    axs.set_ylabel("peak |θ| (deg)")
    axs.set_title("Balancing from an initial tilt")
    axs.legend(fontsize=9)
    axs.grid(True, axis="y", alpha=0.3)

    xi = np.arange(len(names))
    axi.bar(xi, [imp[n] for n in names], 0.6, color=[COL[n] for n in names])
    axi.set_xticks(xi)
    axi.set_xticklabels(["RL", "LQR\nfixed", "LQR\nCARE"], fontsize=9)
    axi.set_ylabel("peak |θ| (deg)")
    axi.set_title("1N × 0.3s impulse (upright)")
    axi.grid(True, axis="y", alpha=0.3)

    fig.suptitle("A CARE-tuned LQR matches RL on tilt; the fixed gains were off-design",
                 fontsize=12)
    fig.tight_layout()
    out = DOCS / "rl_vs_lqr.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[compare] {out}")


def plot_robustness():
    # Region-of-attraction sweeps (the field-standard robustness measure).
    os.chdir(_REPO_ROOT / "mujoco_sim")
    ctrls = make_controllers()
    labels = ["RL\n(PPO)", "LQR\n(fixed)", "LQR\n(CARE-tuned)"]
    colors = ["#185FA5", "#D85A30", "#1D9E75"]
    tilt = [roa_tilt_deg(c) for _, c in ctrls]
    force = [roa_force_N(c) for _, c in ctrls]
    print(f"[roa] max tilt={[round(v) for v in tilt]}°  max force={[round(v) for v in force]}N")

    x = np.arange(len(ctrls))
    fig, (at, af) = plt.subplots(1, 2, figsize=(9, 4))
    at.bar(x, tilt, 0.6, color=colors)
    for i, v in enumerate(tilt):
        cap = "≥30° (sim cone)" if v >= 30 else f"{v:.0f}°"
        at.text(i, v + 0.4, cap, ha="center", fontsize=9)
    at.set_xticks(x)
    at.set_xticklabels(labels)
    at.set_ylabel("max recoverable tilt (deg)")
    at.set_title("Region of attraction (higher = more robust)")
    at.grid(True, axis="y", alpha=0.3)

    af.bar(x, force, 0.6, color=colors)
    for i, v in enumerate(force):
        cap = "≥160 N (test max)" if v >= 160 else f"{v:.0f} N"
        af.text(i, v + 2, cap, ha="center", fontsize=9)
    af.set_xticks(x)
    af.set_xticklabels(labels)
    af.set_ylabel("max recoverable impulse (N · 0.3s)")
    af.set_title("Disturbance rejection (higher = more robust)")
    af.grid(True, axis="y", alpha=0.3)

    fig.suptitle("RL has by far the smallest region of attraction; the CARE-tuned LQR the largest",
                 fontsize=11)
    fig.tight_layout()
    out = DOCS / "rl_robustness.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[robust] {out}")


def plot_disturbance_response():
    # The paper-standard artifact: theta(t) recovery after a defined push.
    os.chdir(_REPO_ROOT / "mujoco_sim")
    ctrls = make_controllers()
    colors = ["#185FA5", "#D85A30", "#1D9E75"]
    panels = [(10.0, "all recover", None), (20.0, "RL falls", (-33, 12))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (force, tag, ylim) in zip(axes, panels):
        ax.axvspan(0, 0.3, color="gray", alpha=0.15)
        for (name, c), col in zip(ctrls, colors):
            t, th, failed = disturbance_response(c, force)
            if failed:
                # truncate at the fail cone so the 180° wrap doesn't draw a spike
                bad = np.where(np.abs(th) >= 30.0)[0]
                if len(bad):
                    t, th = t[: bad[0] + 1], th[: bad[0] + 1]
                lbl = f"{name}: FELL"
            else:
                out = np.where(np.abs(th) >= 0.5)[0]
                settle = t[out[-1]] if len(out) else 0.0
                lbl = f"{name}: peak {float(np.max(np.abs(th))):.1f}°, settle {settle:.1f}s"
            ax.plot(t, th, color=col, lw=1.8, label=lbl)
        if ylim is not None:
            ax.axhline(-30, ls="--", color=FAIL_COLOR, lw=1)
            ax.text(0.05, -30.6, "fall (±30°)", color=FAIL_COLOR, fontsize=8, va="top")
            ax.set_ylim(*ylim)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("time after push (s)")
        ax.set_ylabel("tilt θ (deg)")
        ax.set_title(f"{force:.0f} N × 0.3 s push — {tag}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Disturbance step-response — how each controller recovers from a push", fontsize=12)
    fig.tight_layout()
    out = DOCS / "rl_disturbance_response.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[response] {out}")


def main():
    DOCS.mkdir(exist_ok=True)
    plot_training_curve()
    plot_comparison()
    plot_robustness()
    plot_disturbance_response()


if __name__ == "__main__":
    main()
