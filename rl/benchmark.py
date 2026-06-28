#!/usr/bin/env python3
"""Objective evaluation battery for the balancing controllers.

Run (repo root, MuJoCo env):  python -m rl.benchmark

Reports, per controller (RL / fixed-gain LQR / CARE-tuned LQR), the metrics the
self-balancing-robot literature uses to judge a controller:
  - peak |theta|, settling time, control-effort RMS, position drift   (a +2° step)
  - max recoverable initial tilt  (region-of-attraction proxy)
  - max recoverable impulse force (disturbance-rejection limit)

The two "max recoverable" sweeps are the field-standard robustness measure —
e.g. "LQR outperforms PID in maximum initial tilt angle" — and they quantify
how small the RL policy's region of attraction is compared with a tuned LQR.

The sim-driving functions are orchestration; only `max_recoverable` is unit
tested (tests/test_benchmark.py). RL/SB3 are loaded lazily.
"""

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "mujoco_sim")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rl.metrics import compute_episode_metrics  # noqa: E402  (pure NumPy)

TILT_SWEEP_DEG = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
FORCE_SWEEP_N = [1, 5, 10, 20, 40, 80, 120, 160]
RECOVER_DEG = 2.0   # "recovered" = survived and settled below this at the end


def max_recoverable(values, survived):
    """Largest value below the first failure (the region-of-attraction boundary).

    `values` ascending, `survived` the matching booleans. Returns 0.0 if the
    first value already fails. Stops at the first failure, so an isolated later
    survival does not count.
    """
    best = 0.0
    for v, ok in zip(values, survived):
        if not ok:
            break
        best = v
    return best


def _recovered(final_theta_rad, failed):
    return (not failed) and (np.degrees(abs(final_theta_rad)) < RECOVER_DEG)


def _run_tilt(controller, pitch_deg, secs=8.0):
    from segway_sim import SegwaySimulation, SIM_DT

    sim = SegwaySimulation(use_ros2=False, controller=controller)
    sim.reset(pitch_deg=pitch_deg)
    thetas, phis, taus = [], [], []
    for _ in range(int(secs / SIM_DT)):
        s, tL, _ = sim.step()
        thetas.append(s[0])
        phis.append(s[2])
        taus.append(tL)
    final = float(sim.ext.get_theta(sim.data))
    final_x = float(sim.data.qpos[0])   # cart x-position (reset starts at 0)
    failed = sim.failed
    sim.close()
    return thetas, phis, taus, final, final_x, failed, SIM_DT


def _run_impulse(controller, force_N):
    from segway_sim import SegwaySimulation

    sim = SegwaySimulation(use_ros2=False, controller=controller)
    sim.reset(pitch_deg=0.0)
    for _ in range(500):
        sim.step()
    sim.apply_disturbance(force_N=force_N, duration_s=0.3)
    tend = sim.data.time + 0.3
    while sim.data.time < tend + 2.5:
        sim.step()
    final = float(sim.ext.get_theta(sim.data))
    failed = sim.failed
    sim.close()
    return final, failed


def step_metrics(controller, pitch_deg=2.0):
    thetas, phis, taus, _final, final_x, failed, dt = _run_tilt(controller, pitch_deg)
    m = compute_episode_metrics(thetas, phis, taus, dt, fell=failed)
    m["drift_m"] = abs(final_x)
    return m


def roa_tilt_deg(controller):
    survived = []
    for p in TILT_SWEEP_DEG:
        r = _run_tilt(controller, float(p))
        survived.append(_recovered(r[3], r[5]))   # final_theta, failed
    return max_recoverable(TILT_SWEEP_DEG, survived)


def roa_force_N(controller):
    survived = [_recovered(*_run_impulse(controller, float(f))) for f in FORCE_SWEEP_N]
    return max_recoverable(FORCE_SWEEP_N, survived)


def disturbance_response(controller, force_N, dur_s=0.3, rec_s=4.0):
    """theta(t) after a force_N x dur_s push from upright (the paper-standard
    disturbance step-response). Returns (times_s, thetas_deg, failed)."""
    from segway_sim import SegwaySimulation, SIM_DT

    sim = SegwaySimulation(use_ros2=False, controller=controller)
    sim.reset(pitch_deg=0.0)
    for _ in range(500):
        sim.step()
    sim.apply_disturbance(force_N=force_N, duration_s=dur_s)
    t0 = sim.data.time
    times, thetas = [], []
    for _ in range(int(rec_s / SIM_DT)):
        s, _, _ = sim.step()
        times.append(sim.data.time - t0)
        thetas.append(np.degrees(s[0]))
    failed = sim.failed
    sim.close()
    return np.array(times), np.array(thetas), failed


def make_controllers():
    """[(name, controller)] for RL, fixed-gain LQR, CARE-tuned LQR."""
    from segway_sim import SegwaySimulation
    from rl.policy_adapter import RLPolicy
    from rl.lqr_tuning import make_care_lqr

    models = _REPO_ROOT / "rl" / "models"
    m0 = SegwaySimulation(use_ros2=False)
    care = make_care_lqr(m0.model)
    m0.close()
    rl = RLPolicy.from_files(models / "ppo_segway.zip", models / "vecnormalize.pkl")
    return [("RL (PPO)", rl), ("LQR (fixed)", None), ("LQR (CARE)", care)]


def main():
    import os

    os.chdir(_REPO_ROOT / "mujoco_sim")  # segway.xml loads via a cwd-relative path
    print(f"{'controller':<13}{'peak|θ|°':>9}{'settle_s':>9}{'τ_RMS':>8}"
          f"{'drift_m':>9}{'maxTilt°':>9}{'maxForce_N':>11}")
    for name, c in make_controllers():
        m = step_metrics(c, 2.0)
        settle = "inf" if m["settling_time_s"] == float("inf") else f"{m['settling_time_s']:.2f}"
        tilt = roa_tilt_deg(c)
        force = roa_force_N(c)
        print(f"{name:<13}{m['peak_theta_deg']:>9.2f}{settle:>9}{m['tau_rms']:>8.2f}"
              f"{m['drift_m']:>9.3f}{tilt:>9.0f}{force:>11.0f}")
    print("\n(+2° step for the first four columns; maxTilt is region-of-attraction, "
          "cone-limited at 30°)")


if __name__ == "__main__":
    main()
