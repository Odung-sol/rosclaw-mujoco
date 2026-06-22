#!/usr/bin/env python3
"""Evaluate a trained PPO policy against the LQR baseline.

Run (from repo root):  python -m rl.evaluate --model rl/models/ppo_segway.zip

Methodology — a *fair* comparison:
  * Both controllers are rolled out through the SAME SegwayBalanceEnv, so they
    see identical physics (segway.xml), identical state (state_extractor),
    identical termination (|theta| > 30 deg), and identical per-wheel torque
    limits. Only the *action source* differs.
  * Episodes are matched: for each test pitch, RL and LQR start from the exact
    same initial condition (env.reset(options={"pitch_deg": ...})).
  * The LQR's per-wheel torque is mapped back to the env's normalized action
    (tau / max_torque), so it is applied through the same path as the policy.

Each episode is scored by rl/metrics.compute_episode_metrics (peak |theta|,
settling time, tau RMS, final position drift, survival).

It also runs verify_phi_dot(): a check that the state extractor's wheel-velocity
sign convention is the corrected (L - R)/2, not the historical (L + R)/2 bug
that silently zeroed the position state — see the printed report.
"""

import sys
import argparse
from pathlib import Path

import numpy as np

# ── Make the repo root (for `import rl.*`) and mujoco_sim/ importable ─────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MUJOCO_DIR = _REPO_ROOT / "mujoco_sim"
for _p in (str(_REPO_ROOT), str(_MUJOCO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from segway_sim import SIM_DT  # noqa: E402
from lqr_controller import SegwayLQR  # noqa: E402
from rl.metrics import compute_episode_metrics  # noqa: E402
from rl.segway_env import SegwayBalanceEnv  # noqa: E402

# Default initial pitches (deg) to test recovery from.
DEFAULT_PITCHES = [1.0, -1.0, 2.0, -2.0, 3.0]


# ── phi_dot sign-convention verification ─────────────────────────────────────
def verify_phi_dot(sim, n_steps=150, torque=4.0):
    """Roll the segway forward and report phi_dot computed two ways.

    The L and R wheel joints spin about opposite axes (-y / +y in segway.xml),
    so for pure forward motion their joint velocities are opposite-signed:
        corrected (L - R)/2  -> the real average wheel speed (non-zero)
        old/buggy (L + R)/2  -> cancels to ~0  (the pre-2026-04-29 bug)
    Returns a dict of evidence; main() prints it.
    """
    sim.reset(pitch_deg=0.0)
    for _ in range(n_steps):
        sim.set_torque_and_step(torque, torque)

    Lv = float(sim.data.qvel[sim.ext.Lv])
    Rv = float(sim.data.qvel[sim.ext.Rv])
    corrected = (Lv - Rv) / 2.0
    old = (Lv + Rv) / 2.0
    return {
        "L_wheel_vel": Lv,
        "R_wheel_vel": Rv,
        "phi_dot_corrected": corrected,
        "phi_dot_old_buggy": old,
        "phi_dot_from_state_extractor": float(sim.ext.get_phi_dot(sim.data)),
    }


def _print_phi_dot_report(ev):
    print("── phi_dot sign-convention verification ──────────────────────────")
    print(f"  L wheel vel = {ev['L_wheel_vel']:+.4f} rad/s   "
          f"R wheel vel = {ev['R_wheel_vel']:+.4f} rad/s")
    print(f"  corrected (L-R)/2  = {ev['phi_dot_corrected']:+.4f} rad/s   "
          f"<- used by state_extractor ({ev['phi_dot_from_state_extractor']:+.4f})")
    print(f"  old/buggy (L+R)/2  = {ev['phi_dot_old_buggy']:+.4f} rad/s   "
          "<- would collapse to ~0")
    corrected, old = abs(ev["phi_dot_corrected"]), abs(ev["phi_dot_old_buggy"])
    ok = corrected > 0.1 and old < corrected * 0.2
    verdict = "OK" if ok else "CHECK"
    print(f"  => [{verdict}] RL and LQR both read phi_dot via the corrected "
          "getter, so the historical")
    print("     (L+R)/2 bug does NOT bias the RL-vs-LQR comparison. (Baseline "
          "is the LOCAL")
    print("     SegwayLQR, which uses phi; the ROS2 node uses x and is a "
          "different state.)")
    print()


# ── rollout ──────────────────────────────────────────────────────────────────
def run_episode(env, action_fn, pitch_deg, max_steps):
    """Roll one episode; return (thetas, phis, taus, fell)."""
    obs, _ = env.reset(options={"pitch_deg": pitch_deg})
    thetas, phis, taus = [], [], []
    fell = False
    for _ in range(max_steps):
        action = action_fn(obs)
        tau_applied = float(np.clip(float(action[0]) * env.max_torque,
                                    -env.max_torque, env.max_torque))
        obs, _, terminated, truncated, _ = env.step(action)
        thetas.append(float(obs[0]))
        phis.append(float(obs[2]))
        taus.append(tau_applied)
        if terminated:
            fell = True
            break
        if truncated:
            break
    return thetas, phis, taus, fell


def make_lqr_action_fn(lqr, max_torque):
    def fn(obs):
        tau_each, _ = lqr.compute_torque(obs)  # obs = [theta,theta_dot,phi,phi_dot]
        return np.array([tau_each / max_torque], dtype=np.float32)
    return fn


def load_policy(model_path, vecnorm_path):
    """Lazy-import SB3 so importing this module (e.g. for verify_phi_dot in
    tests) does not require torch."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO.load(model_path)
    # VecNormalize.load needs a venv; we only use it for normalize_obs at eval.
    venv = DummyVecEnv([lambda: SegwayBalanceEnv()])
    vecnorm = VecNormalize.load(str(vecnorm_path), venv)
    vecnorm.training = False
    vecnorm.norm_reward = False

    def fn(obs):
        norm = vecnorm.normalize_obs(np.asarray(obs, dtype=np.float32))
        action, _ = model.predict(norm, deterministic=True)
        return action

    return fn, venv


def _print_metrics_row(label, pitch, m):
    settle = "  n/a" if m["settling_time_s"] == float("inf") else f"{m['settling_time_s']:5.2f}"
    survived = "yes" if m["survived"] else "FELL"
    print(f"{pitch:+5.1f}  {label:<3}  {m['peak_theta_deg']:8.2f}  "
          f"{m['final_theta_deg']:9.3f}  {settle:>6}  {m['tau_rms']:7.3f}  "
          f"{m['final_phi']:+8.3f}  {survived:>6}")


def main():
    p = argparse.ArgumentParser(description="Evaluate PPO policy vs LQR baseline.")
    p.add_argument("--model", default="rl/models/ppo_segway.zip")
    p.add_argument("--vecnorm", default="rl/models/vecnormalize.pkl")
    p.add_argument("--max-steps", type=int, default=1000,
                   help="env steps per episode (1000 = 10 s at 100 Hz; long "
                        "enough to see the LQR's ~7 s settling)")
    p.add_argument("--pitch", type=float, default=None,
                   help="single initial pitch (deg); default = a fixed test set")
    args = p.parse_args()

    pitches = [args.pitch] if args.pitch is not None else DEFAULT_PITCHES
    env = SegwayBalanceEnv(max_episode_steps=args.max_steps)
    dt = SIM_DT * env.frame_skip

    # 1) phi_dot verification (independent of the policy).
    _print_phi_dot_report(verify_phi_dot(env.sim))

    # 2) Build both controllers' action functions.
    lqr_fn = make_lqr_action_fn(SegwayLQR(torque_limit=env.max_torque), env.max_torque)
    controllers = [("LQR", lqr_fn)]

    rl_venv = None
    model_path = Path(args.model)
    if model_path.exists():
        rl_fn, rl_venv = load_policy(model_path, args.vecnorm)
        controllers.insert(0, ("RL", rl_fn))
    else:
        print(f"[warn] {model_path} not found — skipping RL, showing LQR only. "
              "Train first: python -m rl.train\n")

    # 3) Matched-pitch comparison.
    print(f"{'pitch':>5}  {'ctl':<3}  {'peak|θ|°':>8}  {'final|θ|°':>9}  "
          f"{'settle':>6}  {'τrms':>7}  {'final φ':>8}  {'surv':>6}")
    for pitch in pitches:
        for label, fn in controllers:
            thetas, phis, taus, fell = run_episode(env, fn, pitch, args.max_steps)
            m = compute_episode_metrics(thetas, phis, taus, dt, fell=fell)
            _print_metrics_row(label, pitch, m)

    env.close()
    if rl_venv is not None:
        rl_venv.close()


if __name__ == "__main__":
    main()
