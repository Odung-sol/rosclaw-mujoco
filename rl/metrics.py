#!/usr/bin/env python3
"""Episode metrics for comparing controllers (RL policy vs LQR baseline).

Pure NumPy: takes the per-step logs of one episode and returns a small summary
dict. evaluate.py runs each controller, collects these logs, and feeds them
here so RL and LQR are scored by exactly the same yardstick.
"""

import numpy as np


def compute_episode_metrics(thetas, phis, taus, dt, *, fell, settle_deg=0.5):
    """Summarize one episode.

    Args:
        thetas: per-step body pitch (rad).
        phis:   per-step wheel angle / position proxy (rad).
        taus:   per-step per-wheel torque (N·m).
        dt:     control timestep in seconds (one env step).
        fell:   True if the episode ended by falling (|theta| > fail cone).
        settle_deg: |theta| band (degrees) used for settling time.

    Returns:
        dict with peak_theta_deg, final_theta_deg, settling_time_s, tau_rms,
        final_phi, n_steps, survived.
    """
    thetas = np.asarray(thetas, dtype=float)
    taus = np.asarray(taus, dtype=float)
    theta_deg = np.abs(np.degrees(thetas))

    # Settling time: the moment after which |theta| stays inside the band for
    # the rest of the episode. 0 if it never left the band; inf if it is still
    # outside at the final step (never settled).
    outside = np.where(theta_deg >= settle_deg)[0]
    if len(outside) == 0:
        settling_time_s = 0.0
    elif outside[-1] == len(thetas) - 1:
        settling_time_s = float("inf")
    else:
        settling_time_s = float((outside[-1] + 1) * dt)

    return {
        "peak_theta_deg": float(np.max(theta_deg)),
        "final_theta_deg": float(theta_deg[-1]),
        "settling_time_s": settling_time_s,
        "tau_rms": float(np.sqrt(np.mean(taus**2))),
        "final_phi": float(phis[-1]),
        "n_steps": int(len(thetas)),
        "survived": not bool(fell),
    }
