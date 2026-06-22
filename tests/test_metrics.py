"""Unit tests for episode metrics (rl/metrics.py).

Pure NumPy — no MuJoCo, no SB3 — so these run on CI. These metrics are what
evaluate.py uses to compare the RL policy against the LQR baseline, so the math
gets locked down here with synthetic episodes.
"""

import numpy as np
import pytest

from rl.metrics import compute_episode_metrics


def test_peak_and_final_theta_in_degrees():
    thetas = [0.1, -0.2, 0.05]  # rad
    m = compute_episode_metrics(thetas, phis=[0, 0, 0], taus=[0, 0, 0], dt=0.01, fell=False)
    assert m["peak_theta_deg"] == pytest.approx(np.degrees(0.2))
    assert m["final_theta_deg"] == pytest.approx(np.degrees(0.05))


def test_tau_rms_is_root_mean_square():
    m = compute_episode_metrics([0, 0], phis=[0, 0], taus=[3.0, -4.0], dt=0.01, fell=False)
    assert m["tau_rms"] == pytest.approx(np.sqrt((9 + 16) / 2))


def test_final_phi_is_last_value():
    m = compute_episode_metrics([0, 0, 0], phis=[0.0, 0.1, 0.5], taus=[0, 0, 0], dt=0.01, fell=False)
    assert m["final_phi"] == pytest.approx(0.5)


def test_survived_reflects_fell_flag():
    kw = dict(phis=[0], taus=[0], dt=0.01)
    assert compute_episode_metrics([0.0], fell=False, **kw)["survived"] is True
    assert compute_episode_metrics([0.0], fell=True, **kw)["survived"] is False


def test_settling_time_is_when_theta_last_left_the_band():
    # |theta| in deg: [5, 2, 0.3, 0.1, 0.2], band = 0.5 deg, dt = 0.1 s.
    # Last step outside the band is index 1 -> settled by t = (1+1)*dt = 0.2 s.
    thetas = np.radians([5.0, 2.0, 0.3, 0.1, 0.2])
    m = compute_episode_metrics(thetas, phis=[0] * 5, taus=[0] * 5, dt=0.1,
                                fell=False, settle_deg=0.5)
    assert m["settling_time_s"] == pytest.approx(0.2)


def test_settling_time_zero_when_already_in_band():
    thetas = np.radians([0.1, 0.2, 0.05])
    m = compute_episode_metrics(thetas, phis=[0] * 3, taus=[0] * 3, dt=0.1,
                                fell=False, settle_deg=0.5)
    assert m["settling_time_s"] == pytest.approx(0.0)


def test_settling_time_inf_when_never_settles():
    thetas = np.radians([5.0, 4.0, 3.0])  # always outside the 0.5 deg band
    m = compute_episode_metrics(thetas, phis=[0] * 3, taus=[0] * 3, dt=0.1,
                                fell=False, settle_deg=0.5)
    assert m["settling_time_s"] == float("inf")
