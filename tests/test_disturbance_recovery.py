"""
Disturbance-recovery tests for SegwaySimulation.

Verifies that the LQR balancing controller recovers from an external impulse
applied at the top of the segway body. The canonical case is the Issue #4
spec: 1 N pushed horizontally at the body's top point for 0.3 s.

These tests run the *local* LQR (no Docker). The ROS2 path is exercised
manually — see PR description for that smoke test.

NB. We import segway_sim from mujoco_sim/. The module mutates cwd-relative
paths (loads segway.xml as a relative path), so each test cd's into
mujoco_sim/ and restores afterwards.
"""

import os
import sys
import contextlib
from pathlib import Path

import numpy as np
import pytest


# ── Locate the mujoco_sim/ source dir and make it importable ──────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_DIR = REPO_ROOT / "mujoco_sim"
sys.path.insert(0, str(MUJOCO_DIR))

# Skip the whole module if mujoco isn't installed (CI Linux runner doesn't
# have it). Local macOS dev environment does.
mujoco = pytest.importorskip("mujoco")


@contextlib.contextmanager
def _cd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


@pytest.fixture
def sim():
    """Build a SegwaySimulation instance from inside mujoco_sim/.

    The XML loader uses a cwd-relative path so we have to enter the
    directory before constructing the model.
    """
    from segway_sim import SegwaySimulation

    with _cd(MUJOCO_DIR):
        s = SegwaySimulation(use_ros2=False)
        try:
            yield s
        finally:
            s.close()


# ── Helpers ───────────────────────────────────────────────────────────────
def _settle(sim, n_steps=500):
    """Drive the sim to steady-state from upright. Returns final theta (rad)."""
    sim.reset(pitch_deg=0.0)
    for _ in range(n_steps):
        sim.step()
    return float(sim.ext.get_theta(sim.data))


# ── Tests ─────────────────────────────────────────────────────────────────
class TestDisturbanceAPI:
    """Surface-level checks on the apply_disturbance() entry point."""

    def test_apply_disturbance_method_exists(self, sim):
        assert hasattr(sim, "apply_disturbance"), (
            "SegwaySimulation must expose apply_disturbance(force_N, duration_s)"
        )

    def test_apply_disturbance_records_pending(self, sim):
        sim.reset(pitch_deg=0.0)
        sim.apply_disturbance(force_N=1.0, duration_s=0.3)
        # Implementation detail: a `pending_disturbance` attribute is exposed
        # for testability and debug logging. If you change the field name,
        # also update this test.
        assert sim.pending_disturbance is not None

    def test_apply_disturbance_clears_after_duration(self, sim):
        sim.reset(pitch_deg=0.0)
        sim.apply_disturbance(force_N=1.0, duration_s=0.3)
        # Run past the disturbance window. Use sim.data.time as the clock.
        target_t = sim.data.time + 0.31
        while sim.data.time < target_t:
            sim.step()
        assert sim.pending_disturbance is None, (
            "Disturbance should auto-clear once data.time exceeds end_time"
        )

    def test_apply_disturbance_validates_inputs(self, sim):
        with pytest.raises((ValueError, TypeError)):
            sim.apply_disturbance(force_N=float("nan"), duration_s=0.3)
        with pytest.raises((ValueError, TypeError)):
            sim.apply_disturbance(force_N=1.0, duration_s=-0.1)
        with pytest.raises((ValueError, TypeError)):
            sim.apply_disturbance(force_N=1.0, duration_s=0.0)


class TestRecoveryBehavior:
    """End-to-end: 1 N × 0.3 s impulse → controller recovers balance."""

    def test_recovers_from_canonical_disturbance(self, sim):
        """The Issue #4 acceptance criteria, codified.

        After settling upright, apply 1 N horizontally at body top for 0.3 s.
        Then:
          - peak |theta| during disturbance < 5 deg
          - |theta| at +2 s after disturbance ends < 0.5 deg
          - |x_drift| at +2 s after disturbance ends < 0.1 m
        """
        # Settle
        _settle(sim, n_steps=500)
        x_pre = float(sim.data.qpos[0])

        # Apply canonical disturbance
        sim.apply_disturbance(force_N=1.0, duration_s=0.3)
        t_start = sim.data.time
        t_end = t_start + 0.3

        peak_theta = 0.0
        # Run the full disturbance window
        while sim.data.time < t_end:
            state, _, _ = sim.step()
            peak_theta = max(peak_theta, abs(state[0]))

        # Run 2 s of recovery after the impulse ends
        recover_until = t_end + 2.0
        while sim.data.time < recover_until:
            state, _, _ = sim.step()

        peak_theta_deg = np.degrees(peak_theta)
        final_theta_deg = abs(np.degrees(float(sim.ext.get_theta(sim.data))))
        x_drift = abs(float(sim.data.qpos[0]) - x_pre)

        assert not sim.failed, "Sim entered fail state — disturbance was too large"
        assert peak_theta_deg < 5.0, (
            f"Peak |theta|={peak_theta_deg:.2f}deg exceeds 5 deg ceiling"
        )
        assert final_theta_deg < 0.5, (
            f"Final |theta|={final_theta_deg:.3f}deg — controller did not recover within 2 s"
        )
        assert x_drift < 0.1, (
            f"x_drift={x_drift:.3f}m — too far from rest position"
        )

    def test_no_disturbance_produces_no_motion(self, sim):
        """Sanity: with no disturbance scheduled, theta stays near zero.

        Catches the case where apply_disturbance accidentally leaks state
        across resets, or step() always applies *some* perturbation.
        """
        _settle(sim, n_steps=500)
        x_pre = float(sim.data.qpos[0])

        # Don't apply anything — just run 2 s
        for _ in range(int(2.0 / 0.002)):
            sim.step()

        final_theta_deg = abs(np.degrees(float(sim.ext.get_theta(sim.data))))
        x_drift = abs(float(sim.data.qpos[0]) - x_pre)
        assert final_theta_deg < 0.1
        assert x_drift < 0.01
