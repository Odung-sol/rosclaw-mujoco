"""Tests for the RL balance env and the SegwaySimulation stepping primitive
it relies on.

Both need MuJoCo (and gymnasium for the env), so the whole module
importorskips them — it skips on CI (no MuJoCo there) and runs locally in the
MuJoCo env, mirroring test_disturbance_recovery.py.
"""

import os
import sys
import contextlib
from pathlib import Path

import numpy as np
import pytest

# ── Make mujoco_sim/ importable (segway_sim, state_extractor, lqr_controller) ──
REPO_ROOT = Path(__file__).resolve().parents[1]
MUJOCO_DIR = REPO_ROOT / "mujoco_sim"
sys.path.insert(0, str(MUJOCO_DIR))

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")


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
    """A SegwaySimulation built from inside mujoco_sim/ (cwd-relative XML)."""
    from segway_sim import SegwaySimulation

    with _cd(MUJOCO_DIR):
        s = SegwaySimulation(use_ros2=False)
        try:
            yield s
        finally:
            s.close()


class TestSetTorqueAndStep:
    """The shared low-level stepping primitive the RL env builds on."""

    def test_advances_sim_time_by_one_timestep(self, sim):
        from segway_sim import SIM_DT

        sim.reset(pitch_deg=0.0)
        t0 = float(sim.data.time)
        sim.set_torque_and_step(1.0, 1.0)
        assert float(sim.data.time) == pytest.approx(t0 + SIM_DT)

    def test_sets_both_wheel_ctrl_inputs(self, sim):
        sim.reset(pitch_deg=0.0)
        sim.set_torque_and_step(2.5, -3.0)
        assert sim.data.ctrl[sim.L_act] == pytest.approx(2.5)
        assert sim.data.ctrl[sim.R_act] == pytest.approx(-3.0)

    def test_processes_pending_disturbances(self, sim):
        # Proves the primitive routes through _apply_pending_disturbance: a
        # scheduled kick auto-clears once its window elapses.
        sim.reset(pitch_deg=0.0)
        sim.apply_disturbance(force_N=1.0, duration_s=0.01)
        for _ in range(20):  # 20 * 0.002 s = 0.04 s > 0.01 s window
            sim.set_torque_and_step(0.0, 0.0)
        assert sim.pending_disturbance is None


@pytest.fixture
def env():
    """A SegwayBalanceEnv. The env locates segway.xml itself, so no cwd dance."""
    from rl.segway_env import SegwayBalanceEnv

    e = SegwayBalanceEnv()
    try:
        yield e
    finally:
        e.close()


class TestSegwayBalanceEnv:
    """Gymnasium env contract for the balancing task."""

    def test_observation_space_is_four_dimensional(self, env):
        assert env.observation_space.shape == (4,)

    def test_action_space_is_one_dimensional_and_normalized(self, env):
        assert env.action_space.shape == (1,)
        assert float(env.action_space.low[0]) == pytest.approx(-1.0)
        assert float(env.action_space.high[0]) == pytest.approx(1.0)

    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset(seed=0)
        assert obs.shape == (4,)
        assert np.all(np.isfinite(obs))
        assert isinstance(info, dict)

    def test_reset_obs_matches_state_extractor(self, env):
        obs, _ = env.reset(seed=0, options={"pitch_deg": 0.0})
        expected = env.sim.ext.get_state(env.sim.data)
        assert np.allclose(obs, expected, atol=1e-5)

    def test_step_returns_gym_five_tuple(self, env):
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step([0.0])
        assert obs.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_scaled_to_per_wheel_torque(self, env):
        # action 1.0 -> per-wheel torque = 1.0 * TORQUE_LIMIT (20), applied
        # equally to both wheels. Matches the actuator ctrlrange and the
        # per-wheel authority SegwayLQR has after its own ±20 clip.
        from segway_sim import TORQUE_LIMIT

        env.reset(seed=0, options={"pitch_deg": 0.0})
        env.step([1.0])
        assert env.sim.data.ctrl[env.sim.L_act] == pytest.approx(TORQUE_LIMIT)
        assert env.sim.data.ctrl[env.sim.R_act] == pytest.approx(TORQUE_LIMIT)

    def test_reward_matches_reward_module(self, env):
        from rl.reward import compute_reward

        env.reset(seed=0, options={"pitch_deg": 0.0})
        action = [0.3]
        obs, reward, _, _, _ = env.step(action)
        assert reward == pytest.approx(compute_reward(obs, action))

    def test_terminates_when_tilted_past_threshold(self, env):
        # Start beyond the 30 deg fail cone -> the step reports done.
        env.reset(seed=0, options={"pitch_deg": 35.0})
        _, _, terminated, _, _ = env.step([0.0])
        assert terminated is True

    def test_does_not_terminate_when_upright(self, env):
        env.reset(seed=0, options={"pitch_deg": 0.0})
        _, _, terminated, _, _ = env.step([0.0])
        assert terminated is False

    def test_truncates_at_max_episode_steps(self):
        from rl.segway_env import SegwayBalanceEnv

        e = SegwayBalanceEnv(max_episode_steps=3)
        try:
            e.reset(seed=0, options={"pitch_deg": 0.0})
            truncs = [e.step([0.0])[3] for _ in range(3)]
        finally:
            e.close()
        assert truncs[0] is False
        assert truncs[-1] is True

    def test_reset_with_options_pitch_is_deterministic(self, env):
        # Same commanded pitch -> same initial state regardless of seed. This
        # is what evaluate.py relies on for fair RL-vs-LQR episode matching.
        obs1, _ = env.reset(seed=1, options={"pitch_deg": 2.0})
        obs2, _ = env.reset(seed=2, options={"pitch_deg": 2.0})
        assert np.allclose(obs1, obs2, atol=1e-6)
