"""Unit tests for the RL balancing reward (rl/reward.py).

Pure NumPy — no MuJoCo, no gymnasium — so these run on CI too (unlike the
env/training tests, which importorskip mujoco). The reward is the heart of
what the policy optimizes, so it gets its own focused test file.

State convention (matches mujoco_sim/state_extractor.get_state):
    obs = [theta, theta_dot, phi, phi_dot]
Action convention (matches SegwayBalanceEnv):
    action = [tau_norm]  in [-1, 1]
"""

import pytest

from rl.reward import compute_reward
from rl import reward


UPRIGHT = [0.0, 0.0, 0.0, 0.0]
NO_TORQUE = [0.0]


def test_upright_with_no_torque_earns_the_alive_bonus():
    # Perfectly balanced, no control effort => zero cost => the per-step
    # maximum, which is exactly the alive bonus.
    assert compute_reward(UPRIGHT, NO_TORQUE) == pytest.approx(reward.ALIVE_BONUS)


def test_reward_equals_alive_bonus_minus_weighted_quadratic_cost():
    # Locks the *shape* of the reward (negative quadratic cost + alive bonus)
    # without hard-coding weight values, so Commit 2 can retune weights freely.
    # phi_dot (obs[3]) is intentionally absent from the cost — see the
    # dedicated test below.
    obs = [0.1, 0.2, 0.3, 0.4]
    action = [0.5]
    expected = reward.ALIVE_BONUS - (
        reward.W_THETA * 0.1**2
        + reward.W_THETA_DOT * 0.2**2
        + reward.W_PHI * 0.3**2
        + reward.W_TAU * 0.5**2
    )
    assert compute_reward(obs, action) == pytest.approx(expected)


def test_tilting_reduces_reward():
    assert compute_reward([0.1, 0.0, 0.0, 0.0], NO_TORQUE) < compute_reward(UPRIGHT, NO_TORQUE)


def test_pitch_penalty_is_symmetric():
    # Quadratic => leaning forward and back by the same angle costs the same.
    assert compute_reward([+0.15, 0, 0, 0], NO_TORQUE) == pytest.approx(
        compute_reward([-0.15, 0, 0, 0], NO_TORQUE)
    )


def test_pitch_rate_is_penalized():
    assert compute_reward([0, 0.5, 0, 0], NO_TORQUE) < compute_reward(UPRIGHT, NO_TORQUE)


def test_wheel_drift_is_penalized():
    assert compute_reward([0, 0, 1.0, 0], NO_TORQUE) < compute_reward(UPRIGHT, NO_TORQUE)


def test_control_effort_is_penalized():
    assert compute_reward(UPRIGHT, [1.0]) < compute_reward(UPRIGHT, [0.0])


def test_wheel_velocity_does_not_affect_reward():
    # phi_dot is observed by the policy but is NOT one of the 4 reward terms
    # (theta, theta_dot, phi, tau). This pins that decision.
    assert compute_reward([0, 0, 0, 5.0], NO_TORQUE) == pytest.approx(
        compute_reward(UPRIGHT, NO_TORQUE)
    )


def test_pitch_weighted_more_heavily_than_position():
    # Balance matters more than position: the same deviation in theta should
    # cost more than in phi (w_theta > w_phi).
    tilt = compute_reward([0.2, 0, 0, 0], NO_TORQUE)
    drift = compute_reward([0, 0, 0.2, 0], NO_TORQUE)
    assert tilt < drift
