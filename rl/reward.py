#!/usr/bin/env python3
"""Reward function for the Segway balancing RL task.

The policy observes ``obs = [theta, theta_dot, phi, phi_dot]`` and emits a
normalized torque action in ``[-1, 1]``. This module turns a single
``(obs, action)`` pair into one scalar reward.

Design — the reward is the discrete-time analogue of the LQR cost:

    reward = alive_bonus - ( w_theta     * theta^2       # stay upright (dominant)
                           + w_theta_dot * theta_dot^2   # don't rock
                           + w_phi       * phi^2         # don't drift away
                           + w_tau       * tau_norm^2 )  # spend little torque

Every term is quadratic, mirroring the LQR objective
``J = integral(x^T Q x + u^T R u)``: the weights below play the role of Q's
diagonal (theta, theta_dot, phi) and R (tau). That correspondence is exactly
why the trained policy and the LQR baseline are comparable — they optimize the
same shape of objective.

The one thing the LQR cost has no equivalent for is ``alive_bonus``: a fixed
positive reward earned on every step the segway has *not* fallen over (the env
terminates at ``|theta| > 30 deg``). It is what makes the agent value survival,
not merely small deviations — early in training, "don't fall" is the signal
that matters most.

``phi_dot`` is observed (the policy may use it) but is deliberately NOT
penalized: the chosen four reward terms are ``theta``, ``theta_dot``, ``phi``,
and control effort.
"""

# ── Reward weights ───────────────────────────────────────────────────────────
# Defaults chosen so that balance (theta) dominates, position (phi) is a mild
# secondary objective, and control effort is only lightly discouraged. These
# are starter values; they are tuned empirically during Commit 2 (training).
ALIVE_BONUS = 1.0     # earned every step the segway survives
W_THETA = 1.0         # body pitch — the primary objective
W_THETA_DOT = 0.05    # pitch rate — damping, suppresses oscillation/overshoot
W_PHI = 0.1           # wheel angle (position proxy) — mild anti-drift
W_TAU = 0.01          # normalized control effort — favors smooth, efficient torque


def compute_reward(obs, action):
    """Reward for a single step.

    Args:
        obs: array-like ``[theta, theta_dot, phi, phi_dot]``.
        action: array-like ``[tau_norm]`` with ``tau_norm`` in ``[-1, 1]`` —
            the *raw* normalized action, not the scaled torque. ``W_TAU`` is
            calibrated to this ``[-1, 1]`` range.

    Returns:
        float reward = ``ALIVE_BONUS`` minus the weighted quadratic cost.
    """
    theta = float(obs[0])
    theta_dot = float(obs[1])
    phi = float(obs[2])
    # obs[3] (phi_dot) is intentionally excluded from the cost.
    tau_norm = float(action[0])

    cost = (
        W_THETA * theta**2
        + W_THETA_DOT * theta_dot**2
        + W_PHI * phi**2
        + W_TAU * tau_norm**2
    )
    return ALIVE_BONUS - cost
