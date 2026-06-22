#!/usr/bin/env python3
"""Gymnasium environment for balancing the Segway with RL.

Wraps mujoco_sim/SegwaySimulation so the policy trains against the *same*
physics (segway.xml) and the *same* state pipeline (state_extractor) as the
LQR. That parity is what makes a trained policy a drop-in alternative and a
fair comparison baseline.

    observation : [theta, theta_dot, phi, phi_dot]              (float32)
    action      : [tau_norm] in [-1, 1] -> per-wheel torque = tau_norm * max
    reward      : rl/reward.compute_reward
    terminated  : |theta| > 30 deg   (the sim's existing fail cone)
    truncated   : episode reached max_episode_steps
"""

import os
import sys
import contextlib
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rl.reward import compute_reward

# ── Make mujoco_sim/ importable, and locate it for the cwd-relative XML load ──
_MUJOCO_DIR = Path(__file__).resolve().parents[1] / "mujoco_sim"
if str(_MUJOCO_DIR) not in sys.path:
    sys.path.insert(0, str(_MUJOCO_DIR))

from segway_sim import SegwaySimulation, TORQUE_LIMIT  # noqa: E402  (after sys.path setup)


@contextlib.contextmanager
def _cd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


# Matches the fail cone in SegwaySimulation.step().
_FAIL_ANGLE_RAD = np.deg2rad(30.0)


class SegwayBalanceEnv(gym.Env):
    """Balance the Segway from a (possibly tilted) start without falling over."""

    metadata = {"render_modes": []}

    def __init__(self, frame_skip=5, max_episode_steps=2000, reset_pitch_deg=3.0):
        super().__init__()
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.reset_pitch_deg = float(reset_pitch_deg)
        self.max_torque = float(TORQUE_LIMIT)

        # segway.xml loads via a cwd-relative path, so construct from inside
        # mujoco_sim/. Only construction needs the cwd; stepping does not.
        with _cd(_MUJOCO_DIR):
            self.sim = SegwaySimulation(use_ros2=False)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._step_count = 0

    def _get_obs(self):
        return self.sim.ext.get_state(self.sim.data).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and "pitch_deg" in options:
            pitch_deg = float(options["pitch_deg"])
        else:
            pitch_deg = float(
                self.np_random.uniform(-self.reset_pitch_deg, self.reset_pitch_deg)
            )
        self.sim.reset(pitch_deg=pitch_deg)
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        # action in [-1, 1] -> per-wheel torque in [-max, max], same command to
        # both wheels (parity with SegwayLQR / step_ros2).
        tau_each = float(np.clip(float(action[0]) * self.max_torque,
                                 -self.max_torque, self.max_torque))

        for _ in range(self.frame_skip):
            self.sim.set_torque_and_step(tau_each, tau_each)

        self._step_count += 1
        obs = self._get_obs()
        reward = float(compute_reward(obs, action))
        terminated = bool(abs(float(obs[0])) > _FAIL_ANGLE_RAD)
        truncated = bool(self._step_count >= self.max_episode_steps)
        return obs, reward, terminated, truncated, {}

    def close(self):
        self.sim.close()
