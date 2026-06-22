#!/usr/bin/env python3
"""Run a trained PPO policy behind the SegwayLQR interface.

`RLPolicy.compute_torque(state) -> (tau_L, tau_R)` mirrors
mujoco_sim/lqr_controller.SegwayLQR, so SegwaySimulation can drive a trained
policy in place of the LQR (the `--rl` flag) with no other changes.

SB3 / torch are imported lazily (only in `from_files`), so importing this
module needs just NumPy — the adapter unit tests inject a fake model and run
on CI without torch.
"""

import numpy as np


class RLPolicy:
    """Adapter: a trained policy with the same call shape as SegwayLQR."""

    def __init__(self, model, vecnorm=None, torque_limit=20.0):
        self.model = model            # anything with .predict(obs, deterministic)
        self.vecnorm = vecnorm        # anything with .normalize_obs(obs), or None
        self.max_torque = float(torque_limit)

    @classmethod
    def from_files(cls, model_path, vecnorm_path=None, torque_limit=20.0):
        """Load an SB3 PPO model (+ VecNormalize stats) from disk."""
        import sys
        from pathlib import Path

        from stable_baselines3 import PPO

        model = PPO.load(str(model_path))

        vecnorm = None
        if vecnorm_path is not None and Path(vecnorm_path).exists():
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

            # VecNormalize.load needs a venv to attach to; build a throwaway one.
            root = str(Path(__file__).resolve().parents[1])
            if root not in sys.path:
                sys.path.insert(0, root)
            from rl.segway_env import SegwayBalanceEnv

            venv = DummyVecEnv([lambda: SegwayBalanceEnv()])
            vecnorm = VecNormalize.load(str(vecnorm_path), venv)
            vecnorm.training = False
            vecnorm.norm_reward = False

        return cls(model, vecnorm=vecnorm, torque_limit=torque_limit)

    def compute_torque(self, state):
        """state = [theta, theta_dot, phi, phi_dot] -> (tau_each, tau_each)."""
        obs = np.asarray(state, dtype=np.float32)
        if self.vecnorm is not None:
            obs = self.vecnorm.normalize_obs(obs)
        action, _ = self.model.predict(obs, deterministic=True)
        tau = float(np.clip(float(action[0]) * self.max_torque,
                            -self.max_torque, self.max_torque))
        return tau, tau
