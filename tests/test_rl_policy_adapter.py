"""Unit tests for rl/policy_adapter.py (RLPolicy).

RLPolicy wraps a trained SB3 model behind the SAME interface as SegwayLQR
(`compute_torque(state) -> (tau_L, tau_R)`), so it drops into SegwaySimulation.
A fake model is injected, so these run on CI without torch/SB3 (the adapter
lazy-imports SB3 only in from_files()).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mujoco_sim"))  # SegwayLQR (numpy-only) for parity test

from rl.policy_adapter import RLPolicy  # noqa: E402


class _FakeModel:
    """Stand-in for an SB3 PPO model: predict() returns a fixed action."""

    def __init__(self, action_value):
        self._a = np.array([action_value], dtype=np.float32)
        self.last_obs = None

    def predict(self, obs, deterministic=True):
        self.last_obs = np.asarray(obs)
        return self._a, None


def test_compute_torque_returns_equal_pair():
    tL, tR = RLPolicy(_FakeModel(0.5), torque_limit=20.0).compute_torque([0, 0, 0, 0])
    assert tL == tR


def test_action_scaled_to_per_wheel_torque():
    tL, _ = RLPolicy(_FakeModel(0.5), torque_limit=20.0).compute_torque([0, 0, 0, 0])
    assert tL == pytest.approx(0.5 * 20.0)  # 10 N·m per wheel


def test_clips_to_torque_limit():
    tL, _ = RLPolicy(_FakeModel(5.0), torque_limit=20.0).compute_torque([0, 0, 0, 0])
    assert tL == pytest.approx(20.0)
    tL2, _ = RLPolicy(_FakeModel(-5.0), torque_limit=20.0).compute_torque([0, 0, 0, 0])
    assert tL2 == pytest.approx(-20.0)


def test_applies_vecnorm_when_present():
    class _FakeVecNorm:
        def __init__(self):
            self.called_with = None

        def normalize_obs(self, obs):
            self.called_with = np.asarray(obs)
            return np.zeros_like(obs)

    model, vn = _FakeModel(0.0), _FakeVecNorm()
    RLPolicy(model, vecnorm=vn, torque_limit=20.0).compute_torque([0.1, 0.2, 0.3, 0.4])
    assert vn.called_with is not None              # normalization was applied
    assert np.allclose(model.last_obs, 0.0)        # model saw the normalized obs


def test_interface_parity_with_segway_lqr():
    from lqr_controller import SegwayLQR

    state = [0.1, 0.0, 0.0, 0.0]
    lqr_out = SegwayLQR(torque_limit=20.0).compute_torque(state)
    rl_out = RLPolicy(_FakeModel(0.3), torque_limit=20.0).compute_torque(state)
    assert len(lqr_out) == len(rl_out) == 2
    assert all(isinstance(x, float) for x in (*lqr_out, *rl_out))
