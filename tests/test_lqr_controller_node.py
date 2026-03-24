"""
LQR Controller Node 단위 테스트
ROS2 의존성을 mock하여 CI에서 실행 가능.
"""

import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from ros2_ws.src.segway_controller.lqr_controller_node import SegwayLQRController

DEFAULTS = {
    "body_mass": 10.0,
    "wheel_mass": 1.0,
    "body_length": 0.5,
    "wheel_radius": 0.1,
    "body_inertia": 0.5,
    "max_torque": 20.0,
    "Q_diag": [100.0, 10.0, 1.0, 5.0],
    "R_val": 1.0,
}


@pytest.fixture
def controller():
    """SegwayLQRController 인스턴스 (ROS2 mocked via conftest)."""
    # Override get_parameter to return proper defaults
    original_get_param = SegwayLQRController.get_parameter

    def mock_get_parameter(self, name):
        if name in DEFAULTS:
            return MagicMock(value=DEFAULTS[name])
        return MagicMock(value=None)

    SegwayLQRController.get_parameter = mock_get_parameter

    node = SegwayLQRController()

    # Restore
    SegwayLQRController.get_parameter = original_get_param

    # Replace publishers with mocks for assertions
    node.cmd_pub = MagicMock()
    node.status_pub = MagicMock()

    return node


def _make_state_msg(theta=0.0, theta_dot=0.0, x=0.0, x_dot=0.0, timestamp=0.0):
    """State 토픽 메시지 생성."""
    msg = MagicMock()
    msg.data = json.dumps({
        "timestamp": timestamp,
        "theta": theta,
        "theta_dot": theta_dot,
        "x": x,
        "x_dot": x_dot,
    })
    return msg


def _make_ref_msg(command_data: dict):
    """cmd_reference 토픽 메시지 생성."""
    msg = MagicMock()
    msg.data = json.dumps(command_data)
    return msg


# ── LQR 게인 계산 테스트 ──

class TestLQRGain:
    def test_gain_shape(self, controller):
        """K 행렬이 (1, 4) shape인지 확인."""
        assert controller.K.shape == (1, 4)

    def test_gain_is_finite(self, controller):
        """K 값이 유한한지 확인."""
        assert np.all(np.isfinite(controller.K))

    def test_gain_stabilizes(self, controller):
        """폐루프 시스템이 안정한지 확인 (모든 고유값 실수부 < 0)."""
        M, m, L, g = controller.M, controller.m, controller.L, controller.g
        I_b = controller.I_b
        denom = I_b * (M + m) - M**2 * L**2

        A = np.array([
            [0, 1, 0, 0],
            [M * g * L * (M + m) / denom, 0, 0, 0],
            [0, 0, 0, 1],
            [-M * g * L * M * L / denom, 0, 0, 0],
        ])
        B = np.array([
            [0],
            [-(M + m) / denom],
            [0],
            [M * L / denom],
        ])

        A_cl = A - B @ controller.K
        eigenvalues = np.linalg.eigvals(A_cl)
        # 수치 오차 허용 (CARE solver 특성상 ~1e-9 수준의 잔차 가능)
        assert np.all(np.real(eigenvalues) < 1e-6), (
            f"불안정 고유값 발견: {eigenvalues}"
        )

    def test_gain_changes_with_Q(self, controller):
        """Q 가중치 변경 시 K가 달라지는지 확인."""
        K_original = controller.K.copy()
        # 대각 값만 스케일링 (CARE solver가 수렴하는 범위)
        controller.Q = np.diag([150.0, 15.0, 1.5, 7.5])
        K_new = controller._compute_lqr_gain()
        assert not np.allclose(K_original, K_new)


# ── 명령 처리 테스트 ──

class TestOnReference:
    def test_move_to(self, controller):
        msg = _make_ref_msg({"command": "move_to", "x": 1.5})
        controller._on_reference(msg)
        assert controller.x_ref == 1.5
        assert controller.v_ref == 0.0

    def test_set_velocity(self, controller):
        msg = _make_ref_msg({"command": "set_velocity", "velocity": 0.8})
        controller._on_reference(msg)
        assert controller.v_ref == 0.8

    def test_enable(self, controller):
        controller.enabled = False
        msg = _make_ref_msg({"command": "enable"})
        controller._on_reference(msg)
        assert controller.enabled is True

    def test_disable(self, controller):
        msg = _make_ref_msg({"command": "disable"})
        controller._on_reference(msg)
        assert controller.enabled is False

    def test_update_gains(self, controller):
        K_before = controller.K.copy()
        msg = _make_ref_msg({
            "command": "update_gains",
            "Q_diag": [100, 10, 1, 5],
            "R_val": 2.0,
        })
        controller._on_reference(msg)
        assert not np.allclose(K_before, controller.K)
        assert controller.R_lqr[0, 0] == 2.0

    def test_reset(self, controller):
        controller.x_ref = 5.0
        controller.v_ref = 1.0
        controller.enabled = False
        msg = _make_ref_msg({"command": "reset"})
        controller._on_reference(msg)
        assert controller.x_ref == 0.0
        assert controller.v_ref == 0.0
        assert controller.enabled is True

    def test_invalid_json_ignored(self, controller):
        msg = MagicMock()
        msg.data = "not json"
        controller._on_reference(msg)


# ── 상태 처리 + 토크 계산 테스트 ──

class TestOnState:
    def test_torque_published_when_enabled(self, controller):
        msg = _make_state_msg(theta=0.05, theta_dot=0.1, x=0.0, x_dot=0.0)
        controller._on_state(msg)

        controller.cmd_pub.publish.assert_called_once()
        published = json.loads(controller.cmd_pub.publish.call_args[0][0].data)
        assert "torque" in published
        assert published["controller_mode"] == "lqr"

    def test_zero_torque_when_disabled(self, controller):
        controller.enabled = False
        msg = _make_state_msg(theta=0.05)
        controller._on_state(msg)

        published = json.loads(controller.cmd_pub.publish.call_args[0][0].data)
        assert published["torque"] == 0.0
        assert published["controller_mode"] == "disabled"

    def test_torque_clamped(self, controller):
        """큰 기울기에서 토크가 max_torque로 클램핑."""
        msg = _make_state_msg(theta=1.0, theta_dot=5.0)
        controller._on_state(msg)

        published = json.loads(controller.cmd_pub.publish.call_args[0][0].data)
        assert abs(published["torque"]) <= controller.max_torque

    def test_upright_produces_near_zero_torque(self, controller):
        """직립 상태에서 토크가 거의 0."""
        msg = _make_state_msg(theta=0.0, theta_dot=0.0, x=0.0, x_dot=0.0)
        controller._on_state(msg)

        published = json.loads(controller.cmd_pub.publish.call_args[0][0].data)
        assert abs(published["torque"]) < 0.01

    def test_position_error_generates_torque(self, controller):
        """x_ref != x일 때 위치 보정 토크 발생."""
        controller.x_ref = 2.0
        msg = _make_state_msg(theta=0.0, theta_dot=0.0, x=0.0, x_dot=0.0)
        controller._on_state(msg)

        published = json.loads(controller.cmd_pub.publish.call_args[0][0].data)
        assert abs(published["torque"]) > 0.01
