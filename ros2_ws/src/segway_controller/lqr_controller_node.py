#!/usr/bin/env python3
"""
Segway LQR Balancing Controller — ROS2 Node
Docker 컨테이너 내부에서 실행.

기존 mujoco_sim/lqr_controller.py의 로직을 ROS2 노드로 래핑.
WebSocket → rosbridge → 이 노드 → 토크 계산 → rosbridge → WebSocket

Subscribes: /segway/state, /segway/cmd_reference
Publishes:  /segway/cmd_torque, /segway/controller/status
"""

import json
import numpy as np
from scipy import linalg

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SegwayLQRController(Node):
    def __init__(self):
        super().__init__("segway_lqr_controller")

        # ── Declare Parameters (params.yaml에서 로드) ──
        self.declare_parameter("body_mass", 10.0)
        self.declare_parameter("wheel_mass", 1.0)
        self.declare_parameter("body_length", 0.5)
        self.declare_parameter("wheel_radius", 0.1)
        self.declare_parameter("body_inertia", 0.5)
        self.declare_parameter("max_torque", 20.0)
        self.declare_parameter("Q_diag", [100.0, 10.0, 1.0, 5.0])
        self.declare_parameter("R_val", 1.0)

        # ── Load Parameters ──
        self.M = self.get_parameter("body_mass").value
        self.m = self.get_parameter("wheel_mass").value
        self.L = self.get_parameter("body_length").value
        self.R_wheel = self.get_parameter("wheel_radius").value
        self.I_b = self.get_parameter("body_inertia").value
        self.max_torque = self.get_parameter("max_torque").value
        self.g = 9.81

        Q_diag = self.get_parameter("Q_diag").value
        R_val = self.get_parameter("R_val").value
        self.Q = np.diag(Q_diag)
        self.R_lqr = np.array([[R_val]])

        # ── Controller State ──
        self.enabled = True
        self.x_ref = 0.0
        self.v_ref = 0.0
        self.state_count = 0

        # ── Compute LQR Gain ──
        self.K = self._compute_lqr_gain()
        self.get_logger().info(f"LQR K = {np.round(self.K, 4).tolist()}")

        # ── ROS2 Pub/Sub ──
        self.state_sub = self.create_subscription(
            String, "/segway/state", self._on_state, 10
        )
        self.ref_sub = self.create_subscription(
            String, "/segway/cmd_reference", self._on_reference, 10
        )
        self.cmd_pub = self.create_publisher(String, "/segway/cmd_torque", 10)
        self.status_pub = self.create_publisher(
            String, "/segway/controller/status", 10
        )

        self.create_timer(0.1, self._publish_status)
        self.last_state_time = self.get_clock().now()

        self.get_logger().info("Segway LQR controller ready.")

    def _compute_lqr_gain(self):
        """Linearized Segway → CARE solver → K gain matrix."""
        M, m, L, g = self.M, self.m, self.L, self.g
        I_b = self.I_b
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

        P = linalg.solve_continuous_are(A, B, self.Q, self.R_lqr)
        K = np.linalg.inv(self.R_lqr) @ B.T @ P
        return K

    def _on_state(self, msg: String):
        """State → LQR → torque."""
        try:
            s = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        self.state_count += 1
        self.last_state_time = self.get_clock().now()

        if not self.enabled:
            self._publish_torque(0.0, s.get("timestamp", 0))
            return

        x_vec = np.array([
            s["theta"],
            s["theta_dot"],
            s["x"] - self.x_ref,
            s["x_dot"] - self.v_ref,
        ])

        torque = float((-self.K @ x_vec).item())
        torque = np.clip(torque, -self.max_torque, self.max_torque)
        self._publish_torque(torque, s.get("timestamp", 0))

    def _on_reference(self, msg: String):
        """OpenClaw / 사용자 명령 처리."""
        try:
            ref = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        cmd = ref.get("command", "")

        if cmd == "move_to":
            self.x_ref = float(ref.get("x", 0.0))
            self.v_ref = 0.0
            self.get_logger().info(f"→ move_to x={self.x_ref:.2f}m")

        elif cmd == "set_velocity":
            self.v_ref = float(ref.get("velocity", 0.0))
            self.get_logger().info(f"→ velocity {self.v_ref:.2f}m/s")

        elif cmd == "enable":
            self.enabled = True
            self.get_logger().info("→ ENABLED")

        elif cmd == "disable":
            self.enabled = False
            self.get_logger().info("→ DISABLED (e-stop)")

        elif cmd == "update_gains":
            Q_new = ref.get("Q_diag")
            R_new = ref.get("R_val")
            Q_backup, R_backup = self.Q.copy(), self.R_lqr.copy()
            if Q_new:
                self.Q = np.diag(Q_new)
            if R_new:
                self.R_lqr = np.array([[R_new]])
            try:
                self.K = self._compute_lqr_gain()
                self.get_logger().info(
                    f"→ gains updated K={np.round(self.K, 4).tolist()}"
                )
            except Exception as e:
                self.Q, self.R_lqr = Q_backup, R_backup
                self.get_logger().error(f"→ gain update failed, reverted: {e}")

        elif cmd == "reset":
            self.x_ref = 0.0
            self.v_ref = 0.0
            self.enabled = True
            self.get_logger().info("→ RESET")

    def _publish_torque(self, torque, timestamp):
        msg = String()
        msg.data = json.dumps({
            "timestamp": timestamp,
            "torque": torque,
            "controller_mode": "lqr" if self.enabled else "disabled",
        })
        self.cmd_pub.publish(msg)

    def _publish_status(self):
        elapsed = (
            self.get_clock().now() - self.last_state_time
        ).nanoseconds / 1e9
        state = "active" if self.enabled and elapsed < 1.0 else (
            "standby" if not self.enabled else "timeout"
        )
        msg = String()
        msg.data = json.dumps({
            "status": state,
            "enabled": self.enabled,
            "state_count": self.state_count,
            "x_ref": self.x_ref,
            "v_ref": self.v_ref,
            "K": self.K.tolist(),
        })
        self.status_pub.publish(msg)


def main():
    rclpy.init()
    node = SegwayLQRController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
