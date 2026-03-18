#!/usr/bin/env python3
"""
ROSClaw Discovery Node
OpenClaw AI 에이전트가 이 로봇을 자동으로 탐색할 수 있도록
토픽/서비스 정보를 /rosclaw/capabilities에 발행.
"""

import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


CAPABILITIES = {
    "robot_name": "segway_balancer",
    "description": "Two-wheeled self-balancing robot (Segway) with LQR controller, MuJoCo simulation",
    "topics": {
        "state": {
            "name": "/segway/state",
            "type": "std_msgs/msg/String",
            "direction": "publish",
            "rate_hz": 100,
            "schema": {
                "timestamp": "float (sim time)",
                "theta": "float (rad, pitch angle, 0=upright)",
                "theta_dot": "float (rad/s)",
                "x": "float (m, horizontal position)",
                "x_dot": "float (m/s)",
                "wheel_angle": "float (rad)",
                "wheel_vel": "float (rad/s)",
            },
        },
        "cmd_torque": {
            "name": "/segway/cmd_torque",
            "type": "std_msgs/msg/String",
            "direction": "publish",
            "rate_hz": 100,
            "schema": {
                "timestamp": "float",
                "torque": "float (N·m, range [-20, 20])",
                "controller_mode": "string (lqr|disabled)",
            },
        },
        "cmd_reference": {
            "name": "/segway/cmd_reference",
            "type": "std_msgs/msg/String",
            "direction": "subscribe",
            "schema": {
                "command": "string (move_to|set_velocity|enable|disable|update_gains|reset)",
                "x": "float (m, for move_to)",
                "velocity": "float (m/s, for set_velocity)",
                "Q_diag": "array[4] (for update_gains)",
                "R_val": "float (for update_gains)",
            },
        },
        "controller_status": {
            "name": "/segway/controller/status",
            "type": "std_msgs/msg/String",
            "direction": "publish",
            "rate_hz": 10,
        },
    },
    "commands": [
        {"intent": "move forward/backward", "command": "move_to", "params": ["x"]},
        {"intent": "set speed", "command": "set_velocity", "params": ["velocity"]},
        {"intent": "start balancing", "command": "enable"},
        {"intent": "emergency stop", "command": "disable"},
        {"intent": "tune controller", "command": "update_gains", "params": ["Q_diag", "R_val"]},
        {"intent": "reset position", "command": "reset"},
    ],
}


class DiscoveryNode(Node):
    def __init__(self):
        super().__init__("rosclaw_discovery")
        self.cap_pub = self.create_publisher(String, "/rosclaw/capabilities", 10)
        self.create_timer(1.0, self._publish_capabilities)
        self.get_logger().info("ROSClaw discovery node active.")

    def _publish_capabilities(self):
        msg = String()
        msg.data = json.dumps(CAPABILITIES)
        self.cap_pub.publish(msg)


def main():
    rclpy.init()
    node = DiscoveryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
