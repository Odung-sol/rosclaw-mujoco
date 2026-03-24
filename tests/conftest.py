"""
ROS2 의존성 mock — rclpy 없는 CI 환경에서 테스트 가능하도록.
conftest.py에서 한 번만 설정하면 모든 테스트 파일에서 공유.
"""

import sys
import types
from unittest.mock import MagicMock


def _setup_ros2_mocks():
    """rclpy, std_msgs를 가짜 모듈로 등록."""
    # rclpy 모듈
    rclpy_mod = types.ModuleType("rclpy")
    rclpy_mod.init = MagicMock()
    rclpy_mod.shutdown = MagicMock()
    rclpy_mod.spin = MagicMock()
    rclpy_mod.ok = MagicMock(return_value=True)

    rclpy_node_mod = types.ModuleType("rclpy.node")

    # 실제 Python 클래스로 Node를 정의 (MagicMock 대신)
    class FakeNode:
        def __init__(self, name="fake_node"):
            self._name = name
            self._logger = MagicMock()
            self._clock = MagicMock()
            self._clock.now.return_value = MagicMock(nanoseconds=0)

        def get_logger(self):
            return self._logger

        def get_clock(self):
            return self._clock

        def create_publisher(self, msg_type, topic, qos):
            return MagicMock()

        def create_subscription(self, msg_type, topic, callback, qos):
            return MagicMock()

        def create_timer(self, period, callback):
            return MagicMock()

        def declare_parameter(self, name, default=None):
            return MagicMock()

        def get_parameter(self, name):
            return MagicMock(value=None)

        def destroy_node(self):
            pass

    rclpy_node_mod.Node = FakeNode

    # std_msgs 모듈
    std_msgs_mod = types.ModuleType("std_msgs")
    std_msgs_msg_mod = types.ModuleType("std_msgs.msg")

    class FakeString:
        def __init__(self):
            self.data = ""

    std_msgs_msg_mod.String = FakeString

    # 모듈 등록
    sys.modules["rclpy"] = rclpy_mod
    sys.modules["rclpy.node"] = rclpy_node_mod
    sys.modules["std_msgs"] = std_msgs_mod
    sys.modules["std_msgs.msg"] = std_msgs_msg_mod

    return FakeNode, FakeString


FakeNode, FakeString = _setup_ros2_mocks()
