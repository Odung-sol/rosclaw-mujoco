#!/usr/bin/env python3
"""
NLP CLI Node — 터미널 입력을 /segway/nlp_input 토픽으로 발행.
gemini_nlp_node와 분리하여 ROS2 spin 블로킹 없이 동작.

사용법:
  python3 nlp_cli_node.py
  [명령 입력] 앞으로 1.5미터 이동해
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class NLPCLINode(Node):
    def __init__(self):
        super().__init__("nlp_cli_node")
        self.pub = self.create_publisher(String, "/segway/nlp_input", 10)
        self.get_logger().info(
            "NLP CLI 노드 시작. 자연어 명령을 입력하세요. (종료: Ctrl+C)"
        )

    def run_input_loop(self):
        """메인 스레드에서 input() 루프 실행."""
        try:
            while rclpy.ok():
                text = input("\n[명령 입력] Segway에게 시킬 일: ").strip()
                if not text:
                    continue
                msg = String()
                msg.data = text
                self.pub.publish(msg)
                self.get_logger().info(f"발행 완료: {text}")
        except (KeyboardInterrupt, EOFError):
            pass


def main(args=None):
    rclpy.init(args=args)
    node = NLPCLINode()
    try:
        node.run_input_loop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
