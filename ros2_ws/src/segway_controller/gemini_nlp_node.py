#!/usr/bin/env python3
"""
Gemini NLP Node — ROS2 네이티브 자연어 제어
/segway/nlp_input 토픽으로 자연어 명령을 수신하면
Gemini API로 파싱하여 /segway/cmd_reference 토픽으로 발행.

Subscribes: /segway/nlp_input (std_msgs/String)
Publishes:  /segway/cmd_reference (std_msgs/String)
"""

import json
import os
import re
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

try:
    import google.generativeai as genai
except ImportError:
    genai = None

SYSTEM_PROMPT = """\
너는 ROS2 기반 Segway(Inverted Pendulum) 로봇의 자율주행 제어기야.
현재 LQR 제어기로 밸런싱 중이며, 사용자의 자연어 명령을 분석해 정해진 명령어 포맷(JSON)으로만 출력해.

[사용 가능한 명령어 및 파라미터 규격]
1. 위치 이동: {"command": "move_to", "x": 1.5}  (단위: m)
2. 속도 제어: {"command": "set_velocity", "velocity": 0.5} (단위: m/s)
3. 제어 시작: {"command": "enable"}
4. 긴급 정지: {"command": "disable"}
5. 게인 변경: {"command": "update_gains", "Q_diag": [200, 20, 2, 10], "R_val": 1.0}
6. 초기화:   {"command": "reset"}

다른 설명은 절대 하지 말고 오직 JSON 포맷만 출력해."""

VALID_COMMANDS = {"move_to", "set_velocity", "enable", "disable", "update_gains", "reset"}


class GeminiNLPNode(Node):
    def __init__(self):
        super().__init__("gemini_nlp_node")

        # ── Parameters ──
        self.declare_parameter("gemini_model", "gemini-2.0-flash")
        self.declare_parameter("min_call_interval", 2.0)

        model_name = self.get_parameter("gemini_model").value
        self._min_interval = self.get_parameter("min_call_interval").value
        self._last_call_time = 0.0

        # ── Gemini API 설정 ──
        if genai is None:
            raise RuntimeError(
                "google-generativeai 패키지가 설치되지 않았습니다. "
                "pip install google-generativeai"
            )

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY 환경변수가 설정되지 않았습니다!"
            )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        # ── ROS2 Pub/Sub ──
        self.cmd_pub = self.create_publisher(String, "/segway/cmd_reference", 10)
        self.input_sub = self.create_subscription(
            String, "/segway/nlp_input", self._on_nlp_input, 10
        )

        self.get_logger().info(
            f"Gemini NLP 노드 시작 (model={model_name}). "
            "/segway/nlp_input 토픽 대기 중..."
        )

    def _on_nlp_input(self, msg: String):
        """자연어 입력 토픽 콜백."""
        text = msg.data.strip()
        if not text:
            return

        # Rate limiting
        now = time.time()
        if now - self._last_call_time < self._min_interval:
            self.get_logger().warn(
                f"API 호출 간격이 {self._min_interval}초 미만입니다. 무시합니다."
            )
            return
        self._last_call_time = now

        self._process_command(text)

    def _process_command(self, text: str):
        """Gemini API 호출 → JSON 파싱 → 토픽 발행."""
        prompt = f'{SYSTEM_PROMPT}\n\n사용자 명령: "{text}"'

        try:
            self.get_logger().info(f"Gemini API 호출 중... 입력: {text}")
            response = self.model.generate_content(prompt)
            raw_text = response.text.strip()

            command_data = self._extract_json(raw_text)
            self._validate_command(command_data)

            msg = String()
            msg.data = json.dumps(command_data)
            self.cmd_pub.publish(msg)
            self.get_logger().info(
                f"명령 발행 완료: {command_data}"
            )

        except (ValueError, KeyError) as e:
            self.get_logger().error(f"파싱 실패: {e} (원본: {raw_text!r})")
        except Exception as e:
            self.get_logger().error(f"API 호출 실패: {e}")

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Gemini 응답에서 JSON 객체를 추출."""
        # 중첩 브레이스를 포함한 JSON 매칭
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
        if not match:
            raise ValueError(f"JSON을 찾을 수 없음: {text!r}")
        return json.loads(match.group())

    @staticmethod
    def _validate_command(data: dict):
        """명령어 스키마 검증."""
        cmd = data.get("command")
        if cmd not in VALID_COMMANDS:
            raise ValueError(f"알 수 없는 명령: {cmd!r}")


def main(args=None):
    rclpy.init(args=args)
    try:
        node = GeminiNLPNode()
    except RuntimeError as e:
        print(f"[ERROR] 노드 초기화 실패: {e}")
        rclpy.shutdown()
        return

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
