"""
Gemini NLP Node 단위 테스트
Gemini API를 mock하여 과금 없이 파싱 로직을 검증.
"""

import json
import os
import sys
import types
from unittest.mock import MagicMock

import pytest

# google.generativeai mock (conftest 이후, import 전에 설정)
mock_genai = types.ModuleType("google.generativeai")
mock_genai.configure = MagicMock()
mock_genai.GenerativeModel = MagicMock()
sys.modules.setdefault("google", types.ModuleType("google"))
sys.modules["google.generativeai"] = mock_genai

os.environ["GOOGLE_API_KEY"] = "test-key-for-ci"

from ros2_ws.src.segway_controller.gemini_nlp_node import (
    GeminiNLPNode,
    VALID_COMMANDS,
)


# ── Fixtures ──

@pytest.fixture
def nlp_node():
    """GeminiNLPNode 인스턴스 (Gemini API mocked)."""
    node = GeminiNLPNode()

    # Override publishers with mocks for assertion
    node.cmd_pub = MagicMock()
    node._min_interval = 0.0
    node._last_call_time = 0.0
    return node


def _make_gemini_response(text: str):
    """Gemini API 응답 객체 mock."""
    response = MagicMock()
    response.text = text
    return response


# ── JSON 추출 테스트 ──

class TestExtractJson:
    def test_clean_json(self):
        result = GeminiNLPNode._extract_json('{"command": "enable"}')
        assert result == {"command": "enable"}

    def test_json_in_markdown_code_block(self):
        text = '```json\n{"command": "move_to", "x": 1.5}\n```'
        result = GeminiNLPNode._extract_json(text)
        assert result["command"] == "move_to"
        assert result["x"] == 1.5

    def test_json_with_surrounding_text(self):
        text = '네, 명령을 처리하겠습니다. {"command": "disable"} 이 명령을 전송합니다.'
        result = GeminiNLPNode._extract_json(text)
        assert result["command"] == "disable"

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="JSON을 찾을 수 없음"):
            GeminiNLPNode._extract_json("이것은 JSON이 아닙니다")

    def test_nested_json_update_gains(self):
        text = '{"command": "update_gains", "Q_diag": [200, 20, 2, 10], "R_val": 1.0}'
        result = GeminiNLPNode._extract_json(text)
        assert result["command"] == "update_gains"
        assert result["Q_diag"] == [200, 20, 2, 10]


# ── 명령어 검증 테스트 ──

class TestValidateCommand:
    @pytest.mark.parametrize("cmd", list(VALID_COMMANDS))
    def test_valid_commands(self, cmd):
        GeminiNLPNode._validate_command({"command": cmd})

    def test_invalid_command_raises(self):
        with pytest.raises(ValueError, match="알 수 없는 명령"):
            GeminiNLPNode._validate_command({"command": "fly_away"})

    def test_missing_command_raises(self):
        with pytest.raises(ValueError):
            GeminiNLPNode._validate_command({"not_command": "move_to"})


# ── 명령 처리 통합 테스트 (Gemini API mocked) ──

class TestProcessCommand:
    def test_move_to(self, nlp_node):
        nlp_node.model.generate_content.return_value = _make_gemini_response(
            '{"command": "move_to", "x": 2.0}'
        )
        nlp_node._process_command("앞으로 2미터 이동해")

        nlp_node.cmd_pub.publish.assert_called_once()
        published = json.loads(nlp_node.cmd_pub.publish.call_args[0][0].data)
        assert published["command"] == "move_to"
        assert published["x"] == 2.0

    def test_set_velocity(self, nlp_node):
        nlp_node.model.generate_content.return_value = _make_gemini_response(
            '{"command": "set_velocity", "velocity": 0.5}'
        )
        nlp_node._process_command("속도 0.5로 설정해")

        nlp_node.cmd_pub.publish.assert_called_once()
        published = json.loads(nlp_node.cmd_pub.publish.call_args[0][0].data)
        assert published["command"] == "set_velocity"
        assert published["velocity"] == 0.5

    def test_emergency_stop(self, nlp_node):
        nlp_node.model.generate_content.return_value = _make_gemini_response(
            '```json\n{"command": "disable"}\n```'
        )
        nlp_node._process_command("긴급 정지!")

        nlp_node.cmd_pub.publish.assert_called_once()
        published = json.loads(nlp_node.cmd_pub.publish.call_args[0][0].data)
        assert published["command"] == "disable"

    def test_update_gains(self, nlp_node):
        nlp_node.model.generate_content.return_value = _make_gemini_response(
            '{"command": "update_gains", "Q_diag": [150, 15, 1.5, 8], "R_val": 0.5}'
        )
        nlp_node._process_command("진동 줄여줘")

        nlp_node.cmd_pub.publish.assert_called_once()
        published = json.loads(nlp_node.cmd_pub.publish.call_args[0][0].data)
        assert published["command"] == "update_gains"
        assert published["Q_diag"] == [150, 15, 1.5, 8]

    def test_reset(self, nlp_node):
        nlp_node.model.generate_content.return_value = _make_gemini_response(
            '{"command": "reset"}'
        )
        nlp_node._process_command("초기화해")

        published = json.loads(nlp_node.cmd_pub.publish.call_args[0][0].data)
        assert published["command"] == "reset"

    def test_invalid_response_not_published(self, nlp_node):
        nlp_node.model.generate_content.return_value = _make_gemini_response(
            "죄송합니다, 이해할 수 없는 명령입니다."
        )
        nlp_node._process_command("하늘을 날아줘")

        nlp_node.cmd_pub.publish.assert_not_called()

    def test_api_exception_handled(self, nlp_node):
        nlp_node.model.generate_content.side_effect = Exception("API 오류")
        nlp_node._process_command("이동해")

        nlp_node.cmd_pub.publish.assert_not_called()


# ── Rate Limiting 테스트 ──

class TestRateLimiting:
    def test_rate_limit_blocks_rapid_calls(self, nlp_node):
        nlp_node.model.generate_content.reset_mock()
        nlp_node._min_interval = 10.0
        nlp_node._last_call_time = 9999999999.0  # 미래 시간

        msg = MagicMock()
        msg.data = "앞으로 이동해"
        nlp_node._on_nlp_input(msg)

        nlp_node.model.generate_content.assert_not_called()
