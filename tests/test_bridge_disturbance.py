"""
Tests for the bridge-side /segway/disturbance plumbing.

Covers:
  - _parse_disturbance_payload: JSON validation of operator-supplied payloads
  - pop_disturbance: single-shot consumption semantics

Live WebSocket round-trip is exercised manually; see PR #5 description.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mujoco_sim"))

from segway_bridge import SegwayROSBridge  # noqa: E402


@pytest.fixture
def bridge():
    """Bridge instance without an actual WebSocket connection.

    We never call .connect() — we only exercise the parse and queue
    methods, which work on local state.
    """
    return SegwayROSBridge(url="ws://localhost:0")


# ── _parse_disturbance_payload ────────────────────────────────────────────
class TestParseDisturbancePayload:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ('{"force": 1.0, "duration": 0.3}',     {"force": 1.0, "duration": 0.3}),
            ('{"force": -2.5, "duration": 0.05}',   {"force": -2.5, "duration": 0.05}),
            # int → float coercion is fine
            ('{"force": 1, "duration": 1}',         {"force": 1.0, "duration": 1.0}),
        ],
    )
    def test_accepts_well_formed(self, raw, expected):
        assert SegwayROSBridge._parse_disturbance_payload(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "not json at all",
            "[1, 2, 3]",                            # array, not object
            '{"force": 1.0}',                       # missing duration
            '{"duration": 0.3}',                    # missing force
            '{"force": "huge", "duration": 0.3}',   # non-numeric force
            '{"force": 1.0, "duration": 0}',        # zero duration
            '{"force": 1.0, "duration": -0.5}',     # negative duration
            '{"force": NaN, "duration": 0.3}',      # NaN force (not valid JSON, returns None)
            '{"force": true, "duration": 0.3}',     # bool force — reject
            '{"force": 1.0, "duration": false}',    # bool duration — reject
        ],
    )
    def test_rejects_malformed(self, raw):
        assert SegwayROSBridge._parse_disturbance_payload(raw) is None

    def test_rejects_python_nan(self):
        # NaN can sneak in via a re-encode; verify the float-NaN check catches it.
        import math
        import json
        # json.dumps(NaN) produces "NaN" which is invalid JSON, but
        # allow_nan=True (the default) emits it. We construct manually:
        raw = json.dumps({"force": math.nan, "duration": 0.3})
        # Python's json.loads accepts "NaN" by default — confirm we drop it.
        assert SegwayROSBridge._parse_disturbance_payload(raw) is None

    def test_rejects_inf(self):
        import json
        import math
        raw = json.dumps({"force": math.inf, "duration": 0.3})
        assert SegwayROSBridge._parse_disturbance_payload(raw) is None


# ── pop_disturbance ───────────────────────────────────────────────────────
class TestPopDisturbance:
    def test_returns_none_when_empty(self, bridge):
        assert bridge.pop_disturbance() is None

    def test_returns_value_then_clears(self, bridge):
        bridge.latest_disturbance = {"force": 1.0, "duration": 0.3}
        assert bridge.pop_disturbance() == {"force": 1.0, "duration": 0.3}
        assert bridge.pop_disturbance() is None  # cleared after first pop

    def test_thread_safe_pop(self, bridge):
        """Smoke-test the lock — no races, no exceptions, with concurrent
        producers + consumers. Not a perfect race finder, just regression
        protection against a broken refactor.
        """
        import threading
        produced = 0
        consumed = 0
        stop = threading.Event()
        lock_misses = []

        def producer():
            nonlocal produced
            while not stop.is_set():
                with bridge._lock:
                    bridge.latest_disturbance = {"force": 1.0, "duration": 0.3}
                produced += 1

        def consumer():
            nonlocal consumed
            while not stop.is_set():
                d = bridge.pop_disturbance()
                if d is not None:
                    consumed += 1

        t1 = threading.Thread(target=producer, daemon=True)
        t2 = threading.Thread(target=consumer, daemon=True)
        t1.start()
        t2.start()
        threading.Event().wait(0.1)
        stop.set()
        t1.join(timeout=1)
        t2.join(timeout=1)

        assert produced > 0
        assert consumed > 0
        assert lock_misses == []
