#!/usr/bin/env python3
"""
segway_bridge.py — MuJoCo ↔ ROS2 WebSocket Bridge
macOS에서 실행. 기존 segway_sim.py와 함께 사용.

사용법:
  from segway_bridge import SegwayROSBridge

  bridge = SegwayROSBridge()
  bridge.connect()
  bridge.start_listener()

  # simulation loop에서:
  bridge.publish_state(theta, theta_dot, x, x_dot)
  torque = bridge.get_torque()
"""

import json
import time
import threading
import websocket  # pip install websocket-client

ROSBRIDGE_URL = "ws://127.0.0.1:9090"
STATE_TOPIC = "/segway/state"
CMD_TOPIC = "/segway/cmd_torque"
REF_TOPIC = "/segway/cmd_reference"
MSG_TYPE = "std_msgs/msg/String"


class SegwayROSBridge:
    """Thread-safe WebSocket bridge to rosbridge."""

    def __init__(self, url=ROSBRIDGE_URL):
        self.url = url
        self.ws = None
        self.latest_torque = 0.0
        self.latest_reference = {"command": "none"}
        self.connected = False
        self._lock = threading.Lock()

    def connect(self, max_retries=10, retry_delay=2.0):
        """Connect to rosbridge with retry logic."""
        for attempt in range(max_retries):
            try:
                self.ws = websocket.create_connection(
                    self.url, timeout=5.0,
                    ping_interval=10, ping_timeout=5,
                )
                self.connected = True
                print(f"[Bridge] Connected to {self.url}")
                self._setup_topics()
                return True
            except Exception as e:
                print(f"[Bridge] Retry {attempt+1}/{max_retries}: {e}")
                time.sleep(retry_delay)
        return False

    def _setup_topics(self):
        """Advertise and subscribe to topics."""
        # Advertise state topic
        self.ws.send(json.dumps({
            "op": "advertise", "topic": STATE_TOPIC, "type": MSG_TYPE,
        }))
        # Subscribe to torque commands
        self.ws.send(json.dumps({
            "op": "subscribe", "topic": CMD_TOPIC, "type": MSG_TYPE,
            "throttle_rate": 0,
        }))
        # Subscribe to reference commands (from OpenClaw)
        self.ws.send(json.dumps({
            "op": "subscribe", "topic": REF_TOPIC, "type": MSG_TYPE,
            "throttle_rate": 0,
        }))
        time.sleep(2.0)  # wait for topic discovery
        print("[Bridge] Topics ready.")

    def start_listener(self):
        """Start background thread to receive commands."""
        def _listen():
            while self.connected:
                try:
                    raw = self.ws.recv()
                    msg = json.loads(raw)
                    topic = msg.get("topic")

                    if topic == CMD_TOPIC:
                        data = json.loads(msg["msg"]["data"])
                        with self._lock:
                            self.latest_torque = float(data.get("torque", 0.0))

                    elif topic == REF_TOPIC:
                        data = json.loads(msg["msg"]["data"])
                        with self._lock:
                            self.latest_reference = data

                except websocket.WebSocketTimeoutException:
                    continue
                except Exception as e:
                    if self.connected:
                        print(f"[Bridge] Listener error: {e}")
                    break

        t = threading.Thread(target=_listen, daemon=True)
        t.start()
        return t

    def publish_state(self, theta, theta_dot, x, x_dot,
                      wheel_angle=0.0, wheel_vel=0.0, sim_time=None):
        """Publish Segway state to ROS2."""
        if not self.connected:
            return
        state = {
            "timestamp": sim_time or time.time(),
            "theta": float(theta), "theta_dot": float(theta_dot),
            "x": float(x), "x_dot": float(x_dot),
            "wheel_angle": float(wheel_angle), "wheel_vel": float(wheel_vel),
        }
        try:
            self.ws.send(json.dumps({
                "op": "publish", "topic": STATE_TOPIC,
                "msg": {"data": json.dumps(state)},
            }))
        except Exception as e:
            print(f"[Bridge] Publish error: {e}")

    def get_torque(self):
        """Get latest torque command (thread-safe)."""
        with self._lock:
            return self.latest_torque

    def get_reference(self):
        """Get latest reference command from OpenClaw."""
        with self._lock:
            return self.latest_reference.copy()

    def close(self):
        """Gracefully shut down."""
        self.connected = False
        if self.ws:
            try:
                self.ws.send(json.dumps({"op": "unadvertise", "topic": STATE_TOPIC}))
                time.sleep(0.3)
                self.ws.close()
            except Exception:
                pass
        print("[Bridge] Closed.")


# ── Standalone Test (dummy physics) ──

if __name__ == "__main__":
    import numpy as np

    bridge = SegwayROSBridge()
    if not bridge.connect():
        print("Failed. Run: docker compose up -d")
        exit(1)

    bridge.start_listener()

    theta, theta_dot, x, x_dot = 0.05, 0.0, 0.0, 0.0
    g, L, dt = 9.81, 0.5, 0.01

    print(f"\n{'step':>6} {'theta':>10} {'x':>10} {'torque':>10}")
    print("-" * 42)

    try:
        for step in range(3000):
            bridge.publish_state(theta, theta_dot, x, x_dot, sim_time=step * dt)
            torque = bridge.get_torque()

            theta_ddot = (g / L) * np.sin(theta) + torque / (10.0 * L**2)
            theta_dot += theta_ddot * dt
            theta += theta_dot * dt
            x_dot += torque * 0.01 * dt
            x += x_dot * dt

            if step % 50 == 0:
                print(f"{step:6d} {theta:+10.4f} {x:+10.4f} {torque:+10.4f}")
            if abs(theta) > 1.0:
                print(f"\n[FAIL] Fell at step {step}")
                break
            time.sleep(dt)
        else:
            print(f"\n[OK] Balanced for 30s!")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        bridge.close()
