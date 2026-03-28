#!/usr/bin/env python3
import time
import json
import socket
import argparse
import numpy as np
import mujoco
import mujoco.viewer

from state_extractor import SegwayStateExtractor
from lqr_controller import SegwayLQR

MODEL_PATH = "segway.xml"
SIM_DT = 0.002
TORQUE_LIMIT = 20.0
UDP_STATE_PORT = 9091


class SegwaySimulation:
    def __init__(self, use_ros2=False):
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data  = mujoco.MjData(self.model)

        self.ext = SegwayStateExtractor(self.model)
        self.lqr = SegwayLQR(torque_limit=TORQUE_LIMIT)

        self.L_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "L_wheel_torque")
        self.R_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "R_wheel_torque")

        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.failed = False
        self.log_data = []

        # ROS2 bridge mode
        self.use_ros2 = use_ros2
        self.bridge = None
        if use_ros2:
            self._init_bridge()

    def _init_bridge(self):
        """Connect to ROS2 via rosbridge WebSocket."""
        from segway_bridge import SegwayROSBridge
        self.bridge = SegwayROSBridge()
        if not self.bridge.connect():
            raise RuntimeError(
                "Failed to connect to rosbridge. "
                "Run: docker compose up -d"
            )
        self.bridge.start_listener()

        # Wait for first torque response to ensure round-trip works
        print("[ROS2] Waiting for LQR controller response...")
        self.bridge.publish_state(0.01, 0.0, 0.0, 0.0, sim_time=0.0)
        for _ in range(50):  # up to 5 seconds
            time.sleep(0.1)
            if abs(self.bridge.get_torque()) > 1e-6:
                break
        print("[ROS2] Bridge connected. LQR runs in Docker.")

    def reset(self, pitch_deg=2.0):
        mujoco.mj_resetData(self.model, self.data)

        pitch = np.deg2rad(pitch_deg)
        self.data.qpos[3] = np.cos(pitch/2)
        self.data.qpos[5] = np.sin(pitch/2)

        self.failed = False
        self.log_data = []
        self.ext.reset()
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        """One simulation step using local LQR."""
        state = self.ext.get_state(self.data)

        if abs(state[0]) > np.deg2rad(30):
            self.failed = True
            self.data.ctrl[self.L_act] = 0.0
            self.data.ctrl[self.R_act] = 0.0
            mujoco.mj_step(self.model, self.data)
            return state, 0.0, 0.0

        tL, tR = self.lqr.compute_torque(state)
        self.data.ctrl[self.L_act] = float(tL)
        self.data.ctrl[self.R_act] = float(tR)

        mujoco.mj_step(self.model, self.data)
        return state, tL, tR

    def step_ros2(self):
        """One simulation step using ROS2 LQR (via bridge)."""
        theta = self.ext.get_theta(self.data)
        theta_dot = self.ext.get_theta_dot(self.data)
        x = float(self.data.qpos[0])
        x_dot = float(self.data.qvel[0])
        wheel_angle = self.ext.get_phi(self.data)
        wheel_vel = self.ext.get_phi_dot(self.data)

        # Send state and wait briefly for response
        self.bridge.publish_state(
            theta, theta_dot, x, x_dot,
            wheel_angle, wheel_vel,
            sim_time=float(self.data.time),
        )
        time.sleep(0.01)  # 10ms for WebSocket round-trip

        # Get latest torque computed by ROS2 LQR node
        torque = self.bridge.get_torque()

        # Fail detection
        if abs(theta) > np.deg2rad(30):
            self.failed = True
            torque = 0.0

        # Apply same torque to both wheels
        torque = float(np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT))
        self.data.ctrl[self.L_act] = torque
        self.data.ctrl[self.R_act] = torque

        mujoco.mj_step(self.model, self.data)
        state = np.array([theta, theta_dot, x, x_dot])
        return state, torque, torque

    def send_state_udp(self, state, tL, tR):
        theta_disp   = self.ext.get_theta_display(self.data)
        phi_disp     = self.ext.get_phi_display(self.data)
        phi_dot_disp = self.ext.get_phi_dot_display(self.data)

        msg = json.dumps({
            "time":      float(self.data.time),
            "theta":     theta_disp,
            "theta_dot": float(state[1]),
            "phi":       phi_disp,
            "phi_dot":   phi_dot_disp,
            "tau_L":     float(tL),
            "tau_R":     float(tR),
        }).encode()

        try:
            self.state_sock.sendto(msg, ("127.0.0.1", UDP_STATE_PORT))
        except Exception:
            pass

    def run_viewer(self, pitch_deg=2.0):
        self.reset(pitch_deg)
        count = 0
        step_fn = self.step_ros2 if self.use_ros2 else self.step

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                t0 = time.time()

                state, tL, tR = step_fn()
                count += 1

                if count % 5 == 0:
                    self.send_state_udp(state, tL, tR)

                if count % 500 == 0:
                    if self.use_ros2:
                        theta_deg = np.degrees(state[0])
                        x = state[2]
                        print(f"t={self.data.time:7.3f}  "
                              f"theta={theta_deg:+7.2f}deg  "
                              f"x={x:+7.3f}m  "
                              f"torque={tL:+7.3f}Nm")
                    else:
                        self.ext.print_state(self.data)

                viewer.sync()

                dt = SIM_DT - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)

    def run_headless(self, duration=10.0, pitch_deg=2.0):
        self.reset(pitch_deg)
        steps = int(duration / SIM_DT)
        step_fn = self.step_ros2 if self.use_ros2 else self.step

        for i in range(steps):
            state, tL, tR = step_fn()
            self.log_data.append({
                "time": float(self.data.time),
                "theta": float(state[0]),
                "theta_dot": float(state[1]),
                "tau_L": float(tL),
                "tau_R": float(tR),
            })

            if i % 5 == 0:
                self.send_state_udp(state, tL, tR)

            if i % 500 == 0:
                theta_deg = np.degrees(state[0])
                print(f"t={self.data.time:7.3f}  "
                      f"theta={theta_deg:+7.2f}deg  "
                      f"torque={tL:+7.3f}Nm")

            if self.failed:
                print(f"FAILED at t={self.data.time:.3f}")
                break

        thetas = [abs(d["theta"]) for d in self.log_data] or [0.0]
        taus   = [abs(d["tau_L"]) for d in self.log_data] or [0.0]
        print(f"\n[Result] steps={len(self.log_data)}, "
              f"theta_max={np.degrees(max(thetas)):.2f}deg, "
              f"tau_RMS={np.sqrt(np.mean(np.array(taus)**2)):.3f}Nm, "
              f"failed={self.failed}")

    def close(self):
        try:
            self.state_sock.close()
        except Exception:
            pass
        if self.bridge:
            self.bridge.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true")
    p.add_argument("--ros2", action="store_true",
                   help="Use ROS2 LQR controller via rosbridge (requires docker compose up)")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--pitch", type=float, default=2.0)
    args = p.parse_args()

    sim = SegwaySimulation(use_ros2=args.ros2)
    try:
        if args.headless:
            sim.run_headless(args.duration, args.pitch)
        else:
            sim.run_viewer(args.pitch)
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        sim.close()
