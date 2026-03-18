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

# 시뮬 타임스텝: 모델 opt.timestep과 다르면 보기가 이상할 수 있음
# 일단 네가 쓰던 값 유지
SIM_DT = 0.002

TORQUE_LIMIT = 20.0

# UDP는 일단 유지하되, 없어도 시뮬은 돌아가게 try/except 처리
UDP_STATE_PORT = 9091

class SegwaySimulation:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data  = mujoco.MjData(self.model)

        self.ext = SegwayStateExtractor(self.model)
        self.lqr = SegwayLQR(torque_limit=TORQUE_LIMIT)

        self.L_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "L_wheel_torque")
        self.R_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "R_wheel_torque")

        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.failed = False
        self.log_data = []

    def reset(self, pitch_deg=2.0):
        mujoco.mj_resetData(self.model, self.data)

        # quaternion로 자세 초기화 (네가 쓰던 방식 유지)
        pitch = np.deg2rad(pitch_deg)
        # qpos[3:7] = quat (w,x,y,z) 라는 가정 하에
        # 여기서는 y축 회전(피치)처럼 넣던 너의 방식 유지
        self.data.qpos[3] = np.cos(pitch/2)
        self.data.qpos[5] = np.sin(pitch/2)

        self.failed = False
        self.log_data = []
        self.ext.reset()
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        state = self.ext.get_state(self.data)

        # 넘어짐 판정
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

    def send_state_udp(self, state, tL, tR):
        # round는 계단 원인 될 수 있어서 "일단" raw float로 보냄
        msg = json.dumps({
            "time": float(self.data.time),
            "theta": float(state[0]),
            "theta_dot": float(state[1]),
            "phi": float(state[2]),
            "phi_dot": float(state[3]),
            "tau_L": float(tL),
            "tau_R": float(tR),
        }).encode()

        try:
            self.state_sock.sendto(msg, ("127.0.0.1", UDP_STATE_PORT))
        except Exception:
            pass

    def run_viewer(self, pitch_deg=2.0):
        self.reset(pitch_deg)
        count = 0

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                t0 = time.time()

                state, tL, tR = self.step()
                count += 1

                # UDP: 너무 자주 보내면 부담될 수 있어서 5 유지
                if count % 5 == 0:
                    self.send_state_udp(state, tL, tR)

                if count % 500 == 0:
                    self.ext.print_state(self.data)

                viewer.sync()

                dt = SIM_DT - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)

    def run_headless(self, duration=10.0, pitch_deg=2.0):
        self.reset(pitch_deg)
        steps = int(duration / SIM_DT)

        for i in range(steps):
            state, tL, tR = self.step()
            self.log_data.append({
                "time": float(self.data.time),
                "theta": float(state[0]),
                "theta_dot": float(state[1]),
                "phi": float(state[2]),
                "phi_dot": float(state[3]),
                "tau_L": float(tL),
                "tau_R": float(tR),
            })

            if i % 5 == 0:
                self.send_state_udp(state, tL, tR)

            if i % 500 == 0:
                self.ext.print_state(self.data)

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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--pitch", type=float, default=2.0)
    args = p.parse_args()

    sim = SegwaySimulation()
    try:
        if args.headless:
            sim.run_headless(args.duration, args.pitch)
        else:
            sim.run_viewer(args.pitch)
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        sim.close()