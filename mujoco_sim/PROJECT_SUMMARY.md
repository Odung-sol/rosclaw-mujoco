# MuJoCo Segway 프로젝트 정리

작성일: 2026-03-15
작업 환경: macOS + segway_env (Python venv) + mjpython

---

## 1. 전체 파일/폴더 구조

```
~/segway_project/
└── mujoco_sim/
    ├── segway.xml                  ← MuJoCo MJCF 모델 (메인 모델 파일)
    ├── segway_sim.py               ← 시뮬레이션 메인 (뷰어 + LQR 루프)
    ├── state_extractor.py          ← MuJoCo 데이터 → LQR 상태벡터 변환
    ├── lqr_controller.py           ← LQR 게인 적용 및 토크 계산
    ├── plot_client.py              ← UDP 수신 실시간 그래프 (matplotlib)
    ├── torque_test.py              ← 토크 방향 검증 스크립트
    ├── meshes/
    │   ├── body.stl                ← 본체 메시 (STL)
    │   ├── L_wheel.stl             ← 왼쪽 바퀴 메시 (STL)
    │   └── R_wheel.stl             ← 오른쪽 바퀴 메시 (STL)
    └── backups/
        ├── graph_working_20260305_190224/   ← 그래프 정상 동작 시점 백업
        │   ├── segway_sim.py
        │   ├── state_extractor.py
        │   ├── lqr_controller.py
        │   └── plot_client.py
        └── uncontrolled_mode_20260306/      ← LQR 없는 자유 낙하 모드 백업
            ├── segway_sim.py
            ├── state_extractor.py
            ├── lqr_controller.py
            └── plot_client.py
```

### 실행 명령

```bash
# 터미널 1: 시뮬레이션 실행
cd ~/segway_project/mujoco_sim
source ~/segway_env/bin/activate
mjpython segway_sim.py

# 터미널 2: 실시간 그래프
cd ~/segway_project/mujoco_sim
source ~/segway_env/bin/activate
python3 plot_client.py
```

---

## 2. URDF 모델 구성

> ⚠️ 이 프로젝트에는 URDF 파일이 없습니다.
> 처음부터 MuJoCo MJCF (segway.xml) 형식으로 직접 작성했습니다.

### 물리적 구성

| 링크 | 질량 | 관성 (Ixx, Iyy, Izz) | 설명 |
|------|------|----------------------|------|
| body | 37.65 kg | 5.75, 5.42, 0.82 kg·m² | 세그웨이 본체 (CoM = z+0.14m) |
| L_wheel | 0.85 kg | 0.01, 0.02, 0.01 kg·m² | 왼쪽 바퀴 (반지름 0.1m) |
| R_wheel | 0.85 kg | 0.01, 0.02, 0.01 kg·m² | 오른쪽 바퀴 (반지름 0.1m) |

| 조인트 | 타입 | 위치 | 축 | 설명 |
|--------|------|------|----|------|
| body_free | freejoint | (0,0,0.2) | 6DOF | 본체 자유 조인트 |
| L_wheel_joint | hinge | (0,+0.29,0) | (0,-1,0) | 왼쪽 바퀴 회전 |
| R_wheel_joint | hinge | (0,-0.29,0) | (0,+1,0) | 오른쪽 바퀴 회전 |

**전체 중량:** 37.65 + 0.85×2 = 39.35 kg
**중력 강성 (linearized):** m·g·l_com ≈ 37.65 × 9.81 × 0.14 ≈ 51.7 Nm/rad

---

## 3. MuJoCo MJCF 전체 코드 (segway.xml)

```xml
<?xml version="1.0" encoding="utf-8"?>
<mujoco model="segway">
  <compiler angle="radian" meshdir="meshes"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4" iterations="100" cone="pyramidal" jacobian="dense">
    <flag contact="enable"/>
  </option>
  <default>
    <joint damping="0.01"/>
    <geom condim="4" friction="1.0 0.005 0.001" solref="0.004 1" solimp="0.9 0.95 0.001"/>
  </default>
  <asset>
    <mesh name="body_mesh" file="body.stl"/>
    <mesh name="L_wheel_mesh" file="L_wheel.stl"/>
    <mesh name="R_wheel_mesh" file="R_wheel.stl"/>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.8 0.8 0.8" rgb2="0.3 0.3 0.3" width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" material="grid_mat"/>
    <light pos="0 0 5" dir="0 0 -1"/>
    <body name="body" pos="0 0 0.2">
      <freejoint name="body_free"/>
      <site name="body_imu" pos="0 0 0.14" size="0.01"/>
      <inertial pos="0 0 0.14" mass="37.65" diaginertia="5.75 5.42 0.82"/>
      <geom name="body_visual" type="mesh" mesh="body_mesh" rgba="0.9 0.92 0.93 1" contype="0" conaffinity="0"/>
      <geom name="body_col" type="box" size="0.15 0.25 0.2" pos="0 0 0.14" rgba="0.9 0.92 0.93 0" contype="0" conaffinity="0" mass="0"/>
      <body name="L_wheel" pos="0 0.29 0">
        <joint name="L_wheel_joint" type="hinge" axis="0 -1 0" damping="0.01"/>
        <inertial pos="0 0 0" mass="0.85" diaginertia="0.01 0.02 0.01"/>
        <geom name="L_wheel_visual" type="mesh" mesh="L_wheel_mesh" rgba="0.7 0.7 0.7 1" contype="0" conaffinity="0"/>
        <geom name="L_wheel_col" type="cylinder" size="0.1 0.03" euler="1.5708 0 0" condim="4" friction="1.0 0.005 0.001" mass="0"/>
      </body>
      <body name="R_wheel" pos="0 -0.29 0">
        <joint name="R_wheel_joint" type="hinge" axis="0 1 0" damping="0.01"/>
        <inertial pos="0 0 0" mass="0.85" diaginertia="0.01 0.02 0.01"/>
        <geom name="R_wheel_visual" type="mesh" mesh="R_wheel_mesh" rgba="0.7 0.7 0.7 1" contype="0" conaffinity="0"/>
        <geom name="R_wheel_col" type="cylinder" size="0.1 0.03" euler="1.5708 0 0" condim="4" friction="1.0 0.005 0.001" mass="0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="L_wheel_torque" joint="L_wheel_joint" gear="1"  ctrllimited="true" ctrlrange="-20 20"/>
    <motor name="R_wheel_torque" joint="R_wheel_joint" gear="-1" ctrllimited="true" ctrlrange="-20 20"/>
  </actuator>
  <sensor>
    <framequat name="body_quat" objtype="body" objname="body"/>
    <gyro      name="body_gyro" site="body_imu"/>
    <jointpos  name="L_wheel_pos" joint="L_wheel_joint"/>
    <jointpos  name="R_wheel_pos" joint="R_wheel_joint"/>
    <jointvel  name="L_wheel_vel" joint="L_wheel_joint"/>
    <jointvel  name="R_wheel_vel" joint="R_wheel_joint"/>
  </sensor>
</mujoco>
```

### 주요 설계 포인트

- **freejoint**: body가 공간에서 자유롭게 운동 (6DOF). qpos[0:3]=xyz, qpos[3:7]=쿼터니언[w,x,y,z]
- **액추에이터 방향**: `L_wheel gear=1`, `R_wheel gear=-1` → ctrl < 0 이면 양쪽 바퀴 앞으로 전진
- **토크 제한**: 각 바퀴 ±20 Nm
- **충돌 형상**: 시각 메시(STL)와 물리 충돌체(cylinder/box) 분리

---

## 4. LQR 컨트롤러 구현

### 상태변수 정의

```
x = [theta, theta_dot, phi, phi_dot]
```

| 변수 | 의미 | MuJoCo 소스 |
|------|------|-------------|
| theta | 본체 피치각 (rad), 앞으로 기울면 양수 | qpos[3:7] 쿼터니언 → pitch 추출 |
| theta_dot | 피치 각속도 (rad/s) | qvel[4] = wy (Y축 각속도) |
| phi | 바퀴 누적 회전각 (rad), 위치 정보 | (Lq - Rq) / 2 + theta |
| phi_dot | 바퀴 각속도 (rad/s) | (Lv - Rv) / 2 + theta_dot |

### A, B 행렬 (선형화 모델, 연속시간)

세그웨이 2륜 도립진자 선형화 모델:

```
상태: x = [theta, theta_dot, phi, phi_dot]^T
입력: u = Tw (전체 토크, Nm)

A = [0,          1,     0,  0]
    [m*g*l/I_eff, 0,     0,  0]   ← 중력 불안정 항
    [0,          0,     0,  1]
    [-m*g*l/I_eff,0,    0,  0]

B = [0]
    [-1/I_eff]
    [0]
    [1/(m_w*r²)]

여기서:
  I_eff ≈ I_body + m*l²  (피치 관성)
  m = 37.65 kg (본체 질량)
  g = 9.81 m/s²
  l = 0.14 m (CoM 높이)
  r = 0.10 m (바퀴 반지름)
```

### MATLAB LQR 결과 (사용자 제공)

```matlab
% MATLAB에서 계산된 값 (연속시간 LQR)
Q = diag([...])   % 상태 가중치
R = [...]          % 입력 가중치

% 결과
K_matlab = [144.8373,  44.0214,  3.1623,  6.0483]

% 제어법칙 (MATLAB 표준): u = -K*x
% 코드 적용: Tw = -(K @ state)  ← 부호 반전 포함
```

### 현재 코드 (lqr_controller.py) - 전체

```python
#!/usr/bin/env python3
import numpy as np

class SegwayLQR:
    """
    Fixed LQR gain (MATLAB).
    state = [theta, theta_dot, phi, phi_dot]
    Tw = -K x
    tau_L = Tw/2, tau_R = Tw/2
    """

    def __init__(self, torque_limit=20.0):
        self.torque_limit = float(torque_limit)

        # MATLAB 출력 기반 K
        # 필요하면 여기만 바꿔가며 비교
        self.K = np.array([[-144.8373, -44.0214, -3.1623, -6.0483]], dtype=float)

        print("[MATLAB LQR] Using fixed K =", self.K)

    def compute_torque(self, state):
        state = np.asarray(state, dtype=float).reshape(-1)
        if state.size != 4:
            raise ValueError("state must be length 4")

        Tw = -(self.K @ state).item()           # scalar
        tau_each = Tw / 2.0
        tau_each = float(np.clip(tau_each, -self.torque_limit, self.torque_limit))
        return tau_each, tau_each
```

### 토크 방향 규칙 (확인된 사항)

```
L_wheel: axis="0 -1 0", gear=1
R_wheel: axis="0 1 0",  gear=-1

ctrl < 0  →  양쪽 바퀴 앞으로 (전진)
ctrl > 0  →  양쪽 바퀴 뒤로 (후진)

theta > 0 (앞으로 기울) → Tw < 0 → ctrl < 0 → 바퀴 전진 → 안정화 ✓
```

---

## 5. 상태 추출기 전체 코드 (state_extractor.py)

```python
#!/usr/bin/env python3
import numpy as np

class SegwayStateExtractor:
    """
    Extract state for LQR:
    x = [theta, theta_dot, phi, phi_dot]

    theta     = body pitch angle (rad)  (Y축 회전, 앞뒤 기울기)
    theta_dot = body pitch rate (rad/s) (qvel[4] = wy)
    phi       = world-frame wheel angle (rad) = joint_angle + theta
    phi_dot   = world-frame wheel velocity (rad/s)
    """

    def __init__(self, model):
        self.model = model

        # Joint indices (by name)
        self.Lq = model.joint("L_wheel_joint").qposadr[0]
        self.Rq = model.joint("R_wheel_joint").qposadr[0]
        self.Lv = model.joint("L_wheel_joint").dofadr[0]
        self.Rv = model.joint("R_wheel_joint").dofadr[0]

    def reset(self):
        pass

    def get_theta(self, data):
        """
        Body PITCH angle (Y축 회전, 앞뒤 기울기).
        MuJoCo quaternion [w, x, y, z]
        앞으로 기울면 theta > 0
        """
        w, x, y, z = data.qpos[3:7]
        sin_pitch = 2.0 * (w * y - x * z)
        cos_pitch = 1.0 - 2.0 * (y * y + x * x)
        return float(np.arctan2(sin_pitch, cos_pitch))

    def get_theta_dot(self, data):
        """
        Body pitch rate.
        freejoint qvel = [vx, vy, vz, wx, wy, wz]
        qvel[4] = wy = pitch rate (Y축 각속도)
        """
        return float(data.qvel[4])

    def get_phi(self, data):
        """
        World-frame 바퀴 각도 = joint_angle + body_pitch
        레퍼런스 Plugin.cc: LR_wheel_angle = joint->GetAngle() + Gyro_Y
        L축 "0 -1 0" 반전 보정: (Lq - Rq)/2
        """
        theta = self.get_theta(data)
        phi_joint = (data.qpos[self.Lq] - data.qpos[self.Rq]) / 2.0
        return float(phi_joint + theta)

    def get_phi_dot(self, data):
        """
        World-frame 바퀴 각속도 = joint_velocity + pitch_rate
        """
        theta_dot = self.get_theta_dot(data)
        phi_joint_dot = (data.qvel[self.Lv] - data.qvel[self.Rv]) / 2.0
        return float(phi_joint_dot + theta_dot)

    # ── 그래프 표시용 (LQR 입력 아님) ──────────────────────

    def get_theta_display(self, data):
        """그래프 표시용 theta = 실제 PITCH 각도"""
        w, x, y, z = data.qpos[3:7]
        sin_pitch = 2.0 * (w * y - x * z)
        cos_pitch = 1.0 - 2.0 * (y * y + x * x)
        return float(np.arctan2(sin_pitch, cos_pitch))

    def get_phi_display(self, data):
        """그래프 표시용 phi = joint angle only (Lq - Rq)/2"""
        return float((data.qpos[self.Lq] - data.qpos[self.Rq]) / 2.0)

    def get_phi_dot_display(self, data):
        return float((data.qvel[self.Lv] - data.qvel[self.Rv]) / 2.0)

    # ── LQR state vector ──────────────────────────────────

    def get_state(self, data):
        return np.array([
            self.get_theta(data),
            self.get_theta_dot(data),
            self.get_phi(data),
            self.get_phi_dot(data),
        ], dtype=float)

    def print_state(self, data):
        s = self.get_state(data)
        print(
            f"t={data.time:7.3f}  "
            f"theta={np.degrees(s[0]):+7.2f}deg  "
            f"theta_dot={s[1]:+8.3f}  "
            f"phi={s[2]:+10.4f}  "
            f"phi_dot={s[3]:+10.4f}"
        )
```

> ⚠️ 현재 파일의 `get_theta`는 ROLL 공식으로 변경되어 있음. 위의 PITCH 공식 버전이 올바른 버전.

---

## 6. 시뮬레이션 메인 전체 코드 (segway_sim.py)

```python
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

SIM_DT       = 0.002
TORQUE_LIMIT = 20.0
UDP_STATE_PORT = 9091

# ── 외력 설정 ──────────────────────────────────────────────
PUSH_FORCE    = 1.0     # N
PUSH_DURATION = 0.3     # 초
POLE_H        = 0.20    # m (막대 끝 높이 = 토크 팔)
AUTO_PUSH_T   = 1.0     # 초 (자동 push 시점)


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

        self._body_id    = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "body")
        self._push_dir   = 0.0
        self._push_timer = 0.0
        self._auto_pushed = False

    def reset(self, pitch_deg=0.0):
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[2] = 0.1  # 바퀴가 바닥에 닿도록 높이 보정

        pitch = np.deg2rad(pitch_deg)
        self.data.qpos[3] = np.cos(pitch / 2)   # w
        self.data.qpos[5] = np.sin(pitch / 2)   # y (Y축 회전 = pitch)

        self.failed = False
        self.log_data = []
        self._auto_pushed = False
        self._push_timer = 0.0
        self.ext.reset()
        mujoco.mj_forward(self.model, self.data)

    def _apply_push(self):
        """막대 끝에 한번 외력 인가"""
        if not self._auto_pushed and self.data.time >= AUTO_PUSH_T:
            self._auto_pushed = True
            self._push_timer = PUSH_DURATION
            self._push_dir = +1.0
            print(f"[PUSH] t={self.data.time:.2f}s  F={PUSH_FORCE}N → ({PUSH_DURATION}s)")

        if self._push_timer > 0:
            F = self._push_dir * PUSH_FORCE
            self.data.xfrc_applied[self._body_id, 0] = F
            self.data.xfrc_applied[self._body_id, 4] = F * POLE_H
            self._push_timer -= SIM_DT
        else:
            self.data.xfrc_applied[self._body_id, :] = 0.0

    def step(self):
        state = self.ext.get_state(self.data)

        self._apply_push()

        if abs(state[0]) > np.deg2rad(45):
            self.failed = True
            self.data.ctrl[self.L_act] = 0.0
            self.data.ctrl[self.R_act] = 0.0
            self.data.xfrc_applied[self._body_id, :] = 0.0
            mujoco.mj_step(self.model, self.data)
            return state, 0.0, 0.0

        tL, tR = self.lqr.compute_torque(state)
        self.data.ctrl[self.L_act] = float(tL)
        self.data.ctrl[self.R_act] = float(tR)

        mujoco.mj_step(self.model, self.data)
        return state, tL, tR

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

    def run_viewer(self, pitch_deg=0.0):
        self.reset(pitch_deg)
        count = 0

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                t0 = time.time()

                state, tL, tR = self.step()
                count += 1

                if count % 5 == 0:
                    self.send_state_udp(state, tL, tR)

                if count % 100 == 0:
                    Tw = tL + tR
                    theta_deg = np.degrees(state[0])
                    wheel_dir = "전진" if Tw < 0 else "후진" if Tw > 0 else "정지"
                    print(f"t={self.data.time:5.2f}  "
                          f"theta={theta_deg:+6.2f}°  "
                          f"Tw={Tw:+7.3f}({wheel_dir})  "
                          f"phi={state[2]:+7.3f}")

                viewer.sync()

                dt = SIM_DT - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)

    def close(self):
        try:
            self.state_sock.close()
        except Exception:
            pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--pitch",    type=float, default=0.0)
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
```

---

## 7. 그래프 클라이언트 전체 코드 (plot_client.py)

```python
#!/usr/bin/env python3
"""
plot_client.py - UDP로 오는 Segway telemetry를 실시간 플롯
- recv는 별도 스레드
- plot은 30Hz 고정 갱신
- 슬라이딩 윈도우(최근 WINDOW_S 초만 표시)
"""

import socket, json, threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# ===== 설정 =====
UDP_PORT = 9091
MAXLEN   = 6000
PLOT_HZ  = 30
WINDOW_S = 10.0

# ===== 버퍼 =====
t_buf     = deque(maxlen=MAXLEN)
theta_buf = deque(maxlen=MAXLEN)
phi_buf   = deque(maxlen=MAXLEN)
tau_buf   = deque(maxlen=MAXLEN)

_lock = threading.Lock()
_recv = 0

def udp_thread():
    global _recv
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
    sock.bind(("127.0.0.1", UDP_PORT))
    sock.settimeout(1.0)
    print(f"[plot_client] UDP listen 127.0.0.1:{UDP_PORT}")

    while True:
        try:
            raw, _ = sock.recvfrom(8192)
        except socket.timeout:
            continue
        except OSError as e:
            print("[plot_client] socket error:", e)
            break

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue

        t     = float(msg.get("time", 0.0))
        theta = float(msg.get("theta", 0.0))
        phi   = float(msg.get("phi", 0.0))
        tau   = float(msg.get("tau_L", 0.0))

        with _lock:
            t_buf.append(t)
            theta_buf.append(theta)
            phi_buf.append(phi)
            tau_buf.append(tau)
            _recv += 1

threading.Thread(target=udp_thread, daemon=True).start()

# ===== matplotlib =====
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 6))
ax[0].set_ylabel("theta (deg)")
ax[1].set_ylabel("phi (deg)")
ax[2].set_ylabel("tau_L (Nm)")
ax[2].set_xlabel("time (s)")

for a in ax:
    a.grid(True, alpha=0.3)

line1, = ax[0].plot([], [], lw=1.2, color='tab:blue')
line2, = ax[1].plot([], [], lw=1.2, color='tab:orange')
line3, = ax[2].plot([], [], lw=1.2, color='tab:green')

txt = ax[2].text(0.01, 0.05, "", transform=ax[2].transAxes, fontsize=9)

def _prepare_arrays():
    with _lock:
        if len(t_buf) < 3:
            return None
        t     = np.asarray(t_buf, dtype=float)
        theta = np.asarray(theta_buf, dtype=float)
        phi   = np.asarray(phi_buf, dtype=float)
        tau   = np.asarray(tau_buf, dtype=float)
        recv  = _recv

    idx = np.argsort(t)
    t, theta, phi, tau = t[idx], theta[idx], phi[idx], tau[idx]

    _, last_idx_rev = np.unique(t[::-1], return_index=True)
    last_idx = np.sort(len(t) - 1 - last_idx_rev)
    t, theta, phi, tau = t[last_idx], theta[last_idx], phi[last_idx], tau[last_idx]

    t_end = t[-1]
    mask  = t >= (t_end - WINDOW_S)
    if mask.sum() < 2:
        return None
    t, theta, phi, tau = t[mask], theta[mask], phi[mask], tau[mask]

    theta_deg = np.degrees(np.unwrap(theta))
    phi_deg   = np.degrees(np.unwrap(phi))

    return t, theta_deg, phi_deg, tau, recv

def update(_frame):
    pack = _prepare_arrays()
    if pack is None:
        return line1, line2, line3

    t, theta_u, phi_u, tau, recv = pack

    line1.set_data(t, theta_u)
    line2.set_data(t, phi_u)
    line3.set_data(t, tau)

    x_end = t[-1]
    for a in ax:
        a.set_xlim(x_end - WINDOW_S, x_end)

    for a, data in zip(ax, [theta_u, phi_u, tau]):
        if len(data) > 0:
            lo, hi = data.min(), data.max()
            margin = max((hi - lo) * 0.1, 0.01)
            a.set_ylim(lo - margin, hi + margin)

    txt.set_text(f"recv={recv}  buf={len(t)}/{MAXLEN}  t={t[-1]:.3f}s")
    return line1, line2, line3

ani = animation.FuncAnimation(fig, update, interval=int(1000/PLOT_HZ),
                               blit=False, cache_frame_data=False)

plt.tight_layout()
plt.show()
```

---

## 8. ROS2 / rosbridge 구성

> ⚠️ 이 프로젝트에는 ROS2 및 rosbridge가 없습니다.
> 상태 전송은 **UDP 직접 통신**으로 구현되어 있습니다.

### UDP 브릿지 구조

```
segway_sim.py ──UDP 127.0.0.1:9091──► plot_client.py
```

### 전송 메시지 형식 (JSON, UDP)

```json
{
  "time":      float,    // 시뮬레이션 시간 (초)
  "theta":     float,    // 본체 피치각 (rad) - 그래프 표시용
  "theta_dot": float,    // 피치 각속도 (rad/s)
  "phi":       float,    // 바퀴 누적 각도 (rad) - 그래프 표시용
  "phi_dot":   float,    // 바퀴 각속도 (rad/s)
  "tau_L":     float,    // 왼쪽 바퀴 토크 (Nm)
  "tau_R":     float     // 오른쪽 바퀴 토크 (Nm)
}
```

| 항목 | 값 |
|------|-----|
| 프로토콜 | UDP |
| 주소 | 127.0.0.1:9091 |
| 전송 주기 | 10ms (5 step × 2ms) |
| 포맷 | JSON (UTF-8) |

---

## 9. Docker 구성

> ⚠️ 이 프로젝트에는 Docker 구성이 없습니다.
> 로컬 macOS 환경에서 Python 가상환경으로 실행합니다.

### 실제 실행 환경

```
OS: macOS
Python 가상환경: ~/segway_env/
실행기: mjpython (MuJoCo 내장 Python 런타임)

설치 패키지:
- mujoco (mjpython 포함)
- numpy
- matplotlib
```

---

## 10. 현재 완성된 것 / 미완성된 것

### ✅ 완성된 것

| 항목 | 상태 |
|------|------|
| MJCF 모델 (segway.xml) | 완성. 물리 파라미터 설정됨 |
| STL 메시 3개 (body, L_wheel, R_wheel) | 완성 |
| MuJoCo 시뮬레이터 기본 구조 | 완성 |
| UDP 상태 전송 | 완성 |
| 실시간 그래프 (plot_client.py) | 완성. 슬라이딩 윈도우, θ/φ/τ 표시 |
| 넘어짐 판정 로직 | 완성 (45° 초과 시 제어 차단) |
| theta 추출 공식 | 완성 (쿼터니언 → PITCH 각도) |
| 자동 push 기능 (t=1.0s, 1N, 0.3s) | 완성 |
| 백업 시스템 (backups/) | 완성. uncontrolled/graph_working 2개 저장 |

### ❌ 미완성된 것 (현재 진행 중)

| 항목 | 문제 | 원인 분석 |
|------|------|-----------|
| **LQR push-recovery** | 밀면 제자리로 안 돌아옴 | phi 게인 부호/크기 튜닝 중 |
| phi 제어 방향 | 양수로 바꾸면 정방향 복귀하지만 phi_dot 게인이 theta 안정화를 방해 | K[3] > 10이면 phi_dot이 theta_dot 항과 충돌 |

### 🔧 현재 튜닝 이력

| 게인 K | 결과 |
|--------|------|
| [-144.8, -44.0, -3.2, -6.0] (MATLAB 원본) | 뺑글뺑글 회전 |
| [-80, -10, 0, 0] | 거의 제자리 (theta 제어만 작동) |
| [-80, -10, -8, -15] | 밀린 방향으로 천천히 이동 (phi 방향 역방향) |
| [-80, -10, +8, +12] | 여전히 원방향 이동 |
| [-80, -10, +17, +22] | 로봇 전복 (K[3]=22 > 10, 불안정) |
| [-80, -10, +20, +25] | 반대 방향으로 너무 크게 이동 |
| **[-80, -10, +18, +6]** | **현재 설정** (K[3]=6 < 10, 테스트 중) |

### 핵심 미해결 원인

`phi_dot = (Lv - Rv)/2 + theta_dot` 이므로:

```
K[1] * theta_dot = -10 * theta_dot  (안정화)
K[3] * phi_dot   = K[3] * (wheel_vel + theta_dot)
                 = K[3]*wheel_vel + K[3]*theta_dot  (theta_dot 중복 포함!)

합산 theta_dot 기여: (-10 + K[3]) * theta_dot
→ K[3] < 10 이어야 전체적으로 theta_dot 안정화 유지
```

### 📋 다음 단계 (TODO)

1. `K = [-80, -10, +18, +6]` 결과 확인
2. K[3] = 5~9 범위에서 phi 위치 복귀 게인 K[2] 최적화
3. 또는 `get_phi_dot()`에서 theta_dot 제거하는 방법 검토:
   ```python
   def get_phi_dot(self, data):
       # theta_dot 제거 → 순수 바퀴 구름 속도만
       return float((data.qvel[self.Lv] - data.qvel[self.Rv]) / 2.0)
   ```
4. 정상 작동 확인 후 uncontrolled_mode 백업 저장

---

*문서 끝 — 2026-03-15*
