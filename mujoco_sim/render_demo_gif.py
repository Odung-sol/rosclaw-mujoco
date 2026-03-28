#!/usr/bin/env python3
"""MuJoCo 오프스크린 렌더링 → demo.gif 생성 스크립트."""

import numpy as np
import mujoco
from PIL import Image

from state_extractor import SegwayStateExtractor
from lqr_controller import SegwayLQR

MODEL_PATH = "segway.xml"
SIM_DT = 0.002
TORQUE_LIMIT = 20.0

# 렌더링 설정
WIDTH, HEIGHT = 480, 360
DURATION = 5.0           # 총 시뮬 시간 (초)
FPS = 25                 # GIF 프레임 레이트
RENDER_EVERY = int(1.0 / (FPS * SIM_DT))  # 몇 스텝마다 프레임 캡처
OUTPUT_PATH = "../docs/demo.gif"


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    ext = SegwayStateExtractor(model)
    lqr = SegwayLQR(torque_limit=TORQUE_LIMIT)

    L_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "L_wheel_torque")
    R_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "R_wheel_torque")

    # 초기화: 크게 기울인 상태에서 시작
    mujoco.mj_resetData(model, data)
    pitch = np.deg2rad(20.0)
    data.qpos[3] = np.cos(pitch / 2)
    data.qpos[5] = np.sin(pitch / 2)
    ext.reset()
    mujoco.mj_forward(model, data)

    # 오프스크린 렌더러
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    # 카메라: 측면에서 기울기가 잘 보이게
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.8
    cam.azimuth = 100      # 약간 비스듬한 측면
    cam.elevation = -15
    cam.lookat[:] = [0.0, 0.0, 0.15]

    frames = []
    steps = int(DURATION / SIM_DT)

    # 외란 스케줄: (시간, 힘) — 주기적으로 밀어서 흔들리게
    # 위치 복원용 K (phi 대신 x, phi_dot 대신 x_dot 사용)
    K_pos = np.array([[-80.0, -25.0, -8.0, -15.0]])

    print(f"렌더링 시작: {DURATION}초, {steps} 스텝, ~{steps // RENDER_EVERY} 프레임")

    for i in range(steps):
        theta = ext.get_theta(data)
        theta_dot = ext.get_theta_dot(data)
        x = float(data.qpos[0])
        x_dot = float(data.qvel[0])
        state_vec = np.array([theta, theta_dot, x, x_dot])

        Tw = (K_pos @ state_vec).item()
        tau = float(np.clip(Tw / 2, -TORQUE_LIMIT, TORQUE_LIMIT))
        data.ctrl[L_act] = tau
        data.ctrl[R_act] = tau

        mujoco.mj_step(model, data)

        # 프레임 캡처
        if i % RENDER_EVERY == 0:
            renderer.update_scene(data, camera=cam)
            pixels = renderer.render()
            frames.append(Image.fromarray(pixels))

            if len(frames) % 25 == 0:
                t = data.time
                pitch_deg = np.degrees(ext.get_theta_display(data))
                x_pos = float(data.qpos[0])
                print(f"  t={t:.1f}s, pitch={pitch_deg:.2f}°, x={x_pos:.2f}m, 프레임 {len(frames)}")

    print(f"\n총 {len(frames)} 프레임 캡처 완료. GIF 저장 중...")

    # GIF 저장 (128색 팔레트로 용량 최적화)
    quantized = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    quantized[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=quantized[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    print(f"저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
