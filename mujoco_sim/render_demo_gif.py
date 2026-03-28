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
FPS = 20                 # GIF 프레임 레이트
RENDER_EVERY = int(1.0 / (FPS * SIM_DT))  # 몇 스텝마다 프레임 캡처
OUTPUT_PATH = "../docs/demo.gif"


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    ext = SegwayStateExtractor(model)
    lqr = SegwayLQR(torque_limit=TORQUE_LIMIT)

    L_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "L_wheel_torque")
    R_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "R_wheel_torque")

    # 초기화: 약간 기울인 상태에서 시작
    mujoco.mj_resetData(model, data)
    pitch = np.deg2rad(3.0)
    data.qpos[3] = np.cos(pitch / 2)
    data.qpos[5] = np.sin(pitch / 2)
    ext.reset()
    mujoco.mj_forward(model, data)

    # 오프스크린 렌더러 설정
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    # 카메라 설정
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 1.8
    cam.azimuth = 135
    cam.elevation = -20
    cam.lookat[:] = [0.0, 0.0, 0.15]

    frames = []
    steps = int(DURATION / SIM_DT)

    print(f"렌더링 시작: {DURATION}초, {steps} 스텝, ~{steps // RENDER_EVERY} 프레임")

    for i in range(steps):
        state = ext.get_state(data)
        tL, tR = lqr.compute_torque(state)
        data.ctrl[L_act] = float(tL)
        data.ctrl[R_act] = float(tR)

        mujoco.mj_step(model, data)

        # 프레임 캡처
        if i % RENDER_EVERY == 0:
            renderer.update_scene(data, camera=cam)
            pixels = renderer.render()
            frames.append(Image.fromarray(pixels))

            if len(frames) % 20 == 0:
                t = data.time
                pitch = ext.get_theta_display(data)
                print(f"  t={t:.1f}s, pitch={np.degrees(pitch):.2f}°, 프레임 {len(frames)}")

    print(f"\n총 {len(frames)} 프레임 캡처 완료. GIF 저장 중...")

    # GIF 저장
    frames[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    print(f"저장 완료: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
