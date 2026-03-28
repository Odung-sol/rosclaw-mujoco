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
DURATION = 6.0           # 총 시뮬 시간 (초)
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

    # 초기화: 크게 기울인 상태에서 시작 (복원 과정이 보이게)
    mujoco.mj_resetData(model, data)
    pitch = np.deg2rad(10.0)
    data.qpos[3] = np.cos(pitch / 2)
    data.qpos[5] = np.sin(pitch / 2)
    ext.reset()
    mujoco.mj_forward(model, data)

    # 오프스크린 렌더러
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)

    # 카메라: 측면에서 기울기가 잘 보이게
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 2.5
    cam.azimuth = 90       # 측면 뷰
    cam.elevation = -10
    cam.lookat[:] = [0.5, 0.0, 0.15]

    frames = []
    steps = int(DURATION / SIM_DT)

    # 시나리오: 밸런싱 복원 (0~3초) → 전진 (3~5초) → 후진 (5~7초) → 정지 (7~8초)
    v_ref = 0.0

    print(f"렌더링 시작: {DURATION}초, {steps} 스텝, ~{steps // RENDER_EVERY} 프레임")

    for i in range(steps):
        t = data.time

        # 시나리오별 속도 목표 변경
        if t < 1.5:
            v_ref = 0.0        # 밸런싱 복원
        elif t < 3.5:
            v_ref = 1.5        # 전진 (빠르게)
        elif t < 5.0:
            v_ref = -1.5       # 후진 (빠르게)
        else:
            v_ref = 0.0        # 정지

        state = ext.get_state(data)
        # v_ref를 phi_dot 목표로 반영
        state_with_ref = state.copy()
        state_with_ref[3] -= v_ref

        tL, tR = lqr.compute_torque(state_with_ref)
        data.ctrl[L_act] = float(tL)
        data.ctrl[R_act] = float(tR)

        mujoco.mj_step(model, data)

        # 카메라 고정 (로봇 이동이 보이게)

        # 프레임 캡처
        if i % RENDER_EVERY == 0:
            renderer.update_scene(data, camera=cam)
            pixels = renderer.render()
            frames.append(Image.fromarray(pixels))

            if len(frames) % 25 == 0:
                pitch_deg = np.degrees(ext.get_theta_display(data))
                print(f"  t={t:.1f}s, pitch={pitch_deg:.2f}°, v_ref={v_ref}, 프레임 {len(frames)}")

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
