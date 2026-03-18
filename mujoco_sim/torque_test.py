import time
import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("segway.xml")
data = mujoco.MjData(model)

L_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "L_wheel_torque")
R_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "R_wheel_torque")

# 살짝 기울여 시작(넘어지지 말고 접촉만 보려고)
p = np.deg2rad(1.0)
data.qpos[3] = np.cos(p/2)
data.qpos[5] = np.sin(p/2)

print("[INFO] 0~2초: 양쪽 +2Nm, 2~6초: 0Nm")
print("[INFO] 목표: 스핀 없이 굴러가려는지 / 파고드는지 / 미끄러지는지 확인")

with mujoco.viewer.launch_passive(model, data) as viewer:
    t0 = time.time()
    while viewer.is_running() and time.time() - t0 < 6.0:
        t = time.time() - t0

        if t < 2.0:
            data.ctrl[L_act] = 2.0
            data.ctrl[R_act] = 2.0
        else:
            data.ctrl[L_act] = 0.0
            data.ctrl[R_act] = 0.0

        mujoco.mj_step(model, data)
        viewer.sync()
