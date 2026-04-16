#!/usr/bin/env python3
import numpy as np

class SegwayStateExtractor:
    """
    Extract state for LQR:
    x = [theta, theta_dot, phi, phi_dot]

    theta     = body pitch angle (rad)  (quat -> pitch, Y축 앞뒤 기울기)
    theta_dot = body pitch rate (rad/s)
    phi       = average wheel angle (rad)
    phi_dot   = average wheel velocity (rad/s)
    """

    def __init__(self, model):
        self.model = model

        # Joint indices (by name)
        self.Lq = model.joint("L_wheel_joint").qposadr[0]
        self.Rq = model.joint("R_wheel_joint").qposadr[0]
        self.Lv = model.joint("L_wheel_joint").dofadr[0]
        self.Rv = model.joint("R_wheel_joint").dofadr[0]

    def reset(self):
        # placeholder for compatibility (네 코드에서 ext.reset() 호출하길래)
        pass

    def get_theta(self, data):
        """
        Body pitch angle from quaternion (앞뒤 기울기).
        MuJoCo quaternion order: [w, x, y, z]
        """
        w, x, y, z = data.qpos[3:7]
        sin_pitch = 2.0 * (w * y - x * z)
        cos_pitch = 1.0 - 2.0 * (y * y + x * x)
        return np.arctan2(sin_pitch, cos_pitch)

    def get_theta_dot(self, data):
        """Body pitch rate (Y축 각속도)."""
        return float(data.qvel[4])

    def get_phi(self, data):
        return float((data.qpos[self.Lq] + data.qpos[self.Rq]) / 2.0)

    def get_phi_dot(self, data):
        return float((data.qvel[self.Lv] + data.qvel[self.Rv]) / 2.0)

    def get_theta_display(self, data):
        """
        그래프 표시용 theta (LQR 입력 아님).
        실제 PITCH 각도 (앞뒤 기울기) - Y축 회전.
        """
        w, x, y, z = data.qpos[3:7]
        sin_pitch = 2.0 * (w * y - x * z)
        cos_pitch = 1.0 - 2.0 * (y * y + x * x)
        return float(np.arctan2(sin_pitch, cos_pitch))

    def get_phi_display(self, data):
        """
        그래프 표시용 phi (LQR 입력 아님).
        L축 "0 -1 0" 반전 보정: 앞으로 굴러가면 양수로 증가.
        """
        return float((data.qpos[self.Lq] - data.qpos[self.Rq]) / 2.0)

    def get_phi_dot_display(self, data):
        return float((data.qvel[self.Lv] - data.qvel[self.Rv]) / 2.0)

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
            f"phi={s[2]:+10.3f}  "
            f"phi_dot={s[3]:+10.3f}"
        )
