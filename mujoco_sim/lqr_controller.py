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

        # ✅ 네가 준 MATLAB 출력 기반 K (잘 됐던 쪽에 맞춰서 이 값부터 씀)
        # 필요하면 여기만 바꿔가며 비교하면 됨
        self.K = np.array([[-144.8373, -44.0214, -3.1623, -6.0483]], dtype=float)

        print("[MATLAB LQR] Using fixed K =", self.K)

    def compute_torque(self, state):
        state = np.asarray(state, dtype=float).reshape(-1)
        if state.size != 4:
            raise ValueError("state must be length 4")

        Tw = (self.K @ state).item()              # scalar
        tau_each = Tw / 2.0
        tau_each = float(np.clip(tau_each, -self.torque_limit, self.torque_limit))
        return tau_each, tau_each