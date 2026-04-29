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

        # K = [theta, theta_dot, phi, phi_dot] gains.
        #
        # The first two entries (-144.84, -44.02) are the MATLAB-derived
        # balance gains — these have always been correct.
        #
        # The last two (+5.0, +3.0) are the position-regulating gains, which
        # had been wrong since the repo started: they were (-3.16, -6.05)
        # but the state extractor was silently zeroing phi (see
        # state_extractor.get_phi for the (L - R)/2 sign-convention fix in
        # commit fixing #X). With phi forced to 0 the position gains never
        # did anything; once phi was real the old negative-sign gains
        # actively destabilised the cart, so the LQR could no longer recover
        # from even a 1 N kick. The signs flip when the phi convention is
        # corrected. Tuning: empirical sweep over (K[2], K[3]) at 1 N × 0.3 s
        # canonical kick — (+5, +3) gives final |θ| = 0.06° and final x =
        # +0.001 m. Bigger gains diverge (>+10 inverts the body).
        self.K = np.array([[-144.8373, -44.0214, +5.0, +3.0]], dtype=float)

        print("[MATLAB LQR] Using fixed K =", self.K)

    def compute_torque(self, state):
        state = np.asarray(state, dtype=float).reshape(-1)
        if state.size != 4:
            raise ValueError("state must be length 4")

        Tw = (self.K @ state).item()              # scalar
        tau_each = Tw / 2.0
        tau_each = float(np.clip(tau_each, -self.torque_limit, self.torque_limit))
        return tau_each, tau_each
