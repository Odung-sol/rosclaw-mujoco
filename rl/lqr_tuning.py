#!/usr/bin/env python3
"""CARE-tuned LQR gain for the [theta, theta_dot, phi, phi_dot] state.

Used to give the fairness comparison a *properly tuned* LQR baseline instead of
the local fixed MATLAB gains (which were tuned for impulse-from-upright and
overshoot badly from a step initial tilt).

Physics constants are read from the loaded MuJoCo model (segway.xml = source of
truth), so this does NOT add a 4th hardcoded copy of the params (CLAUDE.md §4).
The linearization mirrors ros2_ws/.../lqr_controller_node.py:_compute_lqr_gain,
which uses state [theta, theta_dot, x, x_dot]; we convert the position gains to
wheel-angle coordinates via x = wheel_radius * phi.
"""

import numpy as np
from scipy import linalg

# Same default weights the ROS2 controller node uses (params.yaml Q_diag / R_val).
DEFAULT_Q_DIAG = (100.0, 10.0, 1.0, 5.0)
DEFAULT_R = 1.0


def _physics_from_model(model):
    """Pull (M, m, L, wheel_radius, I_body_pitch, g) from the MuJoCo model."""
    M = float(model.body("body").mass[0])
    m = float(model.body("L_wheel").mass[0])
    I_b = float(model.body("body").inertia[1])    # diaginertia pitch (y) axis
    L = float(model.body("body").ipos[2])         # CoM height above the axle
    Rw = float(model.geom("L_wheel_col").size[0])  # wheel cylinder radius
    g = float(-model.opt.gravity[2])
    return M, m, L, Rw, I_b, g


def care_gain(model, q_diag=DEFAULT_Q_DIAG, r=DEFAULT_R):
    """Return the LQR gain K for state [theta, theta_dot, phi, phi_dot].

    Convention matches SegwayLQR / the ROS2 node: torque Tw = K @ state (no
    extra negation). Raises numpy.linalg.LinAlgError if CARE has no solution.
    """
    M, m, L, Rw, I_b, g = _physics_from_model(model)
    denom = I_b * (M + m) - M**2 * L**2
    A = np.array([[0, 1, 0, 0],
                  [M * g * L * (M + m) / denom, 0, 0, 0],
                  [0, 0, 0, 1],
                  [-M * g * L * M * L / denom, 0, 0, 0]])
    B = np.array([[0], [-(M + m) / denom], [0], [M * L / denom]])
    R = np.array([[r]])
    P = linalg.solve_continuous_are(A, B, np.diag(q_diag), R)
    Kx = (np.linalg.inv(R) @ B.T @ P)[0]          # gain for [theta,theta_dot,x,x_dot]
    return np.array([Kx[0], Kx[1], Rw * Kx[2], Rw * Kx[3]])  # x = Rw * phi


def make_care_lqr(model, q_diag=DEFAULT_Q_DIAG, r=DEFAULT_R, torque_limit=20.0):
    """A SegwayLQR whose fixed gain is replaced by the CARE-tuned gain."""
    from lqr_controller import SegwayLQR

    c = SegwayLQR(torque_limit=torque_limit)
    c.K = np.array([care_gain(model, q_diag, r)], dtype=float)
    return c
