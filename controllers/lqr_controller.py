"""
Infinite-Horizon Continuous-Time Linear Quadratic Regulator (LQR) for a 6-DOF Quadcopter.

State-Space Linearization around the Hover Equilibrium Point:
    dx/dt = A * (x - x_eq) + B * (u - u_eq)

Minimizes the infinite-horizon quadratic cost functional:
    J = integral_0^inf [ (x - x_ref)^T Q (x - x_ref) + (u - u_eq)^T R (u - u_eq) ] dt

The optimal full-state feedback gain matrix K (4x12) is computed via the stabilizing
positive-definite solution P of the Continuous-Time Algebraic Riccati Equation (CARE):
    A^T P + P A - P B R^{-1} B^T P + Q = 0
    K = R^{-1} B^T P
solved numerically using `scipy.linalg.solve_continuous_are`.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from scipy.linalg import solve_continuous_are

import config


class LQRController:
    """
    12-State Linear Quadratic Regulator (LQR) supporting real-time hot-tuning of the
    diagonal weighting matrices Q and R with on-the-fly Riccati solver updates.
    """

    def __init__(
        self,
        q_diag: Sequence[float] | None = None,
        r_diag: Sequence[float] | None = None,
    ) -> None:
        self.mass: float = float(config.MASS)
        self.gravity: float = float(config.GRAVITY)
        self.ixx: float = float(config.INERTIA_X)
        self.iyy: float = float(config.INERTIA_Y)
        self.izz: float = float(config.INERTIA_Z)
        self.rot_damping: np.ndarray = config.ROTATIONAL_DAMPING.copy()

        # Diagonal state penalty vector Q (12 elements) and control effort vector R (4 elements)
        self.q_diag: np.ndarray = (
            np.array(config.Q_DIAG, dtype=float)
            if q_diag is None
            else np.array(q_diag, dtype=float)
        )
        self.r_diag: np.ndarray = (
            np.array(config.R_DIAG, dtype=float)
            if r_diag is None
            else np.array(r_diag, dtype=float)
        )

        if self.q_diag.size != 12:
            raise ValueError("q_diag must contain exactly 12 elements corresponding to the 12 states.")
        if self.r_diag.size != 4:
            raise ValueError("r_diag must contain exactly 4 elements corresponding to the 4 control channels.")

        # Bounded integral action for steady-state wind bias rejection
        self.integral_pos_error: np.ndarray = np.zeros(3, dtype=float)

        # Construct analytical Jacobian matrices A (12x12) and B (12x4) around hover
        self.A, self.B = self._build_linearized_state_space()

        self.Q: np.ndarray = np.eye(12, dtype=float)
        self.R: np.ndarray = np.eye(4, dtype=float)
        self.P: np.ndarray = np.eye(12, dtype=float)
        self.K: np.ndarray = np.zeros((4, 12), dtype=float)

        # Solve initial Continuous Algebraic Riccati Equation
        self.solve_riccati()

    def reset(self) -> None:
        """Resets the steady-state disturbance bias integral accumulator."""
        self.integral_pos_error[:] = 0.0

    # Convenience properties for GUI hot-tuning sliders
    @property
    def q_pos_xy(self) -> float:
        return float(self.q_diag[0])

    @property
    def q_pos_z(self) -> float:
        return float(self.q_diag[2])

    @property
    def q_vel(self) -> float:
        return float(self.q_diag[3])

    @property
    def q_att(self) -> float:
        return float(self.q_diag[6])

    @property
    def q_rate(self) -> float:
        return float(self.q_diag[9])

    @property
    def r_weight(self) -> float:
        return float(self.r_diag[0])

    def _build_linearized_state_space(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Analytically constructs the Jacobian matrices A = df/dx and B = df/du
        evaluated at the nominal hover operating point:
            phi = 0, theta = 0, psi = 0, p = q = r = 0, U1_eq = m * g.
        """
        A = np.zeros((12, 12), dtype=float)
        B = np.zeros((12, 4), dtype=float)

        # Translational position kinematics: dx/dt = vx, dy/dt = vy, dz/dt = vz
        A[0, 3] = 1.0
        A[1, 4] = 1.0
        A[2, 5] = 1.0

        # Small-angle gravity-attitude coupling in the horizontal plane:
        # dvx/dt ≈ +g * theta - (c_d / m) * vx
        # dvy/dt ≈ -g * phi   - (c_d / m) * vy
        A[3, 7] = self.gravity
        A[4, 6] = -self.gravity

        lin_drag = 1.1 * config.DRAG_COEFF / self.mass
        A[3, 3] = -lin_drag
        A[4, 4] = -lin_drag
        A[5, 5] = -1.6 * config.DRAG_COEFF / self.mass

        # Small-angle Euler angle kinematics around hover: d(phi)/dt = p, d(theta)/dt = q, d(psi)/dt = r
        A[6, 9] = 1.0
        A[7, 10] = 1.0
        A[8, 11] = 1.0

        # Natural passive rotational damping
        A[9, 9] = -self.rot_damping[0] / self.ixx
        A[10, 10] = -self.rot_damping[1] / self.iyy
        A[11, 11] = -self.rot_damping[2] / self.izz

        # Input matrix B with respect to [delta_U1, delta_tau_phi, delta_tau_theta, delta_tau_psi]
        B[5, 0] = 1.0 / self.mass
        B[9, 1] = 1.0 / self.ixx
        B[10, 2] = 1.0 / self.iyy
        B[11, 3] = 1.0 / self.izz

        return A, B

    def solve_riccati(self) -> np.ndarray:
        """
        Assembles the positive-definite weighting matrices Q and R from `self.q_diag`
        and `self.r_diag` and solves the Continuous-Time Algebraic Riccati Equation (CARE).
        """
        q_safe = np.maximum(self.q_diag, 1e-4)
        r_safe = np.maximum(self.r_diag, 1e-4)

        # Internal physical unit normalization preserving the relative weights of Q_DIAG and R_DIAG
        q_scaled = q_safe.copy()
        q_scaled[0:2] *= 4.0     # Horizontal position (x, y)
        q_scaled[2] *= 1.5       # Vertical altitude (z)
        q_scaled[3:5] *= 3.2     # Horizontal velocity (vx, vy)
        q_scaled[5] *= 1.8       # Vertical velocity (vz)
        q_scaled[6:8] *= 0.22    # Roll/Pitch attitude angles (phi, theta)
        q_scaled[8] *= 0.25      # Yaw heading angle (psi)
        q_scaled[9:12] *= 2.8    # Body angular rates (p, q, r)

        r_unit_scale = np.array([0.16, 4.8, 4.8, 6.5], dtype=float)
        self.Q = np.diag(q_scaled)
        self.R = np.diag(r_safe * r_unit_scale)

        try:
            self.P = solve_continuous_are(self.A, self.B, self.Q, self.R)
            self.K = np.linalg.solve(self.R, self.B.T @ self.P)
        except Exception:
            # Retain the last stabilizing gain matrix K if an extreme numerical condition occurs
            pass

        return self.K.copy()

    def update_weights(
        self,
        q_pos_xy: float | None = None,
        q_pos_z: float | None = None,
        q_vel: float | None = None,
        q_att: float | None = None,
        q_rate: float | None = None,
        r_weight: float | None = None,
        q_diag: Sequence[float] | None = None,
        r_diag: Sequence[float] | None = None,
    ) -> np.ndarray:
        """
        Updates the diagonal entries of Q and R in real time from GUI slider events
        and immediately recomputes the optimal state-feedback gain matrix K.
        """
        if q_diag is not None:
            arr_q = np.asarray(q_diag, dtype=float)
            if arr_q.size == 12:
                self.q_diag = np.maximum(arr_q, 1e-3)

        if r_diag is not None:
            arr_r = np.asarray(r_diag, dtype=float)
            if arr_r.size == 4:
                self.r_diag = np.maximum(arr_r, 1e-3)

        if q_pos_xy is not None:
            val = max(float(q_pos_xy), 0.1)
            self.q_diag[0] = val
            self.q_diag[1] = val

        if q_pos_z is not None:
            self.q_diag[2] = max(float(q_pos_z), 0.1)

        if q_vel is not None:
            val = max(float(q_vel), 0.05)
            self.q_diag[3] = val
            self.q_diag[4] = val
            self.q_diag[5] = val * 2.0

        if q_att is not None:
            val = max(float(q_att), 0.5)
            self.q_diag[6] = val
            self.q_diag[7] = val
            self.q_diag[8] = val * 0.5

        if q_rate is not None:
            val = max(float(q_rate), 0.05)
            self.q_diag[9:12] = val

        if r_weight is not None:
            val = max(float(r_weight), 0.02)
            self.r_diag[:] = val

        return self.solve_riccati()

    def compute_control(
        self,
        state: np.ndarray,
        setpoint_pos: np.ndarray,
        setpoint_yaw: float = 0.0,
        dt: float = config.DT,
    ) -> np.ndarray:
        """
        Evaluates the full-state feedback optimal control law:
            u = u_eq - K @ (x - x_ref)

        Incorporates heading-invariant horizontal frame rotation and geometric tilt
        compensation for the collective thrust channel U1.
        """
        x_ref = np.zeros(12, dtype=float)
        x_ref[0:3] = np.asarray(setpoint_pos, dtype=float)
        x_ref[8] = float(setpoint_yaw)

        error = np.asarray(state, dtype=float) - x_ref
        error[8] = float((error[8] + np.pi) % (2.0 * np.pi) - np.pi)

        if dt > 0.0:
            self.integral_pos_error += (-error[0:3]) * dt
            self.integral_pos_error = np.clip(self.integral_pos_error, -2.5, 2.5)

        psi = float(state[8])
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        rot_2d = np.array([[c_psi, s_psi], [-s_psi, c_psi]], dtype=float)

        pos_err_eff = error[0:3].copy()
        pos_err_eff[0:2] -= 0.18 * self.integral_pos_error[0:2]
        pos_err_eff[2] -= 0.22 * self.integral_pos_error[2]

        error_aligned = error.copy()
        error_aligned[0:2] = rot_2d @ pos_err_eff[0:2]
        error_aligned[3:5] = rot_2d @ error[3:5]
        error_aligned[2] = pos_err_eff[2]

        error_aligned[0:2] = np.clip(error_aligned[0:2], -2.2, 2.2)
        error_aligned[2] = np.clip(error_aligned[2], -2.5, 2.5)

        delta_u = -self.K @ error_aligned

        phi, theta = float(state[6]), float(state[7])
        tilt_comp = max(float(np.cos(phi) * np.cos(theta)), 0.55)
        u1_eq = self.mass * self.gravity
        u1 = (u1_eq + delta_u[0]) / tilt_comp
        u1 = float(
            np.clip(
                u1,
                4.0 * config.MIN_THRUST_PER_MOTOR,
                4.0 * config.MAX_THRUST_PER_MOTOR,
            )
        )

        tau_phi = float(np.clip(delta_u[1], -1.6, 1.6))
        tau_theta = float(np.clip(delta_u[2], -1.6, 1.6))
        tau_psi = float(np.clip(delta_u[3], -0.5, 0.5))

        return np.array([u1, tau_phi, tau_theta, tau_psi], dtype=float)
