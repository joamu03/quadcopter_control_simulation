"""
Six-Degree-of-Freedom (6-DOF) Nonlinear Quadcopter Dynamics Model with 4th-Order Runge-Kutta (RK4)
Numerical Integration and Atmospheric Wind Disturbance Generator (Dryden Turbulence + Discrete Gusts).

System state vector (12 state variables):
    x = [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]^T
where:
    - [x, y, z]       : Inertial position in the East-North-Up (ENU) reference frame (m)
    - [vx, vy, vz]    : Inertial linear velocity vector (m/s)
    - [phi, theta, psi]: Z-Y-X Euler angles: Roll (phi), Pitch (theta), Yaw (psi) (rad)
    - [p, q, r]       : Angular velocity components in the body-fixed reference frame (rad/s)
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import config


@dataclass
class GustEvent:
    """
    Represents a discrete wind gust disturbance (impulse, 1-cosine, or sustained step with decay)
    injected into the flight environment along a specified 3D unit vector.
    """

    direction: np.ndarray
    amplitude: float
    duration: float
    elapsed: float = 0.0
    profile: str = "step_decay"

    def evaluate(self, dt: float) -> np.ndarray:
        """
        Evaluates the instantaneous wind velocity contribution of this gust event
        and advances its internal elapsed timer by `dt` seconds.
        """
        self.elapsed += dt
        if self.elapsed > self.duration:
            return np.zeros(3, dtype=float)

        tau = self.elapsed / max(self.duration, 1e-6)
        if self.profile == "one_minus_cos":
            # Classical MIL-F-8785C / NASA (1 - cos) gust envelope
            envelope = 0.5 * (1.0 - np.cos(2.0 * np.pi * tau))
        elif self.profile == "impulse":
            # Rapid attack impulse with exponential relaxation
            attack = min(self.elapsed / 0.06, 1.0)
            decay = np.exp(-3.2 * max(0.0, self.elapsed - 0.06) / max(self.duration - 0.06, 0.1))
            envelope = attack * decay
        else:
            # Sustained step disturbance with smooth cosine ramp-down
            if tau < 0.05:
                envelope = tau / 0.05
            elif tau < 0.72:
                envelope = 1.0
            else:
                rem = (tau - 0.72) / 0.28
                envelope = 0.5 * (1.0 + np.cos(np.pi * rem))

        return self.direction * (self.amplitude * float(envelope))

    @property
    def is_active(self) -> bool:
        """Returns True if the gust event has not yet expired."""
        return self.elapsed <= self.duration


class WindDisturbance:
    """
    Three-dimensional atmospheric wind disturbance generator and aerodynamic drag model.

    Superimposes three wind velocity components in the inertial reference frame:
        v_wind(t) = v_steady + v_dryden(t) + sum_i(v_gust_i(t))

    Aerodynamic drag force acting on the airframe:
        v_rel = v_drone - v_wind
        F_d = -1/2 * rho * C_d * A_eff * v_rel * |v_rel| - C_flap * v_rel
    """

    def __init__(
        self,
        rho: float = config.AIR_DENSITY,
        drag_coeff: float = config.DRAG_COEFF,
        effective_area: np.ndarray | None = None,
        turbulence_intensity: float = config.DEFAULT_TURBULENCE_STD,
        seed: int | None = 42,
    ) -> None:
        self.rho: float = float(rho)
        self.drag_coeff: float = float(drag_coeff)
        self.effective_area: np.ndarray = (
            config.EFFECTIVE_AREA.copy()
            if effective_area is None
            else np.asarray(effective_area, dtype=float)
        )
        # Center-of-pressure offset relative to the center of gravity (rotor plane above CoG)
        self.cp_offset: np.ndarray = np.array([0.0, 0.0, 0.022], dtype=float)

        self.rng = np.random.default_rng(seed)
        self.steady_wind: np.ndarray = np.zeros(3, dtype=float)
        self.random_wind_enabled: bool = False
        self.turbulence_intensity: float = float(turbulence_intensity)

        # Internal state vectors for the low-altitude Dryden spectral shaping filter
        self._dryden_state: np.ndarray = np.zeros(3, dtype=float)
        self._dryden_aux: np.ndarray = np.zeros(3, dtype=float)
        self.scale_lengths: np.ndarray = np.array([15.0, 15.0, 7.5], dtype=float)
        self.mean_convective_speed: float = 8.0

        self.active_gusts: list[GustEvent] = []

        # Instantaneous values stored for telemetry plotting and 3D vector rendering
        self.current_wind_velocity: np.ndarray = np.zeros(3, dtype=float)
        self.current_drag_force: np.ndarray = np.zeros(3, dtype=float)
        self.current_wind_torque: np.ndarray = np.zeros(3, dtype=float)

    def reset(self) -> None:
        """Resets all stochastic Dryden filter states and clears active discrete gusts."""
        self._dryden_state[:] = 0.0
        self._dryden_aux[:] = 0.0
        self.active_gusts.clear()
        self.current_wind_velocity[:] = self.steady_wind.copy()
        self.current_drag_force[:] = 0.0
        self.current_wind_torque[:] = 0.0

    def set_random_wind(self, enabled: bool) -> None:
        """Enables or disables continuous stochastic Dryden wind turbulence."""
        self.random_wind_enabled = bool(enabled)
        if not self.random_wind_enabled:
            self._dryden_state[:] = 0.0
            self._dryden_aux[:] = 0.0

    def trigger_gust(
        self,
        axis: str = "X",
        amplitude: float = config.DEFAULT_GUST_SPEED,
        duration: float = config.DEFAULT_GUST_DURATION,
        profile: str = "step_decay",
    ) -> None:
        """
        Injects a discrete wind gust along the specified inertial axis ('X', 'Y', or 'Z').
        """
        axis_u = axis.upper().strip()
        direction = np.zeros(3, dtype=float)
        if axis_u == "X":
            direction[0] = 1.0
        elif axis_u == "Y":
            direction[1] = 1.0
        elif axis_u == "Z":
            direction[2] = -1.0  # Vertical downdraft disturbance
        else:
            direction = np.array([1.0, 0.5, 0.0], dtype=float)
            direction /= np.linalg.norm(direction)

        self.active_gusts.append(
            GustEvent(
                direction=direction,
                amplitude=float(amplitude),
                duration=float(duration),
                profile=profile,
            )
        )

    def _step_dryden_turbulence(self, dt: float) -> np.ndarray:
        """
        Advances the discrete-time Dryden stochastic turbulence differential equations by `dt`.
        """
        if not self.random_wind_enabled or dt <= 0.0:
            self._dryden_state *= np.exp(-dt / 0.25)
            self._dryden_aux *= np.exp(-dt / 0.25)
            return self._dryden_state.copy()

        omega_c = self.mean_convective_speed / self.scale_lengths
        sigma = np.array([1.0, 1.0, 0.50], dtype=float) * self.turbulence_intensity
        white_noise = self.rng.standard_normal(3) / np.sqrt(max(dt, 1e-5))

        alpha = np.exp(-omega_c * dt)
        gain = sigma * np.sqrt(2.0 * omega_c)
        self._dryden_aux = alpha * self._dryden_aux + (1.0 - alpha) * (gain / omega_c) * white_noise

        beta = np.exp(-dt / 0.18)
        self._dryden_state = beta * self._dryden_state + (1.0 - beta) * self._dryden_aux
        return self._dryden_state.copy()

    def update(self, dt: float = config.DT) -> np.ndarray:
        """
        Advances the atmospheric wind state by `dt` seconds and returns `v_wind = [vx, vy, vz]` (m/s).
        """
        v_turb = self._step_dryden_turbulence(dt)
        v_gust = np.zeros(3, dtype=float)
        if self.active_gusts:
            for gust in self.active_gusts:
                v_gust += gust.evaluate(dt)
            self.active_gusts = [g for g in self.active_gusts if g.is_active]

        self.current_wind_velocity = self.steady_wind + v_turb + v_gust
        return self.current_wind_velocity.copy()

    def compute_aerodynamic_effects(
        self,
        drone_velocity: np.ndarray,
        rotation_matrix: np.ndarray,
        wind_velocity: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the translational aerodynamic drag force F_d (in the inertial frame)
        and the induced aerodynamic disturbance moments tau_wind (in the body frame).
        """
        v_w = self.current_wind_velocity if wind_velocity is None else np.asarray(wind_velocity, dtype=float)
        v_rel = np.asarray(drone_velocity, dtype=float) - v_w

        # Quadratic form drag F_d = -1/2 * rho * C_d * A_eff * v_rel * |v_rel| + linear blade-flapping drag
        cd_vec = np.array([4.2, 4.2, 6.0], dtype=float) * self.drag_coeff
        quad_factor = 0.5 * self.rho * cd_vec * self.effective_area * 2.8
        lin_factor = np.array([1.1, 1.1, 1.6], dtype=float) * self.drag_coeff

        speed_rel = float(np.linalg.norm(v_rel))
        f_drag_quad = -quad_factor * (0.65 * v_rel * np.abs(v_rel) + 0.35 * v_rel * speed_rel)
        f_drag_lin = -lin_factor * v_rel
        f_drag_inertial = f_drag_quad + f_drag_lin

        # Project drag force into the body frame and compute overturning moment r_cp x F_drag_body
        f_drag_body = rotation_matrix.T @ f_drag_inertial
        tau_cp = np.cross(self.cp_offset, f_drag_body)

        v_rel_body = rotation_matrix.T @ v_rel
        tau_inflow = np.array(
            [
                0.0012 * v_rel_body[1] * abs(v_rel_body[1]),
                -0.0012 * v_rel_body[0] * abs(v_rel_body[0]),
                0.0003 * v_rel_body[0] * v_rel_body[1],
            ],
            dtype=float,
        )
        tau_wind_body = tau_cp + tau_inflow

        self.current_drag_force = f_drag_inertial
        self.current_wind_torque = tau_wind_body
        return f_drag_inertial, tau_wind_body


class QuadcopterModel:
    """
    Nonlinear 6-DOF Rigid-Body Dynamic Model of a Quadcopter UAV,
    numerically integrated via the explicit 4th-Order Runge-Kutta (RK4) method.

    Governing Newton-Euler Equations:
    ---------------------------------
    1. Translational Dynamics (Inertial ENU Frame):
        m * dv/dt = R(phi, theta, psi) * [0, 0, U1]^T - [0, 0, m*g]^T + F_drag

    2. Euler Angle Kinematics (Intrinsic Z-Y-X Convention):
        d(eta)/dt = W(phi, theta) * omega

    3. Rotational Rigid-Body Dynamics with Rotor Gyroscopic Precession (Body Frame):
        J * d(omega)/dt + omega x (J * omega) + tau_gyro = [tau_phi, tau_theta, tau_psi]^T + tau_wind - D_rot * omega
    """

    def __init__(
        self,
        wind: WindDisturbance | None = None,
        initial_state: np.ndarray | None = None,
    ) -> None:
        self.mass: float = float(config.MASS)
        self.gravity: float = float(config.GRAVITY)
        self.arm_length: float = float(config.ARM_LENGTH)
        self.kf: float = float(config.KF)
        self.km: float = float(config.KM)

        self.ixx: float = float(config.INERTIA_X)
        self.iyy: float = float(config.INERTIA_Y)
        self.izz: float = float(config.INERTIA_Z)
        self.J: np.ndarray = config.J.copy()
        self.J_inv: np.ndarray = np.linalg.inv(self.J)
        self.jr: float = float(config.ROTOR_INERTIA_JR)

        self.min_thrust: float = float(config.MIN_THRUST_PER_MOTOR)
        self.max_thrust: float = float(config.MAX_THRUST_PER_MOTOR)
        self.motor_tau: float = float(config.MOTOR_TIME_CONSTANT)
        self.rot_damping: np.ndarray = config.ROTATIONAL_DAMPING.copy()

        # Equivalent rotor angular velocity limits derived from thrust coefficient KF
        self.omega_min: float = float(np.sqrt(self.min_thrust / max(self.kf, 1e-12)))
        self.omega_max: float = float(np.sqrt(self.max_thrust / max(self.kf, 1e-12)))
        self.hover_thrust_total: float = self.mass * self.gravity
        self.hover_thrust_per_motor: float = self.hover_thrust_total / 4.0
        self.hover_omega: float = float(np.sqrt(self.hover_thrust_per_motor / max(self.kf, 1e-12)))
        self.hover_rpm: float = self.hover_omega * 60.0 / (2.0 * np.pi)

        self.wind: WindDisturbance = WindDisturbance() if wind is None else wind

        self.default_initial_state: np.ndarray = (
            config.INITIAL_STATE.copy()
            if initial_state is None
            else np.asarray(initial_state, dtype=float).copy()
        )
        self.state: np.ndarray = self.default_initial_state.copy()
        self.time: float = 0.0

        # X-Configuration geometry: moment arm projection at 45 degrees onto body X_b and Y_b axes
        self.arm_proj: float = self.arm_length / np.sqrt(2.0)
        self.c_m: float = self.km / max(self.kf, 1e-12)

        # Control Allocation Matrix M_alloc:
        # [U1, tau_phi, tau_theta, tau_psi]^T = M_alloc @ [F1, F2, F3, F4]^T
        # Rotor 1: Front-Left  (+x, +y), CW  (+yaw reactive torque)
        # Rotor 2: Front-Right (+x, -y), CCW (-yaw reactive torque)
        # Rotor 3: Rear-Right  (-x, -y), CW  (+yaw reactive torque)
        # Rotor 4: Rear-Left   (-x, +y), CCW (-yaw reactive torque)
        l = self.arm_proj
        cm = self.c_m
        self.allocation_matrix: np.ndarray = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [l, -l, -l, l],
                [-l, -l, l, l],
                [cm, -cm, cm, -cm],
            ],
            dtype=float,
        )
        self.mixing_matrix_inv: np.ndarray = np.linalg.inv(self.allocation_matrix)

        # Actuator states for the 4 brushless rotors
        self.motor_thrusts: np.ndarray = np.full(4, self.hover_thrust_per_motor, dtype=float)
        self.motor_omega: np.ndarray = np.full(4, self.hover_omega, dtype=float)
        self.motor_rpm: np.ndarray = np.full(4, self.hover_rpm, dtype=float)
        self.motor_pwm: np.ndarray = np.full(4, self._thrust_to_pwm(self.hover_thrust_per_motor), dtype=float)
        self.control_input: np.ndarray = np.array([self.hover_thrust_total, 0.0, 0.0, 0.0], dtype=float)

        self.last_drag_force: np.ndarray = np.zeros(3, dtype=float)
        self.last_wind_torque: np.ndarray = np.zeros(3, dtype=float)

    def _thrust_to_pwm(self, thrust: np.ndarray | float) -> np.ndarray | float:
        """Maps rotor thrust [N] and angular speed to normalized PWM effort percentage [0 - 100 %]."""
        omega = np.sqrt(np.maximum(thrust, 0.0) / max(self.kf, 1e-12))
        norm = (omega - self.omega_min) / max(self.omega_max - self.omega_min, 1e-6)
        return np.clip(norm * 100.0, 0.0, 100.0)

    def reset(self, initial_state: np.ndarray | None = None) -> np.ndarray:
        """Resets kinematic states, dynamic states, and rotor actuators to nominal hover equilibrium."""
        if initial_state is not None:
            self.state = np.asarray(initial_state, dtype=float).copy()
        else:
            self.state = self.default_initial_state.copy()

        self.time = 0.0
        self.motor_thrusts[:] = self.hover_thrust_per_motor
        self.motor_omega[:] = self.hover_omega
        self.motor_rpm[:] = self.hover_rpm
        self.motor_pwm[:] = self._thrust_to_pwm(self.hover_thrust_per_motor)
        self.control_input[:] = [self.hover_thrust_total, 0.0, 0.0, 0.0]
        self.last_drag_force[:] = 0.0
        self.last_wind_torque[:] = 0.0
        return self.state.copy()

    @staticmethod
    def rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
        """
        Computes the orthogonal rotation matrix R_B^I (from Body frame to Inertial frame)
        using intrinsic Z-Y-X Euler angles:
            R = R_z(psi) @ R_y(theta) @ R_x(phi)
        """
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_th, s_th = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)

        return np.array(
            [
                [
                    c_th * c_psi,
                    s_phi * s_th * c_psi - c_phi * s_psi,
                    c_phi * s_th * c_psi + s_phi * s_psi,
                ],
                [
                    c_th * s_psi,
                    s_phi * s_th * s_psi + c_phi * c_psi,
                    c_phi * s_th * s_psi - s_phi * c_psi,
                ],
                [
                    -s_th,
                    s_phi * c_th,
                    c_phi * c_th,
                ],
            ],
            dtype=float,
        )

    @staticmethod
    def euler_kinematics_matrix(phi: float, theta: float) -> np.ndarray:
        """
        Computes the Euler kinematic transformation matrix W(phi, theta) satisfying:
            [d(phi)/dt, d(theta)/dt, d(psi)/dt]^T = W(phi, theta) @ [p, q, r]^T
        """
        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_th = np.cos(theta)
        if abs(c_th) < 1e-4:
            c_th = 1e-4 if c_th >= 0.0 else -1e-4
        t_th = np.sin(theta) / c_th

        return np.array(
            [
                [1.0, s_phi * t_th, c_phi * t_th],
                [0.0, c_phi, -s_phi],
                [0.0, s_phi / c_th, c_phi / c_th],
            ],
            dtype=float,
        )

    def mix_motors(
        self,
        control_cmd: np.ndarray,
        dt: float = config.DT,
        input_is_motor_thrusts: bool = False,
    ) -> np.ndarray:
        """
        Motor Mixing Matrix and first-order brushless rotor actuator dynamics.

        Accepts either:
        - `input_is_motor_thrusts=False`: generalized wrench vector
          `u = [U1 (N), tau_phi (N*m), tau_theta (N*m), tau_psi (N*m)]^T`.
        - `input_is_motor_thrusts=True`: direct individual rotor thrust commands
          `F = [F1, F2, F3, F4]^T` in Newtons.

        Enforces per-rotor physical thrust saturation and returns the realized control wrench
        `u_actual = [U1, tau_phi, tau_theta, tau_psi]^T`.
        """
        cmd = np.asarray(control_cmd, dtype=float)
        if input_is_motor_thrusts:
            t_desired = cmd.copy()
        else:
            t_desired = self.mixing_matrix_inv @ cmd

        t_desired = np.clip(t_desired, self.min_thrust, self.max_thrust)
        omega_target = np.sqrt(t_desired / max(self.kf, 1e-12))

        if dt > 0.0 and self.motor_tau > 0.0:
            alpha = np.exp(-dt / self.motor_tau)
            self.motor_omega = alpha * self.motor_omega + (1.0 - alpha) * omega_target
        else:
            self.motor_omega = omega_target

        self.motor_omega = np.clip(self.motor_omega, self.omega_min, self.omega_max)
        self.motor_thrusts = self.kf * (self.motor_omega ** 2)
        self.motor_rpm = self.motor_omega * 60.0 / (2.0 * np.pi)
        self.motor_pwm = self._thrust_to_pwm(self.motor_thrusts)

        u_actual = self.allocation_matrix @ self.motor_thrusts
        self.control_input = u_actual
        return u_actual

    def derivatives(
        self,
        state: np.ndarray,
        u_actual: np.ndarray,
        wind_vel: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluates the continuous-time state derivative vector dx/dt = f(x, u, v_wind)
        governed by the 6-DOF nonlinear Newton-Euler equations of motion.
        """
        _x, _y, _z, vx, vy, vz, phi, theta, psi, p_rate, q_rate, r_rate = state

        phi_c = float(np.clip(phi, -1.35, 1.35))
        theta_c = float(np.clip(theta, -1.35, 1.35))

        rot = self.rotation_matrix(phi_c, theta_c, psi)
        vel_inertial = np.array([vx, vy, vz], dtype=float)
        f_drag, tau_wind = self.wind.compute_aerodynamic_effects(vel_inertial, rot, wind_vel)

        # 1. Translational kinematics
        dx, dy, dz = vx, vy, vz

        # 2. Translational dynamics (Newton's Second Law in Inertial Frame)
        u1, tau_phi, tau_theta, tau_psi = u_actual
        thrust_inertial = rot @ np.array([0.0, 0.0, u1], dtype=float)
        gravity_inertial = np.array([0.0, 0.0, -self.mass * self.gravity], dtype=float)
        accel = (thrust_inertial + gravity_inertial + f_drag) / self.mass

        # 3. Rotational Euler kinematics
        w_mat = self.euler_kinematics_matrix(phi_c, theta_c)
        omega_body = np.array([p_rate, q_rate, r_rate], dtype=float)
        d_euler = w_mat @ omega_body

        # 4. Rotational Euler dynamics + Rotor gyroscopic precession coupling
        # Net residual angular speed of the 4 counter-rotating propellers: Omega_r = +w1 - w2 + w3 - w4
        omega_r = float(
            self.motor_omega[0] - self.motor_omega[1] + self.motor_omega[2] - self.motor_omega[3]
        )
        tau_gyro = np.array([-self.jr * q_rate * omega_r, +self.jr * p_rate * omega_r, 0.0], dtype=float)

        tau_ctrl = np.array([tau_phi, tau_theta, tau_psi], dtype=float)
        coriolis = np.cross(omega_body, self.J @ omega_body)
        damping = self.rot_damping * omega_body

        d_omega = self.J_inv @ (tau_ctrl + tau_wind - coriolis - tau_gyro - damping)

        return np.array(
            [
                dx,
                dy,
                dz,
                accel[0],
                accel[1],
                accel[2],
                d_euler[0],
                d_euler[1],
                d_euler[2],
                d_omega[0],
                d_omega[1],
                d_omega[2],
            ],
            dtype=float,
        )

    def step(
        self,
        control_cmd: np.ndarray,
        dt: float = config.DT,
        wind_velocity: np.ndarray | None = None,
        input_is_motor_thrusts: bool = False,
    ) -> np.ndarray:
        """
        Advances the 6-DOF physical simulation by a fixed time step `dt` using the
        classical 4th-Order Runge-Kutta (RK4) numerical integration scheme:
            k1 = f(x_k, u_k)
            k2 = f(x_k + 0.5*dt*k1, u_k)
            k3 = f(x_k + 0.5*dt*k2, u_k)
            k4 = f(x_k + dt*k3, u_k)
            x_{k+1} = x_k + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        """
        if wind_velocity is None:
            v_wind = self.wind.update(dt)
        else:
            v_wind = np.asarray(wind_velocity, dtype=float)
            self.wind.current_wind_velocity = v_wind.copy()

        u_actual = self.mix_motors(
            control_cmd,
            dt=dt,
            input_is_motor_thrusts=input_is_motor_thrusts,
        )

        # Explicit 4th-Order Runge-Kutta (RK4) integration
        k1 = self.derivatives(self.state, u_actual, v_wind)
        k2 = self.derivatives(self.state + 0.5 * dt * k1, u_actual, v_wind)
        k3 = self.derivatives(self.state + 0.5 * dt * k2, u_actual, v_wind)
        k4 = self.derivatives(self.state + dt * k3, u_actual, v_wind)
        next_state = self.state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Evaluate aerodynamic forces at the updated state for exact telemetry reporting
        rot_final = self.rotation_matrix(next_state[6], next_state[7], next_state[8])
        f_drag, tau_wind = self.wind.compute_aerodynamic_effects(next_state[3:6], rot_final, v_wind)
        self.last_drag_force = f_drag
        self.last_wind_torque = tau_wind

        # Ground plane contact constraint (z >= 0)
        if next_state[2] < 0.0:
            next_state[2] = 0.0
            if next_state[5] < 0.0:
                next_state[5] = 0.0

        # Attitude bounds and yaw angle wrapping to [-pi, pi]
        next_state[6] = float(np.clip(next_state[6], -1.35, 1.35))
        next_state[7] = float(np.clip(next_state[7], -1.35, 1.35))
        next_state[8] = float((next_state[8] + np.pi) % (2.0 * np.pi) - np.pi)

        self.state = next_state
        self.time += dt
        return self.state.copy()

    @property
    def position(self) -> np.ndarray:
        """Returns current inertial position vector `[x, y, z]` in meters."""
        return self.state[0:3]

    @property
    def velocity(self) -> np.ndarray:
        """Returns current inertial linear velocity vector `[vx, vy, vz]` in m/s."""
        return self.state[3:6]

    @property
    def euler_angles(self) -> np.ndarray:
        """Returns current Euler attitude angles `[phi, theta, psi]` in radians."""
        return self.state[6:9]

    @property
    def euler_angles_deg(self) -> np.ndarray:
        """Returns current Euler attitude angles `[phi, theta, psi]` in degrees."""
        return np.degrees(self.state[6:9])

    @property
    def angular_rates(self) -> np.ndarray:
        """Returns current body-frame angular velocities `[p, q, r]` in rad/s."""
        return self.state[9:12]

    @property
    def total_motor_power_pct(self) -> float:
        """Returns mean actuator effort across all 4 rotors in PWM percentage [0 - 100 %]."""
        return float(np.mean(self.motor_pwm))


# Alias for backwards compatibility
QuadcopterSim = QuadcopterModel
