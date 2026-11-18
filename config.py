"""
Global system configuration for the 6-DOF Quadcopter 3D Flight Control Simulation (PID vs. LQR).
Defines physical parameters, actuator saturation bounds, aerodynamic constants, and initial tuning gains.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any
import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# --- Numerical Integration Parameters ---
DT: float = 0.01            # Fixed integration time step (s) -> 10 ms (100 Hz)
GRAVITY: float = 9.81       # Gravitational acceleration (m/s^2)
AIR_DENSITY: float = 1.225  # Sea-level air density rho (kg/m^3)

# --- Quadcopter Rigid-Body Physical Parameters ---
MASS: float = 1.0           # Total vehicle mass (kg)
ARM_LENGTH: float = 0.22    # Rotor hub distance to center of mass (m)
KF: float = 3.13e-5         # Rotor thrust coefficient (N / (rad/s)^2)
KM: float = 7.50e-7         # Rotor reactive torque coefficient (N*m / (rad/s)^2)

# Principal moments of inertia matrix (kg*m^2)
INERTIA_X: float = 0.012
INERTIA_Y: float = 0.012
INERTIA_Z: float = 0.020
J: np.ndarray = np.diag([INERTIA_X, INERTIA_Y, INERTIA_Z])

# Rotor-propeller rotational inertia for gyroscopic precession coupling (kg*m^2)
ROTOR_INERTIA_JR: float = 6.0e-5

# Translational and rotational aerodynamic damping parameters
DRAG_COEFF: float = 0.1
EFFECTIVE_AREA: np.ndarray = np.array([0.055, 0.055, 0.095], dtype=float)  # Projected area [Ax, Ay, Az] (m^2)
ROTATIONAL_DAMPING: np.ndarray = np.array([0.010, 0.010, 0.015], dtype=float)  # Damping [dp, dq, dr] (N*m*s/rad)

# --- Actuator Physical Saturation Limits (4 Brushless Rotors) ---
MIN_THRUST_PER_MOTOR: float = 0.15   # Minimum armed idle thrust per rotor (N)
MAX_THRUST_PER_MOTOR: float = 9.50   # Maximum thrust per rotor (N) -> 38.0 N total (~3.8x weight)
MOTOR_TIME_CONSTANT: float = 0.015   # First-order ESC/motor actuator time constant tau_m (s)
MAX_TILT_ANGLE_RAD: float = float(np.radians(35.0))  # Maximum commanded Roll/Pitch angle (rad)

# --- Default Initial State and Reference Setpoint ---
# 12D State Vector: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
INITIAL_STATE: np.ndarray = np.array(
    [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=float,
)
DEFAULT_SETPOINT: np.ndarray = np.array([0.0, 0.0, 2.0], dtype=float)
DEFAULT_YAW_REF: float = 0.0

# --- Default Atmospheric Disturbance Parameters ---
DEFAULT_GUST_SPEED: float = 10.5     # Peak velocity for discrete wind gusts (m/s)
DEFAULT_GUST_DURATION: float = 1.5   # Duration of discrete wind gust events (s)
DEFAULT_TURBULENCE_STD: float = 3.2  # Dryden continuous turbulence intensity sigma_w (m/s)

# --- Initial Tuning: Cascade PID Controller ---
PID_GAINS: dict[str, dict[str, float]] = {
    "pos_z": {"kp": 2.0, "ki": 0.1, "kd": 1.5},
    "pos_xy": {"kp": 1.0, "ki": 0.0, "kd": 0.8},
    "att_roll_pitch": {"kp": 10.0, "ki": 0.5, "kd": 4.0},
    "att_yaw": {"kp": 5.0, "ki": 0.0, "kd": 1.0},
}

# --- Initial Tuning: Linear Quadratic Regulator (LQR State-Space) ---
# Diagonal state penalty weights Q and control effort penalty weights R
# State order: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
Q_DIAG: list[float] = [
    10.0, 10.0, 50.0,    # Position (x, y, z)
    5.0, 5.0, 10.0,      # Linear velocities (vx, vy, vz)
    100.0, 100.0, 50.0,  # Euler attitude angles (roll, pitch, yaw)
    1.0, 1.0, 1.0,       # Body angular rates (p, q, r)
]

R_DIAG: list[float] = [1.0, 1.0, 1.0, 1.0]  # Control effort penalty across the 4 actuation channels


def load_system_parameters(yaml_path: str | Path | None = None) -> dict[str, Any]:
    """
    Loads and validates physical and control parameters.
    If a valid YAML configuration path is provided, overrides the default parameters.
    Any I/O, formatting, or domain validation exceptions are caught and logged,
    falling back safely to the nominal configuration.

    Parameters
    ----------
    yaml_path : str | Path | None
        Optional path to an external YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Validated dictionary containing simulation, physical, and controller parameters.
    """
    params: dict[str, Any] = {
        "dt": DT,
        "gravity": GRAVITY,
        "mass": MASS,
        "arm_length": ARM_LENGTH,
        "kf": KF,
        "km": KM,
        "inertia": (INERTIA_X, INERTIA_Y, INERTIA_Z),
        "drag_coeff": DRAG_COEFF,
        "pid_gains": copy.deepcopy(PID_GAINS),
        "q_diag": list(Q_DIAG),
        "r_diag": list(R_DIAG),
    }

    if yaml_path is None:
        return params

    try:
        path_obj = Path(yaml_path)
        if not path_obj.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path_obj}")
        if yaml is None:
            raise RuntimeError("PyYAML is not installed in the active Python environment.")

        with path_obj.open("r", encoding="utf-8") as fh:
            user_data = yaml.safe_load(fh)

        if not isinstance(user_data, dict):
            raise ValueError("Root YAML structure must be a key-value mapping.")

        if "mass" in user_data:
            m_val = float(user_data["mass"])
            if m_val <= 0.0:
                raise ValueError("Quadcopter mass must be strictly positive.")
            params["mass"] = m_val

        if "dt" in user_data:
            dt_val = float(user_data["dt"])
            if not (1e-4 <= dt_val <= 0.1):
                raise ValueError("Integration time step DT must lie within [0.0001, 0.1] seconds.")
            params["dt"] = dt_val

        if "pid_gains" in user_data and isinstance(user_data["pid_gains"], dict):
            for loop_key, loop_dict in user_data["pid_gains"].items():
                if loop_key in params["pid_gains"] and isinstance(loop_dict, dict):
                    for gain_k in ("kp", "ki", "kd"):
                        if gain_k in loop_dict:
                            params["pid_gains"][loop_key][gain_k] = max(0.0, float(loop_dict[gain_k]))

        if "q_diag" in user_data:
            q_list = [max(1e-4, float(v)) for v in user_data["q_diag"]]
            if len(q_list) != 12:
                raise ValueError("Q_DIAG must contain exactly 12 diagonal elements.")
            params["q_diag"] = q_list

        if "r_diag" in user_data:
            r_list = [max(1e-4, float(v)) for v in user_data["r_diag"]]
            if len(r_list) != 4:
                raise ValueError("R_DIAG must contain exactly 4 diagonal elements.")
            params["r_diag"] = r_list

    except Exception as exc:
        logger.warning(
            "Failed to load external configuration (%s). Reverting to nominal default parameters.",
            exc,
        )

    return params
