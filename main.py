"""
Main Entry Point — 6-DOF Quadcopter 3D Flight Control Simulation:
Real-Time Comparative Analysis of Cascade PID and Linear Quadratic Regulator (LQR) Controllers.

Repository Package Architecture:
- `config.py`                    : Physical constants, actuator limits, and initial tuning gains.
- `physics/quadcopter_model.py`  : 6-DOF Newton-Euler dynamics, RK4 solver, and Dryden/gust wind model.
- `controllers/pid_controller.py`: Hierarchical Cascade PID controller (Position -> Attitude -> Rotors).
- `controllers/lqr_controller.py`: State-space LQR controller (Continuous Algebraic Riccati Equation).
- `ui/main_window.py`            : PySide6 desktop interface, 3D viewport, custom icon, and PyQtGraph telemetry.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys


def _ensure_scientific_python_env() -> None:
    """
    Verifies that the active Python interpreter provides the required scientific stack
    (PySide6, SciPy, PyQtGraph, NumPy). If the `python` command invokes an environment
    lacking these packages (e.g., MSYS2) and the standard Python 3.10 installation is
    present on the host, transparently re-executes the process with Python 3.10.
    """
    try:
        import numpy  # noqa: F401
        import PySide6  # noqa: F401
        import pyqtgraph  # noqa: F401
        import scipy  # noqa: F401
    except ImportError:
        fallback_py = os.path.expandvars(
            r"%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
        )
        if os.path.isfile(fallback_py) and os.path.normcase(sys.executable) != os.path.normcase(
            fallback_py
        ):
            result = subprocess.run([fallback_py, *sys.argv], check=False)
            sys.exit(result.returncode)
        raise


_ensure_scientific_python_env()

from PySide6.QtWidgets import QApplication

import config
from controllers.lqr_controller import LQRController
from controllers.pid_controller import PIDController
from physics.quadcopter_model import QuadcopterModel, QuadcopterSim, WindDisturbance
from ui.main_window import MainWindow, Quadcopter3DViewer, create_drone_app_icon

__all__ = [
    "config",
    "QuadcopterModel",
    "QuadcopterSim",
    "WindDisturbance",
    "PIDController",
    "LQRController",
    "Quadcopter3DViewer",
    "MainWindow",
    "create_drone_app_icon",
    "main",
]


def _set_windows_taskbar_app_id() -> None:
    """
    Registers an explicit Application User Model ID (AppUserModelID) with the Windows
    Shell so the custom quadcopter icon is displayed in the Windows taskbar rather than
    being grouped under the generic python.exe executable icon.
    """
    if sys.platform == "win32":
        try:
            app_id = "uav.flightcontrol.simulator.6dof.v1"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception:
            pass


def main() -> int:
    """Initializes the Qt application, configures the custom drone icon, and launches MainWindow."""
    _set_windows_taskbar_app_id()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("UAV 3D Flight Control Simulation (6-DOF) — PID vs. LQR")
    app.setOrganizationName("Control Systems and Flight Dynamics Laboratory")

    drone_icon = create_drone_app_icon()
    app.setWindowIcon(drone_icon)

    window = MainWindow()
    window.setWindowIcon(drone_icon)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
