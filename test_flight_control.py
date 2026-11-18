"""
Unit and Integration Verification Suite for the UAV 3D Flight Control Simulation Repository.

Validates:
1. `config.py`: Parameter loading, domain validation, and exception handling for malformed/missing YAML files.
2. `physics/quadcopter_model.py`: 6-DOF Newton-Euler RK4 solver, hover equilibrium, rotor allocation,
   and atmospheric wind disturbance generation (`WindDisturbance`).
3. `controllers/pid_controller.py` and `controllers/lqr_controller.py`: Closed-loop 3D gust rejection,
   asymptotic setpoint convergence, and real-time Riccati (`solve_continuous_are`) hot-tuning.
4. `ui/main_window.py`: Custom drone application icon generation (`assets/drone_icon.ico` & `.png`),
   PySide6 `MainWindow` lifecycle, 3D perspective viewport rendering, and `pyqtgraph` telemetry curves.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

import config
from controllers.lqr_controller import LQRController
from controllers.pid_controller import PIDController
from physics.quadcopter_model import QuadcopterModel, WindDisturbance
from ui.main_window import MainWindow, create_drone_app_icon


def test_config_validation_and_exception_handling() -> None:
    """Verifies default parameter loading and graceful fallback when an invalid YAML path is supplied."""
    nominal = config.load_system_parameters()
    assert nominal["mass"] == config.MASS
    assert nominal["dt"] == config.DT
    assert len(nominal["q_diag"]) == 12
    assert len(nominal["r_diag"]) == 4

    fallback = config.load_system_parameters("nonexistent_config_file_12345.yaml")
    assert fallback["mass"] == config.MASS


def test_hover_equilibrium_rk4() -> None:
    """Under zero ambient wind and collective thrust U1 = m*g, the RK4 solver preserves exact hover equilibrium."""
    wind = WindDisturbance(seed=1)
    model = QuadcopterModel(wind=wind)

    u_hover = np.array([model.hover_thrust_total, 0.0, 0.0, 0.0], dtype=float)
    for _ in range(50):
        model.step(u_hover, dt=config.DT)

    assert np.allclose(model.position, [0.0, 0.0, 2.0], atol=1e-4)
    assert np.allclose(model.velocity, [0.0, 0.0, 0.0], atol=1e-4)
    assert np.allclose(model.euler_angles, [0.0, 0.0, 0.0], atol=1e-5)


def test_wind_disturbance_and_aerodynamic_drag() -> None:
    """Verifies discrete gust injection along X/Y, Dryden turbulence, and quadratic drag force F_d."""
    wind = WindDisturbance(seed=123)
    wind.trigger_gust("X", amplitude=11.0, duration=1.5)
    v_w = wind.update(0.1)
    assert v_w[0] > 4.0
    assert abs(v_w[1]) < 1e-6

    f_drag, tau_wind = wind.compute_aerodynamic_effects(np.zeros(3), np.eye(3), v_w)
    assert f_drag[0] > 0.4
    assert np.linalg.norm(tau_wind) > 0.0


def test_pid_and_lqr_closed_loop_gust_rejection() -> None:
    """Verifies that both Cascade PID and LQR reject 3D wind gusts and converge to the reference Setpoint."""
    wind = WindDisturbance(seed=7)
    sim_pid = QuadcopterModel(wind=wind)
    sim_lqr = QuadcopterModel(wind=wind)

    pid = PIDController()
    lqr = LQRController()
    setpoint = np.array([0.5, -0.4, 2.2], dtype=float)
    dt = config.DT

    wind.trigger_gust("X", amplitude=9.0, duration=1.2)
    wind.trigger_gust("Y", amplitude=7.0, duration=1.2)

    for _ in range(500):
        v_w = wind.update(dt)
        u_pid = pid.compute_control(sim_pid.state, setpoint, 0.0, dt)
        u_lqr = lqr.compute_control(sim_lqr.state, setpoint, 0.0, dt)
        sim_pid.step(u_pid, dt, wind_velocity=v_w)
        sim_lqr.step(u_lqr, dt, wind_velocity=v_w)

    err_pid = float(np.linalg.norm(sim_pid.position - setpoint))
    err_lqr = float(np.linalg.norm(sim_lqr.position - setpoint))

    assert err_pid < 0.25, f"Steady-state PID position error too large: {err_pid:.3f} m"
    assert err_lqr < 0.25, f"Steady-state LQR position error too large: {err_lqr:.3f} m"


def test_ui_main_window_icon_hot_tuning_and_rendering() -> None:
    """Verifies custom drone icon generation, MainWindow initialization, hot-tuning, and 3D rendering."""
    app = QApplication.instance() or QApplication([])

    icon = create_drone_app_icon()
    assert not icon.isNull()
    assets_dir = Path(__file__).resolve().parent / "assets"
    assert (assets_dir / "drone_icon.ico").is_file()
    assert (assets_dir / "drone_icon.png").is_file()

    window = MainWindow()
    assert not window.windowIcon().isNull()
    window.show()

    # Verify both collapsible side drawers start folded (collapsed) by default
    assert not window.telemetry_drawer.is_expanded
    assert window.telemetry_drawer.panelWidth == 0
    assert not window.control_drawer.is_expanded
    assert window.control_drawer.panelWidth == 0

    # Test expanding and collapsing drawers via toggle and set_expanded
    window.telemetry_drawer.set_expanded(True, animate=False)
    assert window.telemetry_drawer.is_expanded
    assert window.telemetry_drawer.panelWidth == window.telemetry_drawer.target_width

    window.control_drawer.set_expanded(True, animate=False)
    assert window.control_drawer.is_expanded
    assert window.control_drawer.panelWidth == window.control_drawer.target_width

    window.trigger_wind_gust("X")
    window.trigger_wind_gust("Y")
    window.btn_random_wind.setChecked(True)

    for _ in range(15):
        window._on_simulation_step()

    for mode_id in (0, 1, 2):
        window._on_mode_changed(mode_id)
        window.viewer_3d.grab()

    # Real-time hot-tuning of PID gains and LQR Q/R weighting matrices
    window.sl_kp_pos.set_value(2.2)
    window.sl_kp_att.set_value(12.0)
    window._update_pid_gains()
    assert np.isclose(window.pid_controller.kp_pos_xy, 2.2, atol=0.05)

    window.sl_q_pos.set_value(25.0)
    window.sl_r_weight.set_value(0.8)
    window._update_lqr_weights()
    assert np.isclose(window.lqr_controller.q_pos_xy, 25.0, atol=0.5)

    window.pause_simulation()
    assert not window.is_running
    window.reset_simulation()
    assert len(window.t_hist) == 1

    window.close()
    app.quit()


if __name__ == "__main__":
    for test_fn in (
        test_config_validation_and_exception_handling,
        test_hover_equilibrium_rk4,
        test_wind_disturbance_and_aerodynamic_drag,
        test_pid_and_lqr_closed_loop_gust_rejection,
        test_ui_main_window_icon_hot_tuning_and_rendering,
    ):
        print(f"Running {test_fn.__name__} ...", end=" ", flush=True)
        test_fn()
        print("PASSED")
    print("\nAll unit, control-theoretic, and GUI integration tests passed successfully.")
