"""
Graphical User Interface (PySide6), Full-Viewport 3D Perspective Visualization,
Animated Collapsible Side Drawers, and High-Frequency Telemetry (PyQtGraph).

Architecture:
- `create_drone_app_icon`: Generates and caches a multi-resolution vector quadcopter emblem
  (`assets/drone_icon.ico` and `assets/drone_icon.png`) for the window title bar and OS taskbar.
- `CollapsibleSideDrawer`: Animated slide-out side panel powered by `QPropertyAnimation`
  and `QEasingCurve.InOutQuad`. Starts collapsed by default at the window edge so the
  3D viewport occupies 100% of the central workspace on launch.
- `Quadcopter3DViewer`: Full-window interactive 60-FPS 3D perspective viewport rendering the
  6-DOF quadcopter airframe, counter-rotating propellers driven by actual rotor RPMs,
  individual rotor thrust vectors (T1..T4), atmospheric wind velocity/force vectors,
  3D flow particles, ground shadows, trajectory trails, and a quantitative HUD.
- `MainWindow`: Clean full-viewport engineering workstation hosting the 100% central 3D canvas
  with two animated collapsible side drawers:
  1. Left Drawer ("Live Telemetry"): Four real-time high-frequency `pyqtgraph` plots.
  2. Right Drawer ("Control Settings"): Simulation controls, controller selector, wind gust
     injection, and real-time hot-tuning sliders for PID gains and LQR Q/R matrices.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable
import numpy as np

from PySide6.QtCore import (
    QEasingCurve,
    QPointF,
    QPropertyAnimation,
    QRectF,
    Qt,
    QTimer,
    Property,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

import config
from controllers.lqr_controller import LQRController
from controllers.pid_controller import PIDController
from physics.quadcopter_model import QuadcopterModel, WindDisturbance


pg.setConfigOptions(antialias=True, background="#0B1120", foreground="#94A3B8")


def _render_drone_icon_pixmap(size: int = 256) -> QPixmap:
    """
    Renders a high-resolution vector quadcopter emblem onto a square QPixmap of dimension `size`.
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)

    s = float(size)
    c = s * 0.5

    # 1. Rounded-square dark aerospace badge background
    bg_rect = QRectF(s * 0.04, s * 0.04, s * 0.92, s * 0.92)
    bg_grad = QLinearGradient(0.0, 0.0, s, s)
    bg_grad.setColorAt(0.0, QColor("#0B132B"))
    bg_grad.setColorAt(0.55, QColor("#111C38"))
    bg_grad.setColorAt(1.0, QColor("#1E293B"))

    painter.setPen(QPen(QColor("#38BDF8"), max(1.5, s * 0.032)))
    painter.setBrush(bg_grad)
    painter.drawRoundedRect(bg_rect, s * 0.20, s * 0.20)

    # Subtle inner coordinate crosshair ring
    painter.setPen(QPen(QColor(56, 189, 248, 55), max(1.0, s * 0.012), Qt.DashLine))
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(QPointF(c, c), s * 0.34, s * 0.34)

    # 2. Quadcopter X-Configuration arms and rotor hubs
    arm_offset = s * 0.245
    rotor_radius = s * 0.135
    rotors = [
        (c - arm_offset, c - arm_offset, QColor("#00E5FF")),  # Front-Left (Cyan)
        (c + arm_offset, c - arm_offset, QColor("#00E5FF")),  # Front-Right (Cyan)
        (c + arm_offset, c + arm_offset, QColor("#FFAB00")),  # Rear-Right (Amber)
        (c - arm_offset, c + arm_offset, QColor("#FFAB00")),  # Rear-Left (Amber)
    ]

    arm_pen = QPen(QColor("#94A3B8"), max(2.0, s * 0.048), Qt.SolidLine, Qt.RoundCap)
    painter.setPen(arm_pen)
    for rx, ry, _ in rotors:
        painter.drawLine(QPointF(c, c), QPointF(rx, ry))

    for idx, (rx, ry, r_col) in enumerate(rotors):
        center_pt = QPointF(rx, ry)
        grad = QRadialGradient(center_pt, rotor_radius)
        c_inner = QColor(r_col)
        c_inner.setAlpha(115)
        c_outer = QColor(r_col)
        c_outer.setAlpha(25)
        grad.setColorAt(0.0, c_inner)
        grad.setColorAt(1.0, c_outer)

        painter.setPen(QPen(r_col, max(1.5, s * 0.024)))
        painter.setBrush(grad)
        painter.drawEllipse(center_pt, rotor_radius, rotor_radius)

        angle = math.radians(35.0 if idx % 2 == 0 else -35.0)
        b_len = rotor_radius * 0.80
        p1 = QPointF(rx - b_len * math.cos(angle), ry - b_len * math.sin(angle))
        p2 = QPointF(rx + b_len * math.cos(angle), ry + b_len * math.sin(angle))
        painter.setPen(QPen(QColor("#F8FAFC"), max(1.5, s * 0.026), Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(p1, p2)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#E2E8F0"))
        painter.drawEllipse(center_pt, max(1.5, s * 0.028), max(1.5, s * 0.028))

    # 3. Central avionics fuselage core
    core_w = s * 0.15
    core_h = s * 0.19
    core_rect = QRectF(c - core_w * 0.5, c - core_h * 0.5, core_w, core_h)
    painter.setPen(QPen(QColor("#38BDF8"), max(1.2, s * 0.022)))
    painter.setBrush(QColor("#0F172A"))
    painter.drawRoundedRect(core_rect, s * 0.04, s * 0.04)

    # Forward heading indicator chevron
    chevron = QPolygonF(
        [
            QPointF(c, c - s * 0.21),
            QPointF(c - s * 0.045, c - s * 0.12),
            QPointF(c + s * 0.045, c - s * 0.12),
        ]
    )
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor("#00E5FF"))
    painter.drawPolygon(chevron)

    painter.end()
    return pixmap


def create_drone_app_icon() -> QIcon:
    """
    Creates a multi-resolution QIcon for the quadcopter flight simulator and persists
    `assets/drone_icon.png` and `assets/drone_icon.ico` on disk so the OS displays
    the custom icon both in the window title bar and in the Windows taskbar.
    """
    assets_dir = Path(__file__).resolve().parent.parent / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    png_path = assets_dir / "drone_icon.png"
    ico_path = assets_dir / "drone_icon.ico"

    icon = QIcon()
    sizes = (16, 24, 32, 48, 64, 128, 256)
    pixmaps: dict[int, QPixmap] = {}

    for sz in sizes:
        pm = _render_drone_icon_pixmap(sz)
        pixmaps[sz] = pm
        icon.addPixmap(pm)

    try:
        pixmaps[256].save(str(png_path), "PNG")
        try:
            from PIL import Image

            with Image.open(png_path) as img:
                img.save(
                    ico_path,
                    format="ICO",
                    sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
                )
        except Exception:
            pixmaps[256].save(str(ico_path), "ICO")
    except Exception:
        pass

    if ico_path.is_file():
        file_icon = QIcon(str(ico_path))
        for sz, pm in pixmaps.items():
            file_icon.addPixmap(pm)
        return file_icon

    return icon


@dataclass
class FloatSliderRow:
    """Interactive QSlider row mapping integer ticks to a floating-point parameter range."""

    slider: QSlider
    value_label: QLabel
    min_val: float
    max_val: float
    decimals: int
    unit: str

    def get_value(self) -> float:
        ratio = self.slider.value() / 1000.0
        return self.min_val + ratio * (self.max_val - self.min_val)

    def set_value(self, val: float) -> None:
        clamped = max(self.min_val, min(self.max_val, val))
        ratio = (clamped - self.min_val) / max(self.max_val - self.min_val, 1e-9)
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(ratio * 1000.0)))
        self.slider.blockSignals(False)
        self.update_label(clamped)

    def update_label(self, val: float | None = None) -> None:
        v = self.get_value() if val is None else val
        suffix = f" {self.unit}" if self.unit else ""
        self.value_label.setText(f"{v:.{self.decimals}f}{suffix}")


class CollapsibleSideDrawer(QWidget):
    """
    Animated collapsible side drawer widget using `QPropertyAnimation` and `QEasingCurve.InOutQuad`.

    Starts collapsed (`panelWidth = 0`) by default at the window edge, exposing a subtle
    floating toggle tab. Clicking the tab smoothly expands or collapses the drawer without
    blocking the 100 Hz physics loop or the 60 FPS 3D viewport.
    """

    widthChanged = Signal(int)
    toggled = Signal(bool)

    def __init__(
        self,
        title: str,
        content_widget: QWidget,
        side: str = "right",
        target_width: int = 480,
        tooltip_text: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.title = title
        self.side = side.lower()
        self.target_width = int(target_width)
        self._is_expanded: bool = False
        self._current_width: int = 0

        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")

        root_hbox = QHBoxLayout(self)
        root_hbox.setContentsMargins(0, 0, 0, 0)
        root_hbox.setSpacing(0)

        # 1. Sliding content container frame (starts collapsed at width = 0)
        self.panel_frame = QFrame(self)
        self.panel_frame.setObjectName("DrawerPanelFrame")
        self.panel_frame.setMinimumWidth(0)
        self.panel_frame.setMaximumWidth(0)
        self.panel_frame.setStyleSheet(
            """
            QFrame#DrawerPanelFrame {
                background-color: rgba(13, 19, 33, 242);
                border: 1px solid rgba(56, 189, 248, 95);
                border-radius: 10px;
            }
            """
        )

        panel_vbox = QVBoxLayout(self.panel_frame)
        panel_vbox.setContentsMargins(8, 8, 8, 8)
        panel_vbox.setSpacing(6)

        # Drawer Header Bar with title and collapse button
        header_bar = QHBoxLayout()
        header_bar.setContentsMargins(6, 2, 6, 4)

        lbl_header = QLabel(self.title.upper())
        lbl_header.setFont(QFont("Segoe UI", 10, QFont.Bold))
        lbl_header.setStyleSheet("color: #38BDF8; letter-spacing: 0.8px; background: transparent;")

        btn_close = QPushButton("Collapse")
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.setToolTip(f"Collapse the {self.title} drawer back to the window edge.")
        btn_close.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(30, 41, 59, 200);
                color: #94A3B8;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #0284C7;
                color: #FFFFFF;
                border-color: #38BDF8;
            }
            """
        )
        btn_close.clicked.connect(lambda: self.set_expanded(False))

        header_bar.addWidget(lbl_header)
        header_bar.addStretch(1)
        header_bar.addWidget(btn_close)

        panel_vbox.addLayout(header_bar)
        panel_vbox.addWidget(content_widget, 1)

        # 2. Subtle floating edge handle tab button
        self.handle_container = QWidget(self)
        self.handle_container.setAttribute(Qt.WA_TranslucentBackground, True)
        self.handle_container.setStyleSheet("background: transparent;")
        handle_vbox = QVBoxLayout(self.handle_container)
        handle_vbox.setContentsMargins(0, 0, 0, 0)
        handle_vbox.addStretch(1)

        self.toggle_button = QPushButton(self)
        self.toggle_button.setCursor(Qt.PointingHandCursor)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        if tooltip_text:
            self.toggle_button.setToolTip(tooltip_text)
        self.toggle_button.clicked.connect(self.toggle)

        self._update_handle_Look()
        handle_vbox.addWidget(self.toggle_button)
        handle_vbox.addStretch(1)

        if self.side == "left":
            root_hbox.addWidget(self.panel_frame)
            root_hbox.addWidget(self.handle_container)
        else:
            root_hbox.addWidget(self.handle_container)
            root_hbox.addWidget(self.panel_frame)

        # 3. Smooth QPropertyAnimation with QEasingCurve.InOutQuad
        self.animation = QPropertyAnimation(self, b"panelWidth", self)
        self.animation.setDuration(280)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)

    def _get_panel_width(self) -> int:
        return self._current_width

    def _set_panel_width(self, width: int) -> None:
        w = max(0, int(width))
        self._current_width = w
        self.panel_frame.setMinimumWidth(w)
        self.panel_frame.setMaximumWidth(w)
        self.widthChanged.emit(w)

    panelWidth = Property(int, _get_panel_width, _set_panel_width)

    @property
    def is_expanded(self) -> bool:
        return self._is_expanded

    def _update_handle_Look(self) -> None:
        if self.side == "left":
            arrow = "<" if self._is_expanded else ">"
            txt = f"{self.title}  {arrow}"
            radius_css = "border-top-right-radius: 8px; border-bottom-right-radius: 8px; border-left: none;"
        else:
            arrow = ">" if self._is_expanded else "<"
            txt = f"{arrow}  {self.title}"
            radius_css = "border-top-left-radius: 8px; border-bottom-left-radius: 8px; border-right: none;"

        self.toggle_button.setText(txt)
        bg = "rgba(2, 132, 199, 235)" if self._is_expanded else "rgba(15, 23, 42, 225)"
        border_col = "#38BDF8" if self._is_expanded else "rgba(100, 116, 139, 145)"
        text_col = "#FFFFFF" if self._is_expanded else "#E2E8F0"

        self.toggle_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {bg};
                color: {text_col};
                border: 1px solid {border_col};
                {radius_css}
                padding: 12px 10px;
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.4px;
            }}
            QPushButton:hover {{
                background-color: rgba(14, 165, 233, 240);
                color: #FFFFFF;
                border-color: #7DD3FC;
            }}
            """
        )

    def toggle(self) -> None:
        """Toggles the drawer between its collapsed and expanded states with smooth animation."""
        self.set_expanded(not self._is_expanded)

    def set_expanded(self, expanded: bool, animate: bool = True) -> None:
        """Expands or collapses the drawer using `QPropertyAnimation` (`QEasingCurve.InOutQuad`)."""
        target_state = bool(expanded)
        self._is_expanded = target_state
        self.toggle_button.blockSignals(True)
        self.toggle_button.setChecked(target_state)
        self.toggle_button.blockSignals(False)
        self._update_handle_Look()

        end_w = self.target_width if target_state else 0
        self.animation.stop()
        if animate:
            self.animation.setStartValue(self._current_width)
            self.animation.setEndValue(end_w)
            self.animation.start()
        else:
            self._set_panel_width(end_w)

        self.toggled.emit(target_state)


class Quadcopter3DViewer(QWidget):
    """
    Full-Window Real-Time 3D Perspective Viewport for the 6-DOF Quadcopter Simulation.
    Occupies 100% of the central workspace and hosts the overlay layout for the collapsible drawers.

    Mouse Camera Controls:
    - Left-click + drag  : Orbit camera (Azimuth / Elevation).
    - Right-click + drag : Pan camera target in 3D space.
    - Scroll wheel       : Zoom in / out.
    """

    def __init__(
        self,
        sim_pid: QuadcopterModel,
        sim_lqr: QuadcopterModel,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.sim_pid = sim_pid
        self.sim_lqr = sim_lqr

        self.setMinimumSize(780, 520)
        self.setMouseTracking(True)

        self.mode: str = "COMPARISON"
        self.setpoint: np.ndarray = config.DEFAULT_SETPOINT.copy()
        self.setpoint_yaw: float = config.DEFAULT_YAW_REF

        self.follow_drone: bool = False
        self.split_comparison_view: bool = False
        self.show_vectors: bool = True

        # Dynamic HUD horizontal margins so the HUD cards stay inside the visible 3D corridor
        self.hud_left_margin: int = 16
        self.hud_right_margin: int = 16

        # Spherical 3D orbit camera parameters
        self.cam_azimuth: float = -42.0
        self.cam_elevation: float = 26.0
        self.cam_distance: float = 6.0
        self.cam_target: np.ndarray = np.array([0.0, 0.0, 1.75], dtype=float)
        self.fov_deg: float = 46.0

        self._last_mouse_pos: QPointF | None = None
        self._mouse_button: Qt.MouseButton = Qt.NoButton

        # Accumulated propeller phase angles [rad]
        self._prop_angles_pid: np.ndarray = np.array([0.0, 0.8, 1.6, 2.4], dtype=float)
        self._prop_angles_lqr: np.ndarray = np.array([0.4, 1.2, 2.0, 2.8], dtype=float)

        # 3D trajectory history buffers
        self.trail_pid: deque[np.ndarray] = deque(maxlen=280)
        self.trail_lqr: deque[np.ndarray] = deque(maxlen=280)

        # 3D atmospheric wind flow particles
        self._rng = np.random.default_rng(101)
        self._wind_particles: np.ndarray = self._rng.uniform(
            low=[-3.2, -3.2, 0.25],
            high=[3.2, 3.2, 3.8],
            size=(44, 3),
        )

        # Cumulative Root-Mean-Square Error (RMSE) accumulators
        self.rmse_pid_sq_sum: float = 0.0
        self.rmse_lqr_sq_sum: float = 0.0
        self.sample_count: int = 0

    def set_hud_left_offset(self, left_drawer_width: int) -> None:
        """Shifts the 3D HUD overlay cards smoothly as the left drawer expands or collapses."""
        self.hud_left_margin = 16 + int(left_drawer_width)
        self.update()

    def set_hud_right_offset(self, right_drawer_width: int) -> None:
        """Updates the right margin of the 3D HUD corridor as the right drawer expands or collapses."""
        self.hud_right_margin = 16 + int(right_drawer_width)
        self.update()

    def _on_toggle_follow(self, checked: bool) -> None:
        self.follow_drone = checked
        self.update()

    def _on_toggle_split(self, checked: bool) -> None:
        self.split_comparison_view = checked
        self.update()

    def set_isometric_view(self) -> None:
        self.cam_azimuth = -42.0
        self.cam_elevation = 26.0
        self.cam_distance = 6.0
        self.cam_target[:] = [0.0, 0.0, 1.75]
        self.update()

    def set_top_view(self) -> None:
        self.cam_azimuth = -90.0
        self.cam_elevation = 85.0
        self.cam_distance = 7.0
        self.cam_target[:] = [0.0, 0.0, 1.8]
        self.update()

    def set_side_view(self) -> None:
        self.cam_azimuth = -90.0
        self.cam_elevation = 6.0
        self.cam_distance = 6.0
        self.cam_target[:] = [0.0, 0.0, 1.85]
        self.update()

    def reset_trails(self) -> None:
        """Clears 3D trajectory trails and resets cumulative RMSE counters."""
        self.trail_pid.clear()
        self.trail_lqr.clear()
        self.rmse_pid_sq_sum = 0.0
        self.rmse_lqr_sq_sum = 0.0
        self.sample_count = 0
        self.update()

    def advance_visual_state(self, dt: float) -> None:
        """Updates propeller rotation angles, 3D trajectory trails, wind streamlines, and RMSE metrics."""
        signs = np.array([1.0, -1.0, 1.0, -1.0], dtype=float)
        self._prop_angles_pid = (
            self._prop_angles_pid + signs * self.sim_pid.motor_omega * dt * 0.35
        ) % (2.0 * np.pi)
        self._prop_angles_lqr = (
            self._prop_angles_lqr + signs * self.sim_lqr.motor_omega * dt * 0.35
        ) % (2.0 * np.pi)

        self.trail_pid.append(self.sim_pid.position.copy())
        self.trail_lqr.append(self.sim_lqr.position.copy())

        err_pid = float(np.linalg.norm(self.sim_pid.position - self.setpoint))
        err_lqr = float(np.linalg.norm(self.sim_lqr.position - self.setpoint))
        self.rmse_pid_sq_sum += err_pid * err_pid
        self.rmse_lqr_sq_sum += err_lqr * err_lqr
        self.sample_count += 1

        v_wind = self.sim_pid.wind.current_wind_velocity
        wind_speed = float(np.linalg.norm(v_wind))
        if wind_speed > 0.05:
            self._wind_particles += v_wind * dt * 0.75
            for axis, (low, high) in enumerate([(-3.2, 3.2), (-3.2, 3.2), (0.2, 3.8)]):
                span = high - low
                mask_low = self._wind_particles[:, axis] < low
                mask_high = self._wind_particles[:, axis] > high
                self._wind_particles[mask_low, axis] += span
                self._wind_particles[mask_high, axis] -= span

        if self.follow_drone:
            if self.mode == "PID":
                focus = self.sim_pid.position
            elif self.mode == "LQR":
                focus = self.sim_lqr.position
            else:
                focus = 0.5 * (self.sim_pid.position + self.sim_lqr.position)
            self.cam_target = 0.88 * self.cam_target + 0.12 * focus

        self.update()

    def _compute_camera_basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        az = math.radians(self.cam_azimuth)
        el = math.radians(self.cam_elevation)

        cx = self.cam_target[0] + self.cam_distance * math.cos(el) * math.cos(az)
        cy = self.cam_target[1] + self.cam_distance * math.cos(el) * math.sin(az)
        cz = self.cam_target[2] + self.cam_distance * math.sin(el)
        cam_pos = np.array([cx, cy, cz], dtype=float)

        forward = self.cam_target - cam_pos
        forward /= max(float(np.linalg.norm(forward)), 1e-6)

        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        right = np.cross(forward, world_up)
        norm_r = float(np.linalg.norm(right))
        if norm_r < 1e-5:
            world_up = np.array([0.0, 1.0, 0.0], dtype=float)
            right = np.cross(forward, world_up)
            norm_r = float(np.linalg.norm(right))
        right /= max(norm_r, 1e-6)
        up = np.cross(right, forward)

        return cam_pos, right, up, forward

    def _project_point(
        self,
        pt3d: np.ndarray,
        cam_pos: np.ndarray,
        right: np.ndarray,
        up: np.ndarray,
        forward: np.ndarray,
        width: float,
        height: float,
        focal: float,
    ) -> tuple[QPointF | None, float]:
        rel = pt3d - cam_pos
        z_cam = float(np.dot(rel, forward))
        if z_cam <= 0.15:
            return None, z_cam

        x_cam = float(np.dot(rel, right))
        y_cam = float(np.dot(rel, up))
        scale = focal / z_cam
        sx = width * 0.5 + x_cam * scale
        sy = height * 0.5 - y_cam * scale
        return QPointF(sx, sy), z_cam

    def _draw_line_3d(
        self,
        painter: QPainter,
        p1: np.ndarray,
        p2: np.ndarray,
        pen: QPen,
        cam_params: tuple,
    ) -> None:
        pt1, _ = self._project_point(p1, *cam_params)
        pt2, _ = self._project_point(p2, *cam_params)
        if pt1 is not None and pt2 is not None:
            painter.setPen(pen)
            painter.drawLine(pt1, pt2)

    def _draw_arrow_3d(
        self,
        painter: QPainter,
        start: np.ndarray,
        end: np.ndarray,
        color: QColor,
        width_px: float,
        cam_params: tuple,
        head_size: float = 7.5,
    ) -> None:
        p1, _ = self._project_point(start, *cam_params)
        p2, _ = self._project_point(end, *cam_params)
        if p1 is None or p2 is None:
            return

        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        length = math.hypot(dx, dy)
        if length < 2.0:
            return

        pen = QPen(color, width_px, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(p1, p2)

        angle = math.atan2(dy, dx)
        h_len = min(head_size, length * 0.45)
        a1 = QPointF(
            p2.x() - h_len * math.cos(angle - math.pi / 6.0),
            p2.y() - h_len * math.sin(angle - math.pi / 6.0),
        )
        a2 = QPointF(
            p2.x() - h_len * math.cos(angle + math.pi / 6.0),
            p2.y() - h_len * math.sin(angle + math.pi / 6.0),
        )
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([p2, a1, a2]))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        w = float(self.width())
        h = float(self.height())

        bg_grad = QLinearGradient(0.0, 0.0, 0.0, h)
        bg_grad.setColorAt(0.0, QColor("#070B12"))
        bg_grad.setColorAt(0.55, QColor("#0D1526"))
        bg_grad.setColorAt(1.0, QColor("#111B30"))
        painter.fillRect(QRectF(0.0, 0.0, w, h), bg_grad)

        focal = (min(w, h) * 0.5) / math.tan(math.radians(self.fov_deg * 0.5))
        cam_pos, right, up, forward = self._compute_camera_basis()
        cam_params = (cam_pos, right, up, forward, w, h, focal)

        self._draw_ground_grid(painter, cam_params)
        self._draw_wind_field(painter, cam_params)
        self._draw_setpoint_marker(painter, cam_params)

        drones_to_draw: list[
            tuple[QuadcopterModel, deque[np.ndarray], np.ndarray, QColor, str, np.ndarray]
        ] = []
        offset_y = 0.85 if (self.mode == "COMPARISON" and self.split_comparison_view) else 0.0

        if self.mode in ("PID", "COMPARISON"):
            drones_to_draw.append(
                (
                    self.sim_pid,
                    self.trail_pid,
                    self._prop_angles_pid,
                    QColor("#00E5FF"),
                    "PID",
                    np.array([0.0, -offset_y, 0.0], dtype=float),
                )
            )
        if self.mode in ("LQR", "COMPARISON"):
            drones_to_draw.append(
                (
                    self.sim_lqr,
                    self.trail_lqr,
                    self._prop_angles_lqr,
                    QColor("#FFAB00"),
                    "LQR",
                    np.array([0.0, +offset_y, 0.0], dtype=float),
                )
            )

        drones_to_draw.sort(
            key=lambda item: float(np.linalg.norm((item[0].position + item[5]) - cam_pos)),
            reverse=True,
        )

        for sim, trail, prop_angles, accent_color, label, visual_offset in drones_to_draw:
            self._draw_drone_shadow(painter, sim, visual_offset, cam_params)
            self._draw_trajectory_trail(painter, trail, visual_offset, accent_color, cam_params)
            self._draw_quadcopter_3d(
                painter, sim, prop_angles, accent_color, label, visual_offset, cam_params
            )

        self._draw_hud_overlay(painter, w, h)
        painter.end()

    def _draw_ground_grid(self, painter: QPainter, cam_params: tuple) -> None:
        grid_extent = 3.5
        step = 0.5
        vals = np.arange(-grid_extent, grid_extent + 1e-5, step)

        minor_pen = QPen(QColor(51, 65, 85, 110), 1.0)
        major_pen = QPen(QColor(71, 85, 105, 175), 1.3)

        for v in vals:
            is_major = abs(v - round(v)) < 1e-4
            pen = major_pen if is_major else minor_pen
            self._draw_line_3d(
                painter,
                np.array([v, -grid_extent, 0.0]),
                np.array([v, grid_extent, 0.0]),
                pen,
                cam_params,
            )
            self._draw_line_3d(
                painter,
                np.array([-grid_extent, v, 0.0]),
                np.array([grid_extent, v, 0.0]),
                pen,
                cam_params,
            )

        pad_pts = []
        for ang in np.linspace(0.0, 2.0 * np.pi, 36):
            pt, _ = self._project_point(
                np.array([0.45 * np.cos(ang), 0.45 * np.sin(ang), 0.002]), *cam_params
            )
            if pt is not None:
                pad_pts.append(pt)
        if len(pad_pts) > 3:
            painter.setPen(QPen(QColor(56, 189, 248, 120), 1.5, Qt.DashLine))
            painter.setBrush(QColor(14, 165, 233, 22))
            painter.drawPolygon(QPolygonF(pad_pts))

        self._draw_arrow_3d(
            painter,
            np.array([0.0, 0.0, 0.005]),
            np.array([1.2, 0.0, 0.005]),
            QColor("#EF4444"),
            2.2,
            cam_params,
        )
        self._draw_arrow_3d(
            painter,
            np.array([0.0, 0.0, 0.005]),
            np.array([0.0, 1.2, 0.005]),
            QColor("#22C55E"),
            2.2,
            cam_params,
        )
        self._draw_arrow_3d(
            painter,
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            QColor("#3B82F6"),
            2.2,
            cam_params,
        )

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        for pos, text, col in [
            (np.array([1.32, 0.0, 0.02]), "+X (East)", QColor("#F87171")),
            (np.array([0.0, 1.32, 0.02]), "+Y (North)", QColor("#4ADE80")),
            (np.array([0.0, 0.0, 1.12]), "+Z (Alt)", QColor("#60A5FA")),
        ]:
            pt, _ = self._project_point(pos, *cam_params)
            if pt is not None:
                painter.setPen(col)
                painter.drawText(pt, text)

    def _draw_wind_field(self, painter: QPainter, cam_params: tuple) -> None:
        v_wind = self.sim_pid.wind.current_wind_velocity
        speed = float(np.linalg.norm(v_wind))
        if speed < 0.15:
            return

        wind_dir = v_wind / speed
        streak_len = min(0.18 + 0.045 * speed, 0.65)
        alpha = int(min(70 + speed * 16, 215))
        wind_color = QColor(56, 189, 248, alpha)

        for p in self._wind_particles:
            p_end = p + wind_dir * streak_len
            self._draw_arrow_3d(painter, p, p_end, wind_color, 1.3, cam_params, head_size=4.5)

    def _draw_setpoint_marker(self, painter: QPainter, cam_params: tuple) -> None:
        sp = self.setpoint
        ground_sp = np.array([sp[0], sp[1], 0.0], dtype=float)

        pillar_pen = QPen(QColor(16, 185, 129, 130), 1.4, Qt.DashLine)
        self._draw_line_3d(painter, ground_sp, sp, pillar_pen, cam_params)

        cross_pen = QPen(QColor(52, 211, 153, 190), 1.6)
        r = 0.22
        for d in [np.array([r, 0, 0]), np.array([0, r, 0]), np.array([0, 0, r])]:
            self._draw_line_3d(painter, sp - d, sp + d, cross_pen, cam_params)

        pt_sp, z_depth = self._project_point(sp, *cam_params)
        if pt_sp is not None:
            rad = max(5.0, min(14.0, 32.0 / max(z_depth, 1.0)))
            grad = QRadialGradient(pt_sp, rad * 2.0)
            grad.setColorAt(0.0, QColor(16, 185, 129, 220))
            grad.setColorAt(0.5, QColor(16, 185, 129, 95))
            grad.setColorAt(1.0, QColor(16, 185, 129, 0))
            painter.setPen(Qt.NoPen)
            painter.setBrush(grad)
            painter.drawEllipse(pt_sp, rad * 2.0, rad * 2.0)

            painter.setPen(QColor("#A7F3D0"))
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.drawText(
                QPointF(pt_sp.x() + 10.0, pt_sp.y() - 8.0),
                f"Setpoint ({sp[0]:.1f}, {sp[1]:.1f}, {sp[2]:.1f} m)",
            )

    def _draw_drone_shadow(
        self,
        painter: QPainter,
        sim: QuadcopterModel,
        visual_offset: np.ndarray,
        cam_params: tuple,
    ) -> None:
        pos = sim.position + visual_offset
        phi, theta, psi = sim.euler_angles
        rot = sim.rotation_matrix(phi, theta, psi)

        l = sim.arm_proj * 1.35
        rotor_body = [
            np.array([+l, +l, 0.0]),
            np.array([+l, -l, 0.0]),
            np.array([-l, -l, 0.0]),
            np.array([-l, +l, 0.0]),
        ]

        shadow_center = np.array([pos[0], pos[1], 0.003], dtype=float)
        shadow_pen = QPen(QColor(2, 6, 23, 125), 4.0, Qt.SolidLine, Qt.RoundCap)

        for rb in rotor_body:
            rw = pos + rot @ rb
            rs = np.array([rw[0], rw[1], 0.003], dtype=float)
            self._draw_line_3d(painter, shadow_center, rs, shadow_pen, cam_params)
            pt_s, z_d = self._project_point(rs, *cam_params)
            if pt_s is not None:
                rad = max(4.0, 22.0 / max(z_d, 1.0))
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(2, 6, 23, 110))
                painter.drawEllipse(pt_s, rad, rad * 0.65)

    def _draw_trajectory_trail(
        self,
        painter: QPainter,
        trail: deque[np.ndarray],
        visual_offset: np.ndarray,
        color: QColor,
        cam_params: tuple,
    ) -> None:
        if len(trail) < 2:
            return

        pts = list(trail)
        n = len(pts)
        step = max(1, n // 90)
        for i in range(step, n, step):
            p1 = pts[i - step] + visual_offset
            p2 = pts[i] + visual_offset
            frac = i / n
            c = QColor(color)
            c.setAlpha(int(25 + 190 * (frac ** 1.5)))
            pen = QPen(c, 1.4 + 1.4 * frac, Qt.SolidLine, Qt.RoundCap)
            self._draw_line_3d(painter, p1, p2, pen, cam_params)

    def _draw_quadcopter_3d(
        self,
        painter: QPainter,
        sim: QuadcopterModel,
        prop_angles: np.ndarray,
        accent_color: QColor,
        label: str,
        visual_offset: np.ndarray,
        cam_params: tuple,
    ) -> None:
        pos = sim.position + visual_offset
        phi, theta, psi = sim.euler_angles
        rot = sim.rotation_matrix(phi, theta, psi)

        def to_world(local_pt: np.ndarray) -> np.ndarray:
            return pos + rot @ local_pt

        vis_scale = 1.45
        l = sim.arm_proj * vis_scale
        prop_radius = 0.135 * vis_scale

        motor_locals = [
            np.array([+l, +l, 0.0]),
            np.array([+l, -l, 0.0]),
            np.array([-l, -l, 0.0]),
            np.array([-l, +l, 0.0]),
        ]

        skid_pen = QPen(QColor("#64748B"), 2.0, Qt.SolidLine, Qt.RoundCap)
        for ml in motor_locals:
            top_pt = to_world(ml * 0.72)
            bot_pt = to_world(ml * 0.78 + np.array([0.0, 0.0, -0.09 * vis_scale]))
            self._draw_line_3d(painter, top_pt, bot_pt, skid_pen, cam_params)

        rear_arm_pen = QPen(QColor("#94A3B8"), 4.2, Qt.SolidLine, Qt.RoundCap)
        front_arm_pen = QPen(accent_color, 4.8, Qt.SolidLine, Qt.RoundCap)

        for idx, ml in enumerate(motor_locals):
            pen = front_arm_pen if idx in (0, 1) else rear_arm_pen
            self._draw_line_3d(painter, pos, to_world(ml), pen, cam_params)

        bx, by, bz = 0.11 * vis_scale, 0.085 * vis_scale, 0.042 * vis_scale
        verts_local = np.array(
            [
                [-bx, -by, -bz],
                [+bx, -by, -bz],
                [+bx, +by, -bz],
                [-bx, +by, -bz],
                [-bx, -by, +bz],
                [+bx, -by, +bz],
                [+bx, +by, +bz],
                [-bx, +by, +bz],
            ],
            dtype=float,
        )
        verts_world = [to_world(v) for v in verts_local]
        faces = [
            ([0, 1, 2, 3], QColor("#1E293B")),
            ([4, 5, 6, 7], accent_color.darker(145)),
            ([1, 2, 6, 5], accent_color),
            ([0, 3, 7, 4], QColor("#334155")),
            ([2, 3, 7, 6], QColor("#475569")),
            ([0, 1, 5, 4], QColor("#475569")),
        ]

        cam_pos = cam_params[0]
        faces_sorted = sorted(
            faces,
            key=lambda f: float(
                np.linalg.norm(np.mean([verts_world[idx] for idx in f[0]], axis=0) - cam_pos)
            ),
            reverse=True,
        )

        painter.setPen(QPen(QColor("#0F172A"), 1.1))
        for idxs, f_color in faces_sorted:
            poly = []
            for i in idxs:
                pt, _ = self._project_point(verts_world[i], *cam_params)
                if pt is not None:
                    poly.append(pt)
            if len(poly) == 4:
                c = QColor(f_color)
                c.setAlpha(235)
                painter.setBrush(c)
                painter.drawPolygon(QPolygonF(poly))

        nose_start = to_world(np.array([bx, 0.0, 0.0]))
        nose_end = to_world(np.array([bx + 0.22 * vis_scale, 0.0, 0.0]))
        self._draw_arrow_3d(
            painter, nose_start, nose_end, accent_color.lighter(130), 2.4, cam_params
        )

        normal_world = rot @ np.array([0.0, 0.0, 1.0])
        hover_t = sim.hover_thrust_per_motor

        for idx, ml in enumerate(motor_locals):
            hub_base = to_world(ml)
            hub_top = to_world(ml + np.array([0.0, 0.0, 0.032 * vis_scale]))

            self._draw_line_3d(
                painter,
                hub_base,
                hub_top,
                QPen(QColor("#E2E8F0"), 4.5, Qt.SolidLine, Qt.RoundCap),
                cam_params,
            )

            ring_pts = []
            for ang in np.linspace(0.0, 2.0 * np.pi, 22):
                p_local = ml + np.array(
                    [prop_radius * np.cos(ang), prop_radius * np.sin(ang), 0.035 * vis_scale]
                )
                pt2d, _ = self._project_point(to_world(p_local), *cam_params)
                if pt2d is not None:
                    ring_pts.append(pt2d)

            if len(ring_pts) > 4:
                disc_color = QColor(accent_color)
                disc_color.setAlpha(45)
                ring_pen_col = QColor(accent_color)
                ring_pen_col.setAlpha(155)
                painter.setPen(QPen(ring_pen_col, 1.2))
                painter.setBrush(disc_color)
                painter.drawPolygon(QPolygonF(ring_pts))

            ang0 = float(prop_angles[idx])
            for blade_offset in (0.0, np.pi * 0.5):
                a = ang0 + blade_offset
                b1 = to_world(
                    ml
                    + np.array(
                        [prop_radius * np.cos(a), prop_radius * np.sin(a), 0.036 * vis_scale]
                    )
                )
                b2 = to_world(
                    ml
                    + np.array(
                        [-prop_radius * np.cos(a), -prop_radius * np.sin(a), 0.036 * vis_scale]
                    )
                )
                blade_pen = QPen(
                    QColor(248, 250, 252, 220 if blade_offset == 0.0 else 110),
                    2.4 if blade_offset == 0.0 else 1.4,
                    Qt.SolidLine,
                    Qt.RoundCap,
                )
                self._draw_line_3d(painter, b1, b2, blade_pen, cam_params)

            if self.show_vectors:
                thrust_i = float(sim.motor_thrusts[idx])
                ratio = thrust_i / max(hover_t, 1e-6)
                vec_len = float(np.clip(0.28 * ratio, 0.12, 0.72))
                t_end = hub_top + normal_world * vec_len

                if ratio <= 1.15:
                    t_col = QColor("#10B981")
                elif ratio <= 1.45:
                    t_col = QColor("#F59E0B")
                else:
                    t_col = QColor("#EC4899")

                self._draw_arrow_3d(painter, hub_top, t_end, t_col, 2.2, cam_params, head_size=6.0)

        v_wind = sim.wind.current_wind_velocity
        wind_speed = float(np.linalg.norm(v_wind))
        if self.show_vectors and wind_speed > 0.15:
            wind_dir = v_wind / wind_speed
            wind_vec_len = min(0.35 + 0.08 * wind_speed, 1.25)
            w_start = pos - wind_dir * (wind_vec_len + 0.25)
            w_end = pos - wind_dir * 0.22
            self._draw_arrow_3d(
                painter, w_start, w_end, QColor("#38BDF8"), 3.2, cam_params, head_size=8.5
            )
            pt_w, _ = self._project_point(w_start, *cam_params)
            if pt_w is not None:
                painter.setPen(QColor("#7DD3FC"))
                painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                f_mag = float(np.linalg.norm(sim.last_drag_force))
                painter.drawText(
                    QPointF(pt_w.x() + 6.0, pt_w.y() - 6.0),
                    f"Wind {wind_speed:.1f} m/s ({f_mag:.1f} N)",
                )

        z_lbl_offset = 0.36 if label == "PID" else 0.47
        pt_label, _ = self._project_point(
            pos + np.array([0.0, 0.0, z_lbl_offset]), *cam_params
        )
        if pt_label is not None:
            err_norm = float(np.linalg.norm(sim.position - self.setpoint))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.setPen(accent_color)
            painter.drawText(
                QPointF(pt_label.x() - 36.0, pt_label.y()),
                f"{label} | |e|={err_norm:.2f} m",
            )

    def _draw_hud_overlay(self, painter: QPainter, w: float, h: float) -> None:
        x0 = float(self.hud_left_margin)
        avail_w = w - float(self.hud_left_margin) - float(self.hud_right_margin)
        if avail_w < 240.0:
            return

        top_w = min(345.0, avail_w)
        card_top = QRectF(x0, 52.0, top_w, 88.0)
        painter.setPen(QPen(QColor(71, 85, 105, 150), 1.0))
        painter.setBrush(QColor(15, 23, 42, 195))
        painter.drawRoundedRect(card_top, 8.0, 8.0)

        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor("#F8FAFC"))
        mode_titles = {
            "PID": "ACTIVE CONTROLLER: CASCADE PID",
            "LQR": "ACTIVE CONTROLLER: OPTIMAL LQR",
            "COMPARISON": "LIVE COMPARISON: CASCADE PID vs. LQR",
        }
        painter.drawText(QPointF(x0 + 10.0, 72.0), mode_titles.get(self.mode, self.mode))

        painter.setFont(QFont("Segoe UI", 8, QFont.Normal))
        painter.setPen(QColor("#94A3B8"))
        painter.drawText(
            QPointF(x0 + 10.0, 90.0),
            f"Sim Time: {self.sim_pid.time:6.2f} s  |  RK4 Step: dt = 10 ms (100 Hz)",
        )

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor("#00E5FF"))
        painter.drawText(QPointF(x0 + 10.0, 110.0), "[PID UAV]")
        painter.setPen(QColor("#FFAB00"))
        painter.drawText(QPointF(x0 + 83.0, 110.0), "[LQR UAV]")
        painter.setPen(QColor("#10B981"))
        painter.drawText(QPointF(x0 + 160.0, 110.0), "Thrust T_i")
        painter.setPen(QColor("#38BDF8"))
        painter.drawText(QPointF(x0 + 243.0, 110.0), "Wind Vector")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#64748B"))
        painter.drawText(
            QPointF(x0 + 10.0, 128.0),
            "Mouse: Left=Orbit 3D | Right=Pan | Wheel=Zoom",
        )

        bot_w = min(495.0, avail_w)
        card_bot = QRectF(x0, h - 108.0, bot_w, 92.0)
        painter.setPen(QPen(QColor(71, 85, 105, 150), 1.0))
        painter.setBrush(QColor(15, 23, 42, 195))
        painter.drawRoundedRect(card_bot, 8.0, 8.0)

        n_samples = max(self.sample_count, 1)
        rmse_pid = math.sqrt(self.rmse_pid_sq_sum / n_samples)
        rmse_lqr = math.sqrt(self.rmse_lqr_sq_sum / n_samples)

        err_pid = float(np.linalg.norm(self.sim_pid.position - self.setpoint))
        err_lqr = float(np.linalg.norm(self.sim_lqr.position - self.setpoint))
        ang_pid = self.sim_pid.euler_angles_deg
        ang_lqr = self.sim_lqr.euler_angles_deg

        painter.setFont(QFont("Consolas", 8, QFont.Bold))
        painter.setPen(QColor("#CBD5E1"))
        painter.drawText(
            QPointF(x0 + 10.0, h - 88.0),
            "CONTROLLER    |e_pos|    3D RMSE    Roll(phi)  Pitch(th)  Motor PWM",
        )

        painter.setPen(QColor("#00E5FF"))
        painter.drawText(
            QPointF(x0 + 10.0, h - 66.0),
            f"Cascade PID   {err_pid:6.3f}m    {rmse_pid:6.3f}m   {ang_pid[0]:+6.1f} deg  {ang_pid[1]:+6.1f} deg     {self.sim_pid.total_motor_power_pct:5.1f}%",
        )

        painter.setPen(QColor("#FFAB00"))
        painter.drawText(
            QPointF(x0 + 10.0, h - 44.0),
            f"Optimal LQR   {err_lqr:6.3f}m    {rmse_lqr:6.3f}m   {ang_lqr[0]:+6.1f} deg  {ang_lqr[1]:+6.1f} deg     {self.sim_lqr.total_motor_power_pct:5.1f}%",
        )

        v_w = self.sim_pid.wind.current_wind_velocity
        w_mag = float(np.linalg.norm(v_w))
        painter.setPen(QColor("#38BDF8") if w_mag > 0.2 else QColor("#94A3B8"))
        painter.drawText(
            QPointF(x0 + 10.0, h - 24.0),
            f"Ambient Wind: [{v_w[0]:+5.1f}, {v_w[1]:+5.1f}, {v_w[2]:+5.1f}] m/s  (|V_w| = {w_mag:.2f} m/s)",
        )

    def mousePressEvent(self, event) -> None:
        self._last_mouse_pos = event.position()
        self._mouse_button = event.button()

    def mouseMoveEvent(self, event) -> None:
        if self._last_mouse_pos is None:
            return
        pos = event.position()
        dx = pos.x() - self._last_mouse_pos.x()
        dy = pos.y() - self._last_mouse_pos.y()
        self._last_mouse_pos = pos

        if self._mouse_button == Qt.LeftButton:
            self.cam_azimuth = (self.cam_azimuth - dx * 0.45) % 360.0
            self.cam_elevation = float(np.clip(self.cam_elevation + dy * 0.35, -10.0, 88.0))
            self.update()
        elif self._mouse_button in (Qt.RightButton, Qt.MiddleButton):
            _, right, up, _ = self._compute_camera_basis()
            pan_speed = self.cam_distance * 0.0018
            self.cam_target -= right * (dx * pan_speed)
            self.cam_target += up * (dy * pan_speed)
            self.update()

    def mouseReleaseEvent(self, _event) -> None:
        self._last_mouse_pos = None
        self._mouse_button = Qt.NoButton

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        factor = 0.88 if delta > 0 else 1.13
        self.cam_distance = float(np.clip(self.cam_distance * factor, 2.0, 18.0))
        self.update()


class MainWindow(QMainWindow):
    """
    Primary Desktop Window for the 6-DOF Quadcopter Flight Control Simulator (Cascade PID vs. LQR).

    Features a 100% full-window 3D perspective viewport as the central element, with two
    animated collapsible side drawers ("Live Telemetry" on the left and "Control Settings"
    on the right) that start collapsed by default and expand smoothly via `QPropertyAnimation`
    (`QEasingCurve.InOutQuad`).
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            "UAV 3D Flight Control Simulation (6-DOF) — Cascade PID vs. LQR under Wind Disturbances"
        )
        self.setWindowIcon(create_drone_app_icon())
        self.resize(1540, 920)
        self.setMinimumSize(1160, 720)

        sys_params = config.load_system_parameters()
        self.dt: float = float(sys_params["dt"])

        self.wind = WindDisturbance(seed=42)
        self.sim_pid = QuadcopterModel(wind=self.wind)
        self.sim_lqr = QuadcopterModel(wind=self.wind)

        self.pid_controller = PIDController(gains=sys_params["pid_gains"])
        self.lqr_controller = LQRController(
            q_diag=sys_params["q_diag"],
            r_diag=sys_params["r_diag"],
        )

        self.setpoint_pos: np.ndarray = config.DEFAULT_SETPOINT.copy()
        self.setpoint_yaw: float = config.DEFAULT_YAW_REF
        self.gust_amplitude: float = config.DEFAULT_GUST_SPEED

        self.active_mode: str = "COMPARISON"
        self.is_running: bool = False

        # Circular telemetry history buffers for PyQtGraph
        self.history_len: int = 800
        self.t_hist: deque[float] = deque(maxlen=self.history_len)

        self.err_x_pid: deque[float] = deque(maxlen=self.history_len)
        self.err_y_pid: deque[float] = deque(maxlen=self.history_len)
        self.err_z_pid: deque[float] = deque(maxlen=self.history_len)
        self.err_norm_pid: deque[float] = deque(maxlen=self.history_len)

        self.err_x_lqr: deque[float] = deque(maxlen=self.history_len)
        self.err_y_lqr: deque[float] = deque(maxlen=self.history_len)
        self.err_z_lqr: deque[float] = deque(maxlen=self.history_len)
        self.err_norm_lqr: deque[float] = deque(maxlen=self.history_len)

        self.roll_pid: deque[float] = deque(maxlen=self.history_len)
        self.pitch_pid: deque[float] = deque(maxlen=self.history_len)
        self.roll_lqr: deque[float] = deque(maxlen=self.history_len)
        self.pitch_lqr: deque[float] = deque(maxlen=self.history_len)

        self.pwm_mean_pid: deque[float] = deque(maxlen=self.history_len)
        self.pwm_max_pid: deque[float] = deque(maxlen=self.history_len)
        self.pwm_mean_lqr: deque[float] = deque(maxlen=self.history_len)
        self.pwm_max_lqr: deque[float] = deque(maxlen=self.history_len)

        self.wind_vx: deque[float] = deque(maxlen=self.history_len)
        self.wind_vy: deque[float] = deque(maxlen=self.history_len)
        self.wind_vz: deque[float] = deque(maxlen=self.history_len)
        self.wind_mag: deque[float] = deque(maxlen=self.history_len)

        self._plot_tick_counter: int = 0

        self._apply_dark_stylesheet()
        self._build_ui()
        self._record_telemetry_sample()
        self._update_telemetry_plots(force=True)

        # High-precision simulation timer synchronized with the RK4 solver (dt = 10 ms)
        self.sim_timer = QTimer(self)
        self.sim_timer.setTimerType(Qt.PreciseTimer)
        self.sim_timer.setInterval(int(round(self.dt * 1000.0)))
        self.sim_timer.timeout.connect(self._on_simulation_step)

        self.start_simulation()

    def _apply_dark_stylesheet(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #070B12;
            }
            QWidget {
                color: #E2E8F0;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
            }
            QToolTip {
                background-color: #0F172A;
                color: #F8FAFC;
                border: 1px solid #38BDF8;
                border-radius: 5px;
                padding: 6px;
                font-size: 11px;
            }
            QGroupBox {
                background-color: rgba(17, 24, 39, 230);
                border: 1px solid #1E293B;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px 8px 8px 8px;
                font-weight: 700;
                color: #F8FAFC;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                color: #38BDF8;
            }
            QTabWidget::pane {
                border: 1px solid #1E293B;
                border-radius: 6px;
                background-color: rgba(17, 24, 39, 220);
            }
            QTabBar::tab {
                background-color: #1E293B;
                color: #94A3B8;
                padding: 6px 12px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 3px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background-color: #0284C7;
                color: #FFFFFF;
            }
            QSlider::groove:horizontal {
                border: 1px solid #334155;
                height: 6px;
                background: #1E293B;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #0EA5E9;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #38BDF8;
                border: 1px solid #E0F2FE;
                width: 14px;
                margin-top: -5px;
                margin-bottom: -5px;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #7DD3FC;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            """
        )

    def _build_ui(self) -> None:
        """
        Configures `Quadcopter3DViewer` as the 100% full-window central widget and mounts
        the floating top command bar and the two animated collapsible side drawers
        (collapsed by default at the left and right edges).
        """
        self.viewer_3d = Quadcopter3DViewer(self.sim_pid, self.sim_lqr, self)
        self.setCentralWidget(self.viewer_3d)

        # Build the contents of the two collapsible drawers
        telemetry_content = self._build_telemetry_drawer_content()
        control_content = self._build_control_drawer_content()

        self.telemetry_drawer = CollapsibleSideDrawer(
            title="Live Telemetry",
            content_widget=telemetry_content,
            side="left",
            target_width=520,
            tooltip_text=(
                "Expand or collapse the real-time PyQtGraph telemetry drawer "
                "(3D Position Error, Roll/Pitch Attitude, Rotor PWM Effort, and Wind Profile)."
            ),
            parent=self.viewer_3d,
        )

        self.control_drawer = CollapsibleSideDrawer(
            title="Control Settings",
            content_widget=control_content,
            side="right",
            target_width=475,
            tooltip_text=(
                "Expand or collapse the Flight Control & Hot-Tuning drawer "
                "(PID Gains, LQR Weighting Matrices Q & R, Wind Gust Injection, and 3D Setpoint)."
            ),
            parent=self.viewer_3d,
        )

        # Smoothly glide the 3D HUD overlay cards alongside the left and right drawers
        self.telemetry_drawer.widthChanged.connect(self.viewer_3d.set_hud_left_offset)
        self.control_drawer.widthChanged.connect(self.viewer_3d.set_hud_right_offset)
        self.telemetry_drawer.toggled.connect(self._on_telemetry_drawer_toggled)
        self.control_drawer.toggled.connect(self._on_control_drawer_toggled)

        # Overlay layout directly on the 100% 3D viewport
        overlay_vbox = QVBoxLayout(self.viewer_3d)
        overlay_vbox.setContentsMargins(0, 8, 0, 8)
        overlay_vbox.setSpacing(6)

        # 1. Sleek Top Floating Workstation Toolbar (Camera Presets + Quick Gusts + Drawer Toggles)
        top_bar = self._build_floating_top_bar()
        overlay_vbox.addLayout(top_bar)

        # 2. Middle row: Left Collapsible Drawer | 100% 3D Interactive Space | Right Collapsible Drawer
        mid_hbox = QHBoxLayout()
        mid_hbox.setContentsMargins(0, 4, 0, 4)
        mid_hbox.setSpacing(0)

        mid_hbox.addWidget(self.telemetry_drawer, 0)
        mid_hbox.addStretch(1)
        mid_hbox.addWidget(self.control_drawer, 0)

        overlay_vbox.addLayout(mid_hbox, 1)

        # Ensure both drawers start collapsed by default
        self.telemetry_drawer.set_expanded(False, animate=False)
        self.control_drawer.set_expanded(False, animate=False)

    def _build_floating_top_bar(self) -> QHBoxLayout:
        """Builds the subtle industrial top command bar over the 3D viewport."""
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(14, 0, 14, 0)
        top_bar.setSpacing(6)

        pill_style = """
            QPushButton {
                background-color: rgba(15, 23, 42, 215);
                color: #CBD5E1;
                border: 1px solid rgba(100, 116, 139, 135);
                border-radius: 5px;
                padding: 5px 10px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: rgba(37, 99, 235, 220);
                color: #FFFFFF;
                border-color: #60A5FA;
            }
            QPushButton:checked {
                background-color: rgba(2, 132, 199, 235);
                color: #FFFFFF;
                border-color: #38BDF8;
            }
        """

        # Left quick toggle for Live Telemetry drawer
        self.btn_top_telemetry = QPushButton("Live Telemetry")
        self.btn_top_telemetry.setCheckable(True)
        self.btn_top_telemetry.setCursor(Qt.PointingHandCursor)
        self.btn_top_telemetry.setStyleSheet(pill_style)
        self.btn_top_telemetry.setToolTip(
            "Toggle the collapsible Live Telemetry side panel (real-time PyQtGraph plots)."
        )
        self.btn_top_telemetry.clicked.connect(self.telemetry_drawer.set_expanded)
        top_bar.addWidget(self.btn_top_telemetry)

        top_bar.addStretch(1)

        # Center 3D Camera & Viewport Preset pills
        btn_iso = QPushButton("3D Isometric")
        btn_iso.setStyleSheet(pill_style)
        btn_iso.setToolTip("Reset the 3D camera to the default isometric perspective angle.")
        btn_iso.clicked.connect(self.viewer_3d.set_isometric_view)

        btn_top = QPushButton("Top View (XY)")
        btn_top.setStyleSheet(pill_style)
        btn_top.setToolTip("Align the 3D camera orthogonally above the horizontal X-Y flight plane.")
        btn_top.clicked.connect(self.viewer_3d.set_top_view)

        btn_side = QPushButton("Side View (XZ)")
        btn_side.setStyleSheet(pill_style)
        btn_side.setToolTip("Align the 3D camera with the lateral X-Z elevation plane.")
        btn_side.clicked.connect(self.viewer_3d.set_side_view)

        self.viewer_3d.btn_follow = QPushButton("Follow UAV")
        self.viewer_3d.btn_follow.setCheckable(True)
        self.viewer_3d.btn_follow.setStyleSheet(pill_style)
        self.viewer_3d.btn_follow.setToolTip(
            "Dynamically track the quadcopter center of mass with the 3D camera."
        )
        self.viewer_3d.btn_follow.toggled.connect(self.viewer_3d._on_toggle_follow)

        self.viewer_3d.btn_split = QPushButton("Separate UAVs")
        self.viewer_3d.btn_split.setCheckable(True)
        self.viewer_3d.btn_split.setStyleSheet(pill_style)
        self.viewer_3d.btn_split.setToolTip(
            "Visually offset the PID and LQR quadcopters by +-0.85 m along Y for side-by-side comparison."
        )
        self.viewer_3d.btn_split.toggled.connect(self.viewer_3d._on_toggle_split)

        for b in (btn_iso, btn_top, btn_side, self.viewer_3d.btn_follow, self.viewer_3d.btn_split):
            b.setCursor(Qt.PointingHandCursor)
            top_bar.addWidget(b)

        top_bar.addStretch(1)

        # Right quick toggle for Control Settings drawer
        self.btn_top_controls = QPushButton("Control Settings")
        self.btn_top_controls.setCheckable(True)
        self.btn_top_controls.setCursor(Qt.PointingHandCursor)
        self.btn_top_controls.setStyleSheet(pill_style)
        self.btn_top_controls.setToolTip(
            "Toggle the collapsible Control Settings side panel (PID gains, LQR Q & R matrices, and Wind Gusts)."
        )
        self.btn_top_controls.clicked.connect(self.control_drawer.set_expanded)
        top_bar.addWidget(self.btn_top_controls)

        return top_bar

    def _on_telemetry_drawer_toggled(self, expanded: bool) -> None:
        self.btn_top_telemetry.blockSignals(True)
        self.btn_top_telemetry.setChecked(expanded)
        self.btn_top_telemetry.blockSignals(False)
        if expanded:
            self._update_telemetry_plots(force=True)

    def _on_control_drawer_toggled(self, expanded: bool) -> None:
        self.btn_top_controls.blockSignals(True)
        self.btn_top_controls.setChecked(expanded)
        self.btn_top_controls.blockSignals(False)

    # =========================================================================
    # RIGHT COLLAPSIBLE DRAWER: FLIGHT CONTROL, WIND & HOT-TUNING
    # =========================================================================
    def _build_control_drawer_content(self) -> QWidget:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        container.setStyleSheet("background-color: transparent;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 2, 4, 4)
        layout.setSpacing(8)

        # 1. Simulation Execution Control
        sim_group = QGroupBox("Simulation Execution Control (RK4 dt = 10 ms)")
        sim_group.setToolTip(
            "Deterministic 100 Hz numerical integration loop solved via 4th-Order Runge-Kutta (RK4)."
        )
        sim_layout = QHBoxLayout(sim_group)
        sim_layout.setSpacing(8)

        self.btn_start = QPushButton("Start")
        self.btn_start.setToolTip("Start or resume the real-time 6-DOF physics integration loop.")
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setToolTip(
            "Freeze the simulation state to inspect instantaneous vectors and telemetry curves."
        )
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setToolTip(
            "Reset quadcopter states, controller integrals, wind gusts, and telemetry history to hover equilibrium."
        )

        self.btn_start.setStyleSheet(
            "QPushButton { background-color: #059669; color: white; font-weight: bold; padding: 7px 12px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #10B981; }"
            "QPushButton:disabled { background-color: #064E3B; color: #6EE7B7; }"
        )
        self.btn_pause.setStyleSheet(
            "QPushButton { background-color: #D97706; color: white; font-weight: bold; padding: 7px 12px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #F59E0B; }"
            "QPushButton:disabled { background-color: #78350F; color: #FCD34D; }"
        )
        self.btn_reset.setStyleSheet(
            "QPushButton { background-color: #334155; color: white; font-weight: bold; padding: 7px 12px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #475569; }"
        )

        self.btn_start.clicked.connect(self.start_simulation)
        self.btn_pause.clicked.connect(self.pause_simulation)
        self.btn_reset.clicked.connect(self.reset_simulation)

        sim_layout.addWidget(self.btn_start)
        sim_layout.addWidget(self.btn_pause)
        sim_layout.addWidget(self.btn_reset)
        layout.addWidget(sim_group)

        # 2. Control Architecture Selection
        mode_group = QGroupBox("Control Architecture Selection")
        mode_group.setToolTip(
            "Select an individual flight controller or run both simultaneously under identical wind disturbances."
        )
        mode_layout = QHBoxLayout(mode_group)
        mode_layout.setSpacing(6)

        self.chk_pid = QCheckBox("PID")
        self.chk_pid.setToolTip(
            "Hierarchical Cascade PID Controller: Outer 3D Position loop generating attitude references for the Inner Euler/Gyro loop."
        )
        self.chk_lqr = QCheckBox("LQR")
        self.chk_lqr.setToolTip(
            "Infinite-Horizon Linear Quadratic Regulator: Optimal full-state feedback u = u_eq - K*(x - x_ref) via CARE Riccati."
        )
        self.chk_comp = QCheckBox("Live Comparison (PID vs. LQR)")
        self.chk_comp.setToolTip(
            "Simulate and render both Cascade PID (Cyan) and Optimal LQR (Amber) simultaneously in the same wind field."
        )
        self.chk_comp.setChecked(True)

        chk_style = """
            QCheckBox {
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 5px 8px;
                font-weight: 700;
            }
            QCheckBox:checked {
                background-color: #0369A1;
                border-color: #38BDF8;
                color: #FFFFFF;
            }
        """
        for chk in (self.chk_pid, self.chk_lqr, self.chk_comp):
            chk.setStyleSheet(chk_style)
            mode_layout.addWidget(chk)

        self.mode_btn_group = QButtonGroup(self)
        self.mode_btn_group.setExclusive(True)
        self.mode_btn_group.addButton(self.chk_pid, 0)
        self.mode_btn_group.addButton(self.chk_lqr, 1)
        self.mode_btn_group.addButton(self.chk_comp, 2)
        self.mode_btn_group.idClicked.connect(self._on_mode_changed)

        layout.addWidget(mode_group)

        # 3. Wind Disturbance Injection
        wind_group = QGroupBox("Wind Disturbance Injection (Dryden / Step Gusts)")
        wind_group.setToolTip(
            "Applies quadratic aerodynamic drag F_d = -0.5 * rho * C_d * A * v_rel * |v_rel| and center-of-pressure moments."
        )
        wind_layout = QVBoxLayout(wind_group)
        wind_layout.setSpacing(6)

        gust_btn_row = QHBoxLayout()
        gust_btn_row.setSpacing(6)
        self.btn_gust_x = QPushButton("Trigger Lateral Gust (X)")
        self.btn_gust_x.setToolTip(
            "Inject a sudden sustained wind gust along the +X (East) axis to test Pitch disturbance rejection."
        )
        self.btn_gust_y = QPushButton("Trigger Frontal Gust (Y)")
        self.btn_gust_y.setToolTip(
            "Inject a sudden sustained wind gust along the +Y (North) axis to test Roll disturbance rejection."
        )

        gust_btn_style = (
            "QPushButton { background-color: #0284C7; color: white; font-weight: 700; "
            "padding: 6px 8px; border-radius: 6px; border: 1px solid #38BDF8; font-size: 11px; }"
            "QPushButton:hover { background-color: #0EA5E9; }"
            "QPushButton:pressed { background-color: #0369A1; }"
        )
        self.btn_gust_x.setStyleSheet(gust_btn_style)
        self.btn_gust_y.setStyleSheet(gust_btn_style)
        self.btn_gust_x.clicked.connect(lambda: self.trigger_wind_gust("X"))
        self.btn_gust_y.clicked.connect(lambda: self.trigger_wind_gust("Y"))

        gust_btn_row.addWidget(self.btn_gust_x)
        gust_btn_row.addWidget(self.btn_gust_y)
        wind_layout.addLayout(gust_btn_row)

        second_wind_row = QHBoxLayout()
        second_wind_row.setSpacing(6)
        self.btn_random_wind = QPushButton("Continuous Stochastic Wind (OFF)")
        self.btn_random_wind.setCheckable(True)
        self.btn_random_wind.setToolTip(
            "Toggle continuous low-altitude stochastic atmospheric turbulence generated via a Dryden spectral filter."
        )
        self.btn_random_wind.setStyleSheet(
            "QPushButton { background-color: #1E293B; color: #CBD5E1; font-weight: 700; "
            "padding: 6px 8px; border-radius: 6px; border: 1px solid #475569; font-size: 11px; }"
            "QPushButton:checked { background-color: #7C3AED; color: white; border-color: #A78BFA; }"
        )
        self.btn_random_wind.toggled.connect(self._on_toggle_random_wind)

        self.btn_gust_z = QPushButton("Vertical Gust (Z)")
        self.btn_gust_z.setToolTip(
            "Inject a vertical downdraft wind gust along -Z to evaluate altitude hold stiffness."
        )
        self.btn_gust_z.setStyleSheet(
            "QPushButton { background-color: #1E293B; color: #93C5FD; font-weight: 600; "
            "padding: 6px 8px; border-radius: 6px; border: 1px solid #3B82F6; font-size: 11px; }"
            "QPushButton:hover { background-color: #1D4ED8; color: white; }"
        )
        self.btn_gust_z.clicked.connect(lambda: self.trigger_wind_gust("Z"))

        second_wind_row.addWidget(self.btn_random_wind, 2)
        second_wind_row.addWidget(self.btn_gust_z, 1)
        wind_layout.addLayout(second_wind_row)

        wind_grid = QGridLayout()
        self.slider_gust_amp = self._create_float_slider_row(
            wind_grid,
            row=0,
            label_text="Peak Gust Velocity:",
            min_val=2.0,
            max_val=18.0,
            initial_val=self.gust_amplitude,
            decimals=1,
            unit="m/s",
            on_change=self._on_gust_amplitude_changed,
            tooltip_text="Configure the peak air velocity amplitude [m/s] of injected discrete wind gusts.",
        )
        wind_layout.addLayout(wind_grid)
        layout.addWidget(wind_group)

        # 4. Real-Time Parameter Tuning Tabs (PID / LQR / Setpoint)
        tuning_tabs = QTabWidget()
        tuning_tabs.addTab(self._build_pid_tuning_tab(), "PID Gains")
        tuning_tabs.addTab(self._build_lqr_tuning_tab(), "LQR Weights (Q && R)")
        tuning_tabs.addTab(self._build_setpoint_tab(), "3D Setpoint")
        layout.addWidget(tuning_tabs, 1)

        scroll.setWidget(container)
        return scroll

    def _create_float_slider_row(
        self,
        grid: QGridLayout,
        row: int,
        label_text: str,
        min_val: float,
        max_val: float,
        initial_val: float,
        decimals: int,
        unit: str,
        on_change: Callable[[float], None],
        tooltip_text: str = "",
    ) -> FloatSliderRow:
        lbl_name = QLabel(label_text)
        lbl_name.setStyleSheet("color: #CBD5E1; font-weight: 600;")

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 1000)

        lbl_val = QLabel()
        lbl_val.setMinimumWidth(68)
        lbl_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lbl_val.setFont(QFont("Consolas", 9, QFont.Bold))
        lbl_val.setStyleSheet("color: #38BDF8;")

        if tooltip_text:
            lbl_name.setToolTip(tooltip_text)
            slider.setToolTip(tooltip_text)
            lbl_val.setToolTip(tooltip_text)

        row_helper = FloatSliderRow(
            slider=slider,
            value_label=lbl_val,
            min_val=min_val,
            max_val=max_val,
            decimals=decimals,
            unit=unit,
        )
        row_helper.set_value(initial_val)

        def _handle_slider_changed(_raw: int) -> None:
            val = row_helper.get_value()
            row_helper.update_label(val)
            on_change(val)

        slider.valueChanged.connect(_handle_slider_changed)

        grid.addWidget(lbl_name, row, 0)
        grid.addWidget(slider, row, 1)
        grid.addWidget(lbl_val, row, 2)
        return row_helper

    def _build_pid_tuning_tab(self) -> QWidget:
        """Real-time hot-tuning panel for `config.PID_GAINS` with didactic tooltips."""
        tab = QWidget()
        vbox = QVBoxLayout(tab)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        pos_box = QGroupBox("Outer Loop — Horizontal Position (XY) && Altitude (Z)")
        pos_grid = QGridLayout(pos_box)
        self.sl_kp_pos = self._create_float_slider_row(
            pos_grid,
            0,
            "Kp Position (XY):",
            0.2,
            5.0,
            self.pid_controller.kp_pos_xy,
            2,
            "",
            self._update_pid_gains,
            "Outer-loop proportional gain (Kp_xy): increases restoring acceleration toward the X-Y setpoint. High values increase tilt aggressiveness.",
        )
        self.sl_ki_pos = self._create_float_slider_row(
            pos_grid,
            1,
            "Ki Position (XY):",
            0.0,
            2.0,
            self.pid_controller.ki_pos_xy,
            2,
            "",
            self._update_pid_gains,
            "Outer-loop integral gain (Ki_xy): eliminates steady-state horizontal drift under sustained wind drag.",
        )
        self.sl_kd_pos = self._create_float_slider_row(
            pos_grid,
            2,
            "Kd Velocity (XY):",
            0.1,
            4.0,
            self.pid_controller.kd_pos_xy,
            2,
            "",
            self._update_pid_gains,
            "Outer-loop derivative gain (Kd_xy): provides translational velocity damping to suppress position overshoot.",
        )
        self.sl_kp_z = self._create_float_slider_row(
            pos_grid,
            3,
            "Kp Altitude (Z):",
            0.4,
            6.0,
            self.pid_controller.kp_pos_z,
            2,
            "",
            self._update_pid_gains,
            "Altitude proportional gain (Kp_z): regulates collective vertical thrust U1 to maintain target altitude.",
        )
        self.sl_kd_z = self._create_float_slider_row(
            pos_grid,
            4,
            "Kd Altitude (Z):",
            0.2,
            5.0,
            self.pid_controller.kd_pos_z,
            2,
            "",
            self._update_pid_gains,
            "Altitude derivative gain (Kd_z): damps vertical climb/descent velocity vz.",
        )
        vbox.addWidget(pos_box)

        att_box = QGroupBox("Inner Loop — Attitude Control (Roll / Pitch)")
        att_grid = QGridLayout(att_box)
        self.sl_kp_att = self._create_float_slider_row(
            att_grid,
            0,
            "Kp Roll/Pitch:",
            1.0,
            25.0,
            self.pid_controller.kp_att,
            2,
            "",
            self._update_pid_gains,
            "Inner-loop attitude proportional gain (Kp_att): determines rotational stiffness tracking reference angles phi_ref and theta_ref.",
        )
        self.sl_ki_att = self._create_float_slider_row(
            att_grid,
            1,
            "Ki Roll/Pitch:",
            0.0,
            5.0,
            self.pid_controller.ki_att,
            2,
            "",
            self._update_pid_gains,
            "Inner-loop attitude integral gain (Ki_att): compensates aerodynamic pitching/rolling moment bias.",
        )
        self.sl_kd_att = self._create_float_slider_row(
            att_grid,
            2,
            "Kd Roll/Pitch:",
            0.5,
            12.0,
            self.pid_controller.kd_att,
            2,
            "",
            self._update_pid_gains,
            "Inner-loop angular rate damping gain (Kd_att): acts directly on body gyroscope rates (p, q) to prevent rotational flutter.",
        )
        vbox.addWidget(att_box)

        btn_reset_pid = QPushButton("Restore Nominal PID Gains (config.py)")
        btn_reset_pid.setToolTip("Reset all outer and inner PID loop gains to their baseline values in config.py.")
        btn_reset_pid.setStyleSheet(
            "QPushButton { background-color: #1E293B; color: #38BDF8; padding: 5px; border-radius: 5px; border: 1px solid #334155; }"
            "QPushButton:hover { background-color: #334155; }"
        )
        btn_reset_pid.clicked.connect(self._restore_default_pid_gains)
        vbox.addWidget(btn_reset_pid)
        vbox.addStretch(1)
        return tab

    def _build_lqr_tuning_tab(self) -> QWidget:
        """Real-time hot-tuning panel for `config.Q_DIAG` and `config.R_DIAG` with didactic tooltips."""
        tab = QWidget()
        vbox = QVBoxLayout(tab)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        q_box = QGroupBox("State Penalty Matrix Q (Position, Velocity, Attitude)")
        q_grid = QGridLayout(q_box)
        self.sl_q_pos = self._create_float_slider_row(
            q_grid,
            0,
            "Q Position (XY):",
            1.0,
            80.0,
            self.lqr_controller.q_pos_xy,
            1,
            "",
            self._update_lqr_weights,
            "Diagonal state weight Q_x, Q_y: higher values penalize horizontal position deviation more heavily in the LQR cost integral J.",
        )
        self.sl_q_pos_z = self._create_float_slider_row(
            q_grid,
            1,
            "Q Altitude (Z):",
            5.0,
            150.0,
            self.lqr_controller.q_pos_z,
            1,
            "",
            self._update_lqr_weights,
            "Diagonal state weight Q_z: penalizes vertical altitude tracking error (z - z_ref)^2.",
        )
        self.sl_q_vel = self._create_float_slider_row(
            q_grid,
            2,
            "Q Velocity (vx,vy):",
            0.5,
            40.0,
            self.lqr_controller.q_vel,
            1,
            "",
            self._update_lqr_weights,
            "Diagonal state weight Q_vx, Q_vy: penalizes linear translational speed to increase closed-loop damping.",
        )
        self.sl_q_att = self._create_float_slider_row(
            q_grid,
            3,
            "Q Attitude (phi,th):",
            10.0,
            300.0,
            self.lqr_controller.q_att,
            1,
            "",
            self._update_lqr_weights,
            "Diagonal state weight Q_phi, Q_theta: penalizes Roll and Pitch excursions from level flight.",
        )
        self.sl_q_rate = self._create_float_slider_row(
            q_grid,
            4,
            "Q Angular Rate (p,q,r):",
            0.1,
            15.0,
            self.lqr_controller.q_rate,
            2,
            "",
            self._update_lqr_weights,
            "Diagonal state weight Q_p, Q_q, Q_r: penalizes high body angular velocities.",
        )
        vbox.addWidget(q_box)

        r_box = QGroupBox("Control Penalty Matrix R (Actuator Effort)")
        r_grid = QGridLayout(r_box)
        self.sl_r_weight = self._create_float_slider_row(
            r_grid,
            0,
            "R Control Effort:",
            0.1,
            10.0,
            self.lqr_controller.r_weight,
            2,
            "",
            self._update_lqr_weights,
            "Diagonal control penalty R: lower values permit aggressive motor thrust/torque commands; higher values enforce conservative energy usage.",
        )
        vbox.addWidget(r_box)

        self.lbl_riccati_info = QLabel()
        self.lbl_riccati_info.setFont(QFont("Consolas", 8))
        self.lbl_riccati_info.setToolTip(
            "Frobenius norm and key feedback gains of K = R^-1 * B^T * P computed live from the Continuous Algebraic Riccati Equation."
        )
        self.lbl_riccati_info.setStyleSheet(
            "background-color: #0F172A; color: #FBBF24; padding: 6px; border-radius: 5px; border: 1px solid #334155;"
        )
        self._refresh_riccati_label()
        vbox.addWidget(self.lbl_riccati_info)

        btn_reset_lqr = QPushButton("Restore Nominal Q && R Weights (config.py)")
        btn_reset_lqr.setToolTip("Reset Q_DIAG and R_DIAG to their baseline values in config.py and re-solve CARE.")
        btn_reset_lqr.setStyleSheet(
            "QPushButton { background-color: #1E293B; color: #FBBF24; padding: 5px; border-radius: 5px; border: 1px solid #334155; }"
            "QPushButton:hover { background-color: #334155; }"
        )
        btn_reset_lqr.clicked.connect(self._restore_default_lqr_weights)
        vbox.addWidget(btn_reset_lqr)
        vbox.addStretch(1)
        return tab

    def _build_setpoint_tab(self) -> QWidget:
        tab = QWidget()
        vbox = QVBoxLayout(tab)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        sp_box = QGroupBox("3D Reference Trajectory Setpoint")
        sp_grid = QGridLayout(sp_box)
        self.sl_sp_x = self._create_float_slider_row(
            sp_grid,
            0,
            "Setpoint X (East):",
            -2.5,
            2.5,
            self.setpoint_pos[0],
            2,
            "m",
            self._update_setpoint,
            "Target X coordinate in the inertial East-North-Up (ENU) reference frame [m].",
        )
        self.sl_sp_y = self._create_float_slider_row(
            sp_grid,
            1,
            "Setpoint Y (North):",
            -2.5,
            2.5,
            self.setpoint_pos[1],
            2,
            "m",
            self._update_setpoint,
            "Target Y coordinate in the inertial East-North-Up (ENU) reference frame [m].",
        )
        self.sl_sp_z = self._create_float_slider_row(
            sp_grid,
            2,
            "Setpoint Z (Altitude):",
            0.5,
            3.5,
            self.setpoint_pos[2],
            2,
            "m",
            self._update_setpoint,
            "Target vertical hover altitude Z above the ground plane [m].",
        )
        self.sl_sp_yaw = self._create_float_slider_row(
            sp_grid,
            3,
            "Heading (Yaw psi_ref):",
            -180.0,
            180.0,
            0.0,
            0,
            "deg",
            self._update_setpoint,
            "Target heading angle psi_ref around the vertical Z axis [deg].",
        )
        vbox.addWidget(sp_box)

        preset_row = QHBoxLayout()
        btn_origin = QPushButton("Origin Hover (0, 0, 2.0 m)")
        btn_origin.setToolTip("Command both quadcopters back to the central hover equilibrium point (0, 0, 2.0 m).")
        btn_step = QPushButton("3D Step (+1.5, -1.0, 2.6 m)")
        btn_step.setToolTip("Apply a simultaneous 3D position and yaw step reference command to compare step-response dynamics.")
        for b in (btn_origin, btn_step):
            b.setStyleSheet(
                "QPushButton { background-color: #1E293B; color: #A7F3D0; padding: 6px; border-radius: 5px; border: 1px solid #059669; }"
                "QPushButton:hover { background-color: #065F46; }"
            )
            preset_row.addWidget(b)

        btn_origin.clicked.connect(lambda: self._apply_setpoint_preset(0.0, 0.0, 2.0, 0.0))
        btn_step.clicked.connect(lambda: self._apply_setpoint_preset(1.5, -1.0, 2.6, 25.0))
        vbox.addLayout(preset_row)
        vbox.addStretch(1)
        return tab

    # =========================================================================
    # LEFT COLLAPSIBLE DRAWER: REAL-TIME PYQTGRAPH TELEMETRY
    # =========================================================================
    def _build_telemetry_drawer_content(self) -> QWidget:
        container = QWidget(self)
        container.setStyleSheet("background-color: transparent;")
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(4, 2, 4, 4)
        vbox.setSpacing(6)

        # 1. 3D Position Error vs. Setpoint
        self.plot_pos_err = pg.PlotWidget(title="1. 3D Position Error vs. Setpoint [m]")
        self.plot_pos_err.setToolTip(
            "Euclidean 3D position tracking error ||e_pos|| and lateral component e_x (Cyan: Cascade PID, Amber: Optimal LQR)."
        )
        self._style_plot(self.plot_pos_err, "Error [m]")
        self.curve_err_norm_pid = self.plot_pos_err.plot(
            pen=pg.mkPen("#00E5FF", width=2.3), name="||e_pos|| PID"
        )
        self.curve_err_norm_lqr = self.plot_pos_err.plot(
            pen=pg.mkPen("#FFAB00", width=2.3), name="||e_pos|| LQR"
        )
        self.curve_err_x_pid = self.plot_pos_err.plot(
            pen=pg.mkPen("#38BDF8", width=1.2, style=Qt.DashLine), name="e_x PID"
        )
        self.curve_err_x_lqr = self.plot_pos_err.plot(
            pen=pg.mkPen("#FDE047", width=1.2, style=Qt.DashLine), name="e_x LQR"
        )

        # 2. Attitude Angles (Roll & Pitch) during Gust Rejection
        self.plot_attitude = pg.PlotWidget(
            title="2. Attitude Angles (Roll phi, Pitch theta) [deg]"
        )
        self.plot_attitude.setToolTip(
            "Euler Roll (phi) and Pitch (theta) angles in degrees as the controllers lean into the wind to counteract aerodynamic drag."
        )
        self._style_plot(self.plot_attitude, "Angle [deg]")
        self.curve_roll_pid = self.plot_attitude.plot(
            pen=pg.mkPen("#00E5FF", width=2.0), name="Roll phi (PID)"
        )
        self.curve_pitch_pid = self.plot_attitude.plot(
            pen=pg.mkPen("#38BDF8", width=1.5, style=Qt.DashLine), name="Pitch theta (PID)"
        )
        self.curve_roll_lqr = self.plot_attitude.plot(
            pen=pg.mkPen("#FFAB00", width=2.0), name="Roll phi (LQR)"
        )
        self.curve_pitch_lqr = self.plot_attitude.plot(
            pen=pg.mkPen("#F97316", width=1.5, style=Qt.DashLine), name="Pitch theta (LQR)"
        )

        # 3. Control Effort / Rotor Motor Power (% PWM)
        self.plot_motors = pg.PlotWidget(
            title="3. Control Effort / Motor Power [% PWM]"
        )
        self.plot_motors.setToolTip(
            "Mean and peak brushless motor effort [% PWM] across the 4 rotors after control allocation (Motor Mixing)."
        )
        self._style_plot(self.plot_motors, "Effort [%]")
        self.plot_motors.setYRange(0.0, 102.0)
        self.curve_pwm_mean_pid = self.plot_motors.plot(
            pen=pg.mkPen("#00E5FF", width=2.0), name="Mean PWM (PID)"
        )
        self.curve_pwm_max_pid = self.plot_motors.plot(
            pen=pg.mkPen("#0284C7", width=1.2, style=Qt.DotLine), name="Peak Rotor (PID)"
        )
        self.curve_pwm_mean_lqr = self.plot_motors.plot(
            pen=pg.mkPen("#FFAB00", width=2.0), name="Mean PWM (LQR)"
        )
        self.curve_pwm_max_lqr = self.plot_motors.plot(
            pen=pg.mkPen("#D97706", width=1.2, style=Qt.DotLine), name="Peak Rotor (LQR)"
        )

        # 4. Applied Wind Velocity Profile [m/s]
        self.plot_wind = pg.PlotWidget(title="4. Applied Wind Velocity Profile [m/s]")
        self.plot_wind.setToolTip(
            "Instantaneous 3D ambient wind velocity components [V_wx, V_wy, V_wz] and total magnitude |V_w| in [m/s]."
        )
        self._style_plot(self.plot_wind, "Wind [m/s]")
        self.curve_wind_vx = self.plot_wind.plot(
            pen=pg.mkPen("#EF4444", width=2.0), name="V_wx (Lateral X)"
        )
        self.curve_wind_vy = self.plot_wind.plot(
            pen=pg.mkPen("#22C55E", width=2.0), name="V_wy (Frontal Y)"
        )
        self.curve_wind_vz = self.plot_wind.plot(
            pen=pg.mkPen("#3B82F6", width=1.5), name="V_wz (Vertical Z)"
        )
        self.curve_wind_mag = self.plot_wind.plot(
            pen=pg.mkPen("#E2E8F0", width=1.4, style=Qt.DashLine), name="|V_w| Magnitude"
        )

        vbox.addWidget(self.plot_pos_err, 1)
        vbox.addWidget(self.plot_attitude, 1)
        vbox.addWidget(self.plot_motors, 1)
        vbox.addWidget(self.plot_wind, 1)
        return container

    @staticmethod
    def _style_plot(plot_widget: pg.PlotWidget, y_label: str) -> None:
        plot_widget.showGrid(x=True, y=True, alpha=0.22)
        plot_widget.setLabel("left", y_label)
        plot_widget.setLabel("bottom", "Time [s]")
        legend = plot_widget.addLegend(offset=(8, 4), labelTextSize="8pt")
        legend.setBrush(pg.mkBrush(15, 23, 42, 190))

    def _on_mode_changed(self, btn_id: int) -> None:
        modes = {0: "PID", 1: "LQR", 2: "COMPARISON"}
        self.active_mode = modes.get(btn_id, "COMPARISON")
        self.viewer_3d.mode = self.active_mode
        self._update_telemetry_plots(force=True)
        self.viewer_3d.update()

    def _on_gust_amplitude_changed(self, val: float) -> None:
        self.gust_amplitude = float(val)

    def trigger_wind_gust(self, axis: str = "X") -> None:
        self.wind.trigger_gust(
            axis=axis,
            amplitude=self.gust_amplitude,
            duration=config.DEFAULT_GUST_DURATION,
            profile="step_decay",
        )

    def _on_toggle_random_wind(self, checked: bool) -> None:
        self.wind.set_random_wind(checked)
        state_txt = "ON" if checked else "OFF"
        self.btn_random_wind.setText(f"Continuous Stochastic Wind ({state_txt})")

    def _update_pid_gains(self, _val: float = 0.0) -> None:
        self.pid_controller.set_gains(
            kp_pos_xy=self.sl_kp_pos.get_value(),
            ki_pos_xy=self.sl_ki_pos.get_value(),
            kd_pos_xy=self.sl_kd_pos.get_value(),
            kp_pos_z=self.sl_kp_z.get_value(),
            kd_pos_z=self.sl_kd_z.get_value(),
            kp_att=self.sl_kp_att.get_value(),
            ki_att=self.sl_ki_att.get_value(),
            kd_att=self.sl_kd_att.get_value(),
        )

    def _restore_default_pid_gains(self) -> None:
        defaults = PIDController()
        self.sl_kp_pos.set_value(defaults.kp_pos_xy)
        self.sl_ki_pos.set_value(defaults.ki_pos_xy)
        self.sl_kd_pos.set_value(defaults.kd_pos_xy)
        self.sl_kp_z.set_value(defaults.kp_pos_z)
        self.sl_kd_z.set_value(defaults.kd_pos_z)
        self.sl_kp_att.set_value(defaults.kp_att)
        self.sl_ki_att.set_value(defaults.ki_att)
        self.sl_kd_att.set_value(defaults.kd_att)
        self._update_pid_gains()

    def _update_lqr_weights(self, _val: float = 0.0) -> None:
        self.lqr_controller.update_weights(
            q_pos_xy=self.sl_q_pos.get_value(),
            q_pos_z=self.sl_q_pos_z.get_value(),
            q_vel=self.sl_q_vel.get_value(),
            q_att=self.sl_q_att.get_value(),
            q_rate=self.sl_q_rate.get_value(),
            r_weight=self.sl_r_weight.get_value(),
        )
        self._refresh_riccati_label()

    def _refresh_riccati_label(self) -> None:
        K = self.lqr_controller.K
        norm_k = float(np.linalg.norm(K, ord="fro"))
        k_pos_xy = float(abs(K[2, 0]))
        k_att_th = float(abs(K[2, 7]))
        self.lbl_riccati_info.setText(
            f"CARE Riccati Solved | ||K||_F = {norm_k:.2f} | "
            f"K_pos = {k_pos_xy:.2f} | K_theta = {k_att_th:.2f}"
        )

    def _restore_default_lqr_weights(self) -> None:
        defaults = LQRController()
        self.sl_q_pos.set_value(defaults.q_pos_xy)
        self.sl_q_pos_z.set_value(defaults.q_pos_z)
        self.sl_q_vel.set_value(defaults.q_vel)
        self.sl_q_att.set_value(defaults.q_att)
        self.sl_q_rate.set_value(defaults.q_rate)
        self.sl_r_weight.set_value(defaults.r_weight)
        self._update_lqr_weights()

    def _update_setpoint(self, _val: float = 0.0) -> None:
        self.setpoint_pos[:] = [
            self.sl_sp_x.get_value(),
            self.sl_sp_y.get_value(),
            self.sl_sp_z.get_value(),
        ]
        self.setpoint_yaw = float(np.radians(self.sl_sp_yaw.get_value()))
        self.viewer_3d.setpoint = self.setpoint_pos.copy()
        self.viewer_3d.setpoint_yaw = self.setpoint_yaw
        self.viewer_3d.update()

    def _apply_setpoint_preset(self, x: float, y: float, z: float, yaw_deg: float) -> None:
        self.sl_sp_x.set_value(x)
        self.sl_sp_y.set_value(y)
        self.sl_sp_z.set_value(z)
        self.sl_sp_yaw.set_value(yaw_deg)
        self._update_setpoint()

    def start_simulation(self) -> None:
        self.is_running = True
        self.sim_timer.start()
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)

    def pause_simulation(self) -> None:
        self.is_running = False
        self.sim_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)

    def reset_simulation(self) -> None:
        was_random = self.wind.random_wind_enabled
        self.wind.reset()
        self.wind.set_random_wind(was_random)

        self.sim_pid.reset()
        self.sim_lqr.reset()
        self.pid_controller.reset()
        self.lqr_controller.reset()

        for buf in (
            self.t_hist,
            self.err_x_pid,
            self.err_y_pid,
            self.err_z_pid,
            self.err_norm_pid,
            self.err_x_lqr,
            self.err_y_lqr,
            self.err_z_lqr,
            self.err_norm_lqr,
            self.roll_pid,
            self.pitch_pid,
            self.roll_lqr,
            self.pitch_lqr,
            self.pwm_mean_pid,
            self.pwm_max_pid,
            self.pwm_mean_lqr,
            self.pwm_max_lqr,
            self.wind_vx,
            self.wind_vy,
            self.wind_vz,
            self.wind_mag,
        ):
            buf.clear()

        self.viewer_3d.reset_trails()
        self._record_telemetry_sample()
        self._update_telemetry_plots(force=True)
        self.viewer_3d.update()

    def _on_simulation_step(self) -> None:
        dt = self.dt
        v_wind = self.wind.update(dt)

        u_pid = self.pid_controller.compute_control(
            self.sim_pid.state, self.setpoint_pos, self.setpoint_yaw, dt
        )
        self.sim_pid.step(u_pid, dt=dt, wind_velocity=v_wind)

        u_lqr = self.lqr_controller.compute_control(
            self.sim_lqr.state, self.setpoint_pos, self.setpoint_yaw, dt
        )
        self.sim_lqr.step(u_lqr, dt=dt, wind_velocity=v_wind)

        self._record_telemetry_sample()

        self._plot_tick_counter += 1
        if self._plot_tick_counter % 2 == 0:
            self.viewer_3d.advance_visual_state(dt * 2.0)
        if self._plot_tick_counter % 3 == 0:
            self._update_telemetry_plots()

    def _record_telemetry_sample(self) -> None:
        self.t_hist.append(self.sim_pid.time)

        ep_pid = self.setpoint_pos - self.sim_pid.position
        ep_lqr = self.setpoint_pos - self.sim_lqr.position

        self.err_x_pid.append(float(ep_pid[0]))
        self.err_y_pid.append(float(ep_pid[1]))
        self.err_z_pid.append(float(ep_pid[2]))
        self.err_norm_pid.append(float(np.linalg.norm(ep_pid)))

        self.err_x_lqr.append(float(ep_lqr[0]))
        self.err_y_lqr.append(float(ep_lqr[1]))
        self.err_z_lqr.append(float(ep_lqr[2]))
        self.err_norm_lqr.append(float(np.linalg.norm(ep_lqr)))

        ang_pid = self.sim_pid.euler_angles_deg
        ang_lqr = self.sim_lqr.euler_angles_deg
        self.roll_pid.append(float(ang_pid[0]))
        self.pitch_pid.append(float(ang_pid[1]))
        self.roll_lqr.append(float(ang_lqr[0]))
        self.pitch_lqr.append(float(ang_lqr[1]))

        self.pwm_mean_pid.append(float(np.mean(self.sim_pid.motor_pwm)))
        self.pwm_max_pid.append(float(np.max(self.sim_pid.motor_pwm)))
        self.pwm_mean_lqr.append(float(np.mean(self.sim_lqr.motor_pwm)))
        self.pwm_max_lqr.append(float(np.max(self.sim_lqr.motor_pwm)))

        vw = self.wind.current_wind_velocity
        self.wind_vx.append(float(vw[0]))
        self.wind_vy.append(float(vw[1]))
        self.wind_vz.append(float(vw[2]))
        self.wind_mag.append(float(np.linalg.norm(vw)))

    def _update_telemetry_plots(self, force: bool = False) -> None:
        """
        Refreshes PyQtGraph curves. When the telemetry drawer is collapsed and `force=False`,
        skips unnecessary canvas redraws to maximize 3D rendering throughput.
        """
        if not force and not self.telemetry_drawer.is_expanded and self.telemetry_drawer.panelWidth == 0:
            return

        t = np.asarray(self.t_hist, dtype=float)
        if t.size == 0:
            return

        show_pid = self.active_mode in ("PID", "COMPARISON")
        show_lqr = self.active_mode in ("LQR", "COMPARISON")

        if show_pid:
            self.curve_err_norm_pid.setData(t, np.asarray(self.err_norm_pid))
            self.curve_err_x_pid.setData(t, np.asarray(self.err_x_pid))
            self.curve_roll_pid.setData(t, np.asarray(self.roll_pid))
            self.curve_pitch_pid.setData(t, np.asarray(self.pitch_pid))
            self.curve_pwm_mean_pid.setData(t, np.asarray(self.pwm_mean_pid))
            self.curve_pwm_max_pid.setData(t, np.asarray(self.pwm_max_pid))
        else:
            self.curve_err_norm_pid.clear()
            self.curve_err_x_pid.clear()
            self.curve_roll_pid.clear()
            self.curve_pitch_pid.clear()
            self.curve_pwm_mean_pid.clear()
            self.curve_pwm_max_pid.clear()

        if show_lqr:
            self.curve_err_norm_lqr.setData(t, np.asarray(self.err_norm_lqr))
            self.curve_err_x_lqr.setData(t, np.asarray(self.err_x_lqr))
            self.curve_roll_lqr.setData(t, np.asarray(self.roll_lqr))
            self.curve_pitch_lqr.setData(t, np.asarray(self.pitch_lqr))
            self.curve_pwm_mean_lqr.setData(t, np.asarray(self.pwm_mean_lqr))
            self.curve_pwm_max_lqr.setData(t, np.asarray(self.pwm_max_lqr))
        else:
            self.curve_err_norm_lqr.clear()
            self.curve_err_x_lqr.clear()
            self.curve_roll_lqr.clear()
            self.curve_pitch_lqr.clear()
            self.curve_pwm_mean_lqr.clear()
            self.curve_pwm_max_lqr.clear()

        self.curve_wind_vx.setData(t, np.asarray(self.wind_vx))
        self.curve_wind_vy.setData(t, np.asarray(self.wind_vy))
        self.curve_wind_vz.setData(t, np.asarray(self.wind_vz))
        self.curve_wind_mag.setData(t, np.asarray(self.wind_mag))
