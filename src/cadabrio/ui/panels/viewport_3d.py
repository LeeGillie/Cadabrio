"""3D Viewport panel for Cadabrio.

Provides the main 3D view with an OpenGL rendering surface,
ground grid, and orbit/pan/zoom camera controls.
"""

import math
import numpy as np

from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtGui import QColor
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from OpenGL.GL import *  # noqa: F403, F401
from OpenGL.GLU import gluPerspective, gluLookAt

from cadabrio.config.manager import ConfigManager


class OrbitCamera:
    """Orbit camera with pan, zoom, and rotate around a target point."""

    def __init__(self):
        self.target = np.array([0.0, 0.0, 0.0])
        self.distance = 20.0
        self.yaw = 45.0     # degrees around Y axis
        self.pitch = 30.0   # degrees above horizon
        self.fov = 45.0

        # Limits
        self.min_distance = 1.0
        self.max_distance = 500.0
        self.min_pitch = -89.0
        self.max_pitch = 89.0

    def rotate(self, dx: float, dy: float):
        """Rotate camera by mouse delta (in pixels)."""
        self.yaw += dx * 0.3
        self.pitch = max(self.min_pitch, min(self.max_pitch, self.pitch - dy * 0.3))

    def pan(self, dx: float, dy: float):
        """Pan camera by mouse delta (in pixels)."""
        scale = self.distance * 0.002
        # Compute right and up vectors relative to camera
        yaw_rad = math.radians(self.yaw)
        right = np.array([math.cos(yaw_rad), 0, -math.sin(yaw_rad)])
        up = np.array([0, 1, 0])
        self.target -= right * dx * scale
        self.target += up * dy * scale

    def zoom(self, delta: float):
        """Zoom by scroll delta."""
        factor = 1.0 - delta * 0.001
        self.distance = max(self.min_distance, min(self.max_distance, self.distance * factor))

    def eye_position(self) -> np.ndarray:
        """Compute eye position from orbit parameters."""
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        x = self.target[0] + self.distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        y = self.target[1] + self.distance * math.sin(pitch_rad)
        z = self.target[2] + self.distance * math.cos(pitch_rad) * math.cos(yaw_rad)
        return np.array([x, y, z])

    def apply(self, width: int, height: int):
        """Set up the OpenGL projection and modelview matrices."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / max(height, 1)
        gluPerspective(self.fov, aspect, 0.1, 1000.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        eye = self.eye_position()
        gluLookAt(
            eye[0], eye[1], eye[2],
            self.target[0], self.target[1], self.target[2],
            0, 1, 0,
        )


class ViewportGLWidget(QOpenGLWidget):
    """OpenGL widget rendering a ground grid with orbit camera."""

    def __init__(self, config: ConfigManager, parent=None):
        super().__init__(parent)
        self._config = config
        self._camera = OrbitCamera()
        self._camera.fov = config.get("viewport", "camera_fov", 45.0)

        self._last_mouse_pos = QPoint()
        self._grid_size = config.get("viewport", "grid_size", 10.0)
        self._grid_subdivisions = config.get("viewport", "grid_subdivisions", 10)

        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def initializeGL(self):
        bg = self._config.get("appearance", "viewport_background", "#0a0a12")
        c = QColor(bg)
        glClearColor(c.redF(), c.greenF(), c.blueF(), 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._camera.apply(self.width(), self.height())
        self._draw_grid()
        self._draw_axis_indicator()

    def _draw_grid(self):
        """Draw a ground grid on the XZ plane."""
        size = self._grid_size
        subdivs = self._grid_subdivisions
        step = size / subdivs
        half = size / 2.0

        # Minor grid lines (dim)
        glLineWidth(1.0)
        glBegin(GL_LINES)
        glColor4f(0.2, 0.2, 0.35, 0.4)
        for i in range(subdivs + 1):
            x = -half + i * step
            glVertex3f(x, 0, -half)
            glVertex3f(x, 0, half)
            glVertex3f(-half, 0, x)
            glVertex3f(half, 0, x)
        glEnd()

        # Center lines (brighter)
        glLineWidth(1.5)
        glBegin(GL_LINES)
        # X center line (subtle red)
        glColor4f(0.6, 0.15, 0.15, 0.7)
        glVertex3f(-half, 0, 0)
        glVertex3f(half, 0, 0)
        # Z center line (subtle blue)
        glColor4f(0.15, 0.15, 0.6, 0.7)
        glVertex3f(0, 0, -half)
        glVertex3f(0, 0, half)
        glEnd()

    def _draw_axis_indicator(self):
        """Draw a small XYZ axis at the origin."""
        glLineWidth(2.0)
        length = self._grid_size * 0.05

        glBegin(GL_LINES)
        # X - Red
        glColor3f(0.9, 0.2, 0.2)
        glVertex3f(0, 0, 0)
        glVertex3f(length, 0, 0)
        # Y - Green
        glColor3f(0.2, 0.9, 0.2)
        glVertex3f(0, 0, 0)
        glVertex3f(0, length, 0)
        # Z - Blue
        glColor3f(0.2, 0.2, 0.9)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, length)
        glEnd()

    # -------------------------------------------------------------------
    # Mouse interaction
    # -------------------------------------------------------------------

    def mousePressEvent(self, event):
        self._last_mouse_pos = event.position().toPoint()

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        dx = pos.x() - self._last_mouse_pos.x()
        dy = pos.y() - self._last_mouse_pos.y()

        if event.buttons() & Qt.MouseButton.LeftButton:
            orbit_sens = self._config.get("viewport", "orbit_sensitivity", 1.0)
            self._camera.rotate(dx * orbit_sens, dy * orbit_sens)
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            self._camera.pan(dx, dy)
        elif event.buttons() & Qt.MouseButton.RightButton:
            self._camera.zoom(dy * 5)

        self._last_mouse_pos = pos
        self.update()

    def wheelEvent(self, event):
        zoom_sens = self._config.get("viewport", "zoom_sensitivity", 1.0)
        delta = event.angleDelta().y() * zoom_sens
        self._camera.zoom(-delta)
        self.update()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for camera."""
        key = event.key()
        if key == Qt.Key.Key_Home:
            # Reset camera
            self._camera.target = np.array([0.0, 0.0, 0.0])
            self._camera.distance = 20.0
            self._camera.yaw = 45.0
            self._camera.pitch = 30.0
            self.update()
        elif key == Qt.Key.Key_5:
            # Toggle ortho/persp (future)
            pass


class Viewport3DPanel(QWidget):
    """3D viewport widget - central workspace panel."""

    def __init__(self, config: ConfigManager, parent=None):
        super().__init__(parent)
        self._config = config

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._gl_widget = ViewportGLWidget(config)
        layout.addWidget(self._gl_widget)

    @property
    def camera(self) -> OrbitCamera:
        return self._gl_widget._camera
