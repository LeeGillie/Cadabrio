"""3D Viewport panel for Cadabrio.

Provides the main 3D view for visualizing and manipulating models.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QColor

from cadabrio.config.manager import ConfigManager


class Viewport3DPanel(QWidget):
    """3D viewport widget - central workspace panel.

    TODO: Replace placeholder with OpenGL/Vulkan 3D rendering surface.
    """

    def __init__(self, config: ConfigManager, parent=None):
        super().__init__(parent)
        self._config = config

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Placeholder until OpenGL viewport is implemented
        self._placeholder = QLabel("3D Viewport")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            "QLabel { color: #39ff14; font-size: 24pt; font-weight: bold; "
            "background-color: #0a0a12; }"
        )
        layout.addWidget(self._placeholder)
