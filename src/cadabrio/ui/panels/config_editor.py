"""Configuration Editor dialog for Cadabrio.

Presents a tabbed interface with one tab per configuration group.
Allows editing all configuration values with appropriate input widgets.
"""

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QWidget,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QPushButton,
    QDialogButtonBox,
    QScrollArea,
    QFileDialog,
    QLabel,
)

from cadabrio.config.manager import ConfigManager
from cadabrio.config.themes.theme_manager import ThemeManager


# Keys that represent file/directory paths and should get browse buttons
_PATH_KEYS = {
    "blender_path",
    "freecad_path",
    "bambu_studio_path",
    "unreal_engine_path",
    "models_directory",
    "tensorrt_cache_dir",
}

# Keys with known enum options shown as combo boxes
_ENUM_OPTIONS = {
    "tensor_precision": ["float16", "bfloat16", "float32"],
    "default_units": ["millimeters", "centimeters", "meters", "inches", "feet"],
    "renderer": ["opengl", "vulkan"],
    "antialiasing": ["none", "msaa_2x", "msaa_4x", "msaa_8x"],
    "feature_detector": ["sift", "orb", "superpoint"],
    "dense_reconstruction": ["mvs", "nerf", "gaussian_splatting"],
    "mesh_quality": ["low", "medium", "high", "ultra"],
    "default_format": ["glb", "gltf", "obj", "stl", "fbx", "3mf", "usd"],
    "default_target": ["general", "print", "unreal", "blender", "freecad", "bambu"],
    "download_source": ["huggingface", "civitai", "local"],
    "language": ["en", "es", "fr", "de", "ja", "zh"],
    "log_level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
}


class ConfigEditorDialog(QDialog):
    """Tabbed configuration editor dialog."""

    def __init__(self, config: ConfigManager, theme_mgr: ThemeManager, parent=None):
        super().__init__(parent)
        self._config = config
        self._theme_mgr = theme_mgr
        self._widgets: dict[tuple[str, str], QWidget] = {}

        self.setWindowTitle("Cadabrio Preferences")
        self.setMinimumSize(750, 550)

        layout = QVBoxLayout(self)

        # Tab widget - one tab per config group
        self._tabs = QTabWidget()
        for group in config.groups():
            tab = self._build_group_tab(group)
            label = config.get_group_label(group)
            self._tabs.addTab(tab, label)
        layout.addWidget(self._tabs)

        # Status label
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #39ff14; padding: 2px;")
        layout.addWidget(self._status_label)

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
        )
        buttons.accepted.connect(self._apply_and_close)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._apply)
        layout.addWidget(buttons)

    def _build_group_tab(self, group: str) -> QWidget:
        """Build a scrollable form for a configuration group."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        container = QWidget()
        form = QFormLayout(container)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setSpacing(8)

        values = self._config.get_group(group)
        for key, value in values.items():
            widget = self._create_widget_for_value(group, key, value)
            label = key.replace("_", " ").title()
            form.addRow(label + ":", widget)
            # Store the actual input widget (not the row container for path fields)
            if key in _PATH_KEYS:
                self._widgets[(group, key)] = widget.findChild(QLineEdit)
            else:
                self._widgets[(group, key)] = widget

        scroll.setWidget(container)
        return scroll

    def _create_widget_for_value(self, group: str, key: str, value: Any) -> QWidget:
        """Create an appropriate input widget based on the value type."""
        # Bool -> checkbox
        if isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
            return widget

        # Int -> spinbox
        if isinstance(value, int):
            widget = QSpinBox()
            widget.setRange(0, 999999)
            widget.setValue(value)
            return widget

        # Float -> double spinbox
        if isinstance(value, float):
            widget = QDoubleSpinBox()
            widget.setRange(0.0, 999999.0)
            widget.setDecimals(3)
            widget.setSingleStep(0.1)
            widget.setValue(value)
            return widget

        if isinstance(value, str):
            # Theme selector
            if group == "appearance" and key == "theme":
                widget = QComboBox()
                for theme_id, theme_name in self._theme_mgr.available_themes():
                    widget.addItem(theme_name, theme_id)
                idx = widget.findData(value)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
                return widget

            # Known enum options -> combo box
            if key in _ENUM_OPTIONS:
                widget = QComboBox()
                for option in _ENUM_OPTIONS[key]:
                    widget.addItem(option.replace("_", " ").title(), option)
                idx = widget.findData(value)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
                return widget

            # Path fields -> line edit + browse button
            if key in _PATH_KEYS:
                container = QWidget()
                row = QHBoxLayout(container)
                row.setContentsMargins(0, 0, 0, 0)
                line_edit = QLineEdit()
                line_edit.setText(value)
                line_edit.setPlaceholderText("(not set)")
                row.addWidget(line_edit)
                browse_btn = QPushButton("Browse...")
                browse_btn.setFixedWidth(80)
                is_dir = key in {"models_directory", "tensorrt_cache_dir", "unreal_engine_path"}
                browse_btn.clicked.connect(
                    lambda checked, le=line_edit, d=is_dir: self._browse_path(le, d)
                )
                row.addWidget(browse_btn)
                return container

            # Default string -> line edit
            widget = QLineEdit()
            widget.setText(value)
            return widget

        # Fallback
        widget = QLineEdit()
        widget.setText(str(value))
        return widget

    def _browse_path(self, line_edit: QLineEdit, is_directory: bool):
        """Open a file/directory picker and set the result."""
        if is_directory:
            path = QFileDialog.getExistingDirectory(self, "Select Directory")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if path:
            line_edit.setText(path)

    def _read_widget_value(self, widget: QWidget) -> Any:
        """Read the current value from an input widget."""
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QSpinBox):
            return widget.value()
        if isinstance(widget, QDoubleSpinBox):
            return widget.value()
        if isinstance(widget, QComboBox):
            return widget.currentData() or widget.currentText()
        if isinstance(widget, QLineEdit):
            return widget.text()
        return None

    def _apply(self):
        """Apply all changed values to the configuration."""
        for (group, key), widget in self._widgets.items():
            value = self._read_widget_value(widget)
            if value is not None:
                self._config.set(group, key, value)
        self._config.save()

        # Re-apply theme if it changed
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app:
            self._theme_mgr.apply_theme(app)

        self._status_label.setText("Preferences saved.")

    def _apply_and_close(self):
        """Apply and close the dialog."""
        self._apply()
        self.accept()
