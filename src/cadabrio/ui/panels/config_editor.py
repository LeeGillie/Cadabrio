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
)

from cadabrio.config.manager import ConfigManager
from cadabrio.config.themes.theme_manager import ThemeManager


class ConfigEditorDialog(QDialog):
    """Tabbed configuration editor dialog."""

    def __init__(self, config: ConfigManager, theme_mgr: ThemeManager, parent=None):
        super().__init__(parent)
        self._config = config
        self._theme_mgr = theme_mgr
        self._widgets: dict[tuple[str, str], QWidget] = {}

        self.setWindowTitle("Cadabrio Preferences")
        self.setMinimumSize(700, 500)

        layout = QVBoxLayout(self)

        # Tab widget - one tab per config group
        self._tabs = QTabWidget()
        for group in config.groups():
            tab = self._build_group_tab(group)
            label = config.get_group_label(group)
            self._tabs.addTab(tab, label)
        layout.addWidget(self._tabs)

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

        values = self._config.get_group(group)
        for key, value in values.items():
            widget = self._create_widget_for_value(group, key, value)
            label = key.replace("_", " ").title()
            form.addRow(label + ":", widget)
            self._widgets[(group, key)] = widget

        scroll.setWidget(container)
        return scroll

    def _create_widget_for_value(self, group: str, key: str, value: Any) -> QWidget:
        """Create an appropriate input widget based on the value type."""
        if isinstance(value, bool):
            widget = QCheckBox()
            widget.setChecked(value)
            return widget

        if isinstance(value, int):
            widget = QSpinBox()
            widget.setRange(-999999, 999999)
            widget.setValue(value)
            return widget

        if isinstance(value, float):
            widget = QDoubleSpinBox()
            widget.setRange(-999999.0, 999999.0)
            widget.setDecimals(3)
            widget.setValue(value)
            return widget

        if isinstance(value, str):
            # Special handling for theme selection
            if group == "appearance" and key == "theme":
                widget = QComboBox()
                for theme_id, theme_name in self._theme_mgr.available_themes():
                    widget.addItem(theme_name, theme_id)
                idx = widget.findData(value)
                if idx >= 0:
                    widget.setCurrentIndex(idx)
                return widget

            widget = QLineEdit()
            widget.setText(value)
            return widget

        # Fallback: string input
        widget = QLineEdit()
        widget.setText(str(value))
        return widget

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

    def _apply_and_close(self):
        """Apply and close the dialog."""
        self._apply()
        self.accept()
