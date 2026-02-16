"""Theme Editor dialog for Cadabrio.

Provides a visual color picker interface for creating, editing,
and saving custom color themes. Colors are organized by category
for easier navigation.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QComboBox,
    QScrollArea,
    QWidget,
    QGroupBox,
    QColorDialog,
    QDialogButtonBox,
    QMessageBox,
)

from cadabrio.config.manager import ConfigManager
from cadabrio.config.themes.theme_manager import ThemeManager

# Color keys grouped by category for the editor UI
_COLOR_GROUPS = {
    "Backgrounds": [
        "background_primary", "background_secondary", "background_tertiary",
        "background_panel", "background_input", "background_hover", "background_selected",
    ],
    "Text": [
        "foreground_primary", "foreground_secondary", "foreground_muted", "foreground_disabled",
    ],
    "Accents": [
        "accent_primary", "accent_secondary", "accent_tertiary",
        "accent_hover", "accent_muted",
    ],
    "Borders": [
        "border_primary", "border_secondary", "border_accent",
    ],
    "Status": [
        "success", "warning", "error", "info",
    ],
    "Scrollbar": [
        "scrollbar_track", "scrollbar_handle", "scrollbar_handle_hover",
    ],
    "Tabs": [
        "tab_active", "tab_inactive", "tab_hover", "tab_indicator",
    ],
    "Bars & Menus": [
        "toolbar_background", "statusbar_background", "statusbar_foreground",
        "menu_background", "menu_hover", "menu_separator",
    ],
    "Tooltips": [
        "tooltip_background", "tooltip_foreground", "tooltip_border",
    ],
    "Splitter": [
        "splitter", "splitter_hover",
    ],
    "Viewport": [
        "viewport_background", "viewport_grid", "viewport_grid_accent",
    ],
}


class ColorSwatchButton(QPushButton):
    """A button that shows a color swatch and opens a color picker on click."""

    def __init__(self, color_key: str, color_hex: str, parent=None):
        super().__init__(parent)
        self.color_key = color_key
        self._color = color_hex
        self.setFixedSize(36, 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._update_style()
        self.clicked.connect(self._pick_color)

    @property
    def color(self) -> str:
        return self._color

    @color.setter
    def color(self, value: str):
        self._color = value
        self._update_style()

    def _update_style(self):
        self.setStyleSheet(
            f"QPushButton {{ background-color: {self._color}; "
            f"border: 1px solid #3a3a55; border-radius: 3px; }} "
            f"QPushButton:hover {{ border-color: #39ff14; }}"
        )
        self.setToolTip(f"{self.color_key}: {self._color}")

    def _pick_color(self):
        color = QColorDialog.getColor(
            QColor(self._color), self, f"Pick color: {self.color_key}",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if color.isValid():
            self.color = color.name()


class ThemeEditorDialog(QDialog):
    """Dialog for creating and editing color themes."""

    def __init__(self, config: ConfigManager, theme_mgr: ThemeManager, parent=None):
        super().__init__(parent)
        self._config = config
        self._theme_mgr = theme_mgr
        self._swatches: dict[str, ColorSwatchButton] = {}

        self.setWindowTitle("Cadabrio Theme Editor")
        self.setMinimumSize(600, 700)

        layout = QVBoxLayout(self)

        # --- Top bar: base theme selector and name ---
        top = QHBoxLayout()

        top.addWidget(QLabel("Base theme:"))
        self._base_combo = QComboBox()
        for theme_id, theme_name in theme_mgr.available_themes():
            self._base_combo.addItem(theme_name, theme_id)
        self._base_combo.currentIndexChanged.connect(self._load_base_theme)
        top.addWidget(self._base_combo)

        top.addSpacing(20)
        top.addWidget(QLabel("Save as:"))
        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("My Custom Theme")
        self._name_input.setMinimumWidth(180)
        top.addWidget(self._name_input)

        layout.addLayout(top)

        # --- Scrollable color grid ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self._grid_layout = QVBoxLayout(container)
        self._grid_layout.setSpacing(12)

        self._build_color_groups()

        scroll.setWidget(container)
        layout.addWidget(scroll)

        # --- Buttons ---
        btn_row = QHBoxLayout()
        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self._preview)
        btn_row.addWidget(preview_btn)

        reset_btn = QPushButton("Reset to Base")
        reset_btn.clicked.connect(self._load_base_theme)
        btn_row.addWidget(reset_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save_theme)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Load the current theme as starting point
        current_id = config.get("appearance", "theme", "cadabrio_dark")
        idx = self._base_combo.findData(current_id)
        if idx >= 0:
            self._base_combo.setCurrentIndex(idx)
        self._load_base_theme()

    def _build_color_groups(self):
        """Build grouped color swatch rows."""
        for group_name, keys in _COLOR_GROUPS.items():
            group = QGroupBox(group_name)
            grid = QGridLayout(group)
            grid.setVerticalSpacing(6)
            grid.setHorizontalSpacing(12)

            for row, key in enumerate(keys):
                label = QLabel(key.replace("_", " ").title())
                label.setStyleSheet("font-size: 9pt;")
                swatch = ColorSwatchButton(key, "#000000")
                hex_label = QLabel("#000000")
                hex_label.setStyleSheet("font-size: 8pt; color: #a0a0b0; font-family: monospace;")

                # Update hex label when swatch color changes
                swatch.clicked.connect(
                    lambda checked=False, s=swatch, h=hex_label: h.setText(s.color)
                )

                grid.addWidget(label, row, 0)
                grid.addWidget(swatch, row, 1)
                grid.addWidget(hex_label, row, 2)
                self._swatches[key] = swatch

            self._grid_layout.addWidget(group)

    def _load_base_theme(self):
        """Load colors from the selected base theme into all swatches."""
        theme_id = self._base_combo.currentData()
        theme = self._theme_mgr.get_theme(theme_id)
        if not theme:
            return

        colors = theme.get("colors", {})
        for key, swatch in self._swatches.items():
            color = colors.get(key, "#ff00ff")
            swatch.color = color
            # Also update the hex label
            group = swatch.parent()
            if group:
                grid = group.layout()
                for row in range(grid.rowCount()):
                    item = grid.itemAtPosition(row, 1)
                    if item and item.widget() is swatch:
                        hex_item = grid.itemAtPosition(row, 2)
                        if hex_item and hex_item.widget():
                            hex_item.widget().setText(color)
                        break

    def _collect_colors(self) -> dict[str, str]:
        """Collect current colors from all swatches."""
        return {key: swatch.color for key, swatch in self._swatches.items()}

    def _preview(self):
        """Apply the current colors as a live preview."""
        from PySide6.QtWidgets import QApplication
        from cadabrio.config.themes.theme_manager import _build_stylesheet

        colors = self._collect_colors()
        stylesheet = _build_stylesheet(colors)
        app = QApplication.instance()
        if app:
            app.setStyleSheet(stylesheet)

    def _save_theme(self):
        """Save the current colors as a new user theme."""
        name = self._name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Name Required", "Please enter a name for your theme.")
            return

        colors = self._collect_colors()
        theme_id = self._theme_mgr.save_user_theme(name, colors)

        # Set as active theme
        self._config.set("appearance", "theme", theme_id)
        self._config.save()

        # Apply it
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            self._theme_mgr.apply_theme(app, theme_id)

        QMessageBox.information(
            self, "Theme Saved",
            f"Theme '{name}' saved and applied.\n\nIt will appear in Preferences > Appearance."
        )
        self.accept()
