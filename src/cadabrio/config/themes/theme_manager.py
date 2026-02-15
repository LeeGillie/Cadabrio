"""Theme manager for Cadabrio.

Handles loading, applying, and managing color themes.
Supports built-in themes, Windows system themes, and user-created themes.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger

from cadabrio.config.manager import ConfigManager


# Directory containing built-in theme JSON files
_BUILTIN_THEMES_DIR = Path(__file__).parent


class ThemeManager:
    """Manages color themes for the application."""

    def __init__(self, config: ConfigManager):
        self._config = config
        self._themes: dict[str, dict[str, Any]] = {}
        self._current_theme: dict[str, Any] = {}
        self._load_builtin_themes()
        self._load_user_themes()

    def _load_builtin_themes(self):
        """Load all built-in .json theme files."""
        for path in _BUILTIN_THEMES_DIR.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    theme = json.load(f)
                theme_id = path.stem
                self._themes[theme_id] = theme
                logger.debug(f"Loaded built-in theme: {theme.get('name', theme_id)}")
            except Exception as e:
                logger.warning(f"Failed to load theme {path.name}: {e}")

    def _load_user_themes(self):
        """Load user-created themes from config directory."""
        user_themes_dir = self._config.config_dir / "themes"
        if not user_themes_dir.exists():
            return
        for path in user_themes_dir.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    theme = json.load(f)
                theme_id = f"user_{path.stem}"
                self._themes[theme_id] = theme
                logger.debug(f"Loaded user theme: {theme.get('name', theme_id)}")
            except Exception as e:
                logger.warning(f"Failed to load user theme {path.name}: {e}")

    def available_themes(self) -> list[tuple[str, str]]:
        """Return list of (theme_id, display_name) for all available themes."""
        return [
            (tid, theme.get("name", tid)) for tid, theme in self._themes.items()
        ]

    def get_theme(self, theme_id: str) -> dict[str, Any] | None:
        """Get a theme by ID."""
        return self._themes.get(theme_id)

    def get_color(self, color_key: str, fallback: str = "#ff00ff") -> str:
        """Get a color value from the current theme."""
        return self._current_theme.get("colors", {}).get(color_key, fallback)

    def apply_theme(self, app, theme_id: str | None = None):
        """Apply a theme to the QApplication."""
        if theme_id is None:
            theme_id = self._config.get("appearance", "theme", "cadabrio_dark")

        theme = self._themes.get(theme_id)
        if theme is None:
            logger.warning(f"Theme '{theme_id}' not found, falling back to cadabrio_dark")
            theme = self._themes.get("cadabrio_dark", {})

        self._current_theme = theme
        colors = theme.get("colors", {})

        stylesheet = _build_stylesheet(colors)
        app.setStyleSheet(stylesheet)
        logger.info(f"Applied theme: {theme.get('name', theme_id)}")

    def save_user_theme(self, name: str, colors: dict[str, str]) -> str:
        """Save a user-created theme. Returns the theme_id."""
        theme_id = f"user_{name.lower().replace(' ', '_')}"
        theme_data = {
            "name": name,
            "description": f"User-created theme: {name}",
            "author": "User",
            "version": "1.0",
            "colors": colors,
        }

        user_themes_dir = self._config.config_dir / "themes"
        user_themes_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{name.lower().replace(' ', '_')}.json"
        with open(user_themes_dir / filename, "w", encoding="utf-8") as f:
            json.dump(theme_data, f, indent=2)

        self._themes[theme_id] = theme_data
        logger.info(f"Saved user theme: {name}")
        return theme_id


def _build_stylesheet(c: dict[str, str]) -> str:
    """Build a Qt stylesheet from a color dictionary."""
    bg1 = c.get("background_primary", "#0d0d1a")
    bg2 = c.get("background_secondary", "#141428")
    bg3 = c.get("background_tertiary", "#1a1a2e")
    bgp = c.get("background_panel", "#12121f")
    bgi = c.get("background_input", "#1e1e35")
    bgh = c.get("background_hover", "#252545")
    bgs = c.get("background_selected", "#1a3a1a")

    fg1 = c.get("foreground_primary", "#e0e0e0")
    fg2 = c.get("foreground_secondary", "#a0a0b0")
    fgd = c.get("foreground_disabled", "#4a4a5a")

    ac1 = c.get("accent_primary", "#39ff14")
    ach = c.get("accent_hover", "#50ff3a")
    acm = c.get("accent_muted", "#1a8a0a")

    bd1 = c.get("border_primary", "#2a2a45")
    bd2 = c.get("border_secondary", "#3a3a55")
    bda = c.get("border_accent", "#39ff14")

    err = c.get("error", "#ff4757")
    wrn = c.get("warning", "#ffd700")
    inf = c.get("info", "#00d4ff")

    sct = c.get("scrollbar_track", "#141428")
    sch = c.get("scrollbar_handle", "#2a2a45")
    schh = c.get("scrollbar_handle_hover", "#39ff14")

    return f"""
    /* === Global === */
    QWidget {{
        background-color: {bg1};
        color: {fg1};
        font-family: "Segoe UI", sans-serif;
        font-size: 10pt;
    }}

    /* === Main Window === */
    QMainWindow {{
        background-color: {bg1};
    }}
    QMainWindow::separator {{
        background-color: {bd1};
        width: 2px;
        height: 2px;
    }}
    QMainWindow::separator:hover {{
        background-color: {ac1};
    }}

    /* === Dock Widgets === */
    QDockWidget {{
        background-color: {bgp};
        titlebar-close-icon: none;
        titlebar-normal-icon: none;
    }}
    QDockWidget::title {{
        background-color: {bg2};
        padding: 6px;
        border-bottom: 1px solid {bd1};
        color: {ac1};
        font-weight: bold;
    }}

    /* === Menu Bar === */
    QMenuBar {{
        background-color: {bg1};
        color: {fg1};
        border-bottom: 1px solid {bd1};
    }}
    QMenuBar::item:selected {{
        background-color: {bgh};
    }}
    QMenu {{
        background-color: {bg2};
        color: {fg1};
        border: 1px solid {bd1};
    }}
    QMenu::item:selected {{
        background-color: {bgh};
        color: {ac1};
    }}
    QMenu::separator {{
        height: 1px;
        background: {bd1};
        margin: 4px 8px;
    }}

    /* === Tab Widget === */
    QTabWidget::pane {{
        border: 1px solid {bd1};
        background-color: {bgp};
    }}
    QTabBar::tab {{
        background-color: {bg1};
        color: {fg2};
        padding: 8px 16px;
        border: 1px solid {bd1};
        border-bottom: none;
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{
        background-color: {bg3};
        color: {ac1};
        border-bottom: 2px solid {ac1};
    }}
    QTabBar::tab:hover {{
        background-color: {bgh};
    }}

    /* === Buttons === */
    QPushButton {{
        background-color: {bg2};
        color: {fg1};
        border: 1px solid {bd1};
        padding: 6px 16px;
        border-radius: 3px;
    }}
    QPushButton:hover {{
        background-color: {bgh};
        border-color: {ac1};
        color: {ac1};
    }}
    QPushButton:pressed {{
        background-color: {acm};
    }}
    QPushButton:disabled {{
        color: {fgd};
        border-color: {bd1};
    }}

    /* === Inputs === */
    QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {bgi};
        color: {fg1};
        border: 1px solid {bd1};
        padding: 4px 8px;
        border-radius: 3px;
        selection-background-color: {acm};
        selection-color: {fg1};
    }}
    QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
        border-color: {ac1};
    }}

    /* === Scrollbar === */
    QScrollBar:vertical {{
        background: {sct};
        width: 10px;
    }}
    QScrollBar::handle:vertical {{
        background: {sch};
        min-height: 20px;
        border-radius: 5px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {schh};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar:horizontal {{
        background: {sct};
        height: 10px;
    }}
    QScrollBar::handle:horizontal {{
        background: {sch};
        min-width: 20px;
        border-radius: 5px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {schh};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}

    /* === Status Bar === */
    QStatusBar {{
        background-color: {c.get("statusbar_background", "#0a0a15")};
        color: {c.get("statusbar_foreground", ac1)};
        border-top: 1px solid {bd1};
    }}

    /* === ToolBar === */
    QToolBar {{
        background-color: {c.get("toolbar_background", bg1)};
        border-bottom: 1px solid {bd1};
        spacing: 4px;
        padding: 2px;
    }}

    /* === ToolTip === */
    QToolTip {{
        background-color: {c.get("tooltip_background", bgi)};
        color: {c.get("tooltip_foreground", fg1)};
        border: 1px solid {c.get("tooltip_border", ac1)};
        padding: 4px;
    }}

    /* === Splitter === */
    QSplitter::handle {{
        background-color: {c.get("splitter", bd1)};
    }}
    QSplitter::handle:hover {{
        background-color: {c.get("splitter_hover", ac1)};
    }}

    /* === Tree / List Views === */
    QTreeView, QListView, QTableView {{
        background-color: {bgp};
        alternate-background-color: {bg2};
        border: 1px solid {bd1};
    }}
    QTreeView::item:selected, QListView::item:selected, QTableView::item:selected {{
        background-color: {bgs};
        color: {ac1};
    }}
    QTreeView::item:hover, QListView::item:hover {{
        background-color: {bgh};
    }}
    QHeaderView::section {{
        background-color: {bg2};
        color: {fg2};
        padding: 4px;
        border: 1px solid {bd1};
    }}

    /* === Progress Bar === */
    QProgressBar {{
        background-color: {bgi};
        border: 1px solid {bd1};
        border-radius: 3px;
        text-align: center;
        color: {fg1};
    }}
    QProgressBar::chunk {{
        background-color: {ac1};
        border-radius: 2px;
    }}

    /* === Group Box === */
    QGroupBox {{
        border: 1px solid {bd1};
        border-radius: 4px;
        margin-top: 8px;
        padding-top: 16px;
        font-weight: bold;
        color: {ac1};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 4px;
    }}

    /* === CheckBox / Radio === */
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 16px;
        height: 16px;
    }}
    QCheckBox::indicator:checked {{
        background-color: {ac1};
        border: 1px solid {ac1};
        border-radius: 3px;
    }}
    QCheckBox::indicator:unchecked {{
        background-color: {bgi};
        border: 1px solid {bd1};
        border-radius: 3px;
    }}
    """
