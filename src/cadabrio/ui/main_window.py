"""Main application window for Cadabrio.

Provides the primary workspace with dockable panels for 3D viewport,
AI chat console, asset browser, and configuration editor.
"""

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow,
    QDockWidget,
    QMenuBar,
    QStatusBar,
    QToolBar,
    QLabel,
)

from cadabrio.version import __version_display__
from cadabrio.config.manager import ConfigManager
from cadabrio.config.themes.theme_manager import ThemeManager


class CadabrioMainWindow(QMainWindow):
    """Primary application window with docking panel layout."""

    def __init__(self, config: ConfigManager, theme_mgr: ThemeManager):
        super().__init__()
        self._config = config
        self._theme_mgr = theme_mgr

        self.setMinimumSize(1280, 800)

        self._resource_monitor = None
        self._integrations: dict = {}

        # Active project
        from cadabrio.core.project import Project

        self._project = Project.new()
        self._update_title()

        self._build_menu_bar()
        self._build_tool_bar()
        self._build_status_bar()
        self._build_panels()

        # Restore window geometry if saved
        self._restore_state()

    def _build_menu_bar(self):
        """Build the main menu bar."""
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self._action("&New Project", "Ctrl+N", self._new_project))
        file_menu.addAction(self._action("&Open Project...", "Ctrl+O", self._open_project))
        file_menu.addSeparator()
        file_menu.addAction(self._action("&Save", "Ctrl+S", self._save_project))
        file_menu.addAction(self._action("Save &As...", "Ctrl+Shift+S", self._save_project_as))
        file_menu.addSeparator()
        file_menu.addAction(self._action("&Import...", "Ctrl+I"))
        file_menu.addAction(self._action("&Export...", "Ctrl+E"))
        file_menu.addSeparator()
        file_menu.addAction(self._action("E&xit", "Alt+F4", self.close))

        # Edit menu
        edit_menu = menu_bar.addMenu("&Edit")
        edit_menu.addAction(self._action("&Undo", "Ctrl+Z"))
        edit_menu.addAction(self._action("&Redo", "Ctrl+Y"))
        edit_menu.addSeparator()
        edit_menu.addAction(self._action("&Preferences...", "Ctrl+,", self._open_preferences))

        # View menu
        view_menu = menu_bar.addMenu("&View")
        view_menu.addAction(self._action("&3D Viewport", "F5"))
        view_menu.addAction(self._action("AI &Chat", "F6"))
        view_menu.addAction(self._action("&Asset Browser", "F7"))
        view_menu.addSeparator()
        self._resource_monitor_action = QAction("&Resource Monitor", self)
        self._resource_monitor_action.setShortcut(QKeySequence("F9"))
        self._resource_monitor_action.setCheckable(True)
        self._resource_monitor_action.toggled.connect(self._toggle_resource_monitor)
        view_menu.addAction(self._resource_monitor_action)
        view_menu.addSeparator()
        view_menu.addAction(self._action("&Full Screen", "F11"))

        # Tools menu
        tools_menu = menu_bar.addMenu("&Tools")
        tools_menu.addAction(self._action("&AI Tools...", "Ctrl+G", self._open_ai_tools))
        tools_menu.addAction(self._action("&Photogrammetry Pipeline..."))
        tools_menu.addAction(self._action("AI &Model Manager...", callback=self._open_model_manager))
        tools_menu.addSeparator()
        tools_menu.addAction(self._action("&Theme Editor...", callback=self._open_theme_editor))

        # Integrations menu — each entry gets a submenu with Launch / Locate
        self._int_menu = menu_bar.addMenu("&Integrations")
        self._int_actions = {}
        self._int_config_keys = {
            "Blender": "blender_path",
            "FreeCAD": "freecad_path",
            "Unreal Engine": "unreal_engine_path",
            "Bambu Studio": "bambu_studio_path",
        }
        for label, key in [
            ("&Blender", "Blender"),
            ("&FreeCAD", "FreeCAD"),
            ("&Unreal Engine", "Unreal Engine"),
            ("Bambu &Studio", "Bambu Studio"),
        ]:
            sub = self._int_menu.addMenu(label)
            launch_action = self._action(
                "Launch", callback=lambda checked=False, k=key: self._launch_integration(k)
            )
            launch_action.setEnabled(False)
            sub.addAction(launch_action)
            locate_action = self._action(
                "Locate...", callback=lambda checked=False, k=key: self._locate_integration(k)
            )
            sub.addAction(locate_action)
            self._int_actions[key] = {"launch": launch_action, "submenu": sub}
        self._int_menu.addSeparator()
        self._int_menu.addAction(
            self._action("&Refresh Detection", callback=self._redetect_integrations)
        )

        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction(self._action("&About Cadabrio", callback=self._show_about))
        help_menu.addAction(self._action("&Attributions", callback=self._show_attributions))

    def _build_tool_bar(self):
        """Build the main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(toolbar)

        # Placeholder toolbar items
        toolbar.addAction(self._action("New"))
        toolbar.addAction(self._action("Open"))
        toolbar.addAction(self._action("Save"))
        toolbar.addSeparator()
        toolbar.addAction(self._action("Import"))
        toolbar.addAction(self._action("Export"))

    def _build_status_bar(self):
        """Build the status bar."""
        status = QStatusBar()
        self.setStatusBar(status)
        self._status_label = QLabel("Ready")
        status.addWidget(self._status_label)

        self._integrations_label = QLabel("Integrations: ...")
        status.addPermanentWidget(self._integrations_label)

        self._gpu_label = QLabel("GPU: Detecting...")
        status.addPermanentWidget(self._gpu_label)
        self._detect_gpu()

    def _detect_gpu(self):
        """Detect and display GPU info in status bar."""
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self._gpu_label.setText(f"GPU: {name} ({mem:.0f} GB)")
            else:
                self._gpu_label.setText("GPU: CUDA not available")
        except ImportError:
            self._gpu_label.setText("GPU: PyTorch not installed")

    def _build_panels(self):
        """Create the dockable panels."""
        from cadabrio.ui.panels.viewport_3d import Viewport3DPanel
        from cadabrio.ui.panels.chat_console import ChatConsolePanel
        from cadabrio.ui.panels.asset_browser import AssetBrowserPanel

        # 3D Viewport (central)
        self._viewport = Viewport3DPanel(self._config)
        self.setCentralWidget(self._viewport)

        # AI Chat Console (right dock)
        self._chat_dock = QDockWidget("AI Chat", self)
        self._chat_panel = ChatConsolePanel(self._config)
        self._chat_dock.setWidget(self._chat_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._chat_dock)

        # Asset Browser (bottom dock)
        self._asset_dock = QDockWidget("Asset Browser", self)
        self._asset_panel = AssetBrowserPanel(self._config)
        self._asset_panel.asset_imported.connect(self._on_asset_imported)
        self._asset_dock.setWidget(self._asset_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._asset_dock)

    def _action(self, text: str, shortcut: str = None, callback=None) -> QAction:
        """Helper to create a QAction."""
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        if callback:
            action.triggered.connect(callback)
        return action

    # -------------------------------------------------------------------
    # Project operations
    # -------------------------------------------------------------------

    def _update_title(self):
        """Update window title to reflect current project."""
        dirty = " *" if self._project.dirty else ""
        self.setWindowTitle(f"{self._project.name}{dirty} — {__version_display__}")

    def _confirm_discard(self) -> bool:
        """If project has unsaved changes, ask user. Returns True to proceed."""
        if not self._project.dirty:
            return True
        from PySide6.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            f"Project '{self._project.name}' has unsaved changes.\n\nDiscard changes?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if reply == QMessageBox.StandardButton.Save:
            self._save_project()
            return True
        if reply == QMessageBox.StandardButton.Discard:
            return True
        return False  # Cancel

    def _new_project(self):
        """Create a new empty project."""
        if not self._confirm_discard():
            return
        from cadabrio.core.project import Project

        self._project = Project.new()
        self._status_label.setText("New project created")
        self._update_title()

    def _open_project(self):
        """Open a project file from disk."""
        if not self._confirm_discard():
            return
        from PySide6.QtWidgets import QFileDialog
        from cadabrio.core.project import Project, PROJECT_EXTENSION

        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "",
            f"Cadabrio Projects (*{PROJECT_EXTENSION});;All Files (*)"
        )
        if not path:
            return
        try:
            self._project = Project.load(path)
            self._status_label.setText(f"Opened: {self._project.name}")
            self._update_title()
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Open Failed", f"Could not open project:\n\n{e}")

    def _save_project(self):
        """Save the current project. Prompts for path if new."""
        if self._project.path is None:
            self._save_project_as()
            return
        try:
            self._project.save()
            self._status_label.setText(f"Saved: {self._project.path.name}")
            self._update_title()
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Save Failed", f"Could not save project:\n\n{e}")

    def _save_project_as(self):
        """Save the current project to a new path."""
        from PySide6.QtWidgets import QFileDialog
        from cadabrio.core.project import PROJECT_EXTENSION

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", self._project.name,
            f"Cadabrio Projects (*{PROJECT_EXTENSION});;All Files (*)"
        )
        if not path:
            return
        try:
            self._project.save(path)
            self._status_label.setText(f"Saved: {self._project.path.name}")
            self._update_title()
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Save Failed", f"Could not save project:\n\n{e}")

    def _open_preferences(self):
        """Open the configuration editor dialog."""
        from cadabrio.ui.panels.config_editor import ConfigEditorDialog

        dialog = ConfigEditorDialog(self._config, self._theme_mgr, self)
        dialog.exec()

    def _show_about(self):
        """Show the About dialog."""
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.about(
            self,
            "About Cadabrio",
            f"<h2>{__version_display__}</h2>"
            "<p>AI-powered, locally-run 3D creation platform.</p>"
            "<p>Cadabrio unifies text, images, video, photogrammetry, and "
            "existing models into a single intelligent workspace.</p>"
            "<hr>"
            "<p><b>Created by:</b> Lee Gillie, CCP</p>"
            "<p><b>AI Assistant:</b> Claude (Anthropic)</p>"
            "<p><b>License:</b> MIT</p>"
            "<hr>"
            "<p>Built with the best of open source.</p>",
        )

    def _show_attributions(self):
        """Show the open-source attributions dialog."""
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.information(
            self,
            "Attributions",
            "<h3>Cadabrio Open Source Attributions</h3>"
            "<p>Cadabrio is built upon the work of many open-source projects:</p>"
            "<ul>"
            "<li><b>Qt / PySide6</b> - UI framework (LGPL)</li>"
            "<li><b>PyTorch</b> - AI inference engine (BSD)</li>"
            "<li><b>OpenCV</b> - Computer vision (Apache 2.0)</li>"
            "<li><b>Open3D</b> - 3D data processing (MIT)</li>"
            "<li><b>trimesh</b> - Mesh operations (MIT)</li>"
            "<li><b>Blender</b> - 3D creation suite (GPL)</li>"
            "<li><b>FreeCAD</b> - Parametric 3D CAD (LGPL)</li>"
            "<li><b>Hugging Face</b> - Model hub & transformers (Apache 2.0)</li>"
            "</ul>"
            "<p>Full attribution details in ATTRIBUTIONS.md</p>",
        )

    def _open_ai_tools(self):
        """Open the AI Tools dialog."""
        from cadabrio.ui.widgets.ai_tools_dialog import AIToolsDialog

        dialog = AIToolsDialog(self._config, self)
        dialog.exec()

    def _open_theme_editor(self):
        """Open the Theme Editor dialog."""
        from cadabrio.ui.widgets.theme_editor import ThemeEditorDialog

        dialog = ThemeEditorDialog(self._config, self._theme_mgr, self)
        dialog.exec()

    def _open_model_manager(self):
        """Open the AI Model Manager dialog."""
        from cadabrio.ui.widgets.model_manager_dialog import ModelManagerDialog

        dialog = ModelManagerDialog(self._config, self)
        dialog.exec()

    def _launch_integration(self, name: str):
        """Launch an external integration application."""
        import subprocess
        from PySide6.QtWidgets import QMessageBox

        integration = self._integrations.get(name)
        if not integration or not integration.available:
            QMessageBox.warning(
                self, "Not Available",
                f"{name} was not detected on this system.\n\n"
                "Check Preferences > Integrations to set the path manually.",
            )
            return

        exe_path = integration.executable_path
        if not exe_path or not exe_path.exists():
            QMessageBox.warning(self, "Not Found", f"Could not find {name} executable.")
            return

        try:
            subprocess.Popen([str(exe_path)], start_new_session=True)
            self._status_label.setText(f"Launched {name}")
        except Exception as e:
            QMessageBox.critical(self, "Launch Failed", f"Could not launch {name}:\n\n{e}")

    def _locate_integration(self, name: str):
        """Let the user browse for an integration executable."""
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, f"Locate {name}", "",
            "Executables (*.exe);;All Files (*)"
        )
        if not path:
            return
        config_key = self._int_config_keys.get(name)
        if config_key:
            self._config.set("integrations", config_key, path)
            self._config.save()
        self._redetect_integrations()

    def _redetect_integrations(self):
        """Re-run integration detection and update the menu."""
        from cadabrio.integrations.blender import BlenderIntegration
        from cadabrio.integrations.freecad import FreecadIntegration
        from cadabrio.integrations.unreal import UnrealIntegration
        from cadabrio.integrations.bambu_studio import BambuStudioIntegration

        integrations = {
            "Blender": BlenderIntegration(self._config),
            "FreeCAD": FreecadIntegration(self._config),
            "Unreal Engine": UnrealIntegration(self._config),
            "Bambu Studio": BambuStudioIntegration(self._config),
        }
        detected = {}
        for iname, integration in integrations.items():
            detected[iname] = integration.detect()
        self.set_integrations_status(detected, integrations)
        self._status_label.setText("Integration detection refreshed")

    def _toggle_resource_monitor(self, checked: bool):
        """Show or hide the floating resource monitor window."""
        if checked:
            if self._resource_monitor is None:
                from cadabrio.ui.panels.resource_monitor import ResourceMonitorWindow

                self._resource_monitor = ResourceMonitorWindow(self)
                self._resource_monitor.on_closed = self._on_resource_monitor_closed
            self._resource_monitor.show()
            self._resource_monitor.raise_()
        else:
            if self._resource_monitor is not None:
                self._resource_monitor.close()
                self._resource_monitor = None

    def _on_resource_monitor_closed(self):
        """Called when the resource monitor is closed via its X button."""
        self._resource_monitor = None
        self._resource_monitor_action.setChecked(False)

    def _on_asset_imported(self, metadata: dict):
        """Handle a newly imported asset from the asset browser."""
        self._project.add_asset(metadata)
        self._update_title()

    def set_integrations_status(self, detected: dict[str, bool], instances: dict | None = None):
        """Update the status bar and menu with detected integrations."""
        if instances is not None:
            self._integrations = instances
        found = [name for name, ok in detected.items() if ok]
        if found:
            self._integrations_label.setText(f"Integrations: {', '.join(found)}")
        else:
            self._integrations_label.setText("Integrations: None detected")
        # Update menu items: enable launch, show status in submenu title
        for name, ok in detected.items():
            if name not in self._int_actions:
                continue
            entry = self._int_actions[name]
            entry["launch"].setEnabled(ok)
            indicator = "\u2714" if ok else "\u2718"  # checkmark / X
            entry["submenu"].setTitle(f"{indicator}  {name}")

    def _restore_state(self):
        """Restore window geometry from config (placeholder)."""
        pass

    def closeEvent(self, event):
        """Save state before closing. Prompt if project is unsaved."""
        if not self._confirm_discard():
            event.ignore()
            return
        if self._resource_monitor is not None:
            self._resource_monitor.close()
            self._resource_monitor = None
        self._config.save()
        event.accept()
