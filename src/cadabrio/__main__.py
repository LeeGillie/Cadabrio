"""Cadabrio application entry point."""

import sys


def main():
    """Launch the Cadabrio application."""
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon

    from cadabrio.ui.splash_screen import show_splash
    from cadabrio.ui.main_window import CadabrioMainWindow
    from cadabrio.config.manager import ConfigManager

    # High-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Cadabrio")
    app.setOrganizationName("Cadabrio")

    # Set application icon
    icon_path = str(
        __import__("pathlib").Path(__file__).parent.parent.parent
        / "Art and Branding"
        / "Cadabrio-02.ico"
    )
    app.setWindowIcon(QIcon(icon_path))

    # Show splash screen and begin initialization
    splash = show_splash(app)

    # Load configuration
    splash.showStatusMessage("Loading configuration...")
    config = ConfigManager()
    config.load()

    # Initialize logging (file + console sinks)
    splash.showStatusMessage("Starting logging...")
    from cadabrio.core.logging import setup_logging

    setup_logging(config)

    # Apply theme
    splash.showStatusMessage("Applying theme...")
    from cadabrio.config.themes.theme_manager import ThemeManager

    theme_mgr = ThemeManager(config)
    theme_mgr.apply_theme(app)

    # Create main window
    splash.showStatusMessage("Building workspace...")
    window = CadabrioMainWindow(config, theme_mgr)

    # Initialize subsystems
    splash.showStatusMessage("Initializing AI engine...")
    # AI engine init will go here

    splash.showStatusMessage("Detecting integrations...")
    from cadabrio.integrations.blender import BlenderIntegration
    from cadabrio.integrations.freecad import FreecadIntegration
    from cadabrio.integrations.unreal import UnrealIntegration
    from cadabrio.integrations.bambu_studio import BambuStudioIntegration

    integrations = {
        "Blender": BlenderIntegration(config),
        "FreeCAD": FreecadIntegration(config),
        "Unreal Engine": UnrealIntegration(config),
        "Bambu Studio": BambuStudioIntegration(config),
    }
    detected = {}
    for name, integration in integrations.items():
        splash.showStatusMessage(f"Detecting {name}...")
        detected[name] = integration.detect()

    window.set_integrations_status(detected, integrations)

    splash.showStatusMessage("Ready.")
    window.show()
    splash.finish(window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
