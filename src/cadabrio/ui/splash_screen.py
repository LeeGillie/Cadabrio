"""Splash screen for Cadabrio.

Displays Art and Branding/promo5.png during application initialization,
with status messages in the bottom-left and version in the upper-right.
"""

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QPainter, QFont, QColor
from PySide6.QtWidgets import QSplashScreen, QApplication

from cadabrio.version import __version_display__


_SPLASH_IMAGE = Path(__file__).parent.parent.parent.parent / "Art and Branding" / "promo5.png"


class CadabrioSplashScreen(QSplashScreen):
    """Custom splash screen with version display and status messages."""

    def __init__(self, pixmap: QPixmap):
        super().__init__(pixmap)
        self._status_message = ""
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)

    def drawContents(self, painter: QPainter):
        """Draw version in upper-right and status in bottom-left."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        # --- Version in upper-right corner ---
        version_font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        painter.setFont(version_font)

        # Draw text shadow
        painter.setPen(QColor(0, 0, 0, 180))
        version_rect = rect.adjusted(0, 10, -12, 0)
        painter.drawText(
            version_rect.adjusted(1, 1, 1, 1),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
            __version_display__,
        )

        # Draw version text
        painter.setPen(QColor(57, 255, 20))  # Neon green #39ff14
        painter.drawText(
            version_rect,
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight,
            __version_display__,
        )

        # --- Status message in bottom-left corner ---
        if self._status_message:
            status_font = QFont("Segoe UI", 8)
            painter.setFont(status_font)

            status_rect = rect.adjusted(10, 0, 0, -8)

            # Draw text shadow
            painter.setPen(QColor(0, 0, 0, 180))
            painter.drawText(
                status_rect.adjusted(1, 1, 1, 1),
                Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
                self._status_message,
            )

            # Draw status text
            painter.setPen(QColor(224, 224, 224))  # Light gray
            painter.drawText(
                status_rect,
                Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
                self._status_message,
            )

    def showStatusMessage(self, message: str):
        """Update the status message and repaint."""
        self._status_message = message
        self.repaint()
        QApplication.processEvents()


def show_splash(app: QApplication) -> CadabrioSplashScreen:
    """Create and show the splash screen. Returns the splash for later finish()."""
    pixmap = QPixmap(str(_SPLASH_IMAGE))
    if pixmap.isNull():
        # Fallback: create a simple dark pixmap if image not found
        pixmap = QPixmap(800, 450)
        pixmap.fill(QColor("#0d0d1a"))
    else:
        # Show the full image at half size
        pixmap = pixmap.scaled(
            pixmap.width() // 2, pixmap.height() // 2,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    splash = CadabrioSplashScreen(pixmap)
    splash.show()
    QApplication.processEvents()
    return splash
