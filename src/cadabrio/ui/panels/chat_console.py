"""AI Chat Console panel for Cadabrio.

Provides a conversational interface to interact with local AI models
for 3D creation guidance, workflow suggestions, and asset generation.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QLabel,
)

from cadabrio.config.manager import ConfigManager


class ChatConsolePanel(QWidget):
    """AI chat/conversation console panel."""

    def __init__(self, config: ConfigManager, parent=None):
        super().__init__(parent)
        self._config = config

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header
        header = QLabel("AI Assistant")
        header.setStyleSheet("font-weight: bold; color: #39ff14; padding: 4px;")
        layout.addWidget(header)

        # Chat history display
        self._chat_display = QTextEdit()
        self._chat_display.setReadOnly(True)
        self._chat_display.setPlaceholderText(
            "Chat with your AI assistant about 3D creation...\n\n"
            "Try: 'Create a low-poly tree' or 'Help me set up photogrammetry'"
        )
        layout.addWidget(self._chat_display)

        # Input area
        input_layout = QHBoxLayout()
        self._input_field = QLineEdit()
        self._input_field.setPlaceholderText("Type a message...")
        self._input_field.returnPressed.connect(self._send_message)
        input_layout.addWidget(self._input_field)

        self._send_btn = QPushButton("Send")
        self._send_btn.clicked.connect(self._send_message)
        self._send_btn.setFixedWidth(60)
        input_layout.addWidget(self._send_btn)

        layout.addLayout(input_layout)

    def _send_message(self):
        """Handle sending a chat message."""
        text = self._input_field.text().strip()
        if not text:
            return

        self._chat_display.append(f'<p style="color: #e0e0e0;"><b>You:</b> {text}</p>')
        self._input_field.clear()

        # TODO: Route to AI engine for inference
        self._chat_display.append(
            '<p style="color: #39ff14;"><b>Cadabrio AI:</b> '
            "AI engine not yet connected. This will use local models for inference.</p>"
        )

    def append_message(self, role: str, message: str):
        """Programmatically append a message to the chat display."""
        if role == "user":
            color = "#e0e0e0"
            label = "You"
        elif role == "assistant":
            color = "#39ff14"
            label = "Cadabrio AI"
        else:
            color = "#a0a0b0"
            label = role
        self._chat_display.append(
            f'<p style="color: {color};"><b>{label}:</b> {message}</p>'
        )
