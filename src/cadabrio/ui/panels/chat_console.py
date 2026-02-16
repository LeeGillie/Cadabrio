"""AI Chat Console panel for Cadabrio.

Provides a conversational interface to interact with local AI models
for 3D creation guidance, workflow suggestions, and asset generation.
"""

import html
import threading

from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QLabel,
    QComboBox,
)

from loguru import logger

from cadabrio.config.manager import ConfigManager


# VRAM fit indicator symbols (matches AI Tools dialog pattern)
_VRAM_SYMBOLS = {
    "good": "\u2705",
    "tight": "\u26a0\ufe0f",
    "bad": "\u274c",
    "unknown": "",
}


class _ChatSignals(QObject):
    """Signals emitted by the chat worker thread."""

    response_ready = Signal(str)
    error = Signal(str)


class ChatConsolePanel(QWidget):
    """AI chat/conversation console panel."""

    def __init__(self, config: ConfigManager, parent=None):
        super().__init__(parent)
        self._config = config
        self._messages: list[dict[str, str]] = []
        self._signals = _ChatSignals()
        self._signals.response_ready.connect(self._on_response)
        self._signals.error.connect(self._on_error)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header
        header = QLabel("AI Assistant")
        header.setStyleSheet("font-weight: bold; color: #39ff14; padding: 4px;")
        layout.addWidget(header)

        # Model selector row
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(250)
        model_layout.addWidget(self._model_combo, 1)
        layout.addLayout(model_layout)

        self._populate_model_selector()
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)

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

    # -------------------------------------------------------------------
    # Model selector
    # -------------------------------------------------------------------

    def _populate_model_selector(self):
        """Fill the model combo with local CHAT models."""
        self._model_combo.blockSignals(True)
        self._model_combo.clear()

        try:
            from cadabrio.ai.model_manager import (
                ModelManager, ModelCategory, get_gpu_vram_gb,
            )
            mgr = ModelManager(self._config)
            mgr.scan_local_models()
            local_models = mgr.list_models(ModelCategory.CHAT)
            vram = get_gpu_vram_gb()

            for m in local_models:
                fit = m.vram_fit
                symbol = _VRAM_SYMBOLS.get(fit, "")
                label = f"{symbol} {m.model_id}  ({m.size_display})".strip()
                self._model_combo.addItem(label, m.model_id)
                idx = self._model_combo.count() - 1
                if m.estimated_vram_gb > 0 and vram > 0:
                    self._model_combo.setItemData(
                        idx,
                        f"Size: {m.size_display}, Est. VRAM: {m.estimated_vram_gb:.1f} GB, GPU: {vram:.0f} GB",
                        Qt.ItemDataRole.ToolTipRole,
                    )
        except Exception as e:
            logger.debug(f"Could not scan chat models: {e}")

        if self._model_combo.count() == 0:
            self._model_combo.addItem("No chat models downloaded", "")

        self._model_combo.addItem("Custom model ID...", "__custom__")

        # Restore saved selection
        saved_id = self._config.get("ai", "selected_chat_model", "")
        if saved_id:
            for i in range(self._model_combo.count()):
                if self._model_combo.itemData(i) == saved_id:
                    self._model_combo.setCurrentIndex(i)
                    break

        self._model_combo.blockSignals(False)

    def _on_model_changed(self):
        """Persist model selection to config."""
        model_id = self._model_combo.currentData()
        if not model_id or model_id == "__custom__":
            return
        self._config.set("ai", "selected_chat_model", model_id)
        self._config.save()

    def _get_selected_model(self) -> str:
        """Return the currently selected model ID, handling custom entry."""
        model_id = self._model_combo.currentData()
        if model_id == "__custom__":
            from PySide6.QtWidgets import QInputDialog
            text, ok = QInputDialog.getText(
                self, "Custom Model",
                "Enter a HuggingFace model ID (e.g. Qwen/Qwen2.5-7B-Instruct):",
            )
            if ok and text.strip():
                return text.strip()
            return ""
        return model_id or ""

    # -------------------------------------------------------------------
    # Chat messaging
    # -------------------------------------------------------------------

    def _send_message(self):
        """Handle sending a chat message."""
        text = self._input_field.text().strip()
        if not text:
            return

        model_id = self._get_selected_model()
        if not model_id:
            self._chat_display.append(
                '<p style="color: #ff6b6b;"><b>Error:</b> '
                "No chat model selected. Download a model via the Model Manager first.</p>"
            )
            return

        # Show user message
        self._chat_display.append(
            f'<p style="color: #e0e0e0;"><b>You:</b> {html.escape(text)}</p>'
        )
        self._input_field.clear()

        # Add to conversation history
        self._messages.append({"role": "user", "content": text})

        # Disable input while generating
        self._input_field.setEnabled(False)
        self._send_btn.setEnabled(False)
        self._chat_display.append(
            '<p style="color: #a0a0b0;"><i>Generating...</i></p>'
        )

        # Launch worker thread
        thread = threading.Thread(
            target=self._worker,
            args=(list(self._messages), model_id),
            daemon=True,
        )
        thread.start()

    def _worker(self, messages: list[dict[str, str]], model_id: str):
        """Run inference in a background thread."""
        try:
            from cadabrio.ai.pipelines import chat_generate
            response = chat_generate(messages, model_id)
            self._signals.response_ready.emit(response)
        except Exception as e:
            logger.exception(f"Chat inference failed: {e!r}")
            self._signals.error.emit(repr(e) if not str(e) else str(e))

    def _on_response(self, text: str):
        """Handle completed inference response (called on main thread)."""
        # Remove "Generating..." indicator
        self._remove_last_paragraph()

        self._messages.append({"role": "assistant", "content": text})
        self._chat_display.append(
            f'<p style="color: #39ff14;"><b>Cadabrio AI:</b> {html.escape(text)}</p>'
        )

        self._input_field.setEnabled(True)
        self._send_btn.setEnabled(True)
        self._input_field.setFocus()

    def _on_error(self, error_msg: str):
        """Handle inference error (called on main thread)."""
        self._remove_last_paragraph()

        self._chat_display.append(
            f'<p style="color: #ff6b6b;"><b>Error:</b> {html.escape(error_msg)}</p>'
        )

        # Remove the failed user message from history so it can be retried
        if self._messages and self._messages[-1]["role"] == "user":
            self._messages.pop()

        self._input_field.setEnabled(True)
        self._send_btn.setEnabled(True)
        self._input_field.setFocus()

    def _remove_last_paragraph(self):
        """Remove the last paragraph (the 'Generating...' indicator) from the display."""
        cursor = self._chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.movePosition(
            cursor.MoveOperation.StartOfBlock, cursor.MoveMode.KeepAnchor,
        )
        # Select to start of block then extend back one more char to get the block break
        cursor.movePosition(
            cursor.MoveOperation.PreviousBlock, cursor.MoveMode.KeepAnchor,
        )
        cursor.movePosition(
            cursor.MoveOperation.EndOfBlock, cursor.MoveMode.KeepAnchor,
        )
        cursor.removeSelectedText()
        # Remove the trailing empty block
        if not cursor.atStart():
            cursor.deletePreviousChar()

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
            f'<p style="color: {color};"><b>{label}:</b> {html.escape(message)}</p>'
        )
