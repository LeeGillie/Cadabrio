"""AI Tools dialog for Cadabrio.

Provides a unified interface for all AI capabilities:
text-to-image, depth estimation, image-to-3D, and segmentation.
Each tab has a model selector showing locally available models for that purpose
with VRAM fit indicators. Generation shows real step progress with cancel.

The Text to Image tab supports a multi-reference image curation workflow:
search for candidates, accept/reject, annotate, and generate with references.
"""

import threading
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QObject, QSize
from PySide6.QtGui import QPixmap, QImage, QGuiApplication, QColor
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QWidget,
    QProgressBar,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QCheckBox,
    QGroupBox,
    QTextEdit,
    QScrollArea,
    QMenu,
    QGridLayout,
    QSizePolicy,
)

from loguru import logger
from PIL import Image

from cadabrio.ai.model_manager import (
    ModelManager, ModelCategory, get_gpu_vram_gb, vram_fit_level,
)
from cadabrio.ai.reference_images import (
    ReferenceImage, ReferenceCollection, ReferenceStatus,
)
from cadabrio.ai import pipelines


# Default model per purpose — used when nothing local is available
_DEFAULTS = {
    ModelCategory.TEXT_TO_IMAGE: pipelines.DEFAULT_TXT2IMG,
    ModelCategory.DEPTH_ESTIMATION: pipelines.DEFAULT_DEPTH,
    ModelCategory.IMAGE_TO_3D: pipelines.DEFAULT_TRIPOSR,
    ModelCategory.SEGMENTATION: pipelines.DEFAULT_SAM,
}

# VRAM fit indicator text for model combo
_VRAM_SYMBOLS = {
    "good": "\u2705",     # green check
    "tight": "\u26a0\ufe0f",  # warning
    "bad": "\u274c",      # red X
    "unknown": "",
}

# Config keys for persisting model selections per category
_MODEL_CONFIG_KEYS = {
    ModelCategory.TEXT_TO_IMAGE: "selected_txt2img_model",
    ModelCategory.DEPTH_ESTIMATION: "selected_depth_model",
    ModelCategory.IMAGE_TO_3D: "selected_img2mesh_model",
    ModelCategory.SEGMENTATION: "selected_segmentation_model",
}


def _qimage_to_pil(qimg: QImage) -> Image.Image | None:
    """Convert a QImage from the clipboard to a PIL Image."""
    if qimg.isNull():
        return None
    qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    width, height = qimg.width(), qimg.height()
    ptr = qimg.bits()
    if ptr is None:
        return None
    # PySide6 returns a memoryview; convert to bytes
    raw = bytes(ptr)
    return Image.frombytes("RGBA", (width, height), raw).convert("RGB")


class _WorkerSignals(QObject):
    started = Signal(str)
    finished = Signal(str, object)
    progress = Signal(int, int)  # step, total_steps
    search_done = Signal(dict)  # prompt enhancement result
    candidates_found = Signal(list)  # list of candidate dicts from image search


class _ImagePreview(QLabel):
    """A label that displays a PIL image scaled to fit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(256, 256)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #0a0a12; border: 1px solid #2a2a40; border-radius: 4px;")
        self.setText("No image")
        self._pil_image = None

    def set_pil_image(self, img: Image.Image):
        self._pil_image = img
        self._update_display()

    def _update_display(self):
        if self._pil_image is None:
            return
        img = self._pil_image.convert("RGBA")
        data = img.tobytes("raw", "RGBA")
        qimg = QImage(data, img.width, img.height, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def contextMenuEvent(self, event):
        if self._pil_image is None:
            return
        menu = QMenu(self)
        copy_action = menu.addAction("Copy Image")
        save_action = menu.addAction("Save As...")
        action = menu.exec(event.globalPos())
        if action == copy_action:
            self._copy_to_clipboard()
        elif action == save_action:
            self._save_as()

    def _copy_to_clipboard(self):
        if self._pil_image is None:
            return
        img = self._pil_image.convert("RGBA")
        data = img.tobytes("raw", "RGBA")
        qimg = QImage(data, img.width, img.height, QImage.Format.Format_RGBA8888)
        QGuiApplication.clipboard().setImage(qimg)

    def _save_as(self):
        if self._pil_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG (*.png);;JPEG (*.jpg);;All Files (*)"
        )
        if path:
            self._pil_image.save(path)

    @property
    def pil_image(self) -> Image.Image | None:
        return self._pil_image


# -------------------------------------------------------------------
# Reference image gallery widgets
# -------------------------------------------------------------------

class _ReferenceCard(QWidget):
    """A ~160x210 card displaying a single reference image candidate."""

    accepted = Signal(object)  # emits the ReferenceImage
    rejected = Signal(object)

    def __init__(self, ref: ReferenceImage, parent=None):
        super().__init__(parent)
        self.ref = ref
        self.setFixedSize(160, 210)
        self.setStyleSheet(
            "background: #12121e; border: 1px solid #2a2a40; border-radius: 4px;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Thumbnail
        self._thumb = QLabel()
        self._thumb.setFixedSize(148, 100)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setStyleSheet("background: #0a0a12; border: 1px solid #1a1a2e;")
        if ref.image is not None:
            self._set_thumbnail(ref.image)
        else:
            self._thumb.setText("No preview")
        layout.addWidget(self._thumb)

        # Title
        title_text = ref.title[:40] if ref.title else "Image"
        self._title = QLabel(title_text)
        self._title.setStyleSheet("color: #c0c0d0; font-size: 8pt;")
        self._title.setWordWrap(True)
        self._title.setMaximumHeight(28)
        if ref.title:
            self._title.setToolTip(ref.title)
        layout.addWidget(self._title)

        # Status / buttons row
        self._btn_row = QWidget()
        btn_layout = QHBoxLayout(self._btn_row)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(4)

        if ref.status == ReferenceStatus.USER_ADDED:
            status_lbl = QLabel("Added")
            status_lbl.setStyleSheet("color: #39ff14; font-size: 8pt; font-weight: bold;")
            btn_layout.addWidget(status_lbl)
            btn_layout.addStretch()
        elif ref.status in (ReferenceStatus.ACCEPTED, ReferenceStatus.REJECTED):
            self._show_decided_state(btn_layout)
        else:
            self._accept_btn = QPushButton("Accept")
            self._accept_btn.setFixedHeight(22)
            self._accept_btn.setStyleSheet(
                "background: #1a3a1a; color: #39ff14; font-size: 8pt; "
                "border: 1px solid #2a4a2a; padding: 1px 8px;"
            )
            self._accept_btn.clicked.connect(self._on_accept)
            btn_layout.addWidget(self._accept_btn)

            self._reject_btn = QPushButton("Reject")
            self._reject_btn.setFixedHeight(22)
            self._reject_btn.setStyleSheet(
                "background: #3a1a1a; color: #ff6060; font-size: 8pt; "
                "border: 1px solid #4a2a2a; padding: 1px 8px;"
            )
            self._reject_btn.clicked.connect(self._on_reject)
            btn_layout.addWidget(self._reject_btn)

        layout.addWidget(self._btn_row)

        # Annotation field (visible for accepted/user-added items)
        self._annotation = QLineEdit()
        self._annotation.setPlaceholderText("Note: what does this show?")
        self._annotation.setStyleSheet("font-size: 8pt; padding: 2px;")
        self._annotation.setMaximumHeight(22)
        self._annotation.textChanged.connect(self._on_annotation_changed)
        if ref.annotation:
            self._annotation.setText(ref.annotation)
        visible = ref.status in (ReferenceStatus.ACCEPTED, ReferenceStatus.USER_ADDED)
        self._annotation.setVisible(visible)
        layout.addWidget(self._annotation)

    def _set_thumbnail(self, img: Image.Image):
        rgba = img.convert("RGBA")
        data = rgba.tobytes("raw", "RGBA")
        qimg = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            QSize(148, 100),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumb.setPixmap(scaled)

    def _show_decided_state(self, btn_layout):
        if self.ref.status == ReferenceStatus.ACCEPTED:
            lbl = QLabel("Accepted")
            lbl.setStyleSheet("color: #39ff14; font-size: 8pt; font-weight: bold;")
            btn_layout.addWidget(lbl)
        else:
            lbl = QLabel("Rejected")
            lbl.setStyleSheet("color: #ff6060; font-size: 8pt;")
            btn_layout.addWidget(lbl)
            self.setStyleSheet(
                "background: #0a0a10; border: 1px solid #1a1a20; border-radius: 4px; opacity: 0.5;"
            )
        btn_layout.addStretch()

    def _on_accept(self):
        self.ref.status = ReferenceStatus.ACCEPTED
        self.accepted.emit(self.ref)
        # Update card UI
        self._accept_btn.setVisible(False)
        self._reject_btn.setVisible(False)
        # Replace buttons with status label
        lbl = QLabel("Accepted")
        lbl.setStyleSheet("color: #39ff14; font-size: 8pt; font-weight: bold;")
        self._btn_row.layout().addWidget(lbl)
        self._annotation.setVisible(True)
        self.setStyleSheet(
            "background: #12121e; border: 2px solid #39ff14; border-radius: 4px;"
        )

    def _on_reject(self):
        self.ref.status = ReferenceStatus.REJECTED
        self.rejected.emit(self.ref)
        self._accept_btn.setVisible(False)
        self._reject_btn.setVisible(False)
        lbl = QLabel("Rejected")
        lbl.setStyleSheet("color: #ff6060; font-size: 8pt;")
        self._btn_row.layout().addWidget(lbl)
        self._annotation.setVisible(False)
        self.setStyleSheet(
            "background: #0a0a10; border: 1px solid #1a1a20; border-radius: 4px;"
        )

    def _on_annotation_changed(self, text: str):
        self.ref.annotation = text


class _ReferenceGallery(QGroupBox):
    """Scrollable gallery of reference image cards with action buttons."""

    search_more = Signal()
    paste_image = Signal()
    browse_image = Signal()
    clear_all = Signal()

    def __init__(self, parent=None):
        super().__init__("Reference Images", parent)
        self._cards: list[_ReferenceCard] = []

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Action bar
        action_row = QHBoxLayout()
        self._search_more_btn = QPushButton("Search for More")
        self._search_more_btn.setStyleSheet("padding: 4px 12px;")
        self._search_more_btn.clicked.connect(self.search_more.emit)
        action_row.addWidget(self._search_more_btn)

        paste_btn = QPushButton("Paste from Clipboard")
        paste_btn.setStyleSheet("padding: 4px 12px;")
        paste_btn.clicked.connect(self.paste_image.emit)
        action_row.addWidget(paste_btn)

        browse_btn = QPushButton("Browse...")
        browse_btn.setStyleSheet("padding: 4px 12px;")
        browse_btn.clicked.connect(self.browse_image.emit)
        action_row.addWidget(browse_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.setStyleSheet("padding: 4px 12px; color: #ff6060;")
        clear_btn.clicked.connect(self.clear_all.emit)
        action_row.addWidget(clear_btn)

        action_row.addStretch()
        layout.addLayout(action_row)

        # Status label
        self._status_label = QLabel("No reference images yet")
        self._status_label.setStyleSheet("color: #6a6a80; font-size: 8pt;")
        layout.addWidget(self._status_label)

        # Scrollable card grid
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setMinimumHeight(230)
        self._scroll.setMaximumHeight(460)

        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setSpacing(6)
        self._grid_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._scroll.setWidget(self._grid_widget)
        layout.addWidget(self._scroll)

    def add_card(self, ref: ReferenceImage) -> _ReferenceCard:
        """Add a reference card to the grid."""
        card = _ReferenceCard(ref)
        self._cards.append(card)
        self._relayout_grid()
        self._update_status()
        return card

    def clear(self):
        """Remove all cards from the gallery."""
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()
        self._update_status()

    def _relayout_grid(self):
        """Reposition all cards in a 4-column grid."""
        # Clear the grid layout
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            # Don't delete — just remove from layout
        cols = 4
        for i, card in enumerate(self._cards):
            row, col = divmod(i, cols)
            self._grid_layout.addWidget(card, row, col)

    def _update_status(self):
        if not self._cards:
            self._status_label.setText("No reference images yet")
            return
        accepted = sum(
            1 for c in self._cards
            if c.ref.status in (ReferenceStatus.ACCEPTED, ReferenceStatus.USER_ADDED)
        )
        candidates = sum(
            1 for c in self._cards if c.ref.status == ReferenceStatus.CANDIDATE
        )
        rejected = sum(
            1 for c in self._cards if c.ref.status == ReferenceStatus.REJECTED
        )
        parts = []
        if accepted:
            parts.append(f"{accepted} accepted")
        if candidates:
            parts.append(f"{candidates} candidates")
        if rejected:
            parts.append(f"{rejected} rejected")
        self._status_label.setText(", ".join(parts))

    def update_status(self):
        """Public method to refresh the status label after external changes."""
        self._update_status()


class AIToolsDialog(QDialog):
    """Unified dialog for AI generation tools with per-task model selection."""

    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cadabrio AI Tools")
        self.setMinimumSize(700, 500)

        # Constrain to screen so the title bar never goes off-screen
        screen = QGuiApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            self.setMaximumSize(avail.width() - 40, avail.height() - 40)
            self.resize(min(850, avail.width() - 80), min(700, avail.height() - 80))

        self._config = config
        self._signals = _WorkerSignals()
        self._signals.started.connect(self._on_task_started)
        self._signals.finished.connect(self._on_task_finished)
        self._signals.progress.connect(self._on_step_progress)
        self._signals.search_done.connect(self._on_search_done)
        self._signals.candidates_found.connect(self._on_candidates_found)
        self._busy = False
        self._cancel_event = threading.Event()

        # Reference image collection
        self._ref_collection = ReferenceCollection()
        self._search_offset = 0  # for "Search for More" pagination

        # Scan available models
        if config:
            self._model_mgr = ModelManager(config)
            self._model_mgr.scan_local_models()
        else:
            self._model_mgr = None

        layout = QVBoxLayout(self)

        # Tabs — each tab's content is wrapped in a scroll area
        self._tabs = QTabWidget()
        self._tabs.addTab(self._scrollable(self._build_txt2img_tab()), "Text to Image")
        self._tabs.addTab(self._scrollable(self._build_depth_tab()), "Depth Estimation")
        self._tabs.addTab(self._scrollable(self._build_img2mesh_tab()), "Image to 3D")
        self._tabs.addTab(self._scrollable(self._build_segment_tab()), "Segmentation")
        layout.addWidget(self._tabs)

        # Progress bar + Cancel button
        progress_row = QHBoxLayout()
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setRange(0, 0)  # indeterminate by default
        progress_row.addWidget(self._progress)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.setStyleSheet("color: #ff6060; font-weight: bold; padding: 4px 16px;")
        self._cancel_btn.clicked.connect(self._cancel_task)
        progress_row.addWidget(self._cancel_btn)
        layout.addLayout(progress_row)

        # Status + VRAM
        bottom = QHBoxLayout()
        self._status = QLabel("Ready — models download automatically on first use")
        self._status.setStyleSheet("color: #a0a0b0; font-size: 9pt;")
        bottom.addWidget(self._status, 1)
        self._vram_label = QLabel()
        self._vram_label.setStyleSheet("color: #6a6a80; font-size: 8pt;")
        bottom.addWidget(self._vram_label)
        layout.addLayout(bottom)
        self._update_vram()

    def _scrollable(self, widget: QWidget) -> QScrollArea:
        """Wrap a widget in a QScrollArea so it can scroll when content grows."""
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        return scroll

    def moveEvent(self, event):
        """Prevent the dialog from moving off-screen (title bar must stay visible)."""
        super().moveEvent(event)
        screen = QGuiApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            pos = self.pos()
            new_x = max(avail.x() - self.width() + 100, min(pos.x(), avail.right() - 100))
            new_y = max(avail.y(), min(pos.y(), avail.bottom() - 40))
            if new_x != pos.x() or new_y != pos.y():
                self.move(new_x, new_y)

    # -------------------------------------------------------------------
    # Model selector helper — now with VRAM fit indicators
    # -------------------------------------------------------------------

    def _create_model_selector(self, category: ModelCategory) -> QComboBox:
        """Create a model selector combo with local models + VRAM fit indicators."""
        combo = QComboBox()
        combo.setMinimumWidth(300)

        default_id = _DEFAULTS.get(category, "")

        local_models = []
        if self._model_mgr:
            local_models = self._model_mgr.list_models(category)

        if local_models:
            for m in local_models:
                fit = m.vram_fit
                symbol = _VRAM_SYMBOLS.get(fit, "")
                label = f"{symbol} {m.model_id}  ({m.size_display})" if symbol else f"{m.model_id}  ({m.size_display})"
                combo.addItem(label.strip(), m.model_id)
                # Set tooltip with VRAM details
                idx = combo.count() - 1
                vram = get_gpu_vram_gb()
                if m.estimated_vram_gb > 0 and vram > 0:
                    combo.setItemData(
                        idx,
                        f"Size: {m.size_display}, Est. VRAM: {m.estimated_vram_gb:.1f} GB, GPU: {vram:.0f} GB",
                        Qt.ItemDataRole.ToolTipRole,
                    )
        else:
            combo.addItem(f"{default_id}  (will download on first use)", default_id)

        combo.addItem("Custom model ID...", "__custom__")

        # Restore saved selection from config, fall back to default
        saved_id = ""
        config_key = _MODEL_CONFIG_KEYS.get(category, "")
        if self._config and config_key:
            saved_id = self._config.get("ai", config_key, "")
        select_id = saved_id or default_id

        for i in range(combo.count()):
            if combo.itemData(i) == select_id:
                combo.setCurrentIndex(i)
                break

        # Save selection to config whenever user changes it
        combo.currentIndexChanged.connect(
            lambda _idx, c=combo, cat=category: self._on_model_selected(c, cat)
        )

        return combo

    def _on_model_selected(self, combo: QComboBox, category: ModelCategory):
        """Persist the model selection to config."""
        model_id = combo.currentData()
        if not model_id or model_id == "__custom__":
            return
        config_key = _MODEL_CONFIG_KEYS.get(category, "")
        if config_key and self._config:
            self._config.set("ai", config_key, model_id)
            self._config.save()

    def _get_selected_model(self, combo: QComboBox, category: ModelCategory) -> str:
        model_id = combo.currentData()
        if model_id == "__custom__":
            from PySide6.QtWidgets import QInputDialog
            text, ok = QInputDialog.getText(
                self, "Custom Model",
                "Enter a HuggingFace model ID (e.g. org/model-name):",
            )
            if ok and text.strip():
                return text.strip()
            return _DEFAULTS.get(category, "")
        return model_id

    def _on_t2i_model_changed(self):
        """Auto-adjust resolution, steps, and guidance when the model changes."""
        model_id = self._t2i_model.currentData()
        if not model_id or model_id == "__custom__":
            return
        defaults = pipelines.model_default_params(model_id)
        self._t2i_w.setValue(defaults["width"])
        self._t2i_h.setValue(defaults["height"])
        self._t2i_steps.setValue(defaults["steps"])
        self._t2i_cfg.setValue(defaults["guidance_scale"])
        arch = "FLUX" if pipelines._is_flux(model_id) else (
            "SDXL" if pipelines._is_sdxl(model_id) else (
                "SD 1.x" if pipelines._is_sd1x(model_id) else "SD 2.x"
            )
        )
        logger.debug(f"Model changed to {model_id} ({arch}), adjusted defaults")

    # -------------------------------------------------------------------
    # Tab builders
    # -------------------------------------------------------------------

    def _build_txt2img_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Model selector
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._t2i_model = self._create_model_selector(ModelCategory.TEXT_TO_IMAGE)
        model_row.addWidget(self._t2i_model, 1)
        layout.addLayout(model_row)

        # Prompt
        layout.addWidget(QLabel("Prompt:"))
        self._t2i_prompt = QTextEdit()
        self._t2i_prompt.setPlaceholderText(
            "A 2022 red and white Sunset Park Sun-Lite Sport 16BH travel trailer with 6 inch lift kit"
        )
        self._t2i_prompt.setMaximumHeight(80)
        layout.addWidget(self._t2i_prompt)

        # Web search checkbox
        self._t2i_web_search = QCheckBox("Search web for reference images")
        self._t2i_web_search.setToolTip(
            "When checked, click Search to find reference images from the web.\n"
            "You can accept/reject candidates and annotate them.\n"
            "Click Generate to create an image using the accepted references."
        )
        self._t2i_web_search.toggled.connect(self._on_web_search_toggled)
        layout.addWidget(self._t2i_web_search)

        # Reference Images gallery (hidden until search or paste)
        self._ref_gallery = _ReferenceGallery()
        self._ref_gallery.search_more.connect(self._on_search_more)
        self._ref_gallery.paste_image.connect(self._on_paste_reference)
        self._ref_gallery.browse_image.connect(self._on_browse_reference)
        self._ref_gallery.clear_all.connect(self._on_clear_references)
        self._ref_gallery.setVisible(False)
        layout.addWidget(self._ref_gallery)

        # Refining instructions (optional, visible when gallery is shown)
        self._t2i_refine_group = QGroupBox("Refining Instructions (optional)")
        refine_layout = QVBoxLayout(self._t2i_refine_group)

        self._t2i_refine_text = QTextEdit()
        self._t2i_refine_text.setPlaceholderText(
            "e.g. Combine the color from reference 1 with the lift kit shown in reference 2. "
            "Show from a 3/4 front angle on a gravel campsite."
        )
        self._t2i_refine_text.setMaximumHeight(60)
        refine_layout.addWidget(self._t2i_refine_text)

        self._t2i_refine_group.setVisible(False)
        layout.addWidget(self._t2i_refine_group)

        # Enhanced prompt (visible when web search has run)
        self._t2i_enhanced_group = QGroupBox("Enhanced Prompt")
        enhanced_layout = QVBoxLayout(self._t2i_enhanced_group)
        self._t2i_enhanced_prompt = QTextEdit()
        self._t2i_enhanced_prompt.setMaximumHeight(80)
        self._t2i_enhanced_prompt.setStyleSheet("font-size: 9pt;")
        self._t2i_enhanced_prompt.setPlaceholderText(
            "Enhanced prompt will appear here after web search runs"
        )
        enhanced_layout.addWidget(self._t2i_enhanced_prompt)

        strength_row = QHBoxLayout()
        strength_row.addWidget(QLabel("Creativity:"))
        self._t2i_strength = QDoubleSpinBox()
        self._t2i_strength.setRange(0.1, 1.0)
        self._t2i_strength.setSingleStep(0.1)
        self._t2i_strength.setValue(0.5)
        self._t2i_strength.setToolTip(
            "How much to deviate from the reference image:\n"
            "Low (0.2-0.3) = stay close to reference photo\n"
            "Medium (0.4-0.6) = balanced (recommended)\n"
            "High (0.7-0.9) = major changes, reference is just a guide"
        )
        strength_row.addWidget(self._t2i_strength)
        strength_row.addWidget(QLabel("(low = closer to reference, high = more creative)"))
        strength_row.addStretch()
        enhanced_layout.addLayout(strength_row)

        self._t2i_search_info = QLabel("")
        self._t2i_search_info.setStyleSheet("color: #6a6a80; font-size: 8pt;")
        self._t2i_search_info.setWordWrap(True)
        enhanced_layout.addWidget(self._t2i_search_info)

        self._t2i_enhanced_group.setVisible(False)
        self._t2i_reference_image = None
        layout.addWidget(self._t2i_enhanced_group)

        neg_label = QLabel("Negative prompt (SDXL only — FLUX ignores this):")
        neg_label.setStyleSheet("color: #8a8a9a;")
        layout.addWidget(neg_label)
        self._t2i_neg = QLineEdit()
        self._t2i_neg.setPlaceholderText("blurry, low quality, watermark, text")
        layout.addWidget(self._t2i_neg)

        # Settings
        settings = QHBoxLayout()
        settings.addWidget(QLabel("Width:"))
        self._t2i_w = QSpinBox()
        self._t2i_w.setRange(512, 2048)
        self._t2i_w.setSingleStep(128)
        self._t2i_w.setValue(1024)
        settings.addWidget(self._t2i_w)

        settings.addWidget(QLabel("Height:"))
        self._t2i_h = QSpinBox()
        self._t2i_h.setRange(512, 2048)
        self._t2i_h.setSingleStep(128)
        self._t2i_h.setValue(1024)
        settings.addWidget(self._t2i_h)

        settings.addWidget(QLabel("Steps:"))
        self._t2i_steps = QSpinBox()
        self._t2i_steps.setRange(1, 100)
        self._t2i_steps.setValue(4)
        self._t2i_steps.setToolTip(
            "FLUX.1-schnell: 4 steps (default, fast)\n"
            "SDXL / Stable Diffusion: 20-30 steps"
        )
        settings.addWidget(self._t2i_steps)

        settings.addWidget(QLabel("Guidance:"))
        self._t2i_cfg = QDoubleSpinBox()
        self._t2i_cfg.setRange(0.0, 30.0)
        self._t2i_cfg.setSingleStep(0.5)
        self._t2i_cfg.setValue(0.0)
        self._t2i_cfg.setToolTip(
            "FLUX.1-schnell: 0.0 (distilled, no guidance needed)\n"
            "SDXL / Stable Diffusion: 7.0-8.5"
        )
        settings.addWidget(self._t2i_cfg)

        settings.addWidget(QLabel("Seed:"))
        self._t2i_seed = QSpinBox()
        self._t2i_seed.setRange(-1, 2147483647)
        self._t2i_seed.setValue(-1)
        self._t2i_seed.setSpecialValueText("Random")
        settings.addWidget(self._t2i_seed)
        layout.addLayout(settings)

        # Buttons: Search + Generate + Save
        btn_row = QHBoxLayout()
        self._t2i_search_btn = QPushButton("Search")
        self._t2i_search_btn.setStyleSheet("padding: 8px 20px;")
        self._t2i_search_btn.setToolTip("Search the web for reference images without generating")
        self._t2i_search_btn.clicked.connect(self._run_search_only)
        btn_row.addWidget(self._t2i_search_btn)

        gen_btn = QPushButton("Generate")
        gen_btn.setStyleSheet("font-weight: bold; padding: 8px 24px;")
        gen_btn.clicked.connect(self._run_txt2img)
        btn_row.addWidget(gen_btn)

        save_btn = QPushButton("Save Image...")
        save_btn.clicked.connect(lambda: self._save_preview(self._t2i_preview))
        btn_row.addWidget(save_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._t2i_preview = _ImagePreview()
        layout.addWidget(self._t2i_preview, 1)

        # Auto-adjust defaults when model changes
        self._t2i_model.currentIndexChanged.connect(self._on_t2i_model_changed)

        return widget

    def _build_depth_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._depth_model = self._create_model_selector(ModelCategory.DEPTH_ESTIMATION)
        model_row.addWidget(self._depth_model, 1)
        layout.addLayout(model_row)

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("Input image:"))
        self._depth_path = QLineEdit()
        self._depth_path.setReadOnly(True)
        input_row.addWidget(self._depth_path, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(lambda: self._browse_image(self._depth_path))
        input_row.addWidget(browse_btn)
        layout.addLayout(input_row)

        btn_row = QHBoxLayout()
        run_btn = QPushButton("Estimate Depth")
        run_btn.setStyleSheet("font-weight: bold; padding: 8px 24px;")
        run_btn.clicked.connect(self._run_depth)
        btn_row.addWidget(run_btn)
        save_btn = QPushButton("Save Depth Map...")
        save_btn.clicked.connect(lambda: self._save_preview(self._depth_preview))
        btn_row.addWidget(save_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        previews = QHBoxLayout()
        self._depth_input_preview = _ImagePreview()
        self._depth_preview = _ImagePreview()
        previews.addWidget(self._depth_input_preview)
        previews.addWidget(self._depth_preview)
        layout.addLayout(previews, 1)
        return widget

    def _build_img2mesh_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._mesh_model = self._create_model_selector(ModelCategory.IMAGE_TO_3D)
        model_row.addWidget(self._mesh_model, 1)
        layout.addLayout(model_row)

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("Input image:"))
        self._mesh_path = QLineEdit()
        self._mesh_path.setReadOnly(True)
        input_row.addWidget(self._mesh_path, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(lambda: self._browse_image(self._mesh_path))
        input_row.addWidget(browse_btn)
        layout.addLayout(input_row)

        btn_row = QHBoxLayout()
        run_btn = QPushButton("Generate 3D Mesh")
        run_btn.setStyleSheet("font-weight: bold; padding: 8px 24px;")
        run_btn.clicked.connect(self._run_img2mesh)
        btn_row.addWidget(run_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._mesh_input_preview = _ImagePreview()
        layout.addWidget(self._mesh_input_preview, 1)

        self._mesh_status = QLabel("")
        self._mesh_status.setStyleSheet("font-size: 10pt;")
        layout.addWidget(self._mesh_status)
        return widget

    def _build_segment_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._seg_model = self._create_model_selector(ModelCategory.SEGMENTATION)
        model_row.addWidget(self._seg_model, 1)
        layout.addLayout(model_row)

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("Input image:"))
        self._seg_path = QLineEdit()
        self._seg_path.setReadOnly(True)
        input_row.addWidget(self._seg_path, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(lambda: self._browse_image(self._seg_path))
        input_row.addWidget(browse_btn)
        layout.addLayout(input_row)

        btn_row = QHBoxLayout()
        run_btn = QPushButton("Segment Image")
        run_btn.setStyleSheet("font-weight: bold; padding: 8px 24px;")
        run_btn.clicked.connect(self._run_segment)
        btn_row.addWidget(run_btn)
        save_btn = QPushButton("Save Result...")
        save_btn.clicked.connect(lambda: self._save_preview(self._seg_preview))
        btn_row.addWidget(save_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        previews = QHBoxLayout()
        self._seg_input_preview = _ImagePreview()
        self._seg_preview = _ImagePreview()
        previews.addWidget(self._seg_input_preview)
        previews.addWidget(self._seg_preview)
        layout.addLayout(previews, 1)
        return widget

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _browse_image(self, line_edit: QLineEdit):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)"
        )
        if path:
            line_edit.setText(path)

    def _save_preview(self, preview: _ImagePreview):
        if preview.pil_image is None:
            QMessageBox.information(self, "No Image", "Generate an image first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "PNG (*.png);;JPEG (*.jpg);;All Files (*)"
        )
        if path:
            preview.pil_image.save(path)
            self._status.setText(f"Saved: {path}")

    def _update_vram(self):
        try:
            import torch
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / (1024**3)
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self._vram_label.setText(f"VRAM: {used:.1f} / {total:.1f} GB")
            else:
                self._vram_label.setText("VRAM: CUDA not available")
        except ImportError:
            self._vram_label.setText("VRAM: PyTorch not installed")

    # -------------------------------------------------------------------
    # Task execution with progress + cancel
    # -------------------------------------------------------------------

    def _run_in_thread(self, task_name: str, fn, *args):
        if self._busy:
            QMessageBox.information(self, "Busy", "Another AI task is running. Please wait.")
            return
        self._busy = True
        self._cancel_event.clear()
        self._signals.started.emit(task_name)
        thread = threading.Thread(target=self._worker, args=(task_name, fn, *args), daemon=True)
        thread.start()

    def _worker(self, task_name, fn, *args):
        try:
            result = fn(*args)
            self._signals.finished.emit(task_name, result)
        except pipelines.GenerationCancelled:
            self._signals.finished.emit(task_name, "CANCELLED")
        except Exception as e:
            logger.error(f"AI task '{task_name}' failed: {e}")
            self._signals.finished.emit(task_name, f"ERROR: {e}")

    def _cancel_task(self):
        """Request cancellation of the current task."""
        self._cancel_event.set()
        self._status.setText("Cancelling...")
        self._cancel_btn.setEnabled(False)

    def _on_task_started(self, name: str):
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)  # indeterminate until first step callback
        self._cancel_btn.setVisible(True)
        self._cancel_btn.setEnabled(True)
        self._status.setText(f"Running: {name}... (first run downloads the model)")

    def _on_step_progress(self, step: int, total: int):
        """Update progress bar with real step info."""
        if total > 0:
            self._progress.setRange(0, total)
            self._progress.setValue(step)
            self._progress.setFormat(f"Step {step}/{total}")
            self._status.setText(f"Generating... step {step}/{total}")

    def _on_task_finished(self, name: str, result):
        self._busy = False
        self._progress.setVisible(False)
        self._progress.setRange(0, 0)  # reset to indeterminate for next time
        self._cancel_btn.setVisible(False)
        self._update_vram()

        if result == "CANCELLED":
            self._status.setText(f"Cancelled: {name}")
            return

        if isinstance(result, str) and result.startswith("ERROR:"):
            self._status.setText(f"Failed: {name}")
            QMessageBox.critical(self, "AI Task Failed", f"{name} failed:\n\n{result}")
            return

        self._status.setText(f"Completed: {name}")

        if name == "Text to Image" and isinstance(result, Image.Image):
            self._t2i_preview.set_pil_image(result)
        elif name == "Reference Search":
            # Search-only task completed — candidates are added via signal
            pass
        elif name == "Depth Estimation" and isinstance(result, Image.Image):
            self._depth_preview.set_pil_image(result)
        elif name == "Image to 3D":
            if result:
                self._mesh_status.setText(f"Mesh saved: {result}")
            else:
                self._mesh_status.setText("Mesh generation failed")
        elif name == "Segmentation" and isinstance(result, Image.Image):
            self._seg_preview.set_pil_image(result)

    # -------------------------------------------------------------------
    # Progress callback for pipelines (called from worker thread)
    # -------------------------------------------------------------------

    def _progress_callback(self, step: int, total: int):
        """Thread-safe progress callback — emits signal to update UI."""
        self._signals.progress.emit(step, total)

    # -------------------------------------------------------------------
    # Reference image gallery handlers
    # -------------------------------------------------------------------

    def _on_web_search_toggled(self, checked: bool):
        """Show/hide gallery and related controls based on web search checkbox."""
        if not checked:
            self._ref_gallery.setVisible(False)
            self._t2i_refine_group.setVisible(False)
            self._t2i_enhanced_group.setVisible(False)
            self._t2i_reference_image = None

    def _on_candidates_found(self, candidates: list):
        """Populate the gallery with search result candidates."""
        added = 0
        for cand in candidates:
            pil_thumb = cand.get("pil_thumbnail")
            if pil_thumb is None:
                continue  # skip candidates with no thumbnail
            ref = self._ref_collection.add_from_search(
                image=pil_thumb,
                source_url=cand.get("image", ""),
                thumbnail_url=cand.get("thumbnail", ""),
                title=cand.get("title", ""),
            )
            if ref is None:
                continue  # duplicate URL
            card = self._ref_gallery.add_card(ref)
            card.accepted.connect(self._on_candidate_accepted)
            card.rejected.connect(self._on_candidate_rejected)
            added += 1

        if added == 0 and not candidates:
            self._status.setText("No reference images found — try a different prompt")
        self._ref_gallery.setVisible(True)
        self._t2i_refine_group.setVisible(True)
        self._ref_gallery.update_status()

    def _on_candidate_accepted(self, ref: ReferenceImage):
        """Handle a candidate being accepted — download full image in background."""
        self._ref_collection.accept(ref)
        self._ref_gallery.update_status()

        # Download full-res image in background
        if ref.source_url and ref.full_image is None:
            def _download():
                from cadabrio.ai.prompt_enhance import download_full_image
                full = download_full_image(ref.source_url)
                if full is not None:
                    ref.full_image = full
                    logger.info(f"Downloaded full reference: {ref.title[:40]}")
            thread = threading.Thread(target=_download, daemon=True)
            thread.start()

    def _on_candidate_rejected(self, ref: ReferenceImage):
        """Handle a candidate being rejected."""
        self._ref_collection.reject(ref)
        self._ref_gallery.update_status()

    def _on_search_more(self):
        """Search for additional candidates with varied query."""
        prompt = self._t2i_prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Prompt Required", "Please enter a text prompt.")
            return
        # Increase offset to vary the search query suffix
        self._search_offset += 1
        max_candidates = 12
        if self._config:
            max_candidates = self._config.get("ai", "max_reference_candidates", 12)
        known = self._ref_collection.known_urls
        self._run_in_thread(
            "Reference Search",
            self._do_search_candidates, prompt, max_candidates, self._search_offset, known,
        )

    def _on_paste_reference(self):
        """Paste an image from the clipboard as a user-added reference."""
        clipboard = QGuiApplication.clipboard()
        qimg = clipboard.image()
        if qimg.isNull():
            QMessageBox.information(
                self, "No Image",
                "No image found on clipboard. Copy an image first."
            )
            return
        pil_img = _qimage_to_pil(qimg)
        if pil_img is None:
            QMessageBox.warning(self, "Error", "Could not convert clipboard image.")
            return
        # Create thumbnail for display
        thumb = pil_img.copy()
        thumb.thumbnail((200, 150), Image.Resampling.LANCZOS)
        ref = self._ref_collection.add_user_image(thumb, title="Pasted image")
        ref.full_image = pil_img
        card = self._ref_gallery.add_card(ref)
        self._ref_gallery.setVisible(True)
        self._t2i_refine_group.setVisible(True)
        self._ref_gallery.update_status()
        self._status.setText("Pasted image added as reference")

    def _on_browse_reference(self):
        """Browse for an image file to add as a user reference."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)"
        )
        if not path:
            return
        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open image: {e}")
            return
        thumb = pil_img.copy()
        thumb.thumbnail((200, 150), Image.Resampling.LANCZOS)
        ref = self._ref_collection.add_user_image(thumb, title=Path(path).name)
        ref.full_image = pil_img
        card = self._ref_gallery.add_card(ref)
        self._ref_gallery.setVisible(True)
        self._t2i_refine_group.setVisible(True)
        self._ref_gallery.update_status()
        self._status.setText(f"Added reference: {Path(path).name}")

    def _on_clear_references(self):
        """Clear all reference images."""
        self._ref_collection.clear()
        self._ref_gallery.clear()
        self._search_offset = 0
        self._t2i_reference_image = None
        self._t2i_enhanced_group.setVisible(False)
        self._ref_gallery.update_status()
        self._status.setText("References cleared")

    # -------------------------------------------------------------------
    # Prompt building
    # -------------------------------------------------------------------

    def _build_combined_prompt(self, base_prompt: str) -> str:
        """Merge base prompt + annotations + refining text + enhanced prompt."""
        parts = [base_prompt.strip()]

        # Add annotation context from accepted references
        ref_context = self._ref_collection.build_refining_context()
        if ref_context:
            parts.append(ref_context)

        # Add user's refining instructions
        refine = self._t2i_refine_text.toPlainText().strip()
        if refine:
            parts.append(refine)

        # If there's an enhanced prompt from web search, prefer it over base
        enhanced = self._t2i_enhanced_prompt.toPlainText().strip()
        if enhanced and enhanced != base_prompt.strip():
            # Prepend enhanced prompt details, keep annotations/refine as additions
            return f"{enhanced}. {'. '.join(parts[1:])}" if len(parts) > 1 else enhanced

        return ". ".join(parts)

    # -------------------------------------------------------------------
    # Search-only task runner
    # -------------------------------------------------------------------

    def _run_search_only(self):
        """Run reference image search without generating."""
        prompt = self._t2i_prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Prompt Required", "Please enter a text prompt.")
            return
        self._search_offset = 0
        max_candidates = 12
        if self._config:
            max_candidates = self._config.get("ai", "max_reference_candidates", 12)
        known = self._ref_collection.known_urls
        self._run_in_thread(
            "Reference Search",
            self._do_search_candidates, prompt, max_candidates, 0, known,
        )

    def _do_search_candidates(self, prompt, max_candidates, offset, known_urls):
        """Worker: search for reference image candidates."""
        from cadabrio.ai.prompt_enhance import (
            search_reference_candidates, _extract_subject, search_and_enhance,
        )

        subject = _extract_subject(prompt)

        # Run image candidate search (with known URLs for dedup)
        candidates = search_reference_candidates(
            subject, max_candidates, offset, known_urls,
        )
        self._signals.candidates_found.emit(candidates)

        # Also run text search for prompt enhancement (only on first search)
        if offset == 0:
            result = search_and_enhance(prompt)
            self._signals.search_done.emit(result)

        return "SEARCH_DONE"

    # -------------------------------------------------------------------
    # Search result handler (text search for prompt enhancement)
    # -------------------------------------------------------------------

    def _on_search_done(self, result: dict):
        """Update the UI with web search results (prompt enhancement)."""
        enhanced = result.get("enhanced", "")
        sources = result.get("sources", [])

        has_content = enhanced and enhanced != result.get("original", "")

        if has_content:
            self._t2i_enhanced_prompt.setPlainText(enhanced)
            self._t2i_enhanced_group.setVisible(True)

            if sources:
                src_text = "Sources: " + " | ".join(s[:60] for s in sources[:3])
                self._t2i_search_info.setText(src_text)
            else:
                self._t2i_search_info.setText("")

    # -------------------------------------------------------------------
    # Task runners — Text to Image
    # -------------------------------------------------------------------

    def _run_txt2img(self):
        prompt = self._t2i_prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Prompt Required", "Please enter a text prompt.")
            return

        seed = self._t2i_seed.value()
        if seed < 0:
            seed = None
        model_id = self._get_selected_model(self._t2i_model, ModelCategory.TEXT_TO_IMAGE)
        neg = self._t2i_neg.text().strip()
        w, h = self._t2i_w.value(), self._t2i_h.value()
        steps = self._t2i_steps.value()
        cfg = self._t2i_cfg.value()
        strength = self._t2i_strength.value()

        accepted_refs = self._ref_collection.accepted
        web_search = self._t2i_web_search.isChecked()
        has_enhanced = (
            self._t2i_enhanced_group.isVisible()
            and self._t2i_enhanced_prompt.toPlainText().strip()
        )

        if accepted_refs:
            # Has curated references — use best reference with combined prompt
            combined = self._build_combined_prompt(prompt)
            best_ref = self._ref_collection.best_reference
            if best_ref is not None:
                self._run_in_thread(
                    "Text to Image",
                    self._do_img2img, combined, best_ref, neg,
                    strength, steps, cfg, seed, model_id,
                )
            else:
                # Accepted refs but no full image downloaded yet — use thumbnail
                first_accepted = accepted_refs[0]
                ref_img = first_accepted.image
                self._run_in_thread(
                    "Text to Image",
                    self._do_img2img, combined, ref_img, neg,
                    strength, steps, cfg, seed, model_id,
                )
        elif web_search and has_enhanced:
            # Has enhanced prompt from search but no accepted references
            enhanced_prompt = self._t2i_enhanced_prompt.toPlainText().strip()
            self._run_in_thread(
                "Text to Image",
                self._do_txt2img, enhanced_prompt, neg, w, h, steps, cfg,
                seed, model_id,
            )
        elif web_search:
            # Web search checked but hasn't been run yet — do combined search+generate
            self._run_in_thread(
                "Text to Image",
                self._do_search_then_generate, prompt, neg, w, h, steps, cfg,
                seed, model_id, strength,
            )
        else:
            # Direct generation without web search or references
            self._run_in_thread(
                "Text to Image",
                self._do_txt2img, prompt, neg, w, h, steps, cfg, seed, model_id,
            )

    def _do_txt2img(self, prompt, neg, w, h, steps, cfg, seed, model_id):
        return pipelines.text_to_image(
            prompt, neg, w, h, steps, cfg, seed, model_id,
            progress_callback=self._progress_callback,
            cancel_event=self._cancel_event,
        )

    def _do_img2img(self, prompt, ref_image, neg, strength, steps, cfg, seed, model_id):
        return pipelines.image_guided_generate(
            prompt, ref_image, neg, strength, steps, cfg, seed, model_id,
            progress_callback=self._progress_callback,
            cancel_event=self._cancel_event,
        )

    def _do_search_then_generate(self, prompt, neg, w, h, steps, cfg, seed, model_id, strength):
        """Combined web search + image generation in one worker thread."""
        from cadabrio.ai.prompt_enhance import search_and_enhance

        result = search_and_enhance(prompt)
        self._signals.search_done.emit(result)  # Update UI with search results

        if self._cancel_event.is_set():
            raise pipelines.GenerationCancelled("Cancelled by user")

        enhanced = result.get("enhanced", prompt)
        ref_image = result.get("reference_image")

        if ref_image is not None:
            return pipelines.image_guided_generate(
                enhanced, ref_image, neg, strength, steps, cfg, seed, model_id,
                progress_callback=self._progress_callback,
                cancel_event=self._cancel_event,
            )
        else:
            return pipelines.text_to_image(
                enhanced, neg, w, h, steps, cfg, seed, model_id,
                progress_callback=self._progress_callback,
                cancel_event=self._cancel_event,
            )

    # -------------------------------------------------------------------
    # Task runners — Other tabs
    # -------------------------------------------------------------------

    def _run_depth(self):
        path = self._depth_path.text()
        if not path:
            QMessageBox.warning(self, "No Image", "Please select an input image.")
            return
        img = Image.open(path).convert("RGB")
        self._depth_input_preview.set_pil_image(img)
        model_id = self._get_selected_model(self._depth_model, ModelCategory.DEPTH_ESTIMATION)
        self._run_in_thread("Depth Estimation", self._do_depth, img, model_id)

    def _do_depth(self, img, model_id):
        return pipelines.estimate_depth(img, model_id)

    def _run_img2mesh(self):
        path = self._mesh_path.text()
        if not path:
            QMessageBox.warning(self, "No Image", "Please select an input image.")
            return
        img = Image.open(path).convert("RGB")
        self._mesh_input_preview.set_pil_image(img)
        out_path = Path(path).with_name(Path(path).stem + "_mesh")
        model_id = self._get_selected_model(self._mesh_model, ModelCategory.IMAGE_TO_3D)
        self._run_in_thread("Image to 3D", self._do_img2mesh, img, out_path, model_id)

    def _do_img2mesh(self, img, out_path, model_id):
        return pipelines.image_to_3d(img, out_path, model_id)

    def _run_segment(self):
        path = self._seg_path.text()
        if not path:
            QMessageBox.warning(self, "No Image", "Please select an input image.")
            return
        img = Image.open(path).convert("RGB")
        self._seg_input_preview.set_pil_image(img)
        model_id = self._get_selected_model(self._seg_model, ModelCategory.SEGMENTATION)
        self._run_in_thread("Segmentation", self._do_segment, img, model_id)

    def _do_segment(self, img, model_id):
        masks = pipelines.segment_image(img, model_id=model_id)
        if masks:
            return pipelines.segment_to_image(img, masks)
        return img
