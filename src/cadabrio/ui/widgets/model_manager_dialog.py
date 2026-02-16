"""AI Model Manager dialog for Cadabrio.

Provides a UI for browsing local models by purpose, searching Hugging Face Hub
by category, downloading models with progress tracking, and showing VRAM fit
indicators so users know which models will work on their GPU.
"""

import threading
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QProgressBar,
    QTabWidget,
    QWidget,
    QMessageBox,
    QComboBox,
    QGroupBox,
)

from loguru import logger
from cadabrio.config.manager import ConfigManager
from cadabrio.ai.model_manager import (
    ModelManager, ModelCategory, ModelInfo,
    get_hf_cache_dir, get_gpu_vram_gb, vram_fit_level, _format_size,
)


# Display labels for categories
_CATEGORY_LABELS = {
    ModelCategory.TEXT_TO_IMAGE: "Text to Image",
    ModelCategory.TEXT_TO_3D: "Text to 3D",
    ModelCategory.IMAGE_TO_3D: "Image to 3D",
    ModelCategory.DEPTH_ESTIMATION: "Depth Estimation",
    ModelCategory.SEGMENTATION: "Segmentation",
    ModelCategory.CHAT: "Chat / LLM",
    ModelCategory.UPSCALE: "Upscale",
    ModelCategory.OTHER: "Other",
}

# VRAM fit colors
_FIT_COLORS = {
    "good": QColor(0, 200, 0),      # Green
    "tight": QColor(255, 180, 0),    # Orange/yellow
    "bad": QColor(255, 60, 60),      # Red
    "unknown": QColor(120, 120, 140), # Gray
}

_FIT_LABELS = {
    "good": "OK",
    "tight": "Tight",
    "bad": "Too large",
    "unknown": "—",
}


class _DownloadSignals(QObject):
    """Signals for cross-thread download updates."""
    progress = Signal(str, float, str)  # repo_id, fraction, message
    finished = Signal(str, str)


class _SearchSignals(QObject):
    """Signals for background HF search."""
    results = Signal(list)  # list[ModelInfo]
    error = Signal(str)


class _SizeFetchSignals(QObject):
    """Signals for async model size fetching."""
    size_ready = Signal(str, float)  # model_id, size_bytes as float (avoids 32-bit int overflow)


class ModelManagerDialog(QDialog):
    """Dialog for managing AI models — local and Hugging Face Hub."""

    def __init__(self, config: ConfigManager, parent=None):
        super().__init__(parent)
        self._config = config
        self._model_mgr = ModelManager(config)
        self._model_mgr.scan_local_models()
        self._dl_signals = _DownloadSignals()
        self._dl_signals.progress.connect(self._on_download_progress)
        self._dl_signals.finished.connect(self._on_download_finished)
        self._search_signals = _SearchSignals()
        self._search_signals.results.connect(self._on_search_results)
        self._search_signals.error.connect(self._on_search_error)
        self._size_signals = _SizeFetchSignals()
        self._size_signals.size_ready.connect(self._on_size_fetched)
        self._active_downloads: dict[str, bool] = {}
        self._hub_results: list[ModelInfo] = []  # Keep for size updates

        self.setWindowTitle("AI Model Manager")
        self.setMinimumSize(900, 600)

        layout = QVBoxLayout(self)

        # HF Cache location + VRAM info
        info_row = QHBoxLayout()
        info_row.addWidget(QLabel("HuggingFace cache:"))
        cache_label = QLabel(str(get_hf_cache_dir()))
        cache_label.setStyleSheet("font-family: monospace; font-size: 9pt;")
        info_row.addWidget(cache_label, 1)
        vram = get_gpu_vram_gb()
        if vram > 0:
            vram_label = QLabel(f"GPU VRAM: {vram:.0f} GB")
            vram_label.setStyleSheet("font-weight: bold; color: #a0c0ff;")
            info_row.addWidget(vram_label)
        layout.addLayout(info_row)

        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self._build_local_tab(), "Local Models")
        tabs.addTab(self._build_hub_tab(), "Hugging Face Hub")
        layout.addWidget(tabs)

        # Download progress area
        self._progress_group = QGroupBox("Downloads")
        self._progress_layout = QVBoxLayout(self._progress_group)
        self._progress_bars: dict[str, QProgressBar] = {}
        self._progress_group.setVisible(False)
        layout.addWidget(self._progress_group)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    # -------------------------------------------------------------------
    # Local Models tab
    # -------------------------------------------------------------------

    def _build_local_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        top = QHBoxLayout()
        top.addWidget(QLabel("Filter by purpose:"))
        self._local_filter = QComboBox()
        self._local_filter.addItem("All", None)
        for cat, label in _CATEGORY_LABELS.items():
            self._local_filter.addItem(label, cat)
        self._local_filter.currentIndexChanged.connect(self._refresh_local)
        top.addWidget(self._local_filter)
        top.addStretch()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_local)
        top.addWidget(refresh_btn)
        layout.addLayout(top)

        self._local_table = QTableWidget()
        self._local_table.setColumnCount(5)
        self._local_table.setHorizontalHeaderLabels([
            "Model", "Purpose", "Size", "VRAM Fit", "Format",
        ])
        header = self._local_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._local_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._local_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._local_table.setSortingEnabled(True)
        layout.addWidget(self._local_table)

        self._local_count = QLabel()
        self._local_count.setStyleSheet("color: #6a6a80; font-size: 8pt;")
        layout.addWidget(self._local_count)

        self._refresh_local()
        return widget

    def _refresh_local(self):
        self._model_mgr.scan_local_models()
        cat_data = self._local_filter.currentData()
        category = cat_data if isinstance(cat_data, ModelCategory) else None
        models = self._model_mgr.list_models(category)
        self._local_table.setSortingEnabled(False)
        self._local_table.setRowCount(len(models))

        total_bytes = 0
        for row, m in enumerate(models):
            self._local_table.setItem(row, 0, QTableWidgetItem(m.name))
            cat_label = _CATEGORY_LABELS.get(m.category, m.category.value)
            self._local_table.setItem(row, 1, QTableWidgetItem(cat_label))
            self._local_table.setItem(row, 2, QTableWidgetItem(m.size_display))

            # VRAM fit indicator
            fit = m.vram_fit
            fit_item = QTableWidgetItem(_FIT_LABELS.get(fit, "—"))
            fit_item.setForeground(_FIT_COLORS.get(fit, QColor(120, 120, 140)))
            if fit != "unknown":
                fit_item.setToolTip(
                    f"Model: {m.size_gb:.1f} GB, "
                    f"Est. VRAM: {m.estimated_vram_gb:.1f} GB, "
                    f"GPU: {get_gpu_vram_gb():.0f} GB"
                )
            self._local_table.setItem(row, 3, fit_item)

            self._local_table.setItem(row, 4, QTableWidgetItem(m.format.value.upper()))
            total_bytes += m.size_bytes

        self._local_table.setSortingEnabled(True)
        total_gb = total_bytes / (1024**3)
        self._local_count.setText(
            f"{len(models)} model{'s' if len(models) != 1 else ''}, {total_gb:.1f} GB total"
        )

    # -------------------------------------------------------------------
    # Hugging Face Hub tab
    # -------------------------------------------------------------------

    def _build_hub_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search:"))
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("e.g. depth estimation, stable diffusion, TripoSR...")
        self._search_input.returnPressed.connect(self._search_hub)
        search_row.addWidget(self._search_input, 1)
        layout.addLayout(search_row)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Purpose:"))
        self._hub_category = QComboBox()
        self._hub_category.addItem("All Categories", None)
        for cat, label in _CATEGORY_LABELS.items():
            if cat != ModelCategory.OTHER:
                self._hub_category.addItem(label, cat)
        filter_row.addWidget(self._hub_category)

        search_btn = QPushButton("Search")
        search_btn.setStyleSheet("font-weight: bold;")
        search_btn.clicked.connect(self._search_hub)
        filter_row.addWidget(search_btn)

        filter_row.addSpacing(12)
        for label, cat in [
            ("Text2Img", ModelCategory.TEXT_TO_IMAGE),
            ("Depth", ModelCategory.DEPTH_ESTIMATION),
            ("Img2 3D", ModelCategory.IMAGE_TO_3D),
            ("Segment", ModelCategory.SEGMENTATION),
        ]:
            btn = QPushButton(label)
            btn.setMaximumWidth(70)
            btn.clicked.connect(
                lambda checked=False, c=cat: self._quick_search(c)
            )
            filter_row.addWidget(btn)

        layout.addLayout(filter_row)

        # Results table — now with Size and VRAM Fit columns
        self._hub_table = QTableWidget()
        self._hub_table.setColumnCount(8)
        self._hub_table.setHorizontalHeaderLabels([
            "Repository", "Purpose", "Size", "VRAM Fit",
            "Downloads", "Likes", "Status", "",
        ])
        header = self._hub_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in range(1, 8):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self._hub_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._hub_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._hub_table)

        self._hub_status = QLabel("Use the search bar or quick-pick buttons to browse models")
        self._hub_status.setStyleSheet("color: #6a6a80; font-size: 8pt;")
        layout.addWidget(self._hub_status)

        return widget

    def _quick_search(self, category: ModelCategory):
        idx = self._hub_category.findData(category)
        if idx >= 0:
            self._hub_category.setCurrentIndex(idx)
        self._search_input.clear()
        self._search_hub()

    def _search_hub(self):
        if self._config.get("network", "offline_mode", False):
            QMessageBox.warning(self, "Offline Mode", "Cannot search in offline mode.")
            return

        query = self._search_input.text().strip()
        cat_data = self._hub_category.currentData()
        category = cat_data if isinstance(cat_data, ModelCategory) else None

        self._hub_status.setText("Searching...")
        self._hub_table.setRowCount(0)

        thread = threading.Thread(
            target=self._search_worker, args=(query, category), daemon=True
        )
        thread.start()

    def _search_worker(self, query: str, category: ModelCategory | None):
        try:
            results = self._model_mgr.search_hub(query, category, limit=30)
            self._search_signals.results.emit(results)
        except Exception as e:
            self._search_signals.error.emit(str(e))

    def _on_search_results(self, results: list):
        self._hub_results = results
        self._hub_table.setRowCount(len(results))

        need_sizes = []  # model_ids that need size fetching

        for row, m in enumerate(results):
            self._hub_table.setItem(row, 0, QTableWidgetItem(m.name))
            cat_label = _CATEGORY_LABELS.get(m.category, m.pipeline_tag or "—")
            self._hub_table.setItem(row, 1, QTableWidgetItem(cat_label))

            # Size — show if known, otherwise "loading..."
            if m.size_bytes > 0:
                self._hub_table.setItem(row, 2, QTableWidgetItem(m.size_display))
                fit = m.vram_fit
                fit_item = QTableWidgetItem(_FIT_LABELS.get(fit, "—"))
                fit_item.setForeground(_FIT_COLORS.get(fit, QColor(120, 120, 140)))
                self._hub_table.setItem(row, 3, fit_item)
            else:
                loading = QTableWidgetItem("loading...")
                loading.setForeground(QColor(120, 120, 140))
                self._hub_table.setItem(row, 2, loading)
                self._hub_table.setItem(row, 3, QTableWidgetItem("—"))
                need_sizes.append(m.model_id)

            self._hub_table.setItem(row, 4, QTableWidgetItem(m.downloads_display))
            self._hub_table.setItem(row, 5, QTableWidgetItem(m.likes_display))

            if m.downloaded:
                status = QTableWidgetItem("Installed")
                status.setForeground(QColor(0, 200, 0))
                self._hub_table.setItem(row, 6, status)
                self._hub_table.setCellWidget(row, 7, QLabel(""))
            else:
                self._hub_table.setItem(row, 6, QTableWidgetItem(""))
                dl_btn = QPushButton("Download")
                dl_btn.clicked.connect(
                    lambda checked=False, r=m.model_id: self._start_download(r)
                )
                self._hub_table.setCellWidget(row, 7, dl_btn)

        self._hub_status.setText(f"{len(results)} results")

        # Fetch sizes asynchronously for models that don't have them
        if need_sizes:
            self._hub_status.setText(f"{len(results)} results — fetching sizes...")
            thread = threading.Thread(
                target=self._fetch_sizes_worker, args=(need_sizes,), daemon=True
            )
            thread.start()

    def _fetch_sizes_worker(self, model_ids: list[str]):
        """Fetch model sizes from HF API in background."""
        for mid in model_ids:
            try:
                size = ModelManager.fetch_model_size(mid)
                if size > 0:
                    self._size_signals.size_ready.emit(mid, float(size))
            except Exception:
                pass

    def _on_size_fetched(self, model_id: str, size_bytes: float):
        """Update the Hub table when a model's size is fetched."""
        for row, m in enumerate(self._hub_results):
            if m.model_id == model_id:
                m.size_bytes = int(size_bytes)
                self._hub_table.setItem(row, 2, QTableWidgetItem(m.size_display))
                fit = m.vram_fit
                fit_item = QTableWidgetItem(_FIT_LABELS.get(fit, "—"))
                fit_item.setForeground(_FIT_COLORS.get(fit, QColor(120, 120, 140)))
                fit_item.setToolTip(
                    f"Model: {m.size_gb:.1f} GB, "
                    f"Est. VRAM: {m.estimated_vram_gb:.1f} GB, "
                    f"GPU: {get_gpu_vram_gb():.0f} GB"
                )
                self._hub_table.setItem(row, 3, fit_item)
                break

        # Update status when all sizes are fetched
        pending = sum(1 for m in self._hub_results if m.size_bytes <= 0)
        if pending == 0:
            self._hub_status.setText(f"{len(self._hub_results)} results")

    def _on_search_error(self, error: str):
        self._hub_status.setText(f"Search failed: {error}")

    # -------------------------------------------------------------------
    # Downloads — with real progress
    # -------------------------------------------------------------------

    def _start_download(self, repo_id: str):
        if repo_id in self._active_downloads:
            return
        if self._config.get("network", "offline_mode", False):
            QMessageBox.warning(self, "Offline Mode", "Cannot download in offline mode.")
            return

        self._active_downloads[repo_id] = True
        bar = QProgressBar()
        bar.setFormat(f"{repo_id}: starting...")
        bar.setMaximum(100)
        bar.setValue(0)
        self._progress_bars[repo_id] = bar
        self._progress_layout.addWidget(bar)
        self._progress_group.setVisible(True)

        thread = threading.Thread(
            target=self._download_worker, args=(repo_id,), daemon=True
        )
        thread.start()

    def _download_worker(self, repo_id: str):
        try:
            from huggingface_hub import snapshot_download
            from tqdm import tqdm as _tqdm_base
            signals = self._dl_signals

            class _SignalTqdm(_tqdm_base):
                """Subclass of real tqdm that also emits Qt progress signals.

                huggingface_hub may pass extra kwargs (e.g. 'name') that
                tqdm doesn't accept — absorb them here.
                """
                def __init__(self, *args, **kwargs):
                    kwargs.pop("name", None)
                    super().__init__(*args, **kwargs)

                def display(self, msg=None, pos=None):
                    # Emit progress via Qt signal
                    if self.total and self.total > 0:
                        frac = min(self.n / self.total, 1.0)
                        size_msg = f"{_format_size(int(self.n))} / {_format_size(int(self.total))}"
                        try:
                            signals.progress.emit(repo_id, frac, size_msg)
                        except Exception:
                            pass
                    return True

            local_dir = snapshot_download(
                repo_id,
                tqdm_class=_SignalTqdm,
            )
            self._dl_signals.finished.emit(repo_id, str(local_dir))
        except Exception as e:
            logger.error(f"Download failed for {repo_id}: {e}")
            self._dl_signals.finished.emit(repo_id, f"ERROR: {e}")

    def _on_download_progress(self, repo_id: str, fraction: float, message: str):
        bar = self._progress_bars.get(repo_id)
        if bar:
            bar.setMaximum(100)
            bar.setValue(int(fraction * 100))
            bar.setFormat(f"{repo_id}: {message} ({int(fraction * 100)}%)")

    def _on_download_finished(self, repo_id: str, result: str):
        self._active_downloads.pop(repo_id, None)
        bar = self._progress_bars.pop(repo_id, None)
        if bar:
            if result.startswith("ERROR:"):
                bar.setFormat(f"{repo_id}: FAILED")
                bar.setMaximum(100)
                bar.setValue(0)
                QMessageBox.warning(self, "Download Failed", result)
            else:
                bar.setMaximum(100)
                bar.setValue(100)
                bar.setFormat(f"{repo_id}: Complete")
                self._refresh_local()

        if not self._active_downloads:
            self._progress_group.setVisible(False)
