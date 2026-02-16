"""Asset Browser panel for Cadabrio.

Provides file browsing and ingestion for images, videos, 3D models,
and other supported file types from disk or network resources.
Routes selected files to the appropriate importer and shows results.
"""

from pathlib import Path

from loguru import logger
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTreeView,
    QLabel,
    QPushButton,
    QLineEdit,
    QFileSystemModel,
    QFileDialog,
    QMessageBox,
)

from cadabrio.config.manager import ConfigManager
from cadabrio.importers import image, video, model_3d


class AssetBrowserPanel(QWidget):
    """Asset browser panel for importing and managing project files."""

    # Signal emitted when an asset is successfully imported: (metadata dict,)
    asset_imported = Signal(dict)

    # Supported file extensions for filtering
    SUPPORTED_EXTENSIONS = [
        # Images
        "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif", "*.webp", "*.exr", "*.hdr",
        # Video
        "*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm",
        # 3D Models
        "*.obj", "*.stl", "*.fbx", "*.glb", "*.gltf", "*.3mf", "*.ply", "*.usd", "*.usda",
        "*.usdz", "*.blend", "*.step", "*.stp", "*.iges", "*.igs",
        # Project
        "*.cadabrio",
    ]

    def __init__(self, config: ConfigManager, parent=None):
        super().__init__(parent)
        self._config = config
        self._imported_assets: list[dict] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()
        self._path_input = QLineEdit()
        self._path_input.setPlaceholderText("Browse path...")
        self._path_input.returnPressed.connect(self._navigate_to_path)
        toolbar.addWidget(self._path_input)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_folder)
        toolbar.addWidget(browse_btn)

        import_btn = QPushButton("Import")
        import_btn.clicked.connect(self._import_selected)
        toolbar.addWidget(import_btn)

        import_file_btn = QPushButton("Import File...")
        import_file_btn.clicked.connect(self._import_file_dialog)
        toolbar.addWidget(import_file_btn)

        layout.addLayout(toolbar)

        # File tree view
        self._fs_model = QFileSystemModel()
        self._fs_model.setNameFilters(self.SUPPORTED_EXTENSIONS)
        self._fs_model.setNameFilterDisables(False)

        self._tree = QTreeView()
        self._tree.setModel(self._fs_model)
        self._tree.setAlternatingRowColors(True)
        self._tree.setSortingEnabled(True)
        self._tree.setSelectionMode(QTreeView.SelectionMode.ExtendedSelection)
        self._tree.doubleClicked.connect(self._on_double_click)
        layout.addWidget(self._tree)

        # Status bar
        self._status = QLabel("No assets imported")
        self._status.setStyleSheet("color: #a0a0b0; padding: 2px; font-size: 9pt;")
        layout.addWidget(self._status)

    def _navigate_to_path(self):
        """Navigate tree to the path typed in the input."""
        path = self._path_input.text().strip()
        if path and Path(path).is_dir():
            self._fs_model.setRootPath(path)
            self._tree.setRootIndex(self._fs_model.index(path))

    def _browse_folder(self):
        """Open a folder selection dialog."""
        folder = QFileDialog.getExistingDirectory(self, "Select Asset Folder")
        if folder:
            self._path_input.setText(folder)
            self._fs_model.setRootPath(folder)
            self._tree.setRootIndex(self._fs_model.index(folder))

    def _import_file_dialog(self):
        """Open a file dialog to import one or more files directly."""
        ext_filter = (
            "All Supported (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp *.exr *.hdr "
            "*.mp4 *.avi *.mov *.mkv *.webm "
            "*.obj *.stl *.fbx *.glb *.gltf *.3mf *.ply *.usd *.usda *.usdz "
            "*.blend *.step *.stp *.iges *.igs);;"
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp *.exr *.hdr);;"
            "Video (*.mp4 *.avi *.mov *.mkv *.webm);;"
            "3D Models (*.obj *.stl *.fbx *.glb *.gltf *.3mf *.ply *.usd *.blend *.step);;"
            "All Files (*)"
        )
        files, _ = QFileDialog.getOpenFileNames(self, "Import Files", "", ext_filter)
        for f in files:
            self._import_file(Path(f))

    def _on_double_click(self, index):
        """Import a file on double-click."""
        path = self._fs_model.filePath(index)
        if path and Path(path).is_file():
            self._import_file(Path(path))

    def _import_selected(self):
        """Import the currently selected file(s) from the tree."""
        indexes = self._tree.selectedIndexes()
        # QTreeView with columns gives multiple indexes per row; take column 0 only
        seen = set()
        for index in indexes:
            if index.column() != 0:
                continue
            path = self._fs_model.filePath(index)
            if path and path not in seen and Path(path).is_file():
                seen.add(path)
                self._import_file(Path(path))

        if not seen:
            self._status.setText("No file selected")

    def _import_file(self, path: Path):
        """Route a file to the appropriate importer and handle the result."""
        logger.info(f"Importing: {path}")

        metadata = None
        error = None

        try:
            if image.can_import(path):
                metadata = image.import_image(path)
            elif video.can_import(path):
                metadata = video.import_video(path)
            elif model_3d.can_import(path):
                metadata = model_3d.import_model(path)
            else:
                error = f"Unsupported file type: {path.suffix}"
        except Exception as e:
            error = str(e)
            logger.error(f"Import failed for {path}: {e}")

        if error:
            self._status.setText(f"Import failed: {error}")
            QMessageBox.warning(self, "Import Failed", f"Could not import:\n{path}\n\n{error}")
            return

        if metadata and "error" in metadata:
            self._status.setText(f"Import error: {metadata['error']}")
            QMessageBox.warning(
                self, "Import Error",
                f"Imported with errors:\n{path}\n\n{metadata['error']}"
            )
            return

        if metadata:
            self._imported_assets.append(metadata)
            asset_type = metadata.get("type", "unknown")
            summary = self._format_summary(metadata)
            self._status.setText(f"Imported {asset_type}: {path.name} — {summary}")
            logger.info(f"Imported {asset_type}: {path.name} — {summary}")
            self.asset_imported.emit(metadata)

    def _format_summary(self, meta: dict) -> str:
        """Format a short summary string from import metadata."""
        t = meta.get("type", "")
        if t == "image":
            return f"{meta.get('width', '?')}x{meta.get('height', '?')} {meta.get('mode', '')}"
        if t == "video":
            dur = meta.get("duration_seconds", 0)
            return (
                f"{meta.get('width', '?')}x{meta.get('height', '?')} "
                f"{meta.get('fps', '?')} fps, {dur:.1f}s"
            )
        if t == "model_3d":
            verts = meta.get("vertices")
            objs = meta.get("objects")
            if verts is not None:
                return f"{verts:,} vertices, {meta.get('faces', '?'):,} faces"
            if objs is not None:
                return f"{objs} objects"
        return "OK"

    @property
    def imported_assets(self) -> list[dict]:
        """Return list of all imported asset metadata."""
        return self._imported_assets
