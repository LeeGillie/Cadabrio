"""Asset Browser panel for Cadabrio.

Provides file browsing and ingestion for images, videos, 3D models,
and other supported file types from disk or network resources.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTreeView,
    QLabel,
    QPushButton,
    QLineEdit,
    QFileSystemModel,
)

from cadabrio.config.manager import ConfigManager


class AssetBrowserPanel(QWidget):
    """Asset browser panel for importing and managing project files."""

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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()
        self._path_input = QLineEdit()
        self._path_input.setPlaceholderText("Browse path...")
        toolbar.addWidget(self._path_input)

        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_folder)
        toolbar.addWidget(browse_btn)

        import_btn = QPushButton("Import")
        import_btn.clicked.connect(self._import_selected)
        toolbar.addWidget(import_btn)

        layout.addLayout(toolbar)

        # File tree view
        self._fs_model = QFileSystemModel()
        self._fs_model.setNameFilters(self.SUPPORTED_EXTENSIONS)
        self._fs_model.setNameFilterDisables(False)

        self._tree = QTreeView()
        self._tree.setModel(self._fs_model)
        self._tree.setAlternatingRowColors(True)
        self._tree.setSortingEnabled(True)
        layout.addWidget(self._tree)

    def _browse_folder(self):
        """Open a folder selection dialog."""
        from PySide6.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(self, "Select Asset Folder")
        if folder:
            self._path_input.setText(folder)
            self._fs_model.setRootPath(folder)
            self._tree.setRootIndex(self._fs_model.index(folder))

    def _import_selected(self):
        """Import the currently selected file(s) into the project."""
        indexes = self._tree.selectedIndexes()
        for index in indexes:
            path = self._fs_model.filePath(index)
            if path:
                # TODO: Route to appropriate importer based on file type
                pass
