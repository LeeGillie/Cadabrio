# CLAUDE.md - Cadabrio Development Context

## Project Overview

**Cadabrio** is an AI-powered, locally-run 3D creation platform that unifies text, images, video, photogrammetry, and existing models into a single intelligent workspace for designing, refining, and exporting production-ready assets across Blender, FreeCAD, Unreal Engine, and Bambu Studio.

- **Creator**: Lee Gillie, CCP
- **AI Development**: Claude (Anthropic) - Opus, Sonnet, Haiku models
- **License**: MIT
- **Version**: 0.1.0

## Target Hardware

- **OS**: Windows 11 Pro
- **CPU**: Intel Core Ultra 9 285K (24 cores, 3700 MHz)
- **RAM**: 64 GB
- **GPU**: NVIDIA GeForce RTX 5090 (Blackwell, compute capability 12.0, 32 GB VRAM)
- **CUDA**: 12.8+ required (Blackwell architecture)

## RTX 5090 Compatibility Notes

**CRITICAL**: The RTX 5090 uses NVIDIA Blackwell architecture (compute capability 12.0). Known compatibility requirements:
- CUDA Toolkit 12.8+ is mandatory
- PyTorch 2.6+ is required for Blackwell support
- Use `bfloat16` as the preferred inference dtype (hardware native)
- Flash Attention and CUDA Graphs are supported and should be enabled
- TensorRT requires version compatible with Blackwell
- Do NOT use older CUDA versions or PyTorch < 2.6 — they will fail silently or crash
- ONNX Runtime GPU must be version 1.17+ for Blackwell support

## Technology Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| UI | PySide6 (Qt for Python) | Best Windows desktop integration, docking, OpenGL viewports, LGPL |
| AI Runtime | PyTorch + CUDA 12.8 | Direct tensor control, all model formats, no Ollama dependency |
| Environment | Conda | Best CUDA/NVIDIA dependency management |
| 3D Processing | Open3D + trimesh | Mesh ops, point clouds, photogrammetry support |
| Computer Vision | OpenCV | Image/video processing, feature detection |
| Python | 3.12 | Current stable, all dependencies compatible |

## Architecture

```
src/cadabrio/
├── __main__.py          # Entry point
├── version.py           # Version constants
├── config/              # Configuration system (grouped, tabbed editor)
│   ├── manager.py       # ConfigManager - load/save/get/set with listeners
│   ├── defaults.py      # Default config values (groups with _label keys)
│   └── themes/          # Color theme system
│       ├── cadabrio_dark.json  # Default theme (dark + neon green)
│       └── theme_manager.py    # Theme loading, applying, saving
├── ui/                  # PySide6 user interface
│   ├── splash_screen.py # Splash with promo5.png, version overlay, status text
│   ├── main_window.py   # Main window with dockable panels
│   └── panels/          # Dockable panel widgets
│       ├── viewport_3d.py    # 3D viewport (OpenGL placeholder)
│       ├── chat_console.py   # AI chat/conversation console
│       ├── asset_browser.py  # File browser with import support
│       ├── config_editor.py  # Tabbed config editor dialog
│       └── resource_monitor.py # Floating system resource monitor (F9)
├── core/                # Core logic
│   ├── project.py       # Project model with target types
│   ├── pipeline.py      # Processing pipeline with stages
│   ├── scale_manager.py # Unit conversion and scale tracking
│   └── logging.py       # Centralized file logging with rotation
├── ai/                  # AI inference engine
│   ├── engine.py        # PyTorch + CUDA inference engine
│   ├── model_manager.py # Model discovery, download, lifecycle
│   └── inference.py     # GPU inference utilities and dtype selection
├── integrations/        # External application interfaces
│   ├── blender.py       # Blender integration with auto-detection
│   ├── freecad.py       # FreeCAD integration
│   ├── unreal.py        # Unreal Engine integration
│   └── bambu_studio.py  # Bambu Studio integration
├── photogrammetry/      # 3D reconstruction
│   └── pipeline.py      # Photogrammetry job management
├── importers/           # File import handlers
│   ├── image.py         # PNG, JPEG, TIFF, EXR, HDR
│   ├── video.py         # MP4, AVI, MOV (including drone footage)
│   ├── model_3d.py      # OBJ, STL, FBX, glTF, 3MF, STEP, PLY, USD
│   └── text.py          # Text prompts and descriptions
└── exporters/           # File export handlers
    └── formats.py       # GLB, glTF, OBJ, STL, FBX, 3MF, USD
```

## Key Design Principles

1. **Configuration everywhere** - All settings are grouped and editable via a tabbed config editor. Groups include: General, Appearance, GPU & Compute, AI Models, 3D Viewport, Photogrammetry, Integrations, Export, Logging, Network.

2. **Theme system** - Default "Cadabrio Dark" theme (dark background + neon green `#39ff14` accents). Supports built-in themes, Windows system themes, and user-created themes via a theme editor.

3. **Splash screen** - Shows `Art and Branding/promo5.png` during initialization with version in upper-right corner and status messages in bottom-left corner. Visible only during component loading.

4. **Scale preservation** - Scale is tracked through every transformational step from import through AI processing to export.

5. **Offline-first** - All AI models run locally. Offline mode is a config toggle.

6. **Direct GPU control** - Uses PyTorch directly (not Ollama) for full control over model formats (SafeTensors, GGUF, ONNX) and GPU memory.

7. **Attribution always** - Every borrowed open-source work gets attribution in ATTRIBUTIONS.md.

8. **Centralized logging** - File-based logging via loguru with rotation and retention. Logs always capture DEBUG level to `%LOCALAPPDATA%/Cadabrio/Cadabrio/Logs/cadabrio.log` for diagnostics, console level controlled by config. Logs are compressed and rotated by size, retained by age.

9. **Resource monitor** - Floating window (View > Resource Monitor, F9) showing real-time CPU, GPU utilization, RAM, VRAM, disk I/O, and network I/O. Uses psutil + pynvml. Toggleable on/off.

## Commands

```bash
# Setup
conda env create -f environment.yml
conda activate cadabrio

# Run
python -m cadabrio

# Test
pytest tests/ -v

# Lint & Format
ruff check src/ tests/
black src/ tests/
```

## Configuration Groups

Each group appears as a tab in the Preferences dialog (Ctrl+,):
- **General**: language, auto-save, recent projects
- **Appearance**: theme, font, UI scale
- **GPU & Compute**: CUDA device, memory limit, precision, TensorRT
- **AI Models**: models directory, default models, download source
- **3D Viewport**: renderer, antialiasing, grid, units, camera
- **Photogrammetry**: feature detector, reconstruction method, quality
- **Integrations**: paths to Blender, FreeCAD, Unreal Engine, Bambu Studio
- **Export**: default format, target, scale factor, metadata
- **Logging**: log level, file logging, retention, max size, console output
- **Network**: offline mode, proxy, download settings

## Current Status

**v0.1.0 - Foundation**
- [x] Project structure and build system
- [x] Conda environment with CUDA 12.8
- [x] Configuration system with grouped defaults
- [x] Theme system with Cadabrio Dark theme
- [x] Splash screen with version and status overlay
- [x] Main window with dockable panel layout
- [x] AI chat console (UI shell)
- [x] Asset browser with file system navigation
- [x] Config editor dialog (tabbed)
- [x] Integration stubs (Blender, FreeCAD, Unreal, Bambu)
- [x] Photogrammetry pipeline stub
- [x] Import/export format handlers
- [x] Scale management with unit conversion
- [x] Test suite foundation (16 tests passing)
- [x] Integration auto-detection at startup (Blender, FreeCAD, Unreal, Bambu)
- [x] Config editor with browse buttons, enum dropdowns, live theme switching
- [x] Background file logging with rotation and retention (loguru)
- [x] Floating Resource Monitor (CPU, GPU, RAM, VRAM, disk I/O, network I/O) — F9
- [ ] OpenGL 3D viewport rendering
- [ ] AI engine model loading and inference
- [ ] Hugging Face model download
- [ ] Photogrammetry reconstruction
- [ ] Integration launch and data exchange
- [ ] Theme editor widget
- [ ] Project save/load serialization

## File Naming Conventions

- Python modules: `snake_case.py`
- Config files: `snake_case.json`
- Test files: `test_<module>.py`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`

## Notes for Claude Sessions

- Always check RTX 5090 / Blackwell / CUDA 12.8 compatibility before adding GPU-related dependencies
- The project uses `src/` layout — imports are `from cadabrio.xxx import yyy`
- Config values are accessed as `config.get("group", "key", default)`
- Theme colors are in the JSON files under `config/themes/`
- The splash image is at `Art and Branding/promo5.png` — do not move or rename
- All integrations auto-detect installation paths on Windows
- Keep ATTRIBUTIONS.md updated when adding new dependencies
