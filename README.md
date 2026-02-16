# Cadabrio

**AI-powered, locally-run 3D creation platform.**

Cadabrio unifies text, images, video, photogrammetry, and existing 3D models into a single intelligent workspace for designing, refining, and exporting production-ready assets across Blender, FreeCAD, Unreal Engine, and Bambu Studio.

## Features

- **Multi-Input Creation** - Generate 3D models from text descriptions, images, video (including drone footage), or existing models
- **AI Chat Console** - Conversational AI assistant powered by local LLMs (Qwen, Llama, Mistral, etc.)
- **Photogrammetry** - Reconstruct 3D models from photos or video with scale preservation
- **Integration Hub** - Direct export to Blender, FreeCAD, Unreal Engine, and Bambu Studio
- **Target-Aware Workflow** - Intelligent suggestions based on your output target (print, game, CAD, etc.)
- **Fully Local** - All AI models run locally on your GPU, with full offline capability
- **RTX 5090 Optimized** - Built for NVIDIA Blackwell architecture with CUDA 12.8

## System Requirements

- **OS**: Windows 11 Pro
- **CPU**: Intel Core Ultra 9 285K (or comparable)
- **RAM**: 64 GB
- **GPU**: NVIDIA GeForce RTX 5090 (32 GB VRAM)
- **CUDA**: 12.8+

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/LeeGillie/cadabrio.git
cd cadabrio

# 2. Create the conda environment
conda env create -f environment.yml

# 3. Activate the environment
conda activate cadabrio

# 4. Launch Cadabrio
python -m cadabrio
```

Or use the setup script:
```bash
scripts/setup_env.bat   # Windows
```

## Project Structure

```
Cadabrio/
├── src/cadabrio/           # Main application source
│   ├── ai/                 # AI engine, model management, inference
│   ├── config/             # Configuration system with themes
│   ├── core/               # Project, pipeline, scale management
│   ├── exporters/          # Export to GLB, STL, FBX, 3MF, USD, etc.
│   ├── importers/          # Import images, video, 3D models, text
│   ├── integrations/       # Blender, FreeCAD, Unreal, Bambu Studio
│   ├── photogrammetry/     # 3D reconstruction from photos/video
│   └── ui/                 # PySide6 UI (main window, panels, widgets)
├── tests/                  # Test suite
├── Art and Branding/       # Logos, icons, splash screen
├── resources/              # Runtime resources (icons, styles)
├── scripts/                # Setup and utility scripts
└── docs/                   # Documentation
```

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
ruff check src/ tests/

# Format
black src/ tests/
```

## Technology Stack

| Component | Technology | License |
|-----------|-----------|---------|
| UI Framework | PySide6 (Qt) | LGPL |
| AI Runtime | PyTorch + CUDA 12.8 | BSD |
| Computer Vision | OpenCV | Apache 2.0 |
| 3D Processing | Open3D, trimesh | MIT |
| Environment | Conda | BSD |

## Authors

- **Lee Gillie, CCP** - Creator and lead developer
- **Claude (Anthropic)** - AI development assistant (Claude Opus, Sonnet, Haiku)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Attributions

Cadabrio is built upon the work of many open-source projects. See [ATTRIBUTIONS.md](ATTRIBUTIONS.md) for the full list.

---

## Development Roadmap

### Phase 1 — Foundation (v0.1.0)

- [x] Project structure, build system, conda environment (CUDA 12.8)
- [x] Configuration system with grouped defaults and tabbed editor
- [x] Theme system with Cadabrio Dark theme and theme editor
- [x] Splash screen with version overlay and initialization status
- [x] Main window with dockable panel layout
- [x] AI chat console (UI shell)
- [x] Asset browser with file system navigation
- [x] Config editor dialog with browse buttons, enum dropdowns, live theme switching
- [x] Integration auto-detection at startup (Blender, FreeCAD, Unreal Engine, Bambu Studio)
- [x] Background file logging with rotation and retention (loguru)
- [x] Floating Resource Monitor — CPU, GPU, RAM, VRAM, disk/network I/O (F9)
- [x] Import/export format handlers (GLB, STL, FBX, OBJ, 3MF, USD, etc.)
- [x] Scale management with unit conversion
- [x] Test suite foundation

### Phase 2 — AI Image Generation (v0.2.0)

- [x] AI inference engine with RTX 5090 / Blackwell optimization (bf16)
- [x] Model Manager — scan HF cache, browse HF Hub, download models, VRAM fit indicators
- [x] AI Tools dialog — unified interface for all AI capabilities
- [x] Text-to-image generation (FLUX.1-schnell, SDXL, SD 1.x auto-detection)
- [x] Image-guided generation (img2img) with reference photos
- [x] Depth estimation pipeline (DPT)
- [x] Image segmentation pipeline (SAM)
- [x] Image-to-3D pipeline (TripoSR)
- [x] Web search prompt enhancement with og:image reference photo discovery
- [x] Streamlined generate workflow — one-click search + generate
- [x] Real progress tracking with step counts and cancel button
- [x] Model selection persistence across sessions
- [x] Auto-adjust generation defaults per model architecture (resolution, steps, guidance)
- [x] VRAM estimation with weight-only sizing (handles duplicate format repos)
- [x] Right-click context menu on image previews (copy, save)

### Phase 3 — AI Chat & LLM Inference (v0.3.0)

- [x] Chat LLM pipeline — `AutoTokenizer` + `AutoModelForCausalLM` with GPU inference
- [x] Chat console model selector with VRAM fit indicators and config persistence
- [x] Threaded chat inference with conversation history
- [x] Model categorization for chat models (Qwen, Llama, Mistral, GPT, etc.)
- [ ] Conversational 3D creation workflow (describe → generate → refine)
- [ ] Chat-driven parameter adjustment ("make it more red", "rotate 45 degrees")
- [ ] Multi-step pipeline chaining (text → image → depth → mesh)

### Phase 4 — 3D Viewport & Visualization

- [ ] OpenGL 3D viewport with model rendering
- [ ] Viewport camera controls (orbit, pan, zoom)
- [ ] Grid, axes, and measurement overlays
- [ ] Import 3D models into viewport (GLB, OBJ, STL, FBX)
- [ ] Generated mesh preview in viewport (from image-to-3D)
- [ ] Depth map to 3D point cloud visualization

### Phase 5 — Photogrammetry

- [ ] Photo import and alignment (SIFT/ORB/SuperPoint feature detection)
- [ ] Dense reconstruction (MVS, NeRF, Gaussian Splatting)
- [ ] Video frame extraction for drone footage
- [ ] Automatic scale detection from reference objects
- [ ] Mesh cleanup and texture baking

### Phase 6 — Integration & Export

- [ ] Blender integration — launch, send/receive models, addon
- [ ] FreeCAD integration — export STEP, import parametric models
- [ ] Unreal Engine integration — export USD/FBX with materials
- [ ] Bambu Studio integration — export 3MF with print settings
- [ ] Target-aware export presets (print, game, CAD, general)

### Phase 7 — Polish & Production

- [ ] Undo/redo system
- [ ] Project save/load serialization
- [ ] Texture generation and UV mapping
- [ ] Material editing and PBR workflow
- [ ] Batch processing and queue management
- [ ] Auto-update system
- [ ] Installer and distribution packaging
