# Cadabrio

**AI-powered, locally-run 3D creation platform.**

Cadabrio unifies text, images, video, photogrammetry, and existing 3D models into a single intelligent workspace for designing, refining, and exporting production-ready assets across Blender, FreeCAD, Unreal Engine, and Bambu Studio.

## Features

- **Multi-Input Creation** - Generate 3D models from text descriptions, images, video (including drone footage), or existing models
- **AI Chat Console** - Conversational AI assistant for guided 3D creation workflows
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
