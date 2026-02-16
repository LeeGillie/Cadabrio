# CLAUDE.md - Cadabrio Development Context

## Project Overview

**Cadabrio** is an AI-powered, locally-run 3D creation platform that unifies text, images, video, photogrammetry, and existing models into a single intelligent workspace for designing, refining, and exporting production-ready assets across Blender, FreeCAD, Unreal Engine, and Bambu Studio.

- **Creator**: Lee Gillie, CCP
- **AI Development**: Claude (Anthropic) - Opus, Sonnet, Haiku models
- **License**: MIT
- **Version**: 0.3.0
- **Repository**: https://github.com/LeeGillie/Cadabrio

## Target Hardware

- **OS**: Windows 11 Pro
- **CPU**: Intel Core Ultra 9 285K (24 cores, 3700 MHz)
- **RAM**: 64 GB
- **GPU**: NVIDIA GeForce RTX 5090 (Blackwell, compute capability 12.0, 32 GB VRAM)
- **CUDA**: 12.8+ required (Blackwell architecture)

## RTX 5090 Compatibility Notes

**CRITICAL**: The RTX 5090 uses NVIDIA Blackwell architecture (compute capability 12.0). Known compatibility requirements:
- CUDA Toolkit 12.8+ is mandatory
- PyTorch 2.6+ is required for Blackwell support (currently using 2.10.0+cu128)
- Use `bfloat16` as the preferred inference dtype (hardware native)
- Flash Attention and CUDA Graphs are supported and should be enabled
- TensorRT requires version compatible with Blackwell
- Do NOT use older CUDA versions or PyTorch < 2.6 — they will fail silently or crash
- ONNX Runtime GPU must be version 1.17+ for Blackwell support

## Technology Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| UI | PySide6 (Qt for Python) | Best Windows desktop integration, docking, OpenGL viewports, LGPL |
| AI Runtime | PyTorch 2.10.0 + CUDA 12.8 | Direct tensor control, all model formats, no Ollama dependency |
| Diffusion | diffusers 0.36.0 | HuggingFace pipelines for FLUX, SDXL, SD 1.x |
| Transformers | transformers | Depth estimation, segmentation (SAM), chat LLM inference |
| Web Search | duckduckgo-search (ddgs) | Prompt enhancement, reference image discovery |
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
│   ├── panels/          # Dockable panel widgets
│   │   ├── viewport_3d.py    # 3D viewport (OpenGL placeholder)
│   │   ├── chat_console.py   # AI chat console with model selector + LLM inference
│   │   ├── asset_browser.py  # File browser with import support
│   │   ├── config_editor.py  # Tabbed config editor dialog
│   │   └── resource_monitor.py # Floating system resource monitor (F9)
│   └── widgets/         # Standalone dialog widgets
│       ├── ai_tools_dialog.py      # AI Tools — txt2img, depth, img2mesh, segmentation
│       └── model_manager_dialog.py # Model Manager — browse, download, manage HF models
├── core/                # Core logic
│   ├── project.py       # Project model with target types
│   ├── pipeline.py      # Processing pipeline with stages
│   ├── scale_manager.py # Unit conversion and scale tracking
│   └── logging.py       # Centralized file logging with rotation
├── ai/                  # AI inference engine
│   ├── engine.py        # PyTorch + CUDA inference engine
│   ├── inference.py     # GPU inference utilities and dtype selection
│   ├── model_manager.py # Model discovery, VRAM estimation, HF Hub search
│   ├── pipelines.py     # Pipeline runners (FLUX, SDXL, SD 1.x, depth, SAM, TripoSR, chat LLM)
│   ├── prompt_enhance.py # Web search prompt enhancement + reference image search
│   └── reference_images.py # Multi-reference image data structures + curation
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

7. **HuggingFace Hub cache** - All models use the standard HF Hub cache directory (`~/.cache/huggingface/hub`). No separate Cadabrio models directory. Model Manager scans this cache for local models.

8. **Attribution always** - Every borrowed open-source work gets attribution in ATTRIBUTIONS.md.

9. **Centralized logging** - File-based logging via loguru with rotation and retention. Logs always capture DEBUG level to `%LOCALAPPDATA%/Cadabrio/Cadabrio/Logs/cadabrio.log` for diagnostics, console level controlled by config. Logs are compressed and rotated by size, retained by age.

10. **Resource monitor** - Floating window (View > Resource Monitor, F9) showing real-time CPU, GPU utilization, RAM, VRAM, disk I/O, and network I/O. Uses psutil + pynvml. Toggleable on/off.

## AI Pipeline Architecture

### Model Architecture Detection

`pipelines.py` auto-detects the model architecture and uses the correct pipeline class:

- **FLUX** (e.g. `black-forest-labs/FLUX.1-schnell`): `FluxPipeline` / `FluxImg2ImgPipeline`, 1024x1024, 4 steps, guidance 0.0
- **SDXL** (e.g. `stabilityai/stable-diffusion-xl-base-1.0`): `DiffusionPipeline` / `StableDiffusionXLImg2ImgPipeline`, 1024x1024, 25 steps, guidance 7.5
- **SD 1.x** (e.g. `CompVis/stable-diffusion-v1-4`): `DiffusionPipeline` / `StableDiffusionImg2ImgPipeline`, 512x512, 25 steps, guidance 7.5
- **SD 2.x** and unknown: 768x768, 25 steps, guidance 7.5

Detection functions: `_is_flux()`, `_is_sdxl()`, `_is_sd1x()`. Default params: `model_default_params()`. Native resolution: `model_native_res()`.

### Generation Workflow (AI Tools Dialog)

**Multi-Reference Image Workflow** (primary path):
1. User types a prompt and checks "Search web for reference images"
2. Click **Search** → `search_reference_candidates()` uses `ddgs.images()` API to find candidates
3. Gallery populates with thumbnail cards — user accepts/rejects, adds annotations
4. User can: Search for More (paginated), Paste from Clipboard, Browse for files, Clear All
5. Annotations describe what each reference shows (e.g. "correct color", "shows lift kit")
6. Click **Generate** → builds combined prompt (base + annotations + refining instructions + enhanced)
7. Uses best accepted reference with `image_guided_generate()` (img2img)
8. `multi_reference_generate()` stub exists for future IP-Adapter multi-image conditioning

**Simple generation** (no web search):
1. User types prompt, clicks **Generate** → direct `text_to_image()` call

**Legacy search+generate** (web search checked, no gallery curation):
1. If Generate clicked before Search → runs `search_and_enhance()` then auto-generates

**Reference data model**: `ReferenceCollection` manages `ReferenceImage` instances with status (CANDIDATE/ACCEPTED/REJECTED/USER_ADDED), annotations, and lazy full-res downloads.

Progress bar shows real step count ("Step 2/4"), with Cancel button. Model selections are persisted to config across sessions.

### VRAM Estimation

`model_manager.py` estimates VRAM requirements:
- Scans safetensors files in HF cache snapshots
- Handles duplicate formats (e.g. FLUX has single-file + split = 54GB disk, but only ~24GB of weights)
- Prefers subdirectory weights (diffusers format) over top-level single-file exports
- Fit levels: good (<70% VRAM), tight (70-100%), bad (>100%), unknown

### Web Search Image Discovery

**Multi-candidate search** (primary, via `search_reference_candidates()`):
- Tries `ddgs.images()` API first (direct image URLs, thumbnails, titles, dimensions)
- If `ddgs.images()` returns HTTP 403 (rate-limited), falls back to text search + og:image scraping
- Downloads thumbnails for gallery preview, full-resolution downloaded lazily on accept
- "Search for More" varies query with keyword suffixes to get different results
- URL deduplication via `ReferenceCollection.known_urls` prevents duplicate cards

**Legacy og:image scraping** (still available via `search_and_enhance()`):
1. Text search via `ddgs.text()` → get result page URLs
2. Fetch first ~100KB of each page
3. Extract `<meta property="og:image" content="URL">` via regex
4. Download and validate (min 200x200, min 5KB)

### Chat / LLM Pipeline

`pipelines.py` section 5 provides local LLM chat via `AutoTokenizer` + `AutoModelForCausalLM`:

- **`_load_chat(model_id)`** — loads tokenizer + model with `dtype=_dtype()`, explicit `.to(_device())`, caches as `_CACHE["chat"]`
- **`chat_generate(messages, model_id, max_new_tokens, temperature)`** — takes a list of `{"role": ..., "content": ...}` dicts, applies chat template, runs `model.generate()`, returns assistant reply
- Model categorization in `model_manager.py` uses keyword matching — `"qwen"` is included alongside `"llama"`, `"mistral"`, etc.
- Chat model selection is persisted via config key `ai.selected_chat_model`

**Chat Console** (`chat_console.py`):
- Model selector dropdown scans `ModelManager.list_models(ModelCategory.CHAT)` for local models
- Shows VRAM fit indicators (same symbols as AI Tools)
- Supports "Custom model ID..." for unlisted models
- Threaded inference via `_ChatSignals` (QObject signals: `response_ready`, `error`)
- Conversation history maintained as `self._messages` list of role/content dicts
- "Generating..." indicator shown during inference, input disabled until complete

### Transformers Compatibility Notes

- **`dtype` vs `torch_dtype`**: Newer transformers deprecates `torch_dtype` in `from_pretrained()` — use `dtype` instead.
- **`apply_chat_template` return type**: With `return_tensors="pt"`, may return a `BatchEncoding` dict (not a raw tensor). Always extract `["input_ids"]` from the result; don't assume `.shape` is available on the top-level return value.

### PySide6 Notes

- **Signal int overflow**: `Signal(str, int)` is 32-bit signed. Model sizes in bytes exceed 2GB max. Use `Signal(str, float)` and emit `float(size)`.
- **tqdm compatibility**: `snapshot_download(tqdm_class=...)` requires full tqdm interface. Subclass real `tqdm` and only override `display()` — don't mock it.
- **QScrollArea wrapping**: All tab content uses `_scrollable()` to wrap in QScrollArea for proper scrolling when content grows dynamically.
- **Screen bounds**: Dialog constrained to screen via `setMaximumSize()` and `moveEvent` override to keep title bar visible.

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
- **AI Models**: default models, model selections (persisted per category), download source
- **3D Viewport**: renderer, antialiasing, grid, units, camera
- **Photogrammetry**: feature detector, reconstruction method, quality
- **Integrations**: paths to Blender, FreeCAD, Unreal Engine, Bambu Studio
- **Export**: default format, target, scale factor, metadata
- **Logging**: log level, file logging, retention, max size, console output
- **Network**: offline mode, proxy, download settings

## File Naming Conventions

- Python modules: `snake_case.py`
- Config files: `snake_case.json`
- Test files: `test_<module>.py`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`

## Notes for Claude Sessions

- Always read this file first to initialize context after resets
- Check the Development Roadmap section in README.md for current task priorities
- Always check RTX 5090 / Blackwell / CUDA 12.8 compatibility before adding GPU-related dependencies
- The project uses `src/` layout — imports are `from cadabrio.xxx import yyy`
- Config values are accessed as `config.get("group", "key", default)`, saved with `config.set()` + `config.save()`
- Theme colors are in the JSON files under `config/themes/`
- The splash image is at `Art and Branding/promo5.png` — do not move or rename
- All integrations auto-detect installation paths on Windows
- All AI models use the HuggingFace Hub cache — no separate Cadabrio models directory
- Model selections in AI Tools are persisted via config keys `ai.selected_txt2img_model`, etc.
- Chat model selection is persisted via `ai.selected_chat_model`; chat console scans for `ModelCategory.CHAT` models
- Multi-reference image workflow: `search_reference_candidates()` uses `ddgs.images()` for candidate thumbnails; `search_and_enhance()` still uses og:image scraping for legacy single-image flow
- `ReferenceCollection` in `reference_images.py` manages the curation state; `_ReferenceGallery` in the dialog displays it
- Config keys `ai.max_reference_candidates` (default 12) and `ai.auto_accept_best_reference` (default True) control reference search behavior
- Keep ATTRIBUTIONS.md updated when adding new dependencies
- Keep this CLAUDE.md updated when architecture or key systems change
- Keep the README.md Development Roadmap updated when tasks are completed or priorities change
