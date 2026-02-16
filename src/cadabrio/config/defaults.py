"""Default configuration values for Cadabrio.

Configuration is organized into groups. Each group becomes a tab
in the configuration editor UI.
"""

DEFAULT_CONFIG = {
    # --- General ---
    "general": {
        "_label": "General",
        "language": "en",
        "auto_save": True,
        "auto_save_interval_seconds": 300,
        "recent_projects_max": 20,
        "check_updates_on_start": False,
    },
    # --- Appearance ---
    "appearance": {
        "_label": "Appearance",
        "theme": "cadabrio_dark",
        "font_family": "Segoe UI",
        "font_size": 10,
        "viewport_background": "#1a1a2e",
        "show_toolbar_labels": True,
        "ui_scale": 1.0,
    },
    # --- GPU / Compute ---
    "gpu": {
        "_label": "GPU & Compute",
        "cuda_device": 0,
        "gpu_memory_limit_gb": 0,  # 0 = no limit
        "tensor_precision": "float16",  # float16, bfloat16, float32
        "enable_flash_attention": True,
        "enable_cuda_graphs": True,
        "tensorrt_enabled": True,
        "tensorrt_cache_dir": "",
    },
    # --- AI Models ---
    "ai": {
        "_label": "AI Models",
        "models_directory": "",  # deprecated — models use HuggingFace cache
        "default_text_to_3d_model": "",
        "default_image_to_3d_model": "",
        "default_chat_model": "",
        "max_context_length": 8192,
        "inference_batch_size": 1,
        "download_source": "huggingface",  # huggingface, civitai, local
        # Persisted model selections from AI Tools dialog
        "selected_txt2img_model": "",
        "selected_depth_model": "",
        "selected_img2mesh_model": "",
        "selected_segmentation_model": "",
        "selected_chat_model": "",
        # Reference image search preferences
        "max_reference_candidates": 12,
        "auto_accept_best_reference": True,
    },
    # --- 3D Viewport ---
    "viewport": {
        "_label": "3D Viewport",
        "renderer": "opengl",  # opengl, vulkan
        "antialiasing": "msaa_4x",
        "show_grid": True,
        "grid_size": 10.0,
        "grid_subdivisions": 10,
        "default_units": "millimeters",  # millimeters, centimeters, meters, inches
        "camera_fov": 45.0,
        "orbit_sensitivity": 1.0,
        "zoom_sensitivity": 1.0,
    },
    # --- Photogrammetry ---
    "photogrammetry": {
        "_label": "Photogrammetry",
        "feature_detector": "sift",  # sift, orb, superpoint
        "match_threshold": 0.7,
        "dense_reconstruction": "mvs",  # mvs, nerf, gaussian_splatting
        "mesh_quality": "high",  # low, medium, high, ultra
        "texture_resolution": 4096,
        "auto_scale_detection": True,
    },
    # --- Integrations ---
    "integrations": {
        "_label": "Integrations",
        "blender_path": "",
        "freecad_path": "",
        "bambu_studio_path": "",
        "unreal_engine_path": "",
        "blender_addon_auto_install": True,
    },
    # --- Export ---
    "export": {
        "_label": "Export",
        "default_format": "glb",  # glb, gltf, obj, stl, fbx, 3mf, usd
        "default_target": "general",  # general, print, unreal, blender, freecad, bambu
        "embed_textures": True,
        "export_scale_factor": 1.0,
        "stl_binary": True,
        "include_metadata": True,
    },
    # --- Logging ---
    "logging": {
        "_label": "Logging",
        "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
        "log_to_file": True,
        "log_retention_days": 30,
        "log_max_size_mb": 50,
        "log_console_output": True,
    },
    # --- Network ---
    "network": {
        "_label": "Network",
        "offline_mode": False,
        "proxy": "",
        "model_download_timeout_seconds": 600,
        "max_concurrent_downloads": 2,
    },
}
