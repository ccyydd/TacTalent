"""Keep native collection initialization separate from evaluation handoff settings."""

LEGACY_KEYS = (
    "decimation",
    "save_frequency",
    "video_frequency",
    "render_frequency",
    "random_texture",
    "save_pre_move",
    "skip_pre_move",
    "step_lim",
    "max_save_frames",
    "reset_final_steps",
)
# scripts/collect_data.py preserves TaskCfg.skip_pre_move. Force tasks apply
# their explicit overrides later through native prepare_force_task_config.
COLLECTION_KEYS = (
    "decimation",
    "save_frequency",
    "video_frequency",
    "render_frequency",
    "random_texture",
    "save_pre_move",
    "reset_time_limit",
    "reset_first_frame_steps",
    "reset_after_actor_steps",
    "reset_final_steps",
    "reset_render_warmup_steps",
    "use_adaptive_grasp",
    "adaptive_grasp_depth_threshold",
)


def apply_settings(cfg, settings, *, benchmark, mode):
    collecting_openvitac = benchmark == "OpenViTac" and mode == "collect"
    keys = COLLECTION_KEYS if collecting_openvitac else LEGACY_KEYS
    for key in keys:
        if key in settings:
            setattr(cfg, key, settings[key])
    if benchmark == "OpenViTac" and not collecting_openvitac:
        cfg.eval_start_delay_steps = settings.get(
            "eval_start_delay_steps", 0 if cfg.skip_pre_move else 20
        )
