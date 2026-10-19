"""Run one upstream scripted TRAINING episode, retaining native success criteria."""

import argparse, importlib, json, os, time, traceback, hashlib, threading
from pathlib import Path
from tactalent.benchmarks.openvitac.profile import ROOT, UPSTREAM_COMMIT as COMMIT
from tactalent.roboicl.benchmarks import binding
from tactalent.utils.files import write_json


def defaults(task, variant):
    spec = binding("OpenViTac", task, variant)
    return dict(
        spec,
        sensor_type=spec["sensor"],
        renderer_settings=dict(
            rendering_mode="quality", enable_dl_denoiser=True, antialiasing_mode="FXAA"
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["OpenViTac"], default="OpenViTac")
    p.add_argument("--task", required=True)
    p.add_argument("--variant")
    p.add_argument("--conditions", type=json.loads, default={})
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--kit_args", default="")
    a = p.parse_args()
    if not 0 <= a.seed < 1000000:
        raise ValueError("Use a training seed below 1000000")
    from tactalent.utils.storage import require_output_path

    config = defaults(a.task, a.variant)
    out = require_output_path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        task=a.task,
        variant=a.variant,
        seed=a.seed,
        status="error",
        split="train",
        upstream_commit=COMMIT,
    )
    app = task = None
    reset = False
    started = time.monotonic()
    try:
        from tactalent.roboicl.upstream import collection_sources

        result["source_fingerprints"] = collection_sources(binding("OpenViTac", a.task, a.variant))
        from isaaclab.app import AppLauncher

        app = AppLauncher(
            dict(
                headless=True,
                enable_cameras=True,
                livestream=2,
                device="cuda:0",
                kit_args=a.kit_args,
            )
        ).app
        from functools import partial
        import tacex.simulation_approaches.gpu_taxim.taxim_sim as optical
        from tacex.simulation_approaches.gpu_taxim.sim.taxim import Taxim

        optical.Taxim = partial(Taxim, backend="torch")
        import yaml
        from isaaclab.sim import RenderCfg

        module = importlib.import_module("envs." + config["native_task"])
        cfg = module.TaskCfg()
        settings_path = ROOT / "task_config" / (config["task_config"] + ".yml")
        settings = yaml.safe_load(settings_path.read_text())
        overrides = a.conditions
        allowed = {
            "weight_label",
            "hardness_label",
            "roughness_label",
            "initial_grasp_side",
            "rough_block_side",
            "empty_can",
        }
        if not isinstance(overrides, dict) or set(overrides) - allowed:
            raise ValueError("Only official training class/branch selectors are allowed")
        if any(not hasattr(cfg, key) for key in overrides):
            raise ValueError("Task does not support requested training selector")
        for key, value in overrides.items():
            setattr(cfg, key, value)
        result["training_condition_overrides"] = overrides
        cfg.sim.render = RenderCfg(**config["renderer_settings"])
        cfg.scene.num_envs = 1
        cfg.sim.device = "cuda:0"
        cfg.sim.enable_scene_query_support = True
        cfg.save_dir = out / "native"
        cfg.tactile_sensor_type = config["sensor_type"]
        from tactalent.roboicl.native_settings import apply_settings

        apply_settings(cfg, settings, benchmark="OpenViTac", mode="collect")
        result["collection_initialization"] = {
            "skip_pre_move": bool(cfg.skip_pre_move),
            "source": "Native collect_data.py initialization: preserve TaskCfg default, then apply official force-task configuration if applicable",
        }
        cfg.obs_data_type = {"camera": ["rgb"], "tactile": ["rgb"], "embodiment": ["joint", "ee"]}
        cfg.tactile_video_key = "rgb"
        if a.task == "insert_block_v1":
            cfg.target_block = "cube"
        if a.task == "insert_block_v2":
            cfg.target_block = a.variant
        if a.task == "grasp_fragile_chip":
            from envs._force_task_utils import prepare_force_task_config

            prepare_force_task_config(cfg, settings, settings_path, seed=a.seed)
        result["collection_initialization"]["skip_pre_move"] = bool(cfg.skip_pre_move)
        import inspect

        result["collection_initialization"].update(
            pre_move_source=inspect.getsource(module.Task.pre_move),
            settings_path=str(settings_path),
            collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        result["collection_initialization"]["native_force_configuration"] = (
            a.task == "grasp_fragile_chip"
        )
        task = module.Task(cfg, mode="collect")
        if a.task == "grasp_fragile_chip":
            from envs._force_task_utils import run_force_task_episode

            row = run_force_task_episode(task, seed=a.seed)
            success = bool(row["succ_num"])
        else:
            task.reset(seed=a.seed)
            reset = True
            task.play_once()
            success = bool(
                task.plan_success and task.check_success() and not task.check_early_stop()
            )
            if success:
                task.save_to_hdf5()
            task.clean_cache(result="success" if success else "fail")
            reset = False
        result.update(
            status="success" if success else "fail",
            instruction=str(task.instruction),
            steps=int(task.step_count),
            save_count=int(task.save_count),
        )
        if success:
            path = Path(task.save_path)
            h = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                    h.update(block)
            result.update(path=str(path), sha256=h.hexdigest())
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}")
        (out / "error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        if task is not None and reset:
            try:
                task.clean_cache(result="error")
            except Exception:
                pass
    finally:
        result["wall_time_s"] = time.monotonic() - started
        write_json(out / "result.json", result)

        def finish():
            os._exit(0 if result["status"] == "success" else 1)

        guard = threading.Timer(20, finish)
        guard.daemon = True
        guard.start()
        try:
            if task is not None:
                task.close()
            if app is not None:
                app.close()
        finally:
            guard.cancel()
    raise SystemExit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
