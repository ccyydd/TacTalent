"""Native task construction, observation and human-review rendering for both benches."""

import importlib
from pathlib import Path
import numpy as np
from .benchmarks import binding


def presentation_task_class(native):
    class ReviewTask(native):
        def get_frame_shot(self, obs):
            from tactalent.benchmarks.cameras import _rgb

            pictures = [_rgb(obs["observation"][n]["rgb"], n) for n in ("head", "wrist")]
            for side in ("left", "right"):
                key = next(k for k in obs["tactile"] if k.startswith(side))
                pictures.append(_rgb(obs["tactile"][key]["rgb"], side))
            w, h = self.cfg.video_size
            canvas = np.zeros((h, w, 3), np.uint8)
            top = max(p.shape[0] for p in pictures[:2])
            for row, y in ((pictures[:2], 0), (pictures[2:], top)):
                x = (w - sum(p.shape[1] for p in row)) // 2
                for p in row:
                    ph, pw = p.shape[:2]
                    if x < 0 or y + ph > h:
                        raise ValueError("Review panel does not fit native images")
                    canvas[y : y + ph, x : x + pw] = p
                    x += pw
            import torch

            return torch.from_numpy(canvas)

    ReviewTask.__name__ = native.__name__
    ReviewTask.__module__ = native.__module__
    return ReviewTask


def configure_evaluation_budget(cfg, spec, mode):
    """Apply reviewed evaluation caps without changing training or control cadence."""
    if mode != "eval" or "evaluation_action_limit" not in spec:
        return
    maximum = spec["evaluation_action_limit"]
    if type(maximum) is not int or maximum <= 0:
        raise ValueError("Evaluation command budget must be a positive integer")
    if spec["benchmark"] == "OpenViTac" and spec["task"] == "grasp_fragile_chip":
        from .chip_evaluation import configure_evaluation_budget as configure_chip_budget

        configure_chip_budget(cfg, spec, mode)
    else:
        cfg.step_lim = maximum


def create(output, seed, mode, kit_args="", spec=None, evaluation_case=None):
    spec = spec or binding()
    from isaaclab.app import AppLauncher

    app = AppLauncher(
        dict(headless=True, enable_cameras=True, livestream=2, device="cuda:0", kit_args=kit_args)
    ).app
    task = None
    try:
        import yaml

        if spec["benchmark"] == "OpenViTac":
            from functools import partial
            import tacex.simulation_approaches.gpu_taxim.taxim_sim as optical
            from tacex.simulation_approaches.gpu_taxim.sim.taxim import Taxim

            optical.Taxim = partial(Taxim, backend="torch")
        module = importlib.import_module("envs." + spec["native_task"])
        origin = Path(module.__file__).resolve()
        if origin != Path(spec["root"]) / "envs" / (spec["native_task"] + ".py"):
            raise ValueError("Wrong benchmark envs package imported")
        cfg = module.TaskCfg()
        settings_path = Path(spec["root"]) / "task_config" / (spec["task_config"] + ".yml")
        settings = yaml.safe_load(settings_path.read_text())
        from .native_settings import apply_settings

        apply_settings(cfg, settings, benchmark=spec["benchmark"], mode=mode)
        if spec["benchmark"] == "OpenViTac":
            from isaaclab.sim import RenderCfg

            cfg.sim.render = RenderCfg(
                rendering_mode="quality", enable_dl_denoiser=True, antialiasing_mode="FXAA"
            )
        cfg.sim.device = "cuda:0"
        cfg.scene.num_envs = 1
        cfg.sim.enable_scene_query_support = True
        cfg.save_dir = Path(output) / "native"
        cfg.tactile_sensor_type = spec["sensor"]
        cfg.obs_data_type = {"camera": ["rgb"], "tactile": ["rgb"], "embodiment": ["joint", "ee"]}
        cfg.tactile_video_key = "rgb"
        head = next(c for c in cfg.cameras if c.name == "head")
        cfg.video_size = (
            max(head.width * 2, 800),
            head.height + (300 if spec["benchmark"] == "UniVTAC" else 240),
        )
        if spec["task"] == "insert_block_v1":
            cfg.target_block = "cube"
        if spec["task"] == "insert_block_v2":
            cfg.target_block = spec["variant"]
        if spec["benchmark"] == "OpenViTac" and spec["task"] == "grasp_fragile_chip":
            from envs._force_task_utils import prepare_force_task_config

            prepare_force_task_config(cfg, settings, settings_path, seed=seed)
        from .evaluation import apply_case

        if evaluation_case is not None and mode != "eval":
            raise ValueError("Evaluation cases cannot be applied to training collection")
        reviewed_case = apply_case(cfg, spec, evaluation_case)
        from tactalent.benchmarks.cameras import configure_camera_geometry

        configure_camera_geometry(cfg, spec["cameras"])
        cls = presentation_task_class(module.Task)
        if spec["benchmark"] == "OpenViTac" and spec["task"] == "grasp_fragile_chip":
            from .openvitac_chip import task_class

            cls = task_class(cls)
            if mode == "eval":
                from .chip_evaluation import evaluation_task_class

                cls = evaluation_task_class(cls, module.ChipPolicyMonitor)
        if spec["benchmark"] == "OpenViTac":
            from tactalent.benchmarks.openvitac.scene_seed import seeded_task_class

            cls = seeded_task_class(cls, seed)
        task = cls(cfg, mode=mode)
        # Native constructors set their own budgets; apply reviewed eval caps afterwards.
        configure_evaluation_budget(task.cfg, spec, mode)
        task._roboicl_binding = spec
        task._roboicl_evaluation_case = reviewed_case
        if spec["benchmark"] == "UniVTAC":
            from tactalent.benchmarks.cameras import task_instructions

            instructions, origin = task_instructions(Path(spec["root"]), spec["task"])
            task.reset(seed=seed, instructions=instructions)
        else:
            task.reset(seed=seed)
        expected_limit = (
            spec.get("evaluation_action_limit", spec["native_action_limit"])
            if mode == "eval"
            else spec["native_action_limit"]
        )
        if int(task.cfg.step_lim) != expected_limit:
            raise ValueError("Native action limit changed; review task binding")
        if abs(float(task.cfg.sim.dt) * task.cfg.decimation - spec["native_dt"]) > 1e-9:
            raise ValueError("Native physical cadence changed")
        if not str(task.instruction).strip():
            from .benchmarks import profile

            fallback = getattr(profile(spec["benchmark"]), "INSTRUCTION_FALLBACKS", {}).get(
                spec["task"]
            )
            if not fallback:
                raise ValueError(
                    "Native task instruction is empty and no declared goal fallback exists"
                )
            task.set_instruction(fallback)
            task.metadata["roboicl_instruction_source"] = "benchmark_profile_goal_fallback"
        return app, task
    except Exception as exc:
        # Kit close may exit the interpreter. Let the runner record the original
        # initialization error before it closes these partially built resources.
        exc._roboicl_app = app
        exc._roboicl_task = task
        raise


def read(task, spec=None, icl_mode=None):
    from tactalent.robot.state import RobotObserver
    from tactalent.robot.transforms import compose

    spec = spec or getattr(task, "_roboicl_binding", binding())
    state = RobotObserver({}).observe_state(task)
    root = state["root_pose_w"]
    if not np.allclose(root["position"], [0, 0, 0], atol=1e-6) or not np.allclose(
        root["quaternion_wxyz"], [1, 0, 0, 0], atol=1e-6
    ):
        raise ValueError("Training assumes the authored base/world alignment")
    p, q = compose(
        root["position"],
        root["quaternion_wxyz"],
        state["hand_pose"]["position"],
        state["hand_pose"]["quaternion_wxyz"],
    )
    m = task._robot_manager
    target = (
        float(m.robot.data.joint_pos_target[0, m._gripper_ids[0]].detach().cpu())
        / m.gripper_max_qpos
    )
    proprio = {
        "arm_joint_state": state["qpos"][:7],
        "ee_joint_state": [target],
        "ee_pose": [*p.tolist(), *q.tolist()],
    }
    obs = task._get_observations()
    from .observations import native_images

    images = native_images(obs, spec, icl_mode)
    return proprio, images
