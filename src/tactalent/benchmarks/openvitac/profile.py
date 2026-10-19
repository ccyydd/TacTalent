"""OpenViTac interfaces for official RoboICL; public task context and disclosed chip protocol."""

from pathlib import Path
import os

ROOT = Path(os.environ.get("ASTRA_OPENVITAC_ROOT", "/opt/tactalent/benchmarks/OpenViTac"))
TASKS = {
    "weight_classify": 300,
    "hardness_classify": 300,
    "roughness_classify": 300,
    "roughness_regrasp": 500,
    "can_empty_select": 400,
    "grasp_fragile_chip": 3600,
    "turn_gear_pair": 400,
    "pull_drawer": 400,
    "insert_USB": 300,
    "insert_block_v1": 300,
    "insert_block_v2": 300,
}
# User-reviewed evaluation budget; native training/official limits remain in TASKS.
EVALUATION_COMMAND_LIMITS = {
    "grasp_fragile_chip": 600,
    "hardness_classify": 100,
    "insert_block_v1": 200,
    "weight_classify": 200,
}

# Native chip task has no TASK_INSTRUCTION. Goal only, no action strategy.
INSTRUCTION_FALLBACKS = {
    "grasp_fragile_chip": "Grasp the fragile chip and place it in the target tray intact."
}
VARIANTS = ("cube", "half_cylinder", "hexagon")

# User-specified chunk lengths; independent of the native episode budget T.
HORIZONS = {task: 5 for task in TASKS}
HORIZONS.update(
    {
        "can_empty_select": 10,
        "grasp_fragile_chip": 10,
        "turn_gear_pair": 10,
        "insert_block_v1": 10,
        "insert_block_v2": 10,
    }
)


def binding(task, variant=None):
    if task not in TASKS:
        raise ValueError("Unknown OpenViTac task")
    if task == "insert_block_v2":
        if variant not in VARIANTS:
            raise ValueError("Insert Block v2 requires cube, half_cylinder or hexagon")
    elif variant is not None:
        raise ValueError("Variant is only supported for Insert Block v2")
    result = dict(
        benchmark="OpenViTac",
        task=task,
        native_task="insert_block" if task.startswith("insert_block_") else task,
        variant=variant,
        root=str(ROOT),
        train_camera_root=str(ROOT),
        cameras=["head", "wrist"],
        native_action_limit=TASKS[task],
        sensor="gsmini",
        task_config="grasp_fragile_chip" if task == "grasp_fragile_chip" else "gelsight",
        native_dt=1 / 120,
        native_actions_per_command=1,
    )
    if task in EVALUATION_COMMAND_LIMITS:
        result["evaluation_action_limit"] = EVALUATION_COMMAND_LIMITS[task]
    return result


def gripper_guidance(upstream_guidance):
    directive = "; keep 0 while holding"
    if upstream_guidance.count(directive) != 1:
        raise ValueError("Pinned RoboICL gripper guidance changed; review the binding")
    return upstream_guidance.replace(directive, "")


# Benchmark task specification reused from the prior branch. This does not
# import its policy prompts, retrieval, learned rules or controller settings.
TASK_CONTEXT_SOURCE = {
    "source_branch": "research/on-demand-retrieval",
    "source_commit": "5a33eca32b6cd767af99480f5c311da2cf88d52d",
    "path": "benchmark_profiles/openvitac.py",
    "fields": ["TASKS"],
}
TASK_CONTEXTS = {
    "weight_classify": {
        "background": "Objects may look similar but differ in weight. Classify by "
        "interaction and place on the instructed plate."
    },
    "hardness_classify": {
        "background": "Objects may look similar but differ in hardness. Classify by "
        "contact and place on the instructed plate."
    },
    "roughness_classify": {
        "background": "Objects may look similar but differ in surface roughness. "
        "Classify by contact and place on the instructed plate."
    },
    "roughness_regrasp": {"background": "Select the rough block and place it on the target plate."},
    "can_empty_select": {
        "background": "Identify the empty can among the filled cans and place it in " "the basket."
    },
    "grasp_fragile_chip": {
        "background": "Grasp and lift the fragile chip, keeping it intact, then transfer it "
        "to the target tray and place it stably inside."
    },
    "turn_gear_pair": {
        "background": "Assemble the gear and rotate it to drive the paired gear "
        "through engagement."
    },
    "pull_drawer": {"background": "Grasp the upper drawer handle and pull the drawer open."},
    "insert_USB": {
        "background": "Align the USB plug with its target socket and complete insertion."
    },
    "insert_block_v1": {"background": "Grasp the cube and insert it into its matching opening."},
    "insert_block_v2": {
        "background": "Select the language-specified object and insert it into its "
        "matching opening."
    },
}

# Sensor selection is an explicit benchmark interface; RoboICL control is unchanged.
DEFAULT_ICL_MODE = "vision_tactile"
TACTILE_FIELD = "rgb"
TACTILE_SIZE_WH = ((320, 240),)
VISION_IMAGE_GUIDANCE = (
    "Images contain {cameras}. Each camera retains its native raster and aspect ratio; "
    "no cropping or resizing. Head calibration uses the head image's own pixel coordinates. "
    "TRAIN and LIVE calibrations are labelled separately."
)
TACTILE_IMAGE_GUIDANCE = (
    "Each observation is one panel: top row {cameras}; bottom row left tactile RGB | right tactile RGB. "
    "Left/right identify the robot's sensors. Tactile RGB has no marker overlay. "
    "All sensor images retain their native pixels, orientation and aspect ratio, with padding only. "
    "The image_layout metadata gives each sensor's rectangle in the panel. "
    "Head calibration uses the head image's own pixel coordinates. "
    "TRAIN and LIVE calibrations are labelled separately."
)

# Class/decision-branch coverage; pose-only variation stays one-shot.
DEMONSTRATION_COVERAGE = {
    "weight_classify": ["light", "heavy"],
    "hardness_classify": ["soft", "hard"],
    "roughness_classify": ["smooth", "rough"],
    "roughness_regrasp": ["retain_initial", "regrasp"],
    "can_empty_select": ["empty_found", "filled_rejected"],
    "insert_block_v2": ["cube", "half_cylinder", "hexagon"],
}
ONE_SHOT_CHUNKS = 12
ONE_SHOT_LIVE_ANCHORS = 12
FEW_SHOT_CHUNKS = 5
FEW_SHOT_LIVE_ANCHORS = 5

# Reviewed full evaluation (2026-10-14); repository command groups plus explicit
# agreements for current two-can ECS, IB v1 and chip's default 100 episodes.
# The stale Coke/7up commands are incompatible with native CAN_BRANDS.
FULL_EVALUATION_CASES = []
for _task, _key, _values in (
    ("weight_classify", "weight_label", ("light", "heavy")),
    ("hardness_classify", "hardness_label", ("soft", "hard")),
    ("roughness_classify", "roughness_label", ("smooth", "rough")),
    ("roughness_regrasp", "rough_block_side", ("left", "right")),
    ("can_empty_select", "empty_can", ("fanta", "pepsi")),
):
    for _value in _values:
        FULL_EVALUATION_CASES.append(
            dict(
                id=_task + "_" + _value,
                task=_task,
                variant=None,
                conditions={_key: _value},
                episodes=50,
                start_seed=10000,
            )
        )
for _task in ("grasp_fragile_chip", "turn_gear_pair", "pull_drawer", "insert_USB"):
    FULL_EVALUATION_CASES.append(
        dict(id=_task, task=_task, variant=None, conditions={}, episodes=100, start_seed=10000)
    )
for _task, _shapes in (("insert_block_v1", ("cube",)), ("insert_block_v2", VARIANTS)):
    for _shape in _shapes:
        for _layout in ((0, 1, 4), (1, 3, 2)):
            FULL_EVALUATION_CASES.append(
                dict(
                    id=_task + "_" + _shape + "_layout_" + "".join(map(str, _layout)),
                    task=_task,
                    variant=_shape if _task == "insert_block_v2" else None,
                    conditions=dict(target_block=_shape, block_base_pose_indices=list(_layout)),
                    episodes=50,
                    start_seed=10000,
                )
            )
del _task, _key, _values, _value, _shapes, _shape, _layout

UPSTREAM_COMMIT = "a2d23ad8ae4351292bb6f51aad7dfdf481f074fe"
