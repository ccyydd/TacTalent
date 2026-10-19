"""UniVTAC interfaces for official RoboICL; public task context, no scoring override."""

from pathlib import Path
import json
import os

ROOT = Path(os.environ.get("ASTRA_UNIVTAC_ROOT", "/opt/tactalent/benchmarks/UniVTAC"))
TRAIN_CAMERA_ROOT = Path(os.environ.get("ASTRA_UNIVTAC_TRAIN_CAMERA_ROOT", str(ROOT)))
TASKS = {
    "lift_bottle": 500,
    "pull_out_key": 300,
    "lift_can": 300,
    "put_bottle_in_shelf": 300,
    "insert_hole": 300,
    "insert_HDMI": 600,
    "insert_tube": 300,
    "grasp_classify": 300,
}

# User-reviewed evaluation budgets; native training/official limits remain in TASKS.
EVALUATION_COMMAND_LIMITS = {"insert_HDMI": 300, "pull_out_key": 200}

# User-specified chunk lengths; independent of the native episode budget T.
HORIZONS = {task: 5 for task in TASKS}
HORIZONS.update({"lift_bottle": 10, "put_bottle_in_shelf": 10, "grasp_classify": 4})


def binding(task, variant=None):
    if task not in TASKS or variant is not None:
        raise ValueError("Unknown UniVTAC task or unsupported variant")
    camera_type = json.loads((ROOT / "policy/task_settings.json").read_text())[task]["camera_type"]
    if camera_type not in ("head", "all"):
        raise ValueError("Unreviewed native camera selection")
    result = dict(
        benchmark="UniVTAC",
        task=task,
        native_task=task,
        variant=None,
        root=str(ROOT),
        train_camera_root=str(TRAIN_CAMERA_ROOT),
        cameras=["head", "wrist"] if camera_type == "all" else ["head"],
        native_action_limit=TASKS[task],
        sensor="gsmini",
        task_config="demo",
        native_dt=1 / 120,
        native_actions_per_command=1,
    )
    if task in EVALUATION_COMMAND_LIMITS:
        result["evaluation_action_limit"] = EVALUATION_COMMAND_LIMITS[task]
    return result


def gripper_guidance(text):
    directive = "; keep 0 while holding"
    if text.count(directive) != 1:
        raise ValueError("Pinned RoboICL gripper guidance changed")
    return text.replace(directive, "")


# Benchmark task specification reused from the prior branch. This does not
# import its policy prompts, retrieval, learned rules or controller settings.
TASK_CONTEXT_SOURCE = {
    "source_branch": "research/on-demand-retrieval",
    "source_commit": "5a33eca32b6cd767af99480f5c311da2cf88d52d",
    "path": "benchmark_profiles/univtac.py",
    "fields": ["INITIAL_CONTEXT", "TASK_INSTRUCTIONS", "BACKGROUNDS"],
}
TASK_CONTEXTS = {
    "grasp_classify": {
        "goal": "Determine whether the grasped object is rough or smooth. Move the "
        "rough object onto the orange pad, or the smooth object onto the green "
        "pad.",
        "background": "Visually similar cylinders differ in surface texture. Identify "
        "the texture through contact and place the object in its class "
        "region. Rough goes to orange; smooth goes to green.",
        "starting_condition": "The native setup already grasps and lifts the object.",
    },
    "insert_HDMI": {
        "goal": "insert the HDMI to the fixed slot.",
        "background": "The held HDMI plug has a random rotational offset. Align it with "
        "the fixed socket and insert it.",
        "starting_condition": "The native setup already grasps the connector near the "
        "fixed slot.",
    },
    "insert_hole": {
        "goal": "insert the stick to the hole.",
        "background": "A block contains a slanted hole with two possible orientations, 60 "
        "or 120 degrees. Infer its direction through contact and insert the "
        "tube.",
        "starting_condition": "The native setup already grasps the stick near the hole "
        "entrance.",
    },
    "insert_tube": {
        "goal": "insert the tube to the fixed slot.",
        "background": "Insert a 2.0 cm tube into a 2.05 cm hole on an inclined surface, "
        "starting near its centre.",
        "starting_condition": "The native setup already grasps the tube near the slot " "entrance.",
    },
    "lift_can": {
        "goal": "Grasp the can and set it upright on the support surface.",
        "background": "Lift the horizontal can without slipping; its diameter varies between "
        "4, 5 and 6 cm. For this installation, finish with the can upright on "
        "the support surface.",
        "starting_condition": "The native setup places the open gripper near the can.",
    },
    "lift_bottle": {
        "goal": "Grasp the bottle and place it upright on the support surface near the " "wall.",
        "background": "A bottle starts beside a wall. Raise it vertically, keeping its "
        "final base within 5 cm of the wall. For this installation, finish "
        "with the bottle upright on the support surface near the wall.",
        "starting_condition": "The native setup places the open gripper near the bottle.",
    },
    "pull_out_key": {
        "goal": "pull out the key.",
        "background": "The key starts at a random angle inside the slot. Turn to "
        "mechanical resistance, then extract straight. The required "
        "turning angle therefore varies between episodes.",
        "starting_condition": "The native setup already grasps the key while it is in " "the lock.",
    },
    "put_bottle_in_shelf": {
        "goal": "Place the bottle inside the shelf cavity, with the bottle "
        "upright at the final position.",
        "background": "Move the initially upright bottle into the shelf cavity. "
        "Finish with the bottle upright inside the shelf cavity.",
        "starting_condition": "The native setup already grasps the bottle.",
    },
}

# Sensor selection is an explicit benchmark interface; RoboICL control is unchanged.
DEFAULT_ICL_MODE = "vision_tactile"
TACTILE_FIELD = "rgb"
TACTILE_SIZE_WH = ((320, 240), (400, 300))
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
DEMONSTRATION_COVERAGE = {"grasp_classify": ["rough", "smooth"]}
ONE_SHOT_CHUNKS = 12
ONE_SHOT_LIVE_ANCHORS = 12
FEW_SHOT_CHUNKS = 5
FEW_SHOT_LIVE_ANCHORS = 5

# Full-evaluation episodes, independent of demonstrations and control horizon.
FULL_EVALUATION_CASES = [
    dict(id=task, task=task, variant=None, conditions={}, episodes=100, start_seed=1000000)
    for task in TASKS
]
