"""Official selector and serializer with task-specific robot/camera data."""

import ast, hashlib, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .upstream import VENDOR, identity, protocol_sha256
from .benchmarks import (
    binding,
    validate_binding,
    horizon_for_task,
    validate_reference_horizon,
    task_context,
    task_prompt,
)
from roboicl.policy.anchored_cache import observation_end

from .observations import encoded_panel, resolve_mode, input_contract, image_names


def pair_url(images, cameras=None, icl_mode="vision", spec=None):
    return encoded_panel(images, tuple(cameras or images.keys()), icl_mode, spec)[0]


def observation(state, images, step, source, cameras=None, icl_mode="vision", spec=None):
    url, layout = encoded_panel(images, tuple(cameras or images.keys()), icl_mode, spec)
    return {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": json.dumps(
                    {
                        "source": source,
                        "observation_step": step,
                        "robot_state": state,
                        "icl_mode": icl_mode,
                        "image_layout": layout,
                    }
                ),
            },
            {"type": "input_image", "detail": "original", "image_url": url},
            observation_end(step),
        ],
    }


def actions_between(data, start, horizon):
    previous = np.array(data["observations"][start]["state"]["ee_pose"])
    rows = []
    for command in data["commands"][start : start + horizon]:
        target = np.array(command["ee_pose"])
        r = (
            Rotation.from_quat(target[3:][[1, 2, 3, 0]])
            * Rotation.from_quat(previous[3:][[1, 2, 3, 0]]).inv()
        )
        rows.append(
            [*(target[:3] - previous[:3]).tolist(), *r.as_rotvec().tolist(), command["gripper"]]
        )
        previous = target
    return np.round(rows, 6).tolist()


def select(data, horizon, chunk_count=12):
    source = (VENDOR / "roboicl/reference.py").read_text()
    tree = ast.parse(source)
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select_starts"
    )

    def valid(_data, start, h, translation, rotation):
        obs = data["observations"][start : start + h + 1]
        ticks = [o.get("control_step", start + i) for i, o in enumerate(obs)]
        if len(ticks) != h + 1 or not np.all(np.diff(ticks) == 1):
            return False  # Never label an omitted settling interval as a one-step action.
        rows = np.asarray(actions_between(data, start, h))
        return (
            len(rows) == h
            and np.isfinite(rows).all()
            and np.max(np.linalg.norm(rows[:, :3], axis=1)) <= translation + 1e-6
            and np.max(np.linalg.norm(rows[:, 3:6], axis=1)) <= rotation + 1e-6
            and np.all((rows[:, 6] >= 0) & (rows[:, 6] <= 1))
        )

    namespace = {"_valid_chunk": valid}
    exec(
        compile(
            ast.Module(body=[function], type_ignores=[]),
            str(VENDOR / "roboicl/reference.py"),
            "exec",
        ),
        namespace,
    )
    starts = namespace["select_starts"](
        {"action/left_ee_poses": data["observations"]}, horizon, chunk_count
    )
    return starts, hashlib.sha256(ast.get_source_segment(source, function).encode()).hexdigest()


def _prepare_episode(
    source, output=None, horizon=None, icl_mode=None, *, chunk_count=12, include_end_marker=True
):
    source = Path(source)
    data = json.loads(source.read_text())
    meta = data["metadata"]
    spec = validate_binding(meta["binding"] if "binding" in meta else binding())
    icl_mode = resolve_mode(spec, icl_mode)
    if meta.get("status") != "success" or (
        meta.get("seed") is not None and not 0 <= meta["seed"] < 1000000
    ):
        raise ValueError("Successful disjoint training demonstration required")
    if meta.get("seed") is None and (meta.get("split") != "train" or not meta.get("provenance")):
        raise ValueError("Unknown seed requires explicit published training-split provenance")
    if len(data["observations"]) != len(data["commands"]) + 1:
        raise ValueError("Unmatched action/results")
    if horizon is not None:
        validate_reference_horizon({"horizon": horizon}, spec)
    horizon = horizon_for_task(spec)
    try:
        starts, selector_sha = select(data, horizon, chunk_count)
    except ValueError as exc:
        raise ValueError(
            f"Configured H={horizon} for {spec['benchmark']}/{spec['task']} cannot form J={chunk_count} "
            "valid chunks from this demonstration. Choose another eligible training "
            "demonstration; do not shorten H, duplicate frames or cross missing intervals. "
            f"Selector: {exc}"
        ) from exc
    from types import FunctionType, SimpleNamespace
    from roboicl.policy import train_tool_reference, train_endpoint_observations
    from .policy import Model
    from .controller import tracking_receipt
    from .calibration import authored

    if meta.get("training_calibration", authored(spec, training=True)) != authored(
        spec, training=True
    ):
        raise ValueError(
            "Recorded training calibration changed; reconstruct from the training source"
        )
    # Filter a richer recording explicitly; missing requested sensors are errors.
    names = image_names(spec, icl_mode)
    for obs in data["observations"]:
        if any(name not in obs["images"] for name in names):
            raise ValueError(
                "Training recording lacks requested ICL sensors; reconstruct from source HDF5"
            )
        obs["images"] = {name: obs["images"][name] for name in names}
    examples = []
    for start in starts:
        before, after = data["observations"][start], data["observations"][start + horizon]
        frame, end = before.get("control_step", start), after.get("control_step", start + horizon)
        examples.append(
            {
                "frame": frame,
                "result_frame": end,
                "action_tensor": actions_between(data, start, horizon),
                "action_step_interval_inclusive": [frame, end - 1],
                "state": before["state"],
                "result_state": after["state"],
                "prompt_images": before["images"],
                "result_prompt_images": after["images"],
            }
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    terminal = data["observations"][-1]
    last_step = terminal.get("control_step", len(data["observations"]) - 1)
    demo = {
        "sha256": digest,
        "examples": examples,
        "total_frames": last_step + 1,
        "terminal_observation": {
            "frame": last_step,
            "state": terminal["state"],
            "prompt_images": terminal["images"],
        },
    }

    def camera_parts(step, images, cameras, **kw):
        url, layout = encoded_panel(images, spec["cameras"], icl_mode, spec)
        return [
            {
                "type": "input_text",
                "text": json.dumps({"icl_mode": icl_mode, "image_layout": layout}),
            },
            {"type": "input_image", "detail": "original", "image_url": url},
        ]

    original = train_tool_reference.tool_reference_messages
    bound = FunctionType(
        original.__code__,
        dict(
            original.__globals__,
            STATE_KEYS=("arm_joint_state", "ee_joint_state", "ee_pose"),
            tracking_receipt=tracking_receipt,
            camera_input_parts=camera_parts,
        ),
        argdefs=original.__defaults__,
    )
    bound.__kwdefaults__ = original.__kwdefaults__
    prior = train_endpoint_observations.result_observations
    train_endpoint_observations.result_observations = lambda demo, **kw: {
        e["result_frame"]: {"state": e["result_state"], "prompt_images": e["result_prompt_images"]}
        for e in demo["examples"]
    }
    converter = SimpleNamespace(
        dialogue_config=SimpleNamespace(
            predict_horizon=horizon, max_translation_m=0.03, max_rotation_rad=0.15
        )
    )
    try:
        messages = bound(
            demo,
            [],
            lambda rows, state: Model._convert_tensor(converter, rows, state),
            cameras=tuple(spec["cameras"]),
            image_profile="native_jpeg",
            image_detail="original",
            image_layout="triptych",
            include_result_observations=True,
            include_end_marker=include_end_marker,
        )
    finally:
        train_endpoint_observations.result_observations = prior
    result = {
        "protocol_sha256": protocol_sha256(spec),
        "binding": spec,
        "icl_mode": icl_mode,
        "input_contract": input_contract(spec, icl_mode),
        "training_calibration": meta.get("training_calibration", authored(spec, training=True)),
        "upstream": identity(),
        "source": str(source),
        "source_sha256": digest,
        "selector_sha256": selector_sha,
        "horizon": horizon,
        "starts": starts,
        "selected_control_steps": [e["frame"] for e in examples],
        "horizon_selection": f"User-specified per-task H; official J{chunk_count} selector with no H fallback",
        "horizon_rule_source": "tactalent/benchmarks/"
        + spec["benchmark"].lower()
        + "/profile.py:HORIZONS",
        "instruction": meta["instruction"],
        "task_context": task_context(spec),
        "task_prompt": task_prompt(spec, meta["instruction"]),
        "frequency": 1 / meta.get("control_dt", meta["native_dt"]),
        "action_repeat": meta.get("action_repeat", 1),
        "messages": messages,
        "state_action_semantics": meta.get("state_action_semantics"),
        "action_provenance": meta.get("action_provenance"),
    }
    if output is not None:
        Path(output).write_text(json.dumps(result))
    return result


def prepare(source, output, horizon=None, icl_mode=None):
    from .demonstrations import plan, descriptor, validate_reference_demonstrations

    data = json.loads(Path(source).read_text())
    meta = data["metadata"]
    spec = validate_binding(meta["binding"] if "binding" in meta else binding())
    if horizon is not None:
        validate_reference_horizon({"horizon": horizon}, spec)
    contract = plan(spec)
    if contract["shots"] != 1:
        raise ValueError("This task requires covered few-shot demonstrations; use prepare_few_shot")
    result = _prepare_episode(
        source, horizon=horizon, icl_mode=icl_mode, chunk_count=contract["chunks_per_demo"]
    )
    result.update(demonstration_contract=contract, demonstrations=[descriptor(source, result)])
    validate_reference_demonstrations(result, spec)
    Path(output).write_text(json.dumps(result))
    return result
