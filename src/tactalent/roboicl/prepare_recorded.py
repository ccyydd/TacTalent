"""Prepare official one-shot references from verified benchmark training recordings."""

import argparse, hashlib, json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from tactalent.benchmarks.openvitac.recordings import decode, ledger, validate_recorded_arrays, sha
from tactalent.data.kinematics import ArmKinematics
from tactalent.utils.storage import require_output_path
from .benchmarks import binding, validate_reference_horizon, profile
from .reference import prepare
from .observations import ICL_MODES, resolve_mode, input_contract, tactile_key
from .calibration import authored
from tactalent.benchmarks.univtac.paths import urdf_path


def training_source(spec, manifest=None, episode_id=0, episode_index=0, source_config=None):
    if spec["benchmark"] == "OpenViTac":
        if source_config is not None:
            raise ValueError("--source-config applies only to UniVTAC")
        if manifest is None:
            raise ValueError("OpenViTac requires an explicit successful training ledger")
        provenance = ledger(manifest, spec)
        row = provenance["episodes"][episode_index]
        return row["path"], dict(
            split="train",
            seed=row["seed"],
            status="success",
            instruction=row.get("instruction", provenance["instruction"]),
            provenance=provenance,
            source_ledger_sha256=sha(manifest),
            source_hdf5_sha256=row["sha256"],
        )
    from tactalent.benchmarks.univtac.sources import build
    from tactalent.benchmarks.cameras import task_instructions

    provenance = build(
        spec["task"],
        source_config=source_config,
        native_root=Path(spec["root"]),
        episode_id=episode_id,
    )
    row = provenance["source"]
    instructions, origin = task_instructions(Path(spec["root"]), spec["task"])
    return row["episode_path"], dict(
        split="train",
        seed=None,
        status="success",
        episode_id=episode_id,
        instruction=instructions[0],
        instruction_source=origin,
        seed_overlap_status=row["seed_overlap_status"],
        provenance=provenance,
        source_hdf5_sha256=row["episode_sha256"],
    )


def reconstruct_gripper_targets(joints):
    """Map measured fingers to the legal command interval, preserving raw values."""
    measured = np.asarray(joints)[:, 7:9].mean(axis=1) / 0.039
    if not np.isfinite(measured).all():
        raise ValueError("Nonfinite measured gripper state")
    return measured, np.clip(measured, 0.0, 1.0)


def convert(
    manifest,
    output,
    *,
    benchmark="OpenViTac",
    task="turn_gear_pair",
    variant=None,
    episode_id=0,
    horizon=None,
    icl_mode=None,
    episode_index=0,
    record_only=False,
    source_config=None,
):
    spec = binding(benchmark, task, variant)
    from .demonstrations import plan

    if not record_only and plan(spec)["shots"] > 1:
        raise ValueError(
            "Task requires covered few-shot sources; use tactalent.roboicl.prepare_few_shot"
        )
    icl_mode = resolve_mode(spec, icl_mode)
    if horizon is not None:
        validate_reference_horizon({"horizon": horizon}, spec)
    source_path, metadata = training_source(
        spec, manifest, episode_id, episode_index, source_config
    )
    output = require_output_path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with h5py.File(source_path) as f:
        steps, ee, joints = validate_recorded_arrays(
            f["step"][:], f["embodiment/ee"][:], f["embodiment/joint"][:]
        )
        robot = (
            urdf_path(spec["root"])
            if benchmark == "UniVTAC"
            else Path(spec["root"]) / "assets/embodiments/franka/panda.urdf"
        )
        fk = ArmKinematics(robot).audit(ee, joints)
        # Native first-frame warmup can leave one off-grid prefix frame.
        offset = 1 if len(steps) > 2 and steps[1] - steps[0] != 2 else 0
        steps, ee, joints = steps[offset:], ee[offset:], joints[offset:]
        if len(steps) < 3 or np.any((steps - steps[0]) % 2):
            raise ValueError("Training timestamps do not lie on the native 60 Hz recording grid")
        images = {
            name: decode(f["observation/" + name + "/rgb"])[offset:] for name in spec["cameras"]
        }
        if icl_mode == "vision_tactile":
            for side in ("left", "right"):
                key = tactile_key(f["tactile"], side)
                field = profile(spec["benchmark"]).TACTILE_FIELD
                images["tactile_" + side] = decode(f["tactile"][key][field])[offset:]
        if any(len(frames) != len(steps) for frames in images.values()):
            raise ValueError("Unsynchronized training images")
    camera = authored(spec, training=True)
    if tuple(images["head"].shape[1:]) != tuple(camera["shape"]):
        raise ValueError("Training camera raster differs from its source calibration")
    measured_gripper, estimated_target = reconstruct_gripper_targets(joints)
    # Measured contact/solver overshoot is not an issued target outside the
    # robot command range. Only reconstructed targets are bounded; preserve
    # every raw measurement and report the projection. LIVE actions are untouched.
    observations = []
    for i in range(len(steps)):
        paths = {}
        for name, frames in images.items():
            p = output / f"{i:04d}_{name}.jpg"
            Image.fromarray(frames[i]).save(p, quality=90, subsampling=0)
            paths[name] = str(p)
        state = {
            "ee_pose": ee[i].tolist(),
            "arm_joint_state": joints[i, :7].tolist(),
            "ee_joint_state": [float(estimated_target[i])],
        }
        observations.append(
            {
                "state": state,
                "images": paths,
                "source_frame": i + offset,
                "measured_gripper_open_fraction": float(measured_gripper[i]),
                "source_sim_step": int(steps[i]),
                "control_step": int((steps[i] - steps[0]) // 2),
            }
        )
    commands = [
        {"mode": "ee", "ee_pose": o["state"]["ee_pose"], "gripper": o["state"]["ee_joint_state"][0]}
        for o in observations[1:]
    ]
    metadata.update(
        icl_mode=icl_mode,
        input_contract=input_contract(spec, icl_mode),
        binding=spec,
        native_dt=1 / 120,
        control_dt=2 / 120,
        action_repeat=2,
        source_hdf5=str(source_path),
        omitted_initial_frames=offset,
        gripper_target_reconstruction={
            "method": "nearest legal normalized target in [0,1]",
            "projected_frames": int(np.count_nonzero(measured_gripper != estimated_target)),
            "max_measured_overshoot_m": float(
                np.max(np.abs(measured_gripper - estimated_target)) * 0.039
            ),
            "original_measurements_preserved": True,
            "live_action_clipping": False,
        },
        first_source_sim_step=int(steps[0]),
        last_source_sim_step=int(steps[-1]),
        missing_recorded_intervals=int(np.sum(np.diff(steps) // 2 - 1)),
        action_provenance="Approximate next measured-state targets, not original expert command logs; force=True, 60 Hz. Mean measured finger joints / 0.039 m, projected to the legal command interval [0,1]; raw measurements retained.",
        state_action_semantics="panda_hand world xyz + wxyz; world rotation-vector increments; normalized position-controlled gripper.",
        frame_note="Authored Franka base coincides with environment origin; FK and runtime checks enforce this.",
        training_calibration=camera,
        fk_audit=fk,
    )
    source = output / "recording.json"
    source.write_text(
        json.dumps({"metadata": metadata, "observations": observations, "commands": commands})
    )
    if record_only:
        return source
    result = prepare(source, output / "reference.json", horizon=horizon, icl_mode=icl_mode)
    (output / "audit.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "selected_starts": result["starts"],
                "selected_control_steps": result["selected_control_steps"],
                "horizon": result["horizon"],
                "horizon_selection": result["horizon_selection"],
                "observations": len(observations),
                "commands": len(commands),
            },
            indent=2,
        )
    )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["UniVTAC", "OpenViTac"], default="OpenViTac")
    p.add_argument("--task", default="turn_gear_pair")
    p.add_argument("--variant")
    p.add_argument("--training-ledger")
    p.add_argument("--training-episode", type=int, default=0)
    p.add_argument(
        "--source-config",
        type=Path,
        help="UniVTAC training-source YAML; defaults to ASTRA_UNIVTAC_SOURCE_CONFIG",
    )
    p.add_argument(
        "--horizon", type=int, help="Optional assertion; must match the configured per-task H"
    )
    p.add_argument("--icl-mode", choices=ICL_MODES, help="Default: benchmark vision_tactile")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = convert(
        a.training_ledger,
        a.output,
        benchmark=a.benchmark,
        task=a.task,
        variant=a.variant,
        episode_id=a.training_episode,
        horizon=a.horizon,
        icl_mode=a.icl_mode,
        source_config=a.source_config,
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("binding", "source", "frequency", "action_repeat", "starts", "horizon")
            }
        )
    )


if __name__ == "__main__":
    main()
