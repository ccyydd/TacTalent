"""Verified UniVTAC training records and their native collector provenance."""

from __future__ import annotations
import json
from pathlib import Path
from .provenance import collector_evidence
from .profile import TASKS
from .paths import (
    native_root as configured_native_root,
    source_config as configured_source_config,
    urdf_path,
)
from tactalent.data.kinematics import (
    ROBOT_FIELDS,
    ArmKinematics,
    read_source_config,
    robot_fields_sha256,
    validate_episode_arrays,
)
from tactalent.utils.files import file_sha256 as _file_sha256, sha
from tactalent.utils.storage import require_output_path

MODE = "univtac_fixed_training_source_v1"


def build(
    task,
    source_config=None,
    native_root=None,
    *,
    kinematics=None,
    episode_id=0,
):
    import h5py

    if task not in TASKS:
        raise ValueError("Unknown UniVTAC task")
    source_config = configured_source_config() if source_config is None else Path(source_config)
    native_root = configured_native_root() if native_root is None else Path(native_root)
    root, selection = read_source_config(Path(source_config))
    if type(episode_id) is not int or episode_id not in range(50):
        raise ValueError("Select an episode from the declared training split")
    relative = f"{task}/clean/{episode_id}.hdf5"
    path = require_output_path(root / relative).resolve(strict=True)
    before = path.stat()
    digest = _file_sha256(path)
    manifest_path = root / "download_manifest.json"
    raw_manifest = manifest_path.read_bytes()
    entry = json.loads(raw_manifest)["files"][relative]
    if (
        entry.get("verified_sha256") != digest
        or entry.get("upstream_sha256") != digest
        or entry.get("size") != before.st_size
    ):
        raise ValueError("Fixed training source differs from download manifest")
    with h5py.File(path, "r") as file:
        if any(str(k).lower() in {"seed", "random_seed", "rng_seed"} for k in file.attrs):
            raise ValueError("Source seed metadata requires explicit handling")
        arrays = {key: file[key][:] for key in ROBOT_FIELDS}
        step, ee, joint = validate_episode_arrays(
            arrays["step"], arrays["embodiment/ee"], arrays["embodiment/joint"]
        )
        kinematics = ArmKinematics(urdf_path(native_root)) if kinematics is None else kinematics
        fk_audit = kinematics.audit(ee, joint)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Training file changed during provenance audit")
    return {
        "schema_version": MODE,
        "task": task,
        "source": {
            "task": task,
            "split": "train",
            "episode_id": episode_id,
            "episode_path": str(path),
            "episode_sha256": digest,
            "source_seed": None,
            "seed_overlap_status": "unavailable_source_seed_metadata",
            "selection": (
                "fixed_train_index_0_before_evaluation"
                if episode_id == 0
                else "declared_training_index_before_evaluation"
            ),
            "successful_source": True,
            "success_evidence": collector_evidence(native_root),
            "training_split_proof": selection,
            "download_manifest_path": str(manifest_path),
            "download_manifest_sha256": sha(raw_manifest),
            "robot_fields_sha256": robot_fields_sha256(arrays),
            "frame_count": len(step),
            "first_saved_frame_timing": "First recorded post-pre_move state; not exact reset.",
            "camera_calibration_available": False,
            "historical_timestamps_available": False,
            "source_commands_recorded": False,
        },
        "robot_audit": fk_audit,
        "offline_kinematics": kinematics.identity(),
        "builder_source_sha256": _file_sha256(Path(__file__)),
    }
