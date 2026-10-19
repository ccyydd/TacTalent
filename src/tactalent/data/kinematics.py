"""URDF kinematics and verified training-record schemas."""

from __future__ import annotations
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from tactalent.benchmarks.univtac.paths import urdf_path, asset_package

SPANS = (1, 6, 30)
GRIPPER_MAX_M = 0.039
FINGER_SYMMETRY_TOLERANCE_M = 0.0001
FINGER_RANGE_TOLERANCE_M = 0.000001
MEASURED_FINGER_MIN_M = 0.0
MEASURED_FINGER_MAX_M = 0.03999999910593033
ROBOT_FIELDS = ("step", "embodiment/ee", "embodiment/joint")
DEFAULT_URDF = urdf_path()
FK_POSITION_TOLERANCE_M = 0.002
FK_ROTATION_TOLERANCE_RAD = 0.01
QUATERNION_NORM_TOLERANCE = 0.001
ASSET_PACKAGE = asset_package()
MECHANICAL_EVIDENCE_FILES = {
    "data/Robots/Franka/GelSight_Mini/Gripper/uipc_gelpads_high_res_wrist.usd": "3047b1b7bfc982ad35222604b36069d30db431bd773b62fc91b11667b219a1d9",
    "data/Robots/Franka/GelSight_Mini/Gripper/panda_arm_hand/panda_arm_hand.usd": "b1c7d82d36286496d70e37ad5e33e9bc7ec9122fe8bbf7d31527717a895f3ee5",
    "robots/franka/franka_gsmini_gripper_uipc_high_res.py": "6cb0924ca39be15a32fe9e36938f845aa7bd46c9903bf254e74168c8095113fd",
}


def mechanical_limit_identity():
    """Pin the exact bytes used in the separate offline authored-USD audit."""
    sources = []
    for relative, expected in MECHANICAL_EVIDENCE_FILES.items():
        path = (ASSET_PACKAGE / relative).resolve(strict=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"Audited mechanical-limit asset changed: {path}")
        sources.append({"path": str(path), "sha256": digest})
    return {
        "audit_kind": "offline_usd_stage_authored_prismatic_limit_audit",
        "sources": sources,
        "stage_meters_per_unit": 1.0,
        "prim_paths": [
            "/panda/panda_hand/panda_finger_joint1",
            "/panda/panda_hand/panda_finger_joint2",
        ],
        "lower_limit_m": MEASURED_FINGER_MIN_M,
        "upper_limit_m": MEASURED_FINGER_MAX_M,
        "measured_range_tolerance_m": FINGER_RANGE_TOLERANCE_M,
        "command_max_m": GRIPPER_MAX_M,
        "data_clipped": False,
    }


def quaternion_matrices(q):
    w, x, y, z = np.asarray(q, float).T
    result = np.empty((len(w), 3, 3), dtype=float)
    result[:, 0, 0], result[:, 0, 1], result[:, 0, 2] = (
        1 - 2 * (y * y + z * z),
        2 * (x * y - z * w),
        2 * (x * z + y * w),
    )
    result[:, 1, 0], result[:, 1, 1], result[:, 1, 2] = (
        2 * (x * y + z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z - x * w),
    )
    result[:, 2, 0], result[:, 2, 1], result[:, 2, 2] = (
        2 * (x * z - y * w),
        2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    )
    return result


def axis_rotations(axis, angles):
    axis = np.asarray(axis, float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
        raise ValueError("invalid URDF joint axis")
    x, y, z = axis / np.linalg.norm(axis)
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], float)
    angles = np.asarray(angles, float)
    return (
        np.eye(3)[None]
        + np.sin(angles)[:, None, None] * skew
        + (1 - np.cos(angles))[:, None, None] * (skew @ skew)
    )


class ArmKinematics:
    """Offline URDF FK verifies base-relative measured panda_hand semantics."""

    def __init__(self, path=DEFAULT_URDF):
        self.path = Path(path).resolve(strict=True)
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        joints = ET.fromstring(raw).findall("joint")
        edges = {joint.find("child").get("link"): joint for joint in joints}
        chain, tip = [], "panda_hand"
        while tip != "panda_link0":
            if tip not in edges:
                raise ValueError("URDF does not connect panda_link0 to panda_hand")
            joint = edges[tip]
            chain.insert(0, joint)
            tip = joint.find("parent").get("link")
            if len(chain) > len(joints):
                raise ValueError("URDF chain contains a cycle")
        names = [j.get("name") for j in chain if j.get("type") != "fixed"]
        if names != [f"panda_joint{i}" for i in range(1, 8)]:
            raise ValueError("URDF arm chain does not match the recorded first seven joint columns")
        self.chain = []
        for joint in chain:
            origin = joint.find("origin")
            xyz = (
                np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                if origin is not None
                else np.zeros(3)
            )
            rpy = (
                np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                if origin is not None
                else np.zeros(3)
            )
            if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite(np.r_[xyz, rpy]).all():
                raise ValueError("invalid URDF origin")
            transform = np.eye(4)
            transform[:3, 3] = xyz
            transform[:3, :3] = (
                axis_rotations([0, 0, 1], [rpy[2]])[0]
                @ axis_rotations([0, 1, 0], [rpy[1]])[0]
                @ axis_rotations([1, 0, 0], [rpy[0]])[0]
            )
            kind, name = joint.get("type"), joint.get("name")
            if kind not in ("fixed", "revolute", "continuous"):
                raise ValueError("unsupported arm URDF joint type")
            axis = joint.find("axis")
            axis = (
                np.fromstring(axis.get("xyz", "1 0 0"), sep=" ")
                if axis is not None
                else np.array([1, 0, 0])
            )
            index = None if kind == "fixed" else names.index(name)
            self.chain.append((transform, index, axis))

    def forward(self, joint):
        joint = np.asarray(joint, float)
        result = np.broadcast_to(np.eye(4), (len(joint), 4, 4)).copy()
        for origin, index, axis in self.chain:
            result = result @ origin
            if index is not None:
                motion = np.broadcast_to(np.eye(4), result.shape).copy()
                motion[:, :3, :3] = axis_rotations(axis, joint[:, index])
                result = result @ motion
        return result

    def audit(self, ee, joint):
        predicted = self.forward(joint)
        position = np.linalg.norm(predicted[:, :3, 3] - ee[:, :3], axis=1)
        rotation = predicted[:, :3, :3].transpose(0, 2, 1) @ quaternion_matrices(ee[:, 3:])
        angles = np.arccos(np.clip((np.trace(rotation, axis1=1, axis2=2) - 1) / 2, -1, 1))
        result = {
            "fk_checked_frames": len(ee),
            "max_hand_position_error_m": float(position.max()),
            "max_hand_orientation_error_rad": float(angles.max()),
        }
        if (
            result["max_hand_position_error_m"] > FK_POSITION_TOLERANCE_M
            or result["max_hand_orientation_error_rad"] > FK_ROTATION_TOLERANCE_RAD
        ):
            raise ValueError(
                "joint FK differs from measured robot-base panda_hand pose: " + json.dumps(result)
            )
        return result

    def identity(self):
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "base_link": "panda_link0",
            "tip_link": "panda_hand",
            "arm_joint_columns": list(range(7)),
            "arm_joint_names": [f"panda_joint{i}" for i in range(1, 8)],
            "position_tolerance_m": FK_POSITION_TOLERANCE_M,
            "rotation_tolerance_rad": FK_ROTATION_TOLERANCE_RAD,
        }


def validate_episode_arrays(step, ee, joint):
    """Validate the recorded schema before deriving robot-only changes."""
    step, ee, joint = np.asarray(step), np.asarray(ee), np.asarray(joint)
    if step.ndim != 1 or step.dtype.kind not in "iu":
        raise ValueError("step must be a one-dimensional integer array")
    count = len(step)
    if count <= max(SPANS):
        raise ValueError("episode has no complete 30-saved-frame window")
    if np.any(step < 0) or not np.all(np.diff(step) == 2):
        raise ValueError("step must increase strictly by exactly 2 throughout")
    if ee.shape != (count, 7) or joint.shape != (count, 9):
        raise ValueError("expected measured ee (N,7) and joint (N,9)")
    if ee.dtype.kind not in "fiu" or joint.dtype.kind not in "fiu":
        raise ValueError("ee and joint must be numeric, excluding booleans")
    ee, joint = ee.astype(np.float64), joint.astype(np.float64)
    if not np.isfinite(ee).all() or not np.isfinite(joint).all():
        raise ValueError("robot fields contain NaN or infinity")
    norms = np.linalg.norm(ee[:, 3:], axis=1)
    if not np.isfinite(norms).all() or np.any(norms < 1e-12):
        raise ValueError("invalid or zero quaternion")
    if np.any(np.abs(norms - 1) > QUATERNION_NORM_TOLERANCE):
        raise ValueError("quaternion norm differs from unity by more than 1e-3")
    ee = ee.copy()
    ee[:, 3:] /= norms[:, None]
    fingers = joint[:, 7:9]
    if np.any(fingers < MEASURED_FINGER_MIN_M - FINGER_RANGE_TOLERANCE_M) or np.any(
        fingers > MEASURED_FINGER_MAX_M + FINGER_RANGE_TOLERANCE_M
    ):
        raise ValueError(
            "measured single-finger positions exceed the audited USD 0..0.04m mechanical range"
        )
    if np.any(np.abs(fingers[:, 0] - fingers[:, 1]) > FINGER_SYMMETRY_TOLERANCE_M):
        raise ValueError("finger positions are not symmetric within 0.1mm")
    return step, ee, joint


def robot_fields_sha256(arrays):
    """Hash only used numeric fields with unambiguous dtype/shape metadata."""
    digest = hashlib.sha256()
    for name in ROBOT_FIELDS:
        value = np.ascontiguousarray(arrays[name])
        metadata = json.dumps(
            {"name": name, "dtype": value.dtype.str, "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = value.tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def read_source_config(path):
    """Require an explicit, frozen source selection from the actual SFT YAML."""
    import yaml

    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError("source YAML requires a data object")
    data = config["data"]
    start, end = data.get("episode_start"), data.get("episode_end")
    if type(start) is not int or type(end) is not int or (start, end) != (0, 49):
        raise ValueError("source YAML must explicitly declare training episodes 0..49")
    root = data.get("train_data_paths")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError("source YAML requires one absolute train_data_paths root")
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("training source root must be a directory")
    selection = {
        "train_episode_ids": list(range(start, end + 1)),
        "source_config_path": str(path),
        "source_config_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return root, selection
