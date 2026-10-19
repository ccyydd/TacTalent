"""Task-specific fixed head-camera calibration, verified against the live sensor."""

import ast
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .benchmarks import binding


def authored(spec=None, *, training=False):
    spec = spec or binding()
    root = Path(spec["train_camera_root"] if training else spec["root"])
    selected = None
    for filename, classname in (
        (spec["native_task"] + ".py", "TaskCfg"),
        ("_base_task.py", "BaseTaskCfg"),
    ):
        path = root / "envs" / filename
        tree = ast.parse(path.read_text())
        cfg = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == classname)
        for n in cfg.body:
            value = None
            if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "cameras" for t in n.targets
            ):
                value = n.value
            if (
                isinstance(n, ast.AnnAssign)
                and isinstance(n.target, ast.Name)
                and n.target.id == "cameras"
            ):
                value = n.value
            if value is not None:
                selected = (path, value)
                break
        if selected:
            break
    if selected is None:
        raise ValueError("No authored camera declaration")
    path, cameras = selected
    calls = [{v.arg: v.value for v in n.keywords} for n in cameras.elts]
    (kw,) = [kw for kw in calls if ast.literal_eval(kw["name"]) == "head"]
    offset = {v.arg: ast.literal_eval(v.value) for v in kw["offset"].keywords}
    optics = {v.arg: ast.literal_eval(v.value) for v in kw["spawn"].keywords}
    w, h = ast.literal_eval(kw["width"]), ast.literal_eval(kw["height"])
    if offset["convention"] != "opengl":
        raise ValueError("Unexpected camera convention")
    f = w * optics["focal_length"] / optics["horizontal_aperture"]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.array(offset["rot"])[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = offset["pos"]
    return {
        "intrinsic_matrix": [[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]],
        "camera_to_environment": T.tolist(),
        "shape": [h, w, 3],
        "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def verify(task, spec=None):
    from tactalent.benchmarks.cameras import current_camera_geometry
    from tactalent.robot.state import _array

    spec = spec or binding()
    cam = task._camera_manager.cameras["head"]
    geometry = current_camera_geometry(cam, task)
    R = Rotation.from_quat(np.array(geometry["quat_w_ros"])[[1, 2, 3, 0]]).as_matrix() @ np.diag(
        [1, -1, -1]
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = geometry["pos_w"]
    K = _array(cam.data.intrinsic_matrices)[0]
    fixed = authored(spec)
    return verified_geometry(fixed, T, K)


def verified_geometry(fixed, T, K):
    """Check the static mount, then use the actual stage/intrinsic calibration.

    USD's authored quaternion roundtrip can differ slightly from a normalized
    source quaternion. This permits <0.001 rad (0.057 degrees), not camera motion.
    """
    expected = np.asarray(fixed["camera_to_environment"])
    position_error = float(np.linalg.norm(T[:3, 3] - expected[:3, 3]))
    angle_error = float(
        (Rotation.from_matrix(T[:3, :3]) * Rotation.from_matrix(expected[:3, :3]).inv()).magnitude()
    )
    if (
        position_error > 1e-5
        or angle_error > 0.001
        or not np.allclose(K, fixed["intrinsic_matrix"], atol=1e-3)
    ):
        raise ValueError(
            "Authored and actual LIVE head calibration differ: "
            + str(
                dict(
                    position_error_m=position_error,
                    angle_error_rad=angle_error,
                    actual_K=K.tolist(),
                    expected_K=fixed["intrinsic_matrix"],
                )
            )
        )
    return dict(
        fixed,
        intrinsic_matrix=K.tolist(),
        camera_to_environment=T.tolist(),
        pose_source="verified_native_stage",
        verification=dict(position_error_m=position_error, angle_error_rad=angle_error),
    )


def message(calibration, source):
    return {
        "type": "input_text",
        "text": __import__("json").dumps(
            {
                "head_camera_calibration": {
                    "source": source,
                    "camera": "head",
                    "reference_step": 0,
                    "image_shape": calibration["shape"],
                    "intrinsic_matrix": calibration["intrinsic_matrix"],
                    "camera_to_environment": calibration["camera_to_environment"],
                }
            }
        ),
    }
