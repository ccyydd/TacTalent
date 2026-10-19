"""Native sensor frames, authored instructions and measured camera mounts."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

TASK_INSTRUCTIONS = {
    "grasp_classify": (
        "Determine whether the grasped object is rough or smooth. Move the rough "
        "object onto the orange pad, or the smooth object onto the green pad."
    ),
    "insert_HDMI": "insert the HDMI to the fixed slot.",
    "insert_hole": "insert the stick to the hole.",
    "insert_tube": "insert the tube to the fixed slot.",
    "lift_can": "Grasp the can and set it upright on the support surface.",
    "lift_bottle": ("Grasp the bottle and place it upright on the support surface near the wall."),
    "pull_out_key": "pull out the key.",
    "put_bottle_in_shelf": (
        "Place the bottle inside the shelf cavity, with the bottle upright at the final position."
    ),
}


def task_instructions(univtac_dir: Path, task_name: str) -> tuple[list[str], str]:
    """Use installed seen instructions, or the audited static goal fallback."""
    path = univtac_dir / "instructions" / f"{task_name}.json"
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        candidates = value.get("seen", [])
        if (
            isinstance(candidates, list)
            and candidates
            and all(isinstance(item, str) and item.strip() for item in candidates)
        ):
            return candidates, str(path)
        raise ValueError(f"No valid seen task instructions in {path}")
    return [TASK_INSTRUCTIONS[task_name]], "audited_static_benchmark_goal"


def _array(value: Any) -> Any:
    """Detach a sensor tensor without importing the simulation stack."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def _plain(value: Any) -> Any:
    """Convert measured calibration arrays to JSON-compatible values."""
    value = _array(value)
    return value.tolist() if hasattr(value, "tolist") else value


def _rgb(value: Any, name: str) -> Any:
    """Normalize a measured RGB sensor frame before saving and inference."""
    import numpy as np

    image = np.asarray(_array(value))
    if image.ndim != 3 or image.shape[-1] != 3 or not all(image.shape[:2]):
        raise ValueError(f"Sensor {name} did not supply a nonempty HWC RGB frame")
    if image.dtype == np.uint8:
        return image
    if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
        raise ValueError(f"Sensor {name} RGB values must be finite numbers")
    if image.min() < 0 or image.max() > 255:
        raise ValueError(f"Sensor {name} RGB values are outside [0, 255]")
    scale = 255.0 if np.issubdtype(image.dtype, np.floating) and image.max() <= 1 else 1.0
    return np.rint(image * scale).astype(np.uint8)


def configure_camera_geometry(task_config: Any, cameras) -> dict:
    """Preserve native optics; obtain moving camera poses from measured bodies."""
    requested = set(cameras)
    found = {}
    for camera in task_config.cameras:
        if camera.name in requested:
            # Both cached camera.data and this build's Fabric XForm view can keep
            # the initialization pose. USD refresh every tick is also expensive.
            camera.update_latest_camera_pose = False
            found[camera.name] = "rigid_attachment_at_observation"
    if set(found) != requested:
        raise ValueError(
            "Allowed camera configuration is missing: " + str(sorted(requested - set(found)))
        )
    return found


def _pose_profile(transform):
    import numpy as np
    from scipy.spatial.transform import Rotation

    rotation = np.asarray(transform)[:3, :3]
    world_axes = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])

    def quat(matrix):
        return Rotation.from_matrix(matrix).as_quat()[[3, 0, 1, 2]].tolist()

    return {
        "pos_w": np.asarray(transform)[:3, 3].tolist(),
        "quat_w_ros": quat(rotation @ np.diag([1, -1, -1])),
        "quat_w_world": quat(rotation @ world_axes),
    }


def current_camera_geometry(camera: Any, task: Any) -> dict:
    """Compose the authored camera mount with its measured PhysX body pose.

    Camera child transforms are calibration, not episode object state. Resolve
    the mount once; read the actual articulation link pose each observation.
    Static scene cameras retain their authored world transform.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    robot = task._robot_manager.robot
    mount = getattr(camera, "_astra_mount_geometry", None)
    if mount is None:
        from pxr import UsdGeom, UsdPhysics

        prim = camera._sensor_prims[0].GetPrim()
        transform = np.eye(4)
        body_id = None
        while prim and not prim.IsPseudoRoot():
            if prim.GetName() in robot.body_names and "/Robot/" in str(prim.GetPath()):
                body_id = robot.body_names.index(prim.GetName())
                break
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError("Camera is attached to an unsupported moving body")
            xform = UsdGeom.Xformable(prim)
            if xform:
                if xform.GetResetXformStack():
                    raise ValueError("Camera mount resets the transform stack")
                # USD uses row vectors; convert to column-vector transforms.
                transform = np.array(xform.GetLocalTransformation()).T @ transform
            prim = prim.GetParent()
        mount = (body_id, transform)
        camera._astra_mount_geometry = mount
    body_id, transform = mount
    if body_id is not None:
        pose = np.eye(4)
        q = np.asarray(_plain(robot.data.body_quat_w[0, body_id]))
        pose[:3, :3] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
        pose[:3, 3] = _plain(robot.data.body_pos_w[0, body_id])
        transform = pose @ transform
    profile = _pose_profile(transform)
    profile["pose_update_mode"] = (
        "measured_rigid_body" if body_id is not None else "static_stage_transform"
    )
    return profile
