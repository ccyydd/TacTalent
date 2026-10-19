"""Measured Panda state and native forward-kinematics validation."""

from __future__ import annotations
from types import SimpleNamespace
from typing import Any
import numpy as np
from .transforms import compose, quaternion, vector, orientation_error


def _array(value: Any) -> np.ndarray:
    """Read arrays or tensors without importing their runtime."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float)


class RobotObserver:
    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)

    def _manager(self, task: Any) -> Any:
        return task._robot_manager

    def _joint_layout(self, task: Any) -> tuple[list[str], list[int], int]:
        manager = self._manager(task)
        names = list(manager.robot.joint_names)
        arm = list(getattr(manager, "_arm_joint_names", [f"panda_joint{i}" for i in range(1, 8)]))
        fingers = list(
            getattr(
                manager,
                "_gripper_joint_names",
                ["panda_finger_joint1", "panda_finger_joint2"],
            )
        )
        if len(arm) != 7 or len(set(names)) != len(names) or len(set(arm)) != 7:
            raise ValueError("Expected seven distinct named arm joints")
        missing = [name for name in arm + fingers if name not in names]
        if missing:
            raise ValueError(f"Missing robot joints: {missing}")
        return arm, [names.index(name) for name in arm], names.index(fingers[0])

    def _offset(self, task: Any) -> float:
        manager = self._manager(task)
        if hasattr(manager.cfg, "gripper_offset"):
            return float(manager.cfg.gripper_offset)
        if hasattr(manager, "_offset"):
            return -float(_array(manager._offset.p)[2])
        raise ValueError("Robot TCP offset is unavailable")

    def observe_state(self, task: Any) -> dict[str, Any]:
        """Read named joints and base-relative TCP pose, without actor state.

        Args:
            task: An initialized UniVTAC task.

        Returns:
            Seven named arm positions and one finger position, TCP pose,
            and the single-finger opening, all as JSON-compatible values.
        """
        manager = self._manager(task)
        arm_names, arm_ids, finger_id = self._joint_layout(task)
        raw = _array(manager.robot.data.joint_pos)[0]
        hand = manager.get_ee_pose()
        p, q = compose(_array(hand.p), _array(hand.q), [0, 0, self._offset(task)], [1, 0, 0, 0])
        grip = float(raw[finger_id])
        return {
            "qpos": raw[arm_ids].tolist() + [grip],
            "arm_joint_names": arm_names,
            "tcp_pose": {"position": p.tolist(), "quaternion_wxyz": q.tolist()},
            "hand_pose": {
                "position": vector(_array(hand.p), 3).tolist(),
                "quaternion_wxyz": quaternion(_array(hand.q)).tolist(),
            },
            "root_pose_w": {
                "position": vector(_array(manager.robot.data.root_link_pos_w)[0], 3).tolist(),
                "quaternion_wxyz": quaternion(
                    _array(manager.robot.data.root_link_quat_w)[0]
                ).tolist(),
            },
            "gripper_qpos": grip,
        }

    def validate_fk(self, task: Any) -> dict[str, Any]:
        """Compare cuRobo model FK with the actual base-relative hand before motion.

        Args:
            task: Initialized live task or a test double.

        Returns:
            Verification state and robot-only position/orientation errors.
            Missing FK support returns verified=False with an explanation.
        """
        try:
            manager = self._manager(task)
            names, ids, _ = self._joint_layout(task)
            reference = manager.robot.data.joint_pos
            positions = _array(reference)[0, ids].reshape(1, 7)
            if hasattr(reference, "new_tensor"):
                from curobo.types.robot import JointState

                state = JointState(position=reference.new_tensor(positions), joint_names=names)
            else:
                state = SimpleNamespace(position=positions, joint_names=names)
            fk = manager.planner.motion_gen.compute_kinematics(state)
            hand = manager.get_ee_pose()
            # MotionGen v0.7.8 returns rollout KinematicModelState, whose
            # canonical pose wraps ee_pos_seq/ee_quat_seq. CudaRobotModel's
            # direct get_state instead returns ee_position/ee_quaternion.
            ee_pose = getattr(fk, "ee_pose", None)
            if ee_pose is not None:
                fk_position, fk_quaternion = ee_pose.position, ee_pose.quaternion
            elif hasattr(fk, "ee_pos_seq") and hasattr(fk, "ee_quat_seq"):
                fk_position, fk_quaternion = fk.ee_pos_seq, fk.ee_quat_seq
            elif hasattr(fk, "ee_position") and hasattr(fk, "ee_quaternion"):
                fk_position, fk_quaternion = fk.ee_position, fk.ee_quaternion
            else:
                raise ValueError("Unsupported FK result: expected a complete end-effector pose")
            pos_error = float(
                np.linalg.norm(_array(fk_position).reshape(-1, 3)[0] - _array(hand.p))
            )
            rot_error = orientation_error(_array(fk_quaternion).reshape(-1, 4)[0], _array(hand.q))
            passed = pos_error <= float(
                self.config.get("fk_position_tolerance_m", 0.01)
            ) and rot_error <= float(self.config.get("fk_orientation_tolerance_rad", 0.1))
            return {
                "verified": True,
                "passed": passed,
                "position_error_m": pos_error,
                "orientation_error_rad": rot_error,
            }
        except Exception as exc:
            return {
                "verified": False,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
