"""Native cuRobo inverse-kinematics interface for one Panda arm."""

from __future__ import annotations
from types import SimpleNamespace
from typing import Any
import numpy as np
from .transforms import quaternion, vector


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class _Backend:
    """Adapt installed cuRobo or numpy test doubles to one named joint layout."""

    def __init__(self, task: Any, arm_names: list[str]):
        manager = task._robot_manager
        self.planner = manager.planner
        self.motion_gen = self.planner.motion_gen
        self.solver = self.motion_gen.ik_solver
        self.arm_names = arm_names
        self.names = list(
            getattr(self.solver, "joint_names", None)
            or getattr(self.planner, "active_joints_name", arm_names)
        )
        if len(self.names) != 7 or set(self.names) != set(arm_names):
            raise ValueError("IK joint names do not match the named robot arm")
        self.to_solver = [arm_names.index(name) for name in self.names]
        reference = manager.robot.data.joint_pos
        self.live = hasattr(reference, "new_tensor")
        self.reference = reference
        self.num_seeds = int(getattr(self.solver, "num_seeds", 1))
        if self.num_seeds < 1:
            raise ValueError("IK requires at least one seed")
        self.return_seeds = min(4, self.num_seeds)
        if self.live:
            from curobo.types.math import Pose
            from curobo.types.robot import JointState

            self.pose_class, self.state_class = Pose, JointState
        else:
            self.pose_class = self.state_class = SimpleNamespace
        self.constraint_checker = getattr(self.motion_gen, "check_constraints", None)

    def tensor(self, value: Any) -> Any:
        array = np.asarray(value, dtype=float)
        return self.reference.new_tensor(array) if self.live else array

    def state(self, arm: np.ndarray) -> Any:
        return self.state_class(
            position=self.tensor(arm[self.to_solver].reshape(1, 7)),
            joint_names=self.names,
        )

    def fk(self, arm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        result = self.motion_gen.compute_kinematics(self.state(arm))
        if hasattr(result, "ee_pose"):
            position, rotation = result.ee_pose.position, result.ee_pose.quaternion
        elif hasattr(result, "ee_pos_seq") and hasattr(result, "ee_quat_seq"):
            position, rotation = result.ee_pos_seq, result.ee_quat_seq
        elif hasattr(result, "ee_position") and hasattr(result, "ee_quaternion"):
            position, rotation = result.ee_position, result.ee_quaternion
        else:
            raise ValueError("FK must return a complete end-effector pose")
        return (
            vector(_array(position).reshape(-1, 3)[0], 3),
            quaternion(_array(rotation).reshape(-1, 4)[0]),
        )

    def feasible(self, arm: np.ndarray) -> bool:
        if self.constraint_checker is None:
            return True
        metrics = self.constraint_checker(self.state(arm))
        if not hasattr(metrics, "feasible"):
            raise ValueError("Constraint checker must return explicit feasibility")
        flags = _array(metrics.feasible).reshape(-1)
        if flags.size != 1:
            raise ValueError("Single-state feasibility must have one flag")
        if (
            flags.dtype.kind not in "biuf"
            or not np.isfinite(flags).all()
            or not np.isin(flags, [0, 1]).all()
        ):
            raise ValueError("Constraint feasibility flags must be finite booleans")
        return bool(flags[0])

    def solve(
        self, position: np.ndarray, rotation: np.ndarray, previous: np.ndarray
    ) -> list[np.ndarray]:
        goal = self.pose_class(
            position=self.tensor(position.reshape(1, 3)),
            quaternion=self.tensor(rotation.reshape(1, 4)),
        )
        previous_solver = previous[self.to_solver].reshape(1, 7)
        # get_seed consumes (batch, n, dof), despite solve_single's older docstring.
        # Supply all seeds locally, so no distant random IK branches are added.
        result = self.solver.solve_single(
            goal,
            retract_config=self.tensor(previous_solver),
            seed_config=self.tensor(np.repeat(previous_solver[:, None, :], self.num_seeds, axis=1)),
            return_seeds=self.return_seeds,
        )
        js = getattr(result, "js_solution", None)
        values = getattr(result, "solution", None)
        if values is None and js is not None:
            values = js.position
        if values is None:
            raise ValueError("IK result has no joint solutions")
        candidates = _array(values).astype(float).reshape(-1, 7)
        flags = _array(result.success).reshape(-1)
        if (
            flags.dtype.kind not in "biuf"
            or not np.isfinite(flags).all()
            or not np.isin(flags, [0, 1]).all()
        ):
            raise ValueError("IK success flags must be finite booleans")
        if len(candidates) != len(flags):
            raise ValueError("IK success flags do not match returned solutions")
        output_names = list(getattr(js, "joint_names", None) or self.names)
        if len(output_names) != 7 or set(output_names) != set(self.arm_names):
            raise ValueError("IK solution joint names do not match the robot")
        order = [output_names.index(name) for name in self.arm_names]
        return [
            candidate[order].copy()
            for candidate, flag in zip(candidates, flags)
            if bool(flag) and np.isfinite(candidate).all()
        ]
