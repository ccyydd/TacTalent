"""Single-arm IK with native, explicitly accounted benchmark execution."""

import numpy as np
from scipy.spatial.transform import Rotation
from tactalent.robot.ik import _Backend
from tactalent.robot.state import RobotObserver
from tactalent.robot.transforms import orientation_error


def native_action_cost(task, repeat):
    # A command consumes one action slot; interpolation consumes physics steps.
    return 1


def command_budget(task, repeat):
    return max(0, int(task.cfg.step_lim) - int(task.take_action_cnt))


def _apply_univtac(task, tensor, repeat):
    """Use one native action slot, then finish its uncounted physics substeps."""
    if (
        task.take_action_cnt >= task.cfg.step_lim
        or task.eval_success
        or not task.plan_success
        or task.check_early_stop()
    ):
        raise ValueError("Cannot execute a command after native termination")
    initial = task._robot_manager.get_observations(["joint"])["joint"][: len(tensor)].clone()
    ok, success = task.take_action(initial + (tensor - initial) / repeat, "qpos", force=True)
    for index in range(1, repeat):
        if not ok or success or not task.plan_success or task.check_early_stop():
            break
        # Reaching the action limit does not cut the final command in half.
        # The first take_action already charged this command's single slot.
        interpolated = initial + (tensor - initial) * ((index + 1) / repeat)
        task._robot_manager.set_arm(interpolated[:-1], force=True)
        task._robot_manager.set_gripper(interpolated[-1], force=True)
        task._step()
        if task.check_success():
            task.eval_success = success = True
    return {
        "exec_success": bool(ok),
        "success": bool(success),
        "truncated": bool(
            not ok
            or not task.plan_success
            or task.take_action_cnt >= task.cfg.step_lim
            or task.check_early_stop()
        ),
    }


def apply_native(task, qpos, repeat):
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
        raise ValueError("action_repeat must be a positive integer")
    before, sim = int(task.take_action_cnt), int(task.step_count)
    tensor = task._robot_manager.robot.data.joint_pos.new_tensor(qpos)
    if hasattr(task, "env_step"):
        if (
            getattr(task.cfg, "final_acceptance_contract", False)
            and int(task.cfg.policy_action_repeat) != repeat
        ):
            raise ValueError("Force-task repeat already lives inside the native controller")
        _, _, success, truncated, info = task.env_step(
            tensor, "qpos", force=True, action_repeat=repeat
        )
        result = dict(info, success=bool(success), truncated=bool(truncated))
    else:
        result = _apply_univtac(task, tensor, repeat)
    result.update(
        native_action_delta=int(task.take_action_cnt) - before,
        physics_step_delta=int(task.step_count) - sim,
    )
    if result["native_action_delta"] != native_action_cost(task, repeat):
        raise ValueError("Unexpected native action counter change")
    if not 1 <= result["physics_step_delta"] <= repeat:
        raise ValueError("Unexpected native physics cadence")
    return result


class Controller:
    def __init__(self, task, action_repeat=2):
        self.task, self.action_repeat = task, action_repeat
        self.observer = RobotObserver({})
        names, _, _ = self.observer._joint_layout(task)
        self.backend = _Backend(task, names)
        self.fk_audit = self.observer.validate_fk(task)
        if not self.fk_audit["passed"]:
            raise ValueError("Native planner/measurement FK mismatch")
        if abs(float(task._robot_manager.gripper_max_qpos) - 0.039) > 1e-8:
            raise ValueError("Unexpected Panda gripper range")
        self.last_result = {}

    def plan(self, command, limit=0.15):
        state = self.observer.observe_state(self.task)
        prior = np.asarray(state["qpos"][:7])
        if command["mode"] == "joint":
            return np.r_[command["arm"], command["gripper"] * 0.039], None
        root = state["root_pose_w"]
        rot = Rotation.from_quat(np.array(root["quaternion_wxyz"])[[1, 2, 3, 0]])
        pose = np.asarray(command["ee_pose"])
        p = rot.inv().apply(pose[:3] - root["position"])
        q = (rot.inv() * Rotation.from_quat(pose[3:][[1, 2, 3, 0]])).as_quat()[[3, 0, 1, 2]]
        bias = np.asarray(getattr(self.backend.planner, "frame_bias", [0, 0, 0]))
        solutions = self.backend.solve(p + bias, q, prior)
        candidates = [x for x in solutions if np.max(np.abs(x - prior)) <= limit + 1e-6]
        if not candidates:
            return None, {
                "reason": "IK unavailable or exceeds continuity limit",
                "max_joint_increment_rad": limit,
                "smallest_candidate_joint_change_rad": min(
                    (float(np.max(np.abs(x - prior))) for x in solutions), default=None
                ),
            }
        target = min(candidates, key=lambda x: np.linalg.norm(x - prior))
        return np.r_[target, command["gripper"] * 0.039], None

    def apply(self, qpos, command):
        self.last_result = apply_native(self.task, qpos, self.action_repeat)
        return self.last_result["exec_success"]

    def tracking(self, command, state):
        return tracking_receipt(command, state)


def tracking_receipt(command, state):
    if command["mode"] == "joint":
        return {
            "arm": {
                "max_joint_error_rad": float(
                    np.max(np.abs(np.asarray(command["arm"]) - state["arm_joint_state"]))
                )
            }
        }
    a, b = np.asarray(command["ee_pose"]), np.asarray(state["ee_pose"])
    return {
        "arm": {
            "position_error_m": float(np.linalg.norm(a[:3] - b[:3])),
            "orientation_error_rad": float(orientation_error(a[3:], b[3:])),
        }
    }
