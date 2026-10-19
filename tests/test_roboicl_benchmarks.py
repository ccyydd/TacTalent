"""Regression checks for benchmark boundaries and real native action accounting."""

import json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
import torch
from tactalent.roboicl.benchmarks import (
    binding,
    all_bindings,
    horizon_for_task,
    task_context,
    task_prompt,
)
from tactalent.roboicl.calibration import authored
from tactalent.roboicl.controller import apply_native, command_budget
from tactalent.roboicl.policy import Model
from tactalent.roboicl.observations import input_contract
from tactalent.roboicl.upstream import protocol_sha256
from tactalent.roboicl.reference import select


def test_task_inventory_and_native_limits():
    rows = all_bindings()
    assert len(rows) == 21
    assert len({r["task"] for r in rows if r["benchmark"] == "OpenViTac"}) == 11
    assert len([r for r in rows if r["benchmark"] == "UniVTAC"]) == 8
    assert binding("OpenViTac", "grasp_fragile_chip")["native_action_limit"] == 3600
    assert binding("OpenViTac", "insert_USB")["native_action_limit"] == 300
    assert binding("UniVTAC", "insert_HDMI")["native_action_limit"] == 600
    with pytest.raises(ValueError):
        binding("OpenViTac", "insert_block_v2")


@pytest.mark.parametrize(
    "spec", all_bindings(), ids=lambda s: s["benchmark"] + "_" + s["task"] + "_" + str(s["variant"])
)
def test_task_calibration_and_model_cameras(spec, tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_API_KEY", "offline-only")
    # These controller/camera tests omit TRAIN; real bundles are covered separately.
    from tactalent.roboicl.demonstrations import plan

    monkeypatch.setattr(
        "tactalent.roboicl.policy.validate_reference_demonstrations", lambda ref, spec: plan(spec)
    )
    reference = tmp_path / "ref.json"
    reference.write_text(
        json.dumps(
            dict(
                binding=spec,
                demonstration_contract=plan(spec),
                icl_mode="vision",
                input_contract=input_contract(spec, "vision"),
                horizon=horizon_for_task(spec),
                frequency=60,
                messages=[],
                protocol_sha256=protocol_sha256(spec),
                instruction="native task",
                task_context=task_context(spec),
                task_prompt=task_prompt(spec, "native task"),
                training_calibration=authored(spec, training=True),
            )
        )
    )
    model = Model(reference, tmp_path, icl_mode="vision")
    expected = horizon_for_task(spec)
    assert (
        model.dialogue_config.predict_horizon == model.dialogue_config.execute_horizon == expected
    )
    shape = authored(spec)["shape"]
    state = dict(arm_joint_state=[0] * 7, ee_joint_state=[0.5], ee_pose=[0.4, 0, 0.2, 1, 0, 0, 0])
    images = {n: np.zeros(shape, np.uint8) for n in spec["cameras"]}
    model.live_calibration = authored(spec)
    model.update_native(state, images, 0, "native task", 150)
    payload = model.build_payload()
    assert model.payload_metrics(payload)["image_count"] == 1
    expected_task = task_prompt(spec, "native task")
    assert model.messages[0]["content"][0]["text"] == expected_task
    assert task_context(spec)["background"] in expected_task
    assert (
        expected_task not in model.prompt
    )  # Task belongs to context, not the method system prompt.
    assert "keep 0 while holding" not in model.prompt
    assert "experience-summary" not in model.prompt
    if spec["cameras"] == ["head"]:
        with pytest.raises(ValueError):
            model.update_native(
                state, dict(images, wrist=np.zeros(shape, np.uint8)), 1, "task", 150
            )


class LegacyTask:
    def __init__(self, limit=3, success_at=None, early_stop_at=None, fail_at=None, exec_ok=True):
        self.cfg = SimpleNamespace(step_lim=limit)
        self.take_action_cnt = self.step_count = 0
        self.eval_success = False
        self.plan_success = True
        self.success_at, self.early_stop_at, self.fail_at = success_at, early_stop_at, fail_at
        self.exec_ok = exec_ok
        self.q = torch.zeros(8)
        self.targets = []
        self._robot_manager = SimpleNamespace(
            robot=SimpleNamespace(data=SimpleNamespace(joint_pos=self.q)),
            get_observations=lambda keys: {"joint": self.q.clone()},
            set_arm=self.set_arm,
            set_gripper=self.set_gripper,
        )

    def set_arm(self, q, force):
        assert force
        self.q[:7] = q

    def set_gripper(self, q, force):
        assert force
        self.q[7] = q

    def _step(self):
        self.step_count += 1
        self.targets.append(self.q.clone())
        if self.step_count == self.fail_at:
            self.plan_success = False

    def check_success(self):
        return self.step_count == self.success_at

    def check_early_stop(self):
        return self.step_count == self.early_stop_at

    def take_action(self, q, action_type, force):
        assert action_type == "qpos" and force
        self.take_action_cnt += 1
        self.set_arm(q[:7], force)
        self.set_gripper(q[7], force)
        self._step()
        self.eval_success = self.check_success()
        return self.exec_ok, self.eval_success


@pytest.mark.parametrize("limit", [1, 3, 300])
def test_univtac_counts_targets_and_finishes_last_target(limit):
    task = LegacyTask(limit=limit)
    for index in range(limit):
        assert command_budget(task, 2) == limit - index
        target = np.ones(8) * ((index + 1) * 0.0001)
        result = apply_native(task, target, 2)
        assert result["native_action_delta"] == 1
        assert result["physics_step_delta"] == 2
        assert result["truncated"] == (index + 1 == limit)
        np.testing.assert_allclose(task.q.numpy(), target, rtol=1e-5)
    assert task.take_action_cnt == limit and task.step_count == 2 * limit
    assert command_budget(task, 2) == 0
    before = task.q.clone()
    with pytest.raises(ValueError, match="after native termination"):
        apply_native(task, np.ones(8), 2)
    torch.testing.assert_close(task.q, before)
    assert task.take_action_cnt == limit and task.step_count == 2 * limit


def test_univtac_interpolation_preserves_motion_and_cadence():
    task = LegacyTask()
    task.q.fill_(0.02)
    result = apply_native(task, np.ones(8) * 0.04, 2)
    torch.testing.assert_close(task.targets[0], torch.full((8,), 0.03))
    torch.testing.assert_close(task.targets[1], torch.full((8,), 0.04))
    assert result["native_action_delta"] == 1 and result["physics_step_delta"] == 2


@pytest.mark.parametrize("at", [1, 2])
def test_native_success_stops_at_physics_substep(at):
    task = LegacyTask(success_at=at)
    got = apply_native(task, np.ones(8) * 0.01, 2)
    assert got["success"] and task.eval_success
    assert task.take_action_cnt == 1 and task.step_count == at
    assert got["physics_step_delta"] == at


@pytest.mark.parametrize(
    "kwargs",
    [
        {"early_stop_at": 1},
        {"early_stop_at": 2},
        {"fail_at": 1},
        {"fail_at": 2},
        {"exec_ok": False},
    ],
)
def test_univtac_native_failure_stops_substeps(kwargs):
    task = LegacyTask(**kwargs)
    got = apply_native(task, np.ones(8) * 0.01, 2)
    assert got["truncated"] and not got["success"]
    assert task.take_action_cnt == 1
    expected_steps = 1 if "exec_ok" in kwargs else next(iter(kwargs.values()))
    assert task.step_count == expected_steps


def test_univtac_binding_preserves_native_limits_but_one_slot_per_target():
    for spec in all_bindings():
        if spec["benchmark"] == "UniVTAC":
            assert spec["native_actions_per_command"] == 1
            task = LegacyTask(limit=spec["native_action_limit"])
            assert command_budget(task, 2) == spec["native_action_limit"]


def test_openvitac_force_repeat_is_not_doubled():
    task = LegacyTask()
    task.cfg.final_acceptance_contract = True
    task.cfg.policy_action_repeat = 2

    def step(q, action_type, force, action_repeat):
        task.take_action_cnt += 1
        task.step_count += action_repeat
        return {}, 0, False, False, {"exec_success": True}

    task.env_step = step
    got = apply_native(task, np.ones(8) * 0.01, 2)
    assert got["native_action_delta"] == 1 and got["physics_step_delta"] == 2
    with pytest.raises(ValueError):
        apply_native(task, np.ones(8) * 0.01, 4)


def test_recording_gaps_cannot_become_fake_one_step_actions():
    state = dict(ee_pose=[0.4, 0, 0.2, 1, 0, 0, 0])
    obs = [dict(state=state, control_step=i + (3 if i >= 200 else 0)) for i in range(401)]
    commands = [dict(ee_pose=state["ee_pose"], gripper=0.5) for _ in range(400)]
    starts, _ = select(dict(observations=obs, commands=commands), 15)
    assert all(not start < 200 <= start + 15 for start in starts)


def test_review_wrapper_preserves_native_instruction_lookup():
    from tactalent.roboicl.environment import presentation_task_class

    class Native:
        pass

    wrapped = presentation_task_class(Native)
    assert wrapped.__module__ == Native.__module__
    assert wrapped.__name__ == Native.__name__
    assert issubclass(wrapped, Native)


def test_measured_overshoot_is_preserved_and_live_targets_are_not_clipped():
    from tactalent.roboicl.prepare_recorded import reconstruct_gripper_targets

    joints = np.zeros((3, 9))
    joints[:, 7:9] = np.array([0.020, 0.0394, -0.0001])[:, None]
    original = joints.copy()
    measured, target = reconstruct_gripper_targets(joints)
    np.testing.assert_array_equal(joints, original)
    assert measured[1] > 1 and measured[2] < 0
    np.testing.assert_allclose(target, [0.020 / 0.039, 1, 0])
    model = SimpleNamespace(
        dialogue_config=SimpleNamespace(
            predict_horizon=1, max_translation_m=0.03, max_rotation_rad=0.15
        )
    )
    state = dict(ee_pose=[0.4, 0, 0.2, 1, 0, 0, 0])
    with pytest.raises(ValueError, match="Gripper must be"):
        Model._convert_tensor(model, [[0, 0, 0, 0, 0, 0, 1.01]], state)


def test_camera_roundtrip_uses_actual_pose_and_rejects_real_mount_changes():
    from tactalent.roboicl.calibration import verified_geometry
    from scipy.spatial.transform import Rotation

    fixed = authored(binding("OpenViTac", "grasp_fragile_chip"))
    actual = np.array(fixed["camera_to_environment"])
    actual[:3, :3] = Rotation.from_rotvec([0.00025, 0, 0]).as_matrix() @ actual[:3, :3]
    K = np.array(fixed["intrinsic_matrix"])
    verified = verified_geometry(fixed, actual, K)
    np.testing.assert_array_equal(verified["camera_to_environment"], actual)
    assert verified["verification"]["angle_error_rad"] == pytest.approx(0.00025)
    actual[:3, :3] = Rotation.from_rotvec([0.01, 0, 0]).as_matrix() @ actual[:3, :3]
    with pytest.raises(ValueError, match="head calibration differ"):
        verified_geometry(fixed, actual, K)


def test_preflight_rejects_cross_task_reference_and_training_seed(tmp_path, monkeypatch):
    from tactalent.roboicl.demonstrations import plan

    # This test isolates task/seed guards; the covered-reference guard has its own tests.
    monkeypatch.setattr(
        "tactalent.roboicl.run.validate_reference_demonstrations", lambda ref, spec: plan(spec)
    )
    from tactalent.roboicl.run import preflight

    spec = binding("UniVTAC", "pull_out_key")
    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps(
            dict(
                binding=spec,
                icl_mode="vision_tactile",
                input_contract=input_contract(spec),
                protocol_sha256=protocol_sha256(spec),
                instruction="native task",
                task_context=task_context(spec),
                task_prompt=task_prompt(spec, "native task"),
                horizon=5,
                frequency=60,
                action_repeat=2,
            )
        )
    )
    args = SimpleNamespace(
        benchmark="UniVTAC",
        task="lift_can",
        variant=None,
        reference=path,
        backend="live",
        seed=1000000,
    )
    with pytest.raises(ValueError, match="version mismatch"):
        preflight(args)
    args.task = "pull_out_key"
    preflight(args)
    args.reference = None
    args.backend = "smoke"
    args.seed = 0
    with pytest.raises(ValueError, match="held out"):
        preflight(args)


def test_repreparation_does_not_relabel_stale_training_calibration(tmp_path):
    from tactalent.roboicl.reference import prepare

    spec = binding("UniVTAC", "pull_out_key")
    state = dict(ee_pose=[0.4, 0, 0.2, 1, 0, 0, 0])
    data = dict(
        metadata=dict(binding=spec, status="success", seed=0, training_calibration={"stale": True}),
        observations=[dict(state=state) for _ in range(401)],
        commands=[dict(ee_pose=state["ee_pose"], gripper=0.5) for _ in range(400)],
    )
    path = tmp_path / "recording.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="calibration changed"):
        prepare(path, tmp_path / "reference.json")


def test_training_ledger_uses_roboicl_binding_only(monkeypatch):
    from tactalent.roboicl import prepare_recorded

    spec = binding("OpenViTac", "insert_block_v2", "hexagon")
    seen = []

    def validate(path, config):
        seen.append(config)
        return dict(
            instruction="native goal",
            episodes=[dict(path="training.hdf5", seed=0, sha256="source")],
        )

    monkeypatch.setattr(prepare_recorded, "ledger", validate)
    monkeypatch.setattr(prepare_recorded, "sha", lambda path: "ledger")
    path, meta = prepare_recorded.training_source(spec, "training_ledger.json")
    assert seen == [spec]
    assert path == "training.hdf5" and meta["instruction"] == "native goal"


@pytest.mark.parametrize(
    "benchmark,task_name,maximum",
    [
        ("OpenViTac", "hardness_classify", 100),
        ("UniVTAC", "insert_HDMI", 300),
        ("OpenViTac", "insert_block_v1", 200),
        ("OpenViTac", "weight_classify", 200),
        ("UniVTAC", "pull_out_key", 200),
    ],
)
@pytest.mark.parametrize("mode", ["train", "eval"])
def test_reviewed_evaluation_caps_stop_exactly_without_changing_training(
    benchmark, task_name, maximum, mode
):
    from tactalent.roboicl.environment import configure_evaluation_budget

    spec = binding(benchmark, task_name)
    task = LegacyTask(limit=spec["native_action_limit"])
    configure_evaluation_budget(task.cfg, spec, mode)
    expected = maximum if mode == "eval" else spec["native_action_limit"]
    assert command_budget(task, 2) == expected
    for _ in range(expected):
        result = apply_native(task, np.zeros(8), 2)
    assert task.take_action_cnt == expected
    assert task.step_count == 2 * expected
    assert result["truncated"] and command_budget(task, 2) == 0
    with pytest.raises(ValueError, match="after native termination"):
        apply_native(task, np.zeros(8), 2)
