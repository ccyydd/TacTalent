import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from tactalent.roboicl.policy import Model
from tactalent.roboicl.observations import input_contract
from tactalent.roboicl.benchmarks import binding, task_context, task_prompt
from tactalent.roboicl.reference import observation, select, actions_between
from tactalent.roboicl.upstream import VENDOR, protocol_sha256
from roboicl.policy.dialogue_policy import Model as Official
from roboicl.policy.trajectory_memory import LiveTrajectoryMemory


def state():
    return {
        "arm_joint_state": [0] * 7,
        "ee_joint_state": [0.5],
        "ee_pose": [0.4, 0, 0.2, 1, 0, 0, 0],
    }


def model(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_API_KEY", "offline-only")
    # These controller/camera tests omit TRAIN; real bundles are covered separately.
    from tactalent.roboicl.demonstrations import plan

    monkeypatch.setattr(
        "tactalent.roboicl.policy.validate_reference_demonstrations", lambda ref, spec: plan(spec)
    )
    ref = tmp_path / "reference.json"
    ref.write_text(
        json.dumps(
            {
                "demonstration_contract": plan(binding()),
                "horizon": 10,
                "frequency": 60,
                "icl_mode": "vision",
                "input_contract": input_contract(binding(), "vision"),
                "messages": [],
                "protocol_sha256": protocol_sha256(),
                "instruction": "task",
                "task_context": task_context(binding()),
                "task_prompt": task_prompt(binding(), "task"),
            }
        )
    )
    return Model(ref, tmp_path, icl_mode="vision")


def test_official_core_is_used():
    for name in (
        "get_action",
        "_get_action_once",
        "_request",
        "build_payload",
        "_request_inputs",
        "_commit_feedback",
        "notify_action_rejection",
    ):
        assert getattr(Model, name) is getattr(Official, name)


def test_world_delta_accumulates_targets(tmp_path, monkeypatch):
    m = model(tmp_path, monkeypatch)
    rows = [[0.001, 0, 0, 0, 0, 0.01, 0.3] for _ in range(10)]
    got = m._convert_tensor(rows, state())
    assert got[-1]["ee_pose"][0] == pytest.approx(0.410)
    assert got[-1]["gripper"] == 0.3
    assert got[-1]["ee_pose"][3] == pytest.approx(np.cos(0.10 / 2))


def test_no_tactile_no_fake_second_arm(tmp_path, monkeypatch):
    m = model(tmp_path, monkeypatch)
    images = {name: np.zeros((270, 480, 3), np.uint8) for name in ("head", "wrist")}
    m.update_native(state(), images, 0, "task", 400)
    from tactalent.roboicl.calibration import authored

    m.live_calibration = authored()
    payload = m.build_payload()
    assert payload["reasoning"]["effort"] == "xhigh"
    assert "max_output_tokens" not in payload
    assert m.frozen_tools[0]["parameters"]["properties"]["actions"]["items"]["maxItems"] == 7
    assert "tactile" not in json.dumps(payload).lower()
    assert "left_gripper" not in json.dumps(payload)
    assert m.payload_metrics(payload)["image_count"] == 1


def test_native_aspect_ratio(tmp_path):
    images = {name: np.zeros((270, 480, 3), np.uint8) for name in ("head", "wrist")}
    msg = observation(state(), images, 0, "TRAIN")
    import base64, io

    pic = Image.open(io.BytesIO(base64.b64decode(msg["content"][1]["image_url"].split(",")[1])))
    assert pic.size == (960, 270)
    with pytest.raises(ValueError):
        observation(
            state(),
            {"head": np.zeros((300, 480, 3), np.uint8), "wrist": images["wrist"]},
            0,
            "TRAIN",
        )


def test_limits_reject_entire_tensor(tmp_path, monkeypatch):
    m = model(tmp_path, monkeypatch)
    rows = [[0, 0, 0, 0, 0, 0, 0.5] for _ in range(10)]
    rows[-1][0] = 0.031
    with pytest.raises(ValueError):
        m._convert_tensor(rows, state())
    rows[-1][0] = False
    with pytest.raises(ValueError):
        m._convert_tensor(rows, state())


def test_official_selector_has_first_last_and_no_overlap():
    observations = [{"state": state()} for _ in range(401)]
    commands = [{"ee_pose": state()["ee_pose"], "gripper": 0.5} for _ in range(400)]
    starts, digest = select({"observations": observations, "commands": commands}, 15)
    assert len(starts) == 12 and starts[0] == 0 and starts[-1] == 385
    assert all(b - a >= 15 for a, b in zip(starts, starts[1:]))
    assert len(digest) == 64


def test_official_history_bounds():
    memory = LiveTrajectoryMemory(count=12, retain_latest=True)
    memory.configure(400)
    for i in range(0, 390, 5):
        memory.capture(i, i + 5, [{"role": "user", "content": str(i)}])
    assert memory.stats["retained_chunks"] <= 12
    assert memory.stats["latest_interval"] == [385, 390]


def test_stale_reference_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_API_KEY", "offline-only")
    # These controller/camera tests omit TRAIN; real bundles are covered separately.
    from tactalent.roboicl.demonstrations import plan

    monkeypatch.setattr(
        "tactalent.roboicl.policy.validate_reference_demonstrations", lambda ref, spec: plan(spec)
    )
    ref = tmp_path / "reference.json"
    ref.write_text(json.dumps({"protocol_sha256": "older-prompt"}))
    with pytest.raises(ValueError, match="version mismatch"):
        Model(ref, tmp_path)


def test_receipt_counts_control_actions_not_physics_substeps(tmp_path, monkeypatch):
    m = model(tmp_path, monkeypatch)
    images = {name: np.zeros((270, 480, 3), np.uint8) for name in ("head", "wrist")}
    m.update_native(state(), images, 0, "task", 400)
    from tactalent.roboicl.calibration import authored

    m.live_calibration = authored()
    m.build_payload()
    m.pending = {
        "start": 0,
        "predicted": 10,
        "returned": 10,
        "call_id": "offline",
        "execution_note": "move",
    }
    commands = m._convert_tensor([[0.001, 0, 0, 0, 0, 0, 0.5] for _ in range(10)], state())
    for step, c in enumerate(commands, 1):
        observed = dict(state(), ee_pose=c["ee_pose"])
        m.update_native(observed, images, step, "task", 400, c, {"arm": {"position_error_m": 0}})
    m._commit_feedback()
    receipts = [
        json.loads(x["output"]) for x in m.messages if x.get("type") == "function_call_output"
    ]
    assert receipts[-1]["executed_steps"] == 10
    assert receipts[-1]["sampled_steps"] == [10]
    assert receipts[-1]["last_submitted_command"]["ee_pose"][0] == pytest.approx(0.410)
    assert m.pending is None
