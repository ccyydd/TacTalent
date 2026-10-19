"""Covered multi-episode context, native H/timing, boundaries and leakage guards."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from tactalent.roboicl.benchmarks import binding, all_bindings, horizon_for_task
from tactalent.roboicl.demonstrations import plan, prepare_bundle, validate_reference_demonstrations
from tactalent.roboicl.prepare_few_shot import select
from tactalent.roboicl.reference import prepare
from tactalent.roboicl.policy import Model
from tactalent.roboicl.run import preflight
from tactalent.roboicl.calibration import authored


def recording(directory, spec, identity, label):
    directory.mkdir()
    h = horizon_for_task(spec)
    n = 5 * h + 31
    images = {}
    for name, shape in [
        ("head", (270, 480, 3)),
        ("wrist", (270, 480, 3)),
        ("tactile_left", (240, 320, 3)),
        ("tactile_right", (240, 320, 3)),
    ]:
        if name == "wrist" and name not in spec["cameras"]:
            continue
        path = directory / (name + ".jpg")
        Image.fromarray(np.full(shape, identity * 20, np.uint8)).save(path)
        images[name] = str(path)
    observations = []
    for i in range(n):
        state = dict(
            ee_pose=[0.4 + i * 0.0001, 0, 0.2, 1, 0, 0, 0],
            arm_joint_state=[0] * 7,
            ee_joint_state=[0.5],
        )
        observations.append(dict(state=state, images=images, control_step=i))
    meta = dict(
        binding=spec,
        seed=identity,
        split="train",
        status="success",
        source_hdf5_sha256=f"{identity:064x}",
        instruction="native goal " + str(spec["variant"]),
        native_dt=1 / 120,
        control_dt=1 / 60,
        action_repeat=2,
    )
    path = directory / "recording.json"
    path.write_text(
        json.dumps(
            dict(
                metadata=meta,
                observations=observations,
                commands=[
                    dict(ee_pose=o["state"]["ee_pose"], gripper=0.5) for o in observations[1:]
                ],
            )
        )
    )
    return dict(
        recording=str(path), coverage=[label], coverage_basis={"audit": "selection_only_secret"}
    )


@pytest.mark.parametrize(
    "task,variants",
    [
        ("hardness_classify", [None, None]),
        ("insert_block_v2", ["cube", "half_cylinder", "hexagon"]),
    ],
)
@pytest.mark.parametrize("mode", ["vision", "vision_tactile"])
def test_direct_bundle_every_request_keeps_boundaries_goals_and_images(
    task, variants, mode, tmp_path, monkeypatch
):
    monkeypatch.setenv("ASTRA_API_KEY", "offline-only")
    spec = binding("OpenViTac", task, variants[0])
    contract = plan(spec)
    entries = [
        recording(tmp_path / str(i), binding("OpenViTac", task, v), i + 1, label)
        for i, (v, label) in enumerate(zip(variants, contract["coverage"]))
    ]
    path = tmp_path / "reference.json"
    ref = prepare_bundle(entries, spec, path, mode)
    assert ref["demonstration_contract"] == contract
    assert all(
        len(d["starts"]) == 5 and d["horizon"] == horizon_for_task(spec)
        for d in ref["demonstrations"]
    )
    raw = json.dumps(ref["messages"])
    assert raw.count("episode_boundary") == len(entries) - 1
    assert raw.count("END OF FIXED TASK REFERENCE") == 1
    assert "selection_only_secret" not in raw
    calls = [m for m in ref["messages"] if m.get("type") == "function_call"]
    receipts = [m for m in ref["messages"] if m.get("type") == "function_call_output"]
    assert len(calls) == len(entries) * 5 and {m["call_id"] for m in calls} == {
        m["call_id"] for m in receipts
    }
    for i, demo in enumerate(ref["demonstrations"]):
        assert f"TRAIN DEMONSTRATION {i+1}/{len(entries)}" in raw
        assert demo["task_prompt"] in "\n".join(
            p.get("text", "")
            for m in ref["messages"]
            for p in m.get("content", [])
            if isinstance(p, dict)
        )
    model = Model(path, tmp_path, mode)
    model.live_calibration = authored(spec)
    state = json.loads(Path(entries[0]["recording"]).read_text())["observations"][0]["state"]
    imgs = {n: np.zeros((270, 480, 3), np.uint8) for n in spec["cameras"]}
    if mode == "vision_tactile":
        imgs.update(
            {n: np.zeros((240, 320, 3), np.uint8) for n in ("tactile_left", "tactile_right")}
        )
    model.update_native(state, imgs, 0, "active goal", 400)
    first = model.build_payload()
    end_index = next(
        i for i, m in enumerate(model.messages) if "END OF FIXED TASK REFERENCE" in json.dumps(m)
    )
    assert model.messages[end_index + 1]["content"][0]["text"].startswith("TASK:\nactive goal")
    assert model.payload_metrics(first)["image_count"] <= 50
    assert json.loads((tmp_path / "harness.json").read_text())["live_anchor_count"] == 5
    assert ("tactile_left" in json.dumps(first)) == (mode == "vision_tactile")
    assert sum(m.get("type") == "function_call" for m in model.messages) == len(entries) * 5
    model.pending = dict(
        start=0,
        predicted=horizon_for_task(spec),
        returned=1,
        call_id="live_1",
        execution_note="hold",
    )
    command = dict(mode="ee", ee_pose=state["ee_pose"], gripper=0.5)
    model.update_native(
        state, imgs, 1, "active goal", 400, command, {"arm": {"position_error_m": 0}}
    )
    model._commit_feedback()
    second = model.build_payload()
    assert json.dumps(second).count("TRAIN DEMONSTRATION") == len(entries)
    assert model.payload_metrics(second)["image_count"] <= 50


def ledger(spec):
    rows = [
        dict(
            seed=i,
            episode_id=i,
            success=True,
            coverage=[label],
            coverage_basis="native training audit",
        )
        for i, label in enumerate(plan(spec)["coverage"])
    ]
    return dict(
        benchmark=spec["benchmark"],
        task=spec["task"],
        variant=spec["variant"],
        split="train",
        episodes=rows,
    )


@pytest.mark.parametrize("change", ["missing", "duplicate", "heldout", "failed", "unknown"])
def test_coverage_and_split_rejected(change):
    spec = binding("OpenViTac", "hardness_classify")
    data = ledger(spec)
    if change == "missing":
        data["episodes"].pop()
    if change == "duplicate":
        data["episodes"][1]["seed"] = 0
    if change == "heldout":
        data["episodes"][1]["seed"] = 1000000
    if change == "failed":
        data["episodes"][1]["success"] = False
    if change == "unknown":
        data["episodes"][1]["coverage"] = ["unknown"]
    with pytest.raises(ValueError):
        select(data, spec)


def test_old_single_demo_is_rejected_before_sim_or_api(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_API_KEY", "offline-only")
    spec = binding("OpenViTac", "hardness_classify")
    entries = [
        recording(tmp_path / str(i), spec, i + 1, label)
        for i, label in enumerate(plan(spec)["coverage"])
    ]
    path = tmp_path / "reference.json"
    ref = prepare_bundle(entries, spec, path, "vision_tactile")
    ref["demonstrations"].pop()
    path.write_text(json.dumps(ref))
    with pytest.raises(ValueError, match="demonstration count"):
        Model(path, tmp_path)
    args = SimpleNamespace(
        benchmark="OpenViTac",
        task=spec["task"],
        variant=None,
        reference=path,
        backend="live",
        seed=1000000,
    )
    with pytest.raises(ValueError, match="demonstration count"):
        preflight(args)
    with pytest.raises(ValueError, match="covered few-shot"):
        prepare(entries[0]["recording"], tmp_path / "single.json")


def test_bundle_duplicate_sources_rejected(tmp_path):
    spec = binding("OpenViTac", "hardness_classify")
    entries = [
        recording(tmp_path / str(i), spec, i + 1, label)
        for i, label in enumerate(plan(spec)["coverage"])
    ]
    entries[1]["recording"] = entries[0]["recording"]
    with pytest.raises(ValueError, match="Repeated"):
        prepare_bundle(entries, spec, tmp_path / "reference.json")


def test_approved_coverage_only_no_pose_variation_expansion():
    expected = {
        ("UniVTAC", "grasp_classify"): 2,
        ("OpenViTac", "weight_classify"): 2,
        ("OpenViTac", "hardness_classify"): 2,
        ("OpenViTac", "roughness_classify"): 2,
        ("OpenViTac", "roughness_regrasp"): 2,
        ("OpenViTac", "can_empty_select"): 2,
        ("OpenViTac", "insert_block_v2"): 3,
    }
    for spec in all_bindings():
        p = plan(spec)
        k = expected.get((spec["benchmark"], spec["task"]), 1)
        assert p["shots"] == k
        assert p["chunks_per_demo"] == p["live_anchor_count"] == (12 if k == 1 else 5)
