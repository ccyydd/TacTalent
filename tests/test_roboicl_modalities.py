"""Sensor selection must match across demos, live frames and stored reference contracts."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from tactalent.roboicl.benchmarks import (
    binding,
    all_bindings,
    horizon_for_task,
    task_context,
    task_prompt,
)
from tactalent.roboicl.observations import (
    input_contract,
    native_images,
    panel,
    resolve_mode,
    image_names,
    validate_reference_inputs,
)
from tactalent.roboicl.reference import prepare
from tactalent.roboicl.policy import Model
from tactalent.roboicl.upstream import protocol_sha256
from tactalent.roboicl.run import preflight
from tactalent.roboicl.calibration import authored


def state():
    return dict(arm_joint_state=[0.0] * 7, ee_joint_state=[0.5], ee_pose=[0.4, 0, 0.2, 1, 0, 0, 0])


def images(spec, mode):
    result = {name: np.full((270, 480, 3), (20, 40, 80), np.uint8) for name in spec["cameras"]}
    if mode == "vision_tactile":
        result.update(
            tactile_left=np.full((240, 320, 3), (240, 20, 10), np.uint8),
            tactile_right=np.full((240, 320, 3), (10, 20, 240), np.uint8),
        )
    return result


def reference(spec, mode):
    from tactalent.roboicl.demonstrations import plan

    result = dict(
        binding=spec,
        horizon=horizon_for_task(spec),
        frequency=60,
        action_repeat=2,
        protocol_sha256=protocol_sha256(spec),
        messages=[],
        instruction="native task",
        task_context=task_context(spec),
        task_prompt=task_prompt(spec, "native task"),
        icl_mode=mode,
        input_contract=input_contract(spec, mode),
    )
    result.update(
        demonstration_contract=plan(spec),
        demonstrations=[
            dict(
                binding=spec,
                source_sha256="0" * 64,
                seed=0,
                split="train",
                status="success",
                coverage=[],
                horizon=horizon_for_task(spec),
                frequency=60,
                action_repeat=2,
                starts=list(range(12)),
                selected_control_steps=list(range(12)),
                instruction="native task",
                task_prompt=task_prompt(spec, "native task"),
            )
        ],
    )
    return result


@pytest.mark.parametrize(
    "spec", all_bindings(), ids=lambda s: s["benchmark"] + "_" + s["task"] + "_" + str(s["variant"])
)
def test_default_tactile_and_native_camera_selection(spec):
    assert resolve_mode(spec) == "vision_tactile"
    if spec["benchmark"] == "OpenViTac":
        assert spec["cameras"] == ["head", "wrist"]
    else:
        assert ("wrist" in spec["cameras"]) == (spec["task"] in ("lift_can", "insert_tube"))
    assert image_names(spec, "vision") == tuple(spec["cameras"])
    assert image_names(spec)[-2:] == ("tactile_left", "tactile_right")


@pytest.mark.parametrize(
    "benchmark,task", [("UniVTAC", "pull_out_key"), ("OpenViTac", "turn_gear_pair")]
)
@pytest.mark.parametrize("mode", ["vision", "vision_tactile"])
def test_panel_keeps_exact_pixels_sides_and_aspect_ratio(benchmark, task, mode):
    spec = binding(benchmark, task)
    values = images(spec, mode)
    canvas, layout = panel(values, spec["cameras"], mode, spec)
    for region in layout:
        x, y, w, h = region["xywh"]
        np.testing.assert_array_equal(
            np.asarray(canvas)[y : y + h, x : x + w], values[region["sensor"]]
        )
    assert layout[0]["xywh"][:2] == [0, 0]  # Existing head calibration remains local to head.
    if mode == "vision_tactile":
        assert canvas.height == 510
        assert layout[-2]["sensor"] == "tactile_left" and layout[-1]["sensor"] == "tactile_right"
    else:
        assert canvas.size == (480 * len(spec["cameras"]), 270)


def test_rgb_only_excludes_tactile_and_unapproved_wrist():
    spec = binding("UniVTAC", "pull_out_key")
    obs = dict(
        observation={
            name: dict(rgb=np.zeros((270, 480, 3), np.uint8)) for name in ("head", "wrist")
        },
        tactile={"hidden_secret": "must not be read"},
    )
    assert set(native_images(obs, spec, "vision")) == {"head"}
    with pytest.raises(ValueError):
        panel(images(spec, "vision_tactile"), spec["cameras"], "vision", spec)
    with pytest.raises(ValueError):
        panel(images(spec, "vision"), spec["cameras"], "vision_tactile", spec)


@pytest.mark.parametrize("suffix", ["_tactile", "_gsmini"])
def test_native_tactile_reads_rgb_not_marker_or_depth(suffix):
    spec = binding()
    values = images(spec, "vision_tactile")
    obs = dict(
        observation={n: dict(rgb=values[n]) for n in spec["cameras"]},
        tactile={
            s + suffix: dict(rgb=values["tactile_" + s], rgb_marker="reject", depth="reject")
            for s in ("left", "right")
        },
    )
    got = native_images(obs, spec)
    for name, value in values.items():
        np.testing.assert_array_equal(got[name], value)


@pytest.mark.parametrize("mode", ["vision", "vision_tactile"])
def test_live_history_retains_the_selected_modalities(mode, tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_API_KEY", "offline-only")
    spec = binding()
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(reference(spec, mode)))
    model = Model(path, tmp_path, icl_mode=mode)
    model.live_calibration = authored(spec)
    value = images(spec, mode)
    model.update_native(state(), value, 0, "native task", 400)
    first = model.build_payload()
    assert model.payload_metrics(first)["image_count"] == 1
    if mode == "vision":
        assert "tactile" not in json.dumps(first).lower()
    else:
        assert "tactile_left" in json.dumps(first)
    model.pending = dict(
        start=0, predicted=10, returned=10, call_id="test_call", execution_note="hold"
    )
    command = dict(mode="ee", ee_pose=state()["ee_pose"], gripper=0.5)
    model.update_native(
        state(), value, 1, "native task", 400, command, {"arm": {"position_error_m": 0}}
    )
    model._commit_feedback()
    second = model.build_payload()
    assert model.payload_metrics(second)["image_count"] == 2
    assert all(
        json.loads(o["message"]["content"][0]["text"])["icl_mode"] == mode
        for o in model.archive.values()
    )


def test_mode_mismatch_rejected_before_api_or_sim(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_API_KEY", "offline-only")
    spec = binding()
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(reference(spec, "vision")))
    # The unspecified default is tactile and never silently downgrades.
    with pytest.raises(ValueError, match="ICL mode"):
        Model(path, tmp_path)
    args = SimpleNamespace(
        benchmark=spec["benchmark"],
        task=spec["task"],
        variant=None,
        reference=path,
        backend="live",
        seed=1000000,
    )
    with pytest.raises(ValueError, match="ICL mode"):
        preflight(args)
    assert not (tmp_path / "harness.json").exists()
    args.icl_mode = "vision"
    preflight(args)
    with pytest.raises(ValueError, match="ICL mode"):
        validate_reference_inputs({}, spec)


def test_training_and_live_serialization_match(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTRA_API_KEY", "offline-only")
    spec = binding("UniVTAC", "pull_out_key")
    values = images(spec, "vision_tactile")
    paths = {}
    for name, value in values.items():
        p = tmp_path / (name + ".png")
        Image.fromarray(value).save(p)
        paths[name] = str(p)
    data = dict(
        metadata=dict(
            binding=spec,
            status="success",
            seed=0,
            instruction="pull out the key.",
            native_dt=1 / 120,
            control_dt=1 / 60,
            action_repeat=2,
        ),
        observations=[dict(state=state(), images=paths, control_step=i) for i in range(81)],
        commands=[dict(ee_pose=state()["ee_pose"], gripper=0.5) for _ in range(80)],
    )
    source = tmp_path / "recording.json"
    source.write_text(json.dumps(data))
    selected = []
    for mode in ("vision", "vision_tactile"):
        out = tmp_path / mode
        out.mkdir()
        ref = prepare(source, out / "reference.json", icl_mode=mode)
        selected.append(ref["starts"])
        model = Model(out / "reference.json", out, icl_mode=mode)
        model.live_calibration = authored(spec)
        model.update_native(state(), images(spec, mode), 0, data["metadata"]["instruction"], 300)
        payload = model.build_payload()
        assert model.payload_metrics(payload)["image_count"] <= 26
        assert ref["input_contract"] == input_contract(spec, mode)
        raw = json.dumps(payload)
        assert ("tactile_left" in raw) == (mode == "vision_tactile")
    assert selected[0] == selected[1]


def test_recorded_tactile_uses_visual_frame_offset_and_rgb(tmp_path, monkeypatch):
    import h5py
    from tactalent.roboicl import prepare_recorded as module

    spec = binding("UniVTAC", "pull_out_key")
    n = 63
    source = tmp_path / "source.hdf5"
    with h5py.File(source, "w") as f:
        f["step"] = np.r_[0, np.arange(n - 1) * 2 + 1].astype(np.int64)
        f["embodiment/ee"] = np.tile(state()["ee_pose"], (n, 1))
        f["embodiment/joint"] = np.zeros((n, 9))
        visual = np.zeros((n, 270, 480, 3), np.uint8)
        f.create_dataset("observation/head/rgb", data=visual, compression="gzip")
        for side in ("left", "right"):
            frames = np.zeros((n, 240, 320, 3), np.uint8)
            frames[0] = [10, 20, 30]
            frames[1:] = [210, 40, 10] if side == "left" else [10, 40, 210]
            f.create_dataset("tactile/" + side + "_gsmini/rgb", data=frames, compression="gzip")
            f["tactile/" + side + "_gsmini/rgb_marker"] = np.zeros(1)  # Reading markers would fail.
    monkeypatch.setattr(
        module,
        "training_source",
        lambda *a, **kw: (source, dict(status="success", seed=0, instruction="pull out the key.")),
    )
    monkeypatch.setattr(module, "require_output_path", lambda p: Path(p))
    monkeypatch.setattr(
        module, "ArmKinematics", lambda p: SimpleNamespace(audit=lambda *a: {"passed": True})
    )
    out = tmp_path / "prepared"
    ref = module.convert(None, out, benchmark="UniVTAC", task="pull_out_key")
    record = json.loads((out / "recording.json").read_text())
    assert ref["icl_mode"] == "vision_tactile"
    assert len(record["observations"]) == n - 1
    first = record["observations"][0]
    assert first["source_frame"] == 1 and first["source_sim_step"] == 1
    for side, color in [("left", [210, 40, 10]), ("right", [10, 40, 210])]:
        image = np.asarray(Image.open(first["images"]["tactile_" + side]))
        assert image.shape == (240, 320, 3)
        np.testing.assert_allclose(image[100, 100], color, atol=2, rtol=0)
