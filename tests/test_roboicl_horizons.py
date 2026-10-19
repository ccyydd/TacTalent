"""Configured task horizons and rejection of stale/adaptive H choices."""

import json
from types import SimpleNamespace
import pytest
from tactalent.roboicl.benchmarks import binding, horizon_for_task
from tactalent.roboicl.reference import prepare, _prepare_episode
from tactalent.roboicl.policy import Model
from tactalent.roboicl.run import preflight
from tactalent.roboicl.upstream import protocol_sha256


@pytest.mark.parametrize("horizon", [5, 15, 4, None, 10.0, True])
def test_reference_horizon_cannot_override_task_rule(horizon, tmp_path, monkeypatch):
    spec = binding("OpenViTac", "turn_gear_pair")
    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps(
            dict(
                binding=spec,
                protocol_sha256=protocol_sha256(spec),
                horizon=horizon,
                frequency=60,
                action_repeat=2,
                messages=[],
            )
        )
    )
    args = SimpleNamespace(
        benchmark=spec["benchmark"],
        task=spec["task"],
        variant=None,
        reference=path,
        backend="live",
        seed=1000000,
    )
    with pytest.raises(ValueError, match="configured horizon mismatch"):
        preflight(args)
    monkeypatch.setenv("ASTRA_API_KEY", "offline-only")
    with pytest.raises(ValueError, match="configured horizon mismatch"):
        Model(path, tmp_path)
    assert not (tmp_path / "harness.json").exists()


def recording(tmp_path, commands=47):
    spec = binding("UniVTAC", "grasp_classify")
    state = dict(ee_pose=[0.4, 0, 0.2, 1, 0, 0, 0])
    data = dict(
        metadata=dict(binding=spec, status="success", seed=0),
        observations=[dict(state=state) for _ in range(commands + 1)],
        commands=[dict(ee_pose=state["ee_pose"], gripper=0.5) for _ in range(commands)],
    )
    path = tmp_path / "recording.json"
    path.write_text(json.dumps(data))
    return path


def test_short_demonstration_does_not_silently_shorten_horizon(tmp_path):
    path = recording(tmp_path)
    output = tmp_path / "reference.json"
    with pytest.raises(ValueError, match="Configured H=4.*cannot form J=12"):
        _prepare_episode(path, output)
    assert not output.exists()


def test_explicit_horizon_must_match_task_choice(tmp_path):
    path = recording(tmp_path)
    with pytest.raises(ValueError, match="configured horizon mismatch"):
        prepare(path, tmp_path / "reference.json", horizon=5)


def test_reviewed_task_horizons_and_independence_from_budget():
    from tactalent.roboicl.benchmarks import all_bindings

    ten = {
        ("UniVTAC", "lift_bottle"),
        ("UniVTAC", "put_bottle_in_shelf"),
        ("OpenViTac", "can_empty_select"),
        ("OpenViTac", "grasp_fragile_chip"),
        ("OpenViTac", "turn_gear_pair"),
        ("OpenViTac", "insert_block_v1"),
        ("OpenViTac", "insert_block_v2"),
    }
    for spec in all_bindings():
        key = (spec["benchmark"], spec["task"])
        expected = 10 if key in ten else 4 if key == ("UniVTAC", "grasp_classify") else 5
        assert horizon_for_task(spec) == expected
        assert horizon_for_task(dict(spec, native_action_limit=9999)) == expected
