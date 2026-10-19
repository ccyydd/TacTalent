"""Task specification is benchmark context, with matched TRAIN and LIVE wording."""

import copy
import json
from types import SimpleNamespace
import pytest
from tactalent.roboicl.benchmarks import binding, task_context, task_prompt, horizon_for_task
from tactalent.roboicl.policy import Model
from tactalent.roboicl.run import preflight
from tactalent.roboicl.upstream import protocol_sha256


@pytest.mark.parametrize("change", ["missing_context", "changed_background", "changed_prompt"])
def test_context_guard_rejects_mismatched_reference_before_api(change, tmp_path, monkeypatch):
    spec = binding("OpenViTac", "turn_gear_pair")
    ref = dict(
        binding=spec,
        protocol_sha256=protocol_sha256(spec),
        horizon=horizon_for_task(spec),
        frequency=60,
        action_repeat=2,
        instruction="Assemble and turn the red gear.",
        messages=[],
        task_context=task_context(spec),
        task_prompt=task_prompt(spec, "Assemble and turn the red gear."),
    )
    ref = copy.deepcopy(ref)
    if change == "missing_context":
        ref.pop("task_context")
    elif change == "changed_background":
        ref["task_context"]["background"] = "Older task background."
    else:
        ref["task_prompt"] = "Older task prompt."
    path = tmp_path / "ref.json"
    path.write_text(json.dumps(ref))
    args = SimpleNamespace(
        benchmark=spec["benchmark"],
        task=spec["task"],
        variant=None,
        reference=path,
        backend="live",
        seed=1000000,
    )
    with pytest.raises(ValueError, match="task context mismatch"):
        preflight(args)
    monkeypatch.setenv("ASTRA_API_KEY", "offline-only")
    with pytest.raises(ValueError, match="task context mismatch"):
        Model(path, tmp_path)
    assert not (tmp_path / "harness.json").exists()


@pytest.mark.parametrize(
    "variant,goal",
    [
        ("cube", "Insert the blue cube into its matching opening."),
        ("half_cylinder", "Insert the blue half cylinder into its matching opening."),
        ("hexagon", "Insert the red hexagonal prism into its matching opening."),
    ],
)
def test_openvitac_preserves_native_variant_goal(variant, goal):
    spec = binding("OpenViTac", "insert_block_v2", variant)
    assert task_prompt(spec, goal).startswith("TASK:\n" + goal + "\n")
    assert task_context(spec)["variant"] == variant


def test_chip_empty_instruction_uses_declared_goal():
    spec = binding("OpenViTac", "grasp_fragile_chip")
    assert task_prompt(spec, "").startswith(
        "TASK:\nGrasp the fragile chip and place it in the target tray intact."
    )


def test_univtac_keeps_reviewed_success_goal_and_initial_condition():
    spec = binding("UniVTAC", "lift_bottle")
    text = task_prompt(spec, "Lift the bottle.")
    assert "place it upright on the support surface near the wall" in text
    assert "The native setup places the open gripper near the bottle." in text
    key = task_prompt(binding("UniVTAC", "pull_out_key"), "pull out the key.")
    assert "required turning angle therefore varies between episodes" in key
    assert "already grasps the key while it is in the lock" in key


def test_context_contains_only_reviewed_public_task_fields():
    from tactalent.roboicl.benchmarks import all_bindings

    allowed = {
        "version",
        "benchmark",
        "task",
        "variant",
        "goal",
        "background",
        "starting_condition",
        "source",
    }
    for spec in all_bindings():
        context = task_context(spec)
        assert context.keys() <= allowed
        assert context["source"]["source_branch"] == "research/on-demand-retrieval"
        assert context["source"]["source_commit"].startswith("5a33eca")
        assert len(task_prompt(spec, "Native task goal.").split()) < 120


@pytest.mark.parametrize("instruction", ["", "Place the chip in the tray."])
def test_chip_task_description_contains_goal_without_scoring_details(instruction):
    spec = binding("OpenViTac", "grasp_fragile_chip")
    text = task_prompt(spec, instruction)
    assert "Grasp and lift" in text
    assert "keeping it intact" in text
    assert "transfer it to the target tray and place it stably inside" in text
    assert "gripper release and withdrawal" not in text
    assert "Support may form after release" not in text
    assert "Nonfracture manipulation errors allow recovery" not in text
    assert "releasing earlier fails" not in text
    assert "ends the episode even before contact" not in text
