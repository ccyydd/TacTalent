"""Explicit task routing. Importing this module never imports a simulator."""

import importlib
from pathlib import Path


def profile(benchmark):
    modules = {"UniVTAC": "univtac", "OpenViTac": "openvitac"}
    if benchmark not in modules:
        raise ValueError("Benchmark must be UniVTAC or OpenViTac")
    return importlib.import_module("tactalent.benchmarks." + modules[benchmark] + ".profile")


def binding(benchmark="OpenViTac", task="turn_gear_pair", variant=None):
    return profile(benchmark).binding(task, variant)


def all_bindings():
    rows = []
    for benchmark in ("UniVTAC", "OpenViTac"):
        p = profile(benchmark)
        for task in p.TASKS:
            variants = p.VARIANTS if task == "insert_block_v2" else (None,)
            rows.extend(binding(benchmark, task, v) for v in variants)
    return rows


def native_source_paths(spec):
    root = Path(spec["root"])
    return [
        root / "envs" / (spec["native_task"] + ".py"),
        root / "envs/_base_task.py",
        root / "envs/robot/robot.py",
        root / "task_config" / (spec["task_config"] + ".yml"),
        root / "policy/task_settings.json",
    ]


def validate_binding(spec):
    current = binding(spec["benchmark"], spec["task"], spec.get("variant"))
    if spec != current:
        raise ValueError("Reference task/camera/native-budget declaration changed; prepare again")
    return current


def horizon_for_task(spec):
    """Reviewed per-task H; shared by demonstration chunks, prediction and execution."""
    horizon = profile(spec["benchmark"]).HORIZONS[spec["task"]]
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("Configured task horizon must be a positive integer")
    return horizon


def validate_reference_horizon(reference, spec):
    expected = horizon_for_task(spec)
    if type(reference.get("horizon")) is not int or reference["horizon"] != expected:
        raise ValueError(
            f"RoboICL configured horizon mismatch: {spec['benchmark']}/{spec['task']} "
            f"requires H={expected}; prepare the reference again"
        )
    return expected


def task_context(spec):
    """Frozen public benchmark context, independent of the RoboICL method."""
    owner = profile(spec["benchmark"])
    return dict(
        version="benchmark_task_context_v1",
        benchmark=spec["benchmark"],
        task=spec["task"],
        variant=spec["variant"],
        **owner.TASK_CONTEXTS[spec["task"]],
        source=dict(owner.TASK_CONTEXT_SOURCE),
    )


def task_prompt(spec, native_instruction):
    context = task_context(spec)
    native = str(native_instruction or "").strip()
    fallback = getattr(profile(spec["benchmark"]), "INSTRUCTION_FALLBACKS", {}).get(
        spec["task"], ""
    )
    goal = context.get("goal") or native or fallback
    if not goal:
        raise ValueError("No public task goal is available")
    text = "TASK:\n" + goal
    if context.get("background"):
        text += "\nTASK BACKGROUND\n" + context["background"]
    if context.get("starting_condition"):
        text += "\nSTARTING CONDITION\n" + context["starting_condition"]
    return text


def validate_reference_task_context(reference, spec):
    expected = task_context(spec)
    if reference.get("task_context") != expected or reference.get("task_prompt") != task_prompt(
        spec, reference.get("instruction")
    ):
        raise ValueError("Reference benchmark task context mismatch; prepare the reference again")
    return expected
