"""Per-task direct ICL contracts; selection metadata never enters model observations."""

import hashlib
import json
from pathlib import Path
from .benchmarks import profile, validate_binding, horizon_for_task, task_context, task_prompt


def plan(spec):
    owner = profile(spec["benchmark"])
    coverage = list(owner.DEMONSTRATION_COVERAGE.get(spec["task"], ()))
    shots = len(coverage) or 1
    return dict(
        version="roboicl_covered_direct_demos_v1",
        shots=shots,
        coverage=coverage,
        chunks_per_demo=owner.FEW_SHOT_CHUNKS if shots > 1 else owner.ONE_SHOT_CHUNKS,
        live_anchor_count=owner.FEW_SHOT_LIVE_ANCHORS if shots > 1 else owner.ONE_SHOT_LIVE_ANCHORS,
    )


def descriptor(source, reference, coverage=None, coverage_basis=None):
    meta = json.loads(Path(source).read_text())["metadata"]
    return dict(
        source=str(Path(source).resolve()),
        source_sha256=reference["source_sha256"],
        source_hdf5=meta.get("source_hdf5"),
        source_hdf5_sha256=meta.get("source_hdf5_sha256"),
        binding=reference["binding"],
        split=meta.get("split", "train"),
        seed=meta.get("seed"),
        episode_id=meta.get("episode_id"),
        status=meta["status"],
        provenance=meta.get("provenance"),
        coverage=list(coverage or ()),
        coverage_basis=coverage_basis,
        instruction=reference["instruction"],
        task_prompt=reference["task_prompt"],
        horizon=reference["horizon"],
        frequency=reference["frequency"],
        action_repeat=reference["action_repeat"],
        starts=reference["starts"],
        selected_control_steps=reference["selected_control_steps"],
    )


def validate_reference_demonstrations(reference, spec):
    expected = plan(spec)
    if reference.get("demonstration_contract") != expected:
        raise ValueError("Reference few-shot contract changed; prepare the reference again")
    demos = reference.get("demonstrations", [])
    if len(demos) != expected["shots"]:
        raise ValueError("Reference demonstration count does not match the task")
    seen_sources, seen_identities, covered = set(), set(), set()
    for demo in demos:
        source_spec = validate_binding(demo["binding"])
        if (source_spec["benchmark"], source_spec["task"]) != (spec["benchmark"], spec["task"]):
            raise ValueError("Cannot mix benchmark tasks in a reference")
        if spec["task"] != "insert_block_v2" and source_spec != spec:
            raise ValueError("Demonstration variant mismatch")
        if demo.get("split") != "train" or demo.get("status") != "success":
            raise ValueError("Successful training demonstrations required")
        seed = demo.get("seed")
        if spec["benchmark"] == "UniVTAC" and seed is None:
            identity = demo.get("episode_id")
            if type(identity) is not int or not 0 <= identity < 50 or not demo.get("provenance"):
                raise ValueError("Published training-split provenance required")
        else:
            identity = seed
            if type(seed) is not int or not 0 <= seed < 1000000:
                raise ValueError("Training seeds must be disjoint from evaluation seeds")
        identity = (source_spec.get("variant"), identity)
        digest = demo.get("source_hdf5_sha256") or demo["source_sha256"]
        if digest in seen_sources or identity in seen_identities:
            raise ValueError("Repeated training demonstration")
        seen_sources.add(digest)
        seen_identities.add(identity)
        coverage = demo.get("coverage", [])
        if not isinstance(coverage, list) or any(c not in expected["coverage"] for c in coverage):
            raise ValueError("Unknown demonstration class/branch")
        if expected["shots"] > 1 and (not coverage or not demo.get("coverage_basis")):
            raise ValueError("Coverage requires auditable source evidence")
        if spec["task"] == "insert_block_v2" and source_spec["variant"] not in coverage:
            raise ValueError("Block demonstration goal/coverage mismatch")
        covered.update(coverage)
        starts = demo.get("starts", [])
        ticks = demo.get("selected_control_steps", [])
        if (
            demo.get("horizon") != horizon_for_task(spec)
            or len(starts) != expected["chunks_per_demo"]
            or len(set(starts)) != len(starts)
            or len(ticks) != len(starts)
            or demo.get("frequency") != 60
            or demo.get("action_repeat") != 2
        ):
            raise ValueError("Demonstration chunks/H/frequency mismatch")
        if demo.get("task_prompt") != task_prompt(source_spec, demo.get("instruction")):
            raise ValueError("Demonstration instruction mismatch")
    if covered != set(expected["coverage"]):
        raise ValueError("Missing demonstration class/branch coverage")
    return expected


def prepare_bundle(entries, spec, output, icl_mode=None):
    """Serialize each episode independently using upstream tool-reference messages."""
    from .reference import _prepare_episode
    from .calibration import message
    from roboicl.policy.trajectory_memory import omitted_interval

    contract = plan(spec)
    if len(entries) != contract["shots"] or contract["shots"] <= 1:
        raise ValueError("Expected the configured number of few-shot training recordings")
    refs, demos = [], []
    for entry in entries:
        source = Path(entry["recording"])
        ref = _prepare_episode(
            source,
            icl_mode=icl_mode,
            chunk_count=contract["chunks_per_demo"],
            include_end_marker=False,
        )
        refs.append(ref)
        demos.append(descriptor(source, ref, entry.get("coverage"), entry.get("coverage_basis")))
    # Choose the active goal from its own variant, never from another block's demo.
    active = next((r for r in refs if r["binding"] == spec), None)
    if active is None:
        raise ValueError("A demonstration of the active variant is required")
    result = dict(active)
    result.update(demonstration_contract=contract, demonstrations=demos)
    validate_reference_demonstrations(result, spec)
    messages = []
    for i, ref in enumerate(refs):
        if i:
            messages.append(omitted_interval("TRAIN", None, None, reason="episode_boundary"))
        # Do not reveal class/branch labels or hidden simulator metadata.
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"TRAIN DEMONSTRATION {i+1}/{len(refs)}\n" + ref["task_prompt"],
                    },
                    message(ref["training_calibration"], "TRAIN"),
                ],
            }
        )
        messages.extend(ref["messages"])
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "END OF FIXED TASK REFERENCE. Subsequent observations belong to the active episode.",
                }
            ],
        }
    )
    call_ids = [m["call_id"] for m in messages if m.get("type") == "function_call"]
    if len(call_ids) != contract["shots"] * contract["chunks_per_demo"] or len(
        set(call_ids)
    ) != len(call_ids):
        raise ValueError("Invalid or duplicated demonstration action calls")
    result.update(
        messages=messages,
        sources=[d["source"] for d in demos],
        demonstrations_sha256=hashlib.sha256(
            json.dumps(demos, sort_keys=True).encode()
        ).hexdigest(),
    )
    # source/starts remain the active-variant demo for legacy video viewers; all
    # sources and per-episode selections are explicitly carried in demonstrations.
    Path(output).write_text(json.dumps(result))
    return result
