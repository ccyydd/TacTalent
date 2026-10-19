"""Reviewed full-evaluation cases, separate from demonstration count and native horizon."""

import copy
import hashlib
import json
import os
from pathlib import Path
from .benchmarks import binding, profile
from .observations import resolve_mode

from tactalent.paths import workspace_root


def default_reference_catalog(icl_mode="vision_tactile"):
    explicit = os.environ.get("ASTRA_REFERENCE_CATALOG")
    if explicit:
        return Path(explicit).expanduser()
    storage = Path(os.environ.get("ASTRA_STORAGE_ROOT", str(workspace_root()))).expanduser()
    return storage / "references" / icl_mode / "catalog.json"


DEFAULT_REFERENCE_CATALOG = default_reference_catalog()
PROTOCOL = "balanced_full_evaluation_v1"


def cases(benchmark):
    owner = profile(benchmark)
    rows = copy.deepcopy(owner.FULL_EVALUATION_CASES)
    if {r["task"] for r in rows} != set(owner.TASKS):
        raise ValueError("Full evaluation must cover every task")
    ids = set()
    for row in rows:
        if row["id"] in ids:
            raise ValueError("Duplicate evaluation case")
        ids.add(row["id"])
        binding(benchmark, row["task"], row.get("variant"))
        if type(row["episodes"]) is not int or row["episodes"] <= 0:
            raise ValueError("Invalid case episode count")
    return rows


def case_for(spec, case_id):
    if case_id is None:
        return None
    row = next((r for r in cases(spec["benchmark"]) if r["id"] == case_id), None)
    if row is None or (row["task"], row.get("variant")) != (spec["task"], spec["variant"]):
        raise ValueError("Evaluation case does not match task/variant")
    return row


def validate_episode(spec, seed, case_id=None, reference=None):
    row = case_for(spec, case_id)
    if type(seed) is not int:
        raise ValueError("Evaluation seed must be an integer")
    if row is None:
        if seed < 1000000:
            raise ValueError("Ad-hoc evaluation seeds must be held out (>=1000000)")
    elif not row["start_seed"] <= seed < row["start_seed"] + row["episodes"]:
        raise ValueError("Seed is outside the reviewed evaluation case")
    # Official OpenViTac runs start at 10000. The previous ad-hoc seed convention
    # alone cannot prove train/test separation; check actual selected demos too.
    if reference and spec["benchmark"] == "OpenViTac":
        demos = reference.get("demonstrations", [])
        if any(d.get("seed") == seed for d in demos):
            raise ValueError("Evaluation seed overlaps a selected training demonstration")
    return row


def apply_case(cfg, spec, case_id):
    row = case_for(spec, case_id)
    if row is None:
        return None
    for key in row["conditions"]:
        if not hasattr(cfg, key):
            raise ValueError("Native task does not support evaluation condition: " + key)
    for key, value in row["conditions"].items():
        setattr(cfg, key, tuple(value) if key == "block_base_pose_indices" else value)
    return row


def plan(reference_catalog, benchmark=None, icl_mode="vision_tactile"):
    reference_catalog = Path(reference_catalog).resolve(strict=True)
    catalog = json.loads(reference_catalog.read_text())
    references = {}
    for row in catalog["episodes"]:
        spec = binding(row["benchmark"], row["task"], row.get("variant"))
        mode = resolve_mode(spec, row.get("icl_mode"))
        key = (spec["benchmark"], spec["task"], spec["variant"], mode)
        value = row.get("reference")
        if key in references and references[key] != value:
            raise ValueError("Conflicting reference paths in catalog")
        references[key] = value
    groups, jobs = {}, []
    selected = [benchmark] if benchmark else ["UniVTAC", "OpenViTac"]
    for bench in selected:
        groups[bench] = cases(bench)
        for case in groups[bench]:
            key = (bench, case["task"], case.get("variant"), icl_mode)
            if not references.get(key):
                raise ValueError("Missing task/mode reference: " + str(key))
            for index in range(case["episodes"]):
                jobs.append(
                    dict(
                        benchmark=bench,
                        task=case["task"],
                        variant=case.get("variant"),
                        evaluation_case=case["id"],
                        conditions=case["conditions"],
                        seed=case["start_seed"] + index,
                        episode_index_in_case=index,
                        icl_mode=icl_mode,
                        reference=references[key],
                    )
                )
    declared = dict(
        protocol=PROTOCOL,
        cases=groups,
        error_policy="record_and_exclude_from_valid_denominator; no_retry_or_seed_replacement",
        aggregation="per_task_then_unweighted_task_mean",
    )
    digest = hashlib.sha256(json.dumps(declared, sort_keys=True).encode()).hexdigest()
    return dict(
        **declared,
        plan_sha256=digest,
        reference_catalog=str(reference_catalog),
        reference_catalog_sha256=hashlib.sha256(reference_catalog.read_bytes()).hexdigest(),
        task_count=sum(len({c["task"] for c in cs}) for cs in groups.values()),
        episode_count=len(jobs),
        episodes=jobs,
    )


def summary(records, backend):
    def counts(rows):
        return {
            s: sum(r["result"]["status"] == s for r in rows)
            for s in ("success", "failed", "error", "smoke_passed")
        }

    tasks = []
    keys = sorted({(r["benchmark"], r["task"]) for r in records})
    for bench, task in keys:
        rows = [r for r in records if (r["benchmark"], r["task"]) == (bench, task)]
        n = counts(rows)
        valid = n["success"] + n["failed"]
        tasks.append(
            dict(
                benchmark=bench,
                task=task,
                requested=len(rows),
                counts=n,
                valid_success_rate=n["success"] / valid if backend == "live" and valid else None,
                cases=sorted({r.get("evaluation_case") or "ad_hoc" for r in rows}),
            )
        )
    out = dict(
        backend=backend,
        requested=len(records),
        counts=counts(records),
        tasks=tasks,
        episodes=records,
    )
    if backend == "live":
        rates = [t["valid_success_rate"] for t in tasks]
        # An all-error task must not disappear from the benchmark average.
        out["macro_success_rate"] = (
            sum(rates) / len(rates) if rates and all(r is not None for r in rates) else None
        )
        valid = out["counts"]["success"] + out["counts"]["failed"]
        out["valid_success_rate"] = out["counts"]["success"] / valid if valid else None
        out["raw_success_rate"] = out["counts"]["success"] / len(records) if records else None
        out["benchmarks"] = {}
        for bench in sorted({t["benchmark"] for t in tasks}):
            subset = [t for t in tasks if t["benchmark"] == bench]
            rs = [t["valid_success_rate"] for t in subset]
            out["benchmarks"][bench] = dict(
                task_count=len(subset),
                macro_success_rate=sum(rs) / len(rs) if all(r is not None for r in rs) else None,
                errors=sum(t["counts"]["error"] for t in subset),
            )
    return out
