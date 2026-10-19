"""Default full evaluation or an explicit small catalog; no seed replacement/retries."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path
from .benchmarks import binding
from .observations import resolve_mode, ICL_MODES
from .run import preflight
from .evaluation import default_reference_catalog, plan, validate_episode, summary
from .sampling import sample_plan, validate_sampling
from tactalent.utils.storage import require_output_path


def job_name(job):
    mode = resolve_mode(
        binding(job["benchmark"], job["task"], job.get("variant")), job.get("icl_mode")
    )
    return "_".join(
        str(v)
        for v in (
            job["benchmark"],
            job["task"],
            job.get("variant"),
            job.get("evaluation_case"),
            job.get("seed", 1000000),
            mode,
        )
        if v is not None
    )


def command(job, output):
    cmd = [
        sys.executable,
        "-m",
        "tactalent",
        "eval",
        "--benchmark",
        job["benchmark"],
        "--task",
        job["task"],
        "--seed",
        str(job.get("seed", 1000000)),
        "--output",
        str(output),
        "--icl-mode",
        resolve_mode(
            binding(job["benchmark"], job["task"], job.get("variant")), job.get("icl_mode")
        ),
    ]
    for field, flag in (
        ("variant", "--variant"),
        ("reference", "--reference"),
        ("evaluation_case", "--evaluation-case"),
    ):
        if job.get(field):
            cmd += [flag, job[field]]
    return cmd


def validate_catalog(catalog, backend):
    validate_sampling(catalog)
    jobs = catalog["episodes"]
    seen, cache = set(), {}
    for job in jobs:
        spec = binding(job["benchmark"], job["task"], job.get("variant"))
        mode = resolve_mode(spec, job.get("icl_mode"))
        case_id, seed = job.get("evaluation_case"), job.get("seed", 1000000)
        row = validate_episode(spec, seed, case_id)
        if "conditions" in job and job["conditions"] != (row["conditions"] if row else {}):
            raise ValueError("Catalog conditions differ from reviewed evaluation case")
        name = job_name(job)
        if name in seen:
            raise ValueError("Duplicate episode in catalog")
        seen.add(name)
        key = (spec["benchmark"], spec["task"], spec["variant"], mode, job.get("reference"))
        if key not in cache:
            _, ref = preflight(
                argparse.Namespace(
                    benchmark=spec["benchmark"],
                    task=spec["task"],
                    variant=spec["variant"],
                    reference=Path(job["reference"]) if job.get("reference") else None,
                    backend=backend,
                    seed=seed,
                    icl_mode=mode,
                    evaluation_case=case_id,
                )
            )
            cache[key] = ref
        validate_episode(spec, seed, case_id, cache[key])
    return dict(episodes=len(jobs), references=len(cache))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--catalog", type=Path, help="Explicit small/special episode catalog")
    source.add_argument(
        "--reference-catalog",
        type=Path,
        help="Defaults to ASTRA_REFERENCE_CATALOG or references/<icl-mode>/catalog.json",
    )
    p.add_argument("--benchmark", choices=["UniVTAC", "OpenViTac"], help="Full-plan filter")
    p.add_argument("--icl-mode", choices=ICL_MODES, default="vision_tactile")
    p.add_argument(
        "--sample-ratio",
        help="Stratified full-plan fraction, e.g. 0.1 or 10%%; default all episodes",
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        help="Reproducible subset seed; defaults to 42 with --sample-ratio",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", type=int, nargs="+")
    p.add_argument("--backend", choices=["live", "smoke"], default="live")
    p.add_argument(
        "--plan-only", action="store_true", help="Validate/save plan without simulator or API"
    )
    a = p.parse_args()
    if a.catalog and (a.benchmark or a.icl_mode != "vision_tactile"):
        p.error("For an explicit catalog, put benchmark and mode on each catalog row")
    if a.catalog and (a.sample_ratio is not None or a.sample_seed is not None):
        p.error("Sample the full plan, or reuse a saved sampled catalog without sampling flags")
    if a.sample_seed is not None and a.sample_ratio is None:
        p.error("--sample-seed requires --sample-ratio")
    if not a.plan_only and not a.gpus:
        p.error("--gpus is required to execute")
    if a.gpus and (len(set(a.gpus)) != len(a.gpus) or min(a.gpus) < 0):
        p.error("GPU IDs must be distinct nonnegative integers")
    if not a.catalog:
        a.reference_catalog = a.reference_catalog or default_reference_catalog(a.icl_mode)
        if not a.reference_catalog.is_file():
            p.error(
                f"Reference catalog not found: {a.reference_catalog}. "
                "Build it with python -m tactalent.roboicl.prepare_bank build, "
                "or supply --reference-catalog."
            )
    catalog = (
        json.loads(a.catalog.read_text())
        if a.catalog
        else plan(a.reference_catalog, a.benchmark, a.icl_mode)
    )
    if a.sample_ratio is not None:
        try:
            catalog = sample_plan(
                catalog, a.sample_ratio, 42 if a.sample_seed is None else a.sample_seed
            )
        except ValueError as exc:
            p.error(str(exc))
    validation = validate_catalog(catalog, a.backend)
    if catalog.get("sampling"):
        validation["sampling"] = {
            k: catalog["sampling"][k]
            for k in ("fraction", "seed", "population_episode_count", "selected_episode_count")
        }
    root = require_output_path(a.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "catalog.snapshot.json").write_text(json.dumps(catalog, indent=2))
    (root / "validation.json").write_text(json.dumps(validation, indent=2))
    if a.plan_only:
        print(json.dumps(dict(**validation, executed_episodes=0, api_calls=0, output=str(root))))
        return
    jobs = catalog["episodes"]

    def worker(gpu, selected):
        records = []
        for job in selected:
            name = job_name(job)
            output = root / name
            cmd = command(job, output) + ["--backend", a.backend]
            with (root / (name + ".log")).open("w") as log:
                proc = subprocess.run(
                    cmd, env=dict(os.environ, GPU_ID=str(gpu)), stdout=log, stderr=subprocess.STDOUT
                )
            result_file = output / "result.json"
            result = (
                json.loads(result_file.read_text())
                if result_file.exists()
                else {
                    "status": "error",
                    "termination": "launcher_error",
                    "exit_code": proc.returncode,
                }
            )
            record = dict(
                benchmark=job["benchmark"],
                task=job["task"],
                variant=job.get("variant"),
                seed=job.get("seed", 1000000),
                icl_mode=job.get("icl_mode", "vision_tactile"),
                evaluation_case=job.get("evaluation_case"),
                conditions=job.get("conditions", {}),
                output=str(output),
                result=result,
            )
            records.append(record)
            (root / ("gpu_" + str(gpu) + ".json")).write_text(json.dumps(records, indent=2))
            print(
                json.dumps(dict(task=job["task"], seed=record["seed"], status=result["status"])),
                flush=True,
            )
        return records

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
        futures = [pool.submit(worker, gpu, jobs[i :: len(a.gpus)]) for i, gpu in enumerate(a.gpus)]
        records = [r for future in futures for r in future.result()]
    result = summary(records, a.backend)
    result["plan_sha256"] = catalog.get("plan_sha256")
    result["sampling"] = catalog.get("sampling")
    (root / "summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
