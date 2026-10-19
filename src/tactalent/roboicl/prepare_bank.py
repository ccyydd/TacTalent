"""Prepare recorded demonstrations and validated reference catalogs without model calls."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from tactalent.benchmarks.univtac.paths import source_config as configured_source_config
from tactalent.utils.storage import require_output_path
from .benchmarks import binding, profile
from .demonstrations import plan
from .evaluation import default_reference_catalog
from .observations import ICL_MODES

SCHEMA = "tactalent_reference_sources_v1"


def write_new(path, value):
    path = require_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path


def training_config(data_root, output):
    """Create an explicit 0..49 source declaration; hashes remain checked on import."""
    root = Path(data_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("UniVTAC data root must be a directory")
    manifest = json.loads((root / "download_manifest.json").read_text())
    if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError(
            "UniVTAC data requires a verified download_manifest.json with file entries"
        )
    # JSON is also valid YAML and can be read by the existing source validator.
    return write_new(
        output, {"data": {"train_data_paths": str(root), "episode_start": 0, "episode_end": 49}}
    )


def source_template(output, benchmark=None):
    rows = []
    for bench in [benchmark] if benchmark else ["UniVTAC", "OpenViTac"]:
        owner = profile(bench)
        for task in owner.TASKS:
            row = {"benchmark": bench, "task": task}
            if bench == "UniVTAC" and not owner.DEMONSTRATION_COVERAGE.get(task):
                row["training_episode"] = 0
            else:
                row["training_ledger"] = f"training_ledgers/{bench}/{task}.json"
            rows.append(row)
    return write_new(output, {"schema_version": SCHEMA, "tasks": rows})


def input_path(value, parent):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else parent / path).resolve(strict=True)


def source_jobs(sources, benchmark=None):
    sources = Path(sources).expanduser().resolve(strict=True)
    data = json.loads(sources.read_text())
    if data.get("schema_version") != SCHEMA or not isinstance(data.get("tasks"), list):
        raise ValueError("Expected a tactalent_reference_sources_v1 tasks manifest")
    jobs, seen = [], set()
    allowed = {
        "benchmark",
        "task",
        "variant",
        "training_episode",
        "training_ledger",
        "source_config",
    }
    for row in data["tasks"]:
        if not isinstance(row, dict) or set(row) - allowed:
            raise ValueError("Unknown source-manifest fields")
        bench, task = row.get("benchmark"), row.get("task")
        owner = profile(bench)
        if benchmark and bench != benchmark:
            continue
        variants = (
            owner.VARIANTS
            if task == "insert_block_v2" and row.get("variant") is None
            else (row.get("variant"),)
        )
        for variant in variants:
            spec = binding(bench, task, variant)
            key = (bench, task, variant)
            if key in seen:
                raise ValueError(f"Repeated source task/variant: {key}")
            seen.add(key)
            contract = plan(spec)
            job = {"spec": spec, "source_config": None, "training_ledger": None}
            if row.get("source_config"):
                if bench != "UniVTAC":
                    raise ValueError("source_config applies only to UniVTAC")
                job["source_config"] = input_path(row["source_config"], sources.parent)
            if contract["shots"] > 1 or bench == "OpenViTac":
                if "training_episode" in row or not row.get("training_ledger"):
                    raise ValueError(f"{bench}/{task} requires an explicit training_ledger")
                path = input_path(row["training_ledger"], sources.parent)
                job["training_ledger"] = path
                ledger = json.loads(path.read_text())
                if contract["shots"] > 1:
                    from .prepare_few_shot import select

                    selected = select(ledger, spec)
                else:
                    from tactalent.benchmarks.openvitac.recordings import ledger as validate_ledger

                    selected = validate_ledger(path, spec)["episodes"][:1]
                for source in selected:
                    # The existing builders verify identities, coverage and file hashes.
                    for field in ("path", "recording"):
                        if source.get(field):
                            candidate = Path(source[field]).expanduser()
                            if not candidate.is_absolute() or not candidate.is_file():
                                raise ValueError(
                                    f"Training ledger {field} is not an existing absolute file: {source[field]}"
                                )
                needs_config = bench == "UniVTAC" and any(
                    not source.get("recording") for source in selected
                )
            else:
                if "training_ledger" in row:
                    raise ValueError("UniVTAC one-shot tasks use training_episode")
                episode = row.get("training_episode")
                if type(episode) is not int or not 0 <= episode < 50:
                    raise ValueError(
                        "UniVTAC training_episode must be explicitly selected from 0..49"
                    )
                job["training_episode"] = episode
                needs_config = True
            if needs_config:
                config = job["source_config"] or configured_source_config()
                from tactalent.data.kinematics import read_source_config

                read_source_config(config)
                job["source_config"] = Path(config).resolve(strict=True)
            jobs.append(job)
    if not jobs:
        raise ValueError("Source manifest selects no tasks")
    return sources, jobs


def checked_catalog(references, mode, output):
    from .run import preflight

    rows, seen = [], set()
    for path in sorted(Path(references).expanduser().resolve(strict=True).rglob("reference.json")):
        reference = json.loads(path.read_text())
        spec = reference["binding"]
        key = (spec["benchmark"], spec["task"], spec.get("variant"))
        if key in seen:
            raise ValueError(f"Repeated reference task/variant: {key}")
        seen.add(key)
        preflight(
            SimpleNamespace(
                benchmark=key[0],
                task=key[1],
                variant=key[2],
                reference=path,
                backend="live",
                seed=1000000,
                icl_mode=mode,
                evaluation_case=None,
            )
        )
        rows.append(
            dict(benchmark=key[0], task=key[1], variant=key[2], icl_mode=mode, reference=str(path))
        )
    if not rows:
        raise ValueError("No prepared reference.json files found")
    write_new(output, {"schema_version": "tactalent_reference_catalog_v1", "episodes": rows})
    return rows


def build(sources, mode, output, benchmark=None, check_only=False, catalog_name="catalog.json"):
    if Path(catalog_name).name != catalog_name:
        raise ValueError("Catalog name must be a filename")
    sources, jobs = source_jobs(sources, benchmark)
    if check_only:
        return {
            "tasks_and_variants": len(jobs),
            "icl_mode": mode,
            "api_calls": 0,
            "validated": "source plan, coverage and input paths",
        }
    output = require_output_path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_new(
        output / "preparation.json",
        {
            "source_manifest": str(sources),
            "source_manifest_sha256": hashlib.sha256(sources.read_bytes()).hexdigest(),
            "icl_mode": mode,
            "benchmark": benchmark,
            "api_calls": 0,
        },
    )
    from .prepare_recorded import convert
    from .prepare_few_shot import prepare as prepare_few_shot

    for job in jobs:
        spec = job["spec"]
        destination = output / spec["benchmark"] / spec["task"] / (spec["variant"] or "default")
        common = dict(
            benchmark=spec["benchmark"],
            task=spec["task"],
            variant=spec["variant"],
            icl_mode=mode,
            source_config=job["source_config"],
        )
        if plan(spec)["shots"] > 1:
            prepare_few_shot(job["training_ledger"], destination, **common)
        else:
            convert(
                job["training_ledger"],
                destination,
                episode_id=job.get("training_episode", 0),
                **common,
            )
    catalog = output / catalog_name
    rows = checked_catalog(output, mode, catalog)
    return {"catalog": str(catalog), "references": len(rows), "icl_mode": mode, "api_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser(
        "training-config", help="Declare the verified UniVTAC 0..49 training source"
    )
    config.add_argument("--data-root", type=Path, required=True)
    config.add_argument("--output", type=Path, default=configured_source_config())
    template = commands.add_parser(
        "template", help="Write the source-selection manifest for the benchmark tasks"
    )
    template.add_argument("--benchmark", choices=["UniVTAC", "OpenViTac"])
    template.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser(
        "build", help="Convert one-shot/few-shot sources and validate the catalog"
    )
    prepare.add_argument("--sources", type=Path, required=True)
    prepare.add_argument("--benchmark", choices=["UniVTAC", "OpenViTac"])
    prepare.add_argument("--icl-mode", choices=ICL_MODES, default="vision_tactile")
    prepare.add_argument("--output", type=Path)
    prepare.add_argument(
        "--check-only",
        action="store_true",
        help="Check the source plan and input paths without converting images",
    )
    catalog = commands.add_parser(
        "catalog", help="Validate existing references and create a catalog"
    )
    catalog.add_argument("--references", type=Path, required=True)
    catalog.add_argument("--icl-mode", choices=ICL_MODES, default="vision_tactile")
    catalog.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "training-config":
            result = {"source_config": str(training_config(args.data_root, args.output))}
        elif args.command == "template":
            result = {"sources": str(source_template(args.output, args.benchmark))}
        elif args.command == "catalog":
            output = args.output or default_reference_catalog(args.icl_mode)
            rows = checked_catalog(args.references, args.icl_mode, output)
            result = {"catalog": str(output), "references": len(rows), "api_calls": 0}
        else:
            output = args.output or default_reference_catalog(args.icl_mode).parent
            catalog_name = (
                "catalog.json" if args.output else default_reference_catalog(args.icl_mode).name
            )
            result = build(
                args.sources, args.icl_mode, output, args.benchmark, args.check_only, catalog_name
            )
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
