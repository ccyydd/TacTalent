"""Build direct few-shot references from a covered, successful training ledger.

No model call, summary generation, retrieval, or evaluation-seed selection.
"""

import argparse
import hashlib
import json
from pathlib import Path
from .benchmarks import binding
from .demonstrations import plan, prepare_bundle
from .observations import ICL_MODES, resolve_mode
from .prepare_recorded import convert
from tactalent.utils.storage import require_output_path


def select(ledger, spec):
    if (ledger.get("benchmark"), ledger.get("task"), ledger.get("split")) != (
        spec["benchmark"],
        spec["task"],
        "train",
    ):
        raise ValueError("Training ledger benchmark/task/split mismatch")
    contract = plan(spec)
    if contract["shots"] <= 1:
        raise ValueError("This task remains one-shot; use prepare_recorded")
    rows = ledger.get("episodes", [])
    if len(rows) > 1000:
        raise ValueError("Narrow the explicit training pool")
    valid, identities = [], set()
    for row in rows:
        variant = row.get("variant", ledger.get("variant"))
        source_spec = binding(spec["benchmark"], spec["task"], variant)
        identity = row.get("episode_id") if spec["benchmark"] == "UniVTAC" else row.get("seed")
        upper = 50 if spec["benchmark"] == "UniVTAC" else 1000000
        if type(identity) is not int or not 0 <= identity < upper:
            raise ValueError("Invalid training identity; evaluation data is forbidden")
        key = (source_spec["variant"], identity)
        if key in identities:
            raise ValueError("Repeated training identity")
        identities.add(key)
        if row.get("success") is not True:
            continue
        coverage = row.get("coverage")
        if (
            not isinstance(coverage, list)
            or not coverage
            or any(c not in contract["coverage"] for c in coverage)
            or len(coverage) != len(set(coverage))
            or not row.get("coverage_basis")
        ):
            raise ValueError("Explicit, evidenced class/branch coverage required")
        if spec["task"] == "insert_block_v2" and source_spec["variant"] not in coverage:
            raise ValueError("Block variant/coverage mismatch")
        valid.append(dict(row, variant=variant))
    states = {(0, frozenset()): ()}
    for i, row in enumerate(valid):
        for (count, covered), indices in list(states.items()):
            if count >= contract["shots"]:
                continue
            key = count + 1, covered | frozenset(row["coverage"])
            candidate = indices + (i,)
            if key not in states or candidate < states[key]:
                states[key] = candidate
    chosen = states.get((contract["shots"], frozenset(contract["coverage"])))
    if chosen is None:
        missing = set(contract["coverage"]) - {c for row in valid for c in row["coverage"]}
        raise ValueError(
            f"Need {contract['shots']} distinct successful demos; missing coverage: {sorted(missing)}"
        )
    return [valid[i] for i in chosen]


def prepare(
    ledger_path, output, *, benchmark, task, variant=None, icl_mode=None, source_config=None
):
    spec = binding(benchmark, task, variant)
    mode = resolve_mode(spec, icl_mode)
    ledger_path = require_output_path(ledger_path).resolve(strict=True)
    data = json.loads(ledger_path.read_text())
    rows = select(data, spec)
    output = require_output_path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for i, row in enumerate(rows):
        source_spec = binding(benchmark, task, row["variant"])
        basis = row["coverage_basis"]
        if isinstance(basis, dict) and basis.get("path"):
            basis_path = require_output_path(basis["path"]).resolve(strict=True)
            if (
                basis.get("sha256")
                and hashlib.sha256(basis_path.read_bytes()).hexdigest() != basis["sha256"]
            ):
                raise ValueError("Coverage evidence changed")
        if row.get("recording"):
            recording = require_output_path(row["recording"]).resolve(strict=True)
            meta = json.loads(recording.read_text())["metadata"]
            if meta.get("binding") != source_spec or meta.get("status") != "success":
                raise ValueError("Cached recording task/status mismatch")
            key = "episode_id" if benchmark == "UniVTAC" else "seed"
            if meta.get(key) != row[key]:
                raise ValueError("Cached recording training identity mismatch")
            if meta.get("split") != "train" or not meta.get("provenance"):
                raise ValueError("Cached recording training provenance missing")
            hdf5 = require_output_path(meta["source_hdf5"]).resolve(strict=True)
            digest = hashlib.sha256(hdf5.read_bytes()).hexdigest()
            if digest != meta["source_hdf5_sha256"] or (
                row.get("sha256") and digest != row["sha256"]
            ):
                raise ValueError("Cached recording training file changed")
        else:
            manifest = None
            if benchmark == "OpenViTac":
                # Existing native ledger validation enforces source hashes and successful train split.
                single = dict(
                    data,
                    variant=row["variant"],
                    episodes=[row],
                    instruction=row.get("instruction", data.get("instruction", "")),
                )
                if task == "insert_block_v2" and not row.get("instruction"):
                    raise ValueError("Each block variant needs its own recorded instruction")
                manifest = output / f"native_ledger_{i}.json"
                manifest.write_text(json.dumps(single, indent=2))
            recording = convert(
                manifest,
                output / f"demo_{i}",
                benchmark=benchmark,
                task=task,
                variant=row["variant"],
                episode_id=row.get("episode_id", 0),
                icl_mode=mode,
                record_only=True,
                source_config=source_config,
            )
        entries.append(
            dict(recording=str(recording), coverage=row["coverage"], coverage_basis=basis)
        )
    result = prepare_bundle(entries, spec, output / "reference.json", mode)
    (output / "preparation.json").write_text(
        json.dumps(
            dict(
                training_ledger=str(ledger_path),
                ledger_sha256=hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
                selected=rows,
                contract=result["demonstration_contract"],
                icl_mode=mode,
                api_calls=0,
            ),
            indent=2,
        )
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", required=True, choices=["UniVTAC", "OpenViTac"])
    p.add_argument("--task", required=True)
    p.add_argument("--variant")
    p.add_argument("--training-ledger", required=True)
    p.add_argument(
        "--source-config",
        type=Path,
        help="UniVTAC training-source YAML; defaults to ASTRA_UNIVTAC_SOURCE_CONFIG",
    )
    p.add_argument("--icl-mode", choices=ICL_MODES)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    result = prepare(
        a.training_ledger,
        a.output,
        benchmark=a.benchmark,
        task=a.task,
        variant=a.variant,
        icl_mode=a.icl_mode,
        source_config=a.source_config,
    )
    print(
        json.dumps(
            dict(
                binding=result["binding"],
                icl_mode=result["icl_mode"],
                contract=result["demonstration_contract"],
                api_calls=0,
            )
        )
    )


if __name__ == "__main__":
    main()
