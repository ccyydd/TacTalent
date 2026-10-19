"""Provenance of the official UniVTAC training collector."""

import ast
import hashlib
from pathlib import Path


def collector_evidence(native_root):
    """Record a current collector contract, without claiming historical replay."""
    path = Path(native_root) / "scripts/collect_data.py"
    raw = path.read_bytes()
    tree = ast.parse(raw)
    expected = ast.dump(
        ast.parse(
            "task.plan_success and task.check_success() and not task.check_early_stop()",
            mode="eval",
        ).body
    )
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and ast.dump(node.test) == expected
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "task"
            and child.func.attr == "save_to_hdf5"
            for statement in node.body
            for child in ast.walk(statement)
        )
    ]
    if not matches:
        raise ValueError("Collector no longer saves under the audited success contract")
    return {
        "status": "current_collector_contract_only",
        "collector_path": str(path),
        "collector_sha256": hashlib.sha256(raw).hexdigest(),
        "line_scope": str(matches[0].lineno),
        "independently_replayed": False,
        "private_values_in_model_artifact": False,
        "limitation": "No historical success flag, collector revision, timestamps or RNG "
        "seed is stored. Current collection code saves successful episodes; "
        "this is provenance evidence, not an independent historical success check.",
    }
